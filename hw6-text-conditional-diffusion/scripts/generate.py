"""Generate the final 2000 images per data/generate.csv.

  python scripts/generate.py --ckpt outputs/run_edm59m/best.pt \
      --out generated_images --steps 30 --guidance 2.0 [--oversample 3]

--oversample K: generate K*20 candidates per pair, keep the 20 with the highest
CLIP-T to their own prompt (boosts the binding CLIP-T metric). K=1 disables it.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, load_cache  # noqa: E402
from dataset import load_vocab, prompt_of  # noqa: E402
from diffusion import EDM  # noqa: E402
from model import UNet, EDMPrecond  # noqa: E402
from sample_utils import sample_pairs, save_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_generate_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


@torch.no_grad()
def clip_select(imgs_uint8, prompt, device, keep):
    """Return indices of the `keep` images with highest CLIP-T to prompt."""
    import open_clip
    if not hasattr(clip_select, "_m"):
        m, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai")
        clip_select._m = m.to(device).eval()
        clip_select._pre = pre
        clip_select._tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    m, pre, tok = clip_select._m, clip_select._pre, clip_select._tok
    from PIL import Image
    ims = torch.stack([pre(Image.fromarray(im.permute(1, 2, 0).numpy()))
                       for im in imgs_uint8]).to(device)
    te = m.encode_text(tok([prompt]).to(device))
    ie = m.encode_image(ims)
    ie = ie / ie.norm(dim=-1, keepdim=True)
    te = te / te.norm(dim=-1, keepdim=True)
    score = (ie @ te.T).squeeze(1)
    return score.topk(min(keep, len(score))).indices.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="generated_images")
    ap.add_argument("--csv", default=os.path.join(ROOT, "data/generate.csv"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_weights", choices=["ema", "model"], default="ema")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = load_vocab()
    a2i, o2i = vocab["a2i"], vocab["o2i"]

    ck = torch.load(args.ckpt, map_location=device)
    if "model_cfg" in ck:
        mc = dict(ck["model_cfg"])
        mc["ch_mult"] = tuple(mc["ch_mult"])
        model = EDMPrecond(UNet(**mc)).to(device)
    else:  # legacy fallback
        cargs = ck.get("args", {})
        model = EDMPrecond(UNet(
            base=cargs.get("base", 128),
            ch_mult=tuple(cargs.get("ch_mult", [1, 2, 2, 2])),
            num_res_blocks=cargs.get("num_res_blocks", 3),
            dropout=cargs.get("dropout", 0.1),
            use_cross_attn=cargs.get("use_cross", True))).to(device)
    key = "ema" if args.use_weights == "ema" and "ema" in ck else "model"
    model.load_state_dict(ck[key])
    model.eval()
    print(f"[gen] loaded {key} weights from {args.ckpt} (it={ck.get('it')})")

    bank = ConditionBank(load_cache(device), device)
    edm = EDM()

    rows = read_generate_csv(args.csv)
    by_pair = defaultdict(list)            # (a,o) -> list of target ids in order
    for r in rows:
        by_pair[(r["animal"], r["object"])].append(r["id"])

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    out_dir = os.path.join(ROOT, args.out)
    total = 0
    for (animal, obj), ids in by_pair.items():
        a, o = a2i[animal], o2i[obj]
        need = len(ids)
        n_cand = need * max(1, args.oversample)
        pl = [(a, o)] * n_cand
        imgs = sample_pairs(model, edm, bank, pl, steps=args.steps,
                            guidance=args.guidance, device=device, batch=args.batch)
        if args.oversample > 1:
            idx = clip_select(imgs, prompt_of(animal, obj), device, need)
            imgs = imgs[idx]
        else:
            imgs = imgs[:need]
        save_images(imgs, ids, out_dir)
        total += len(ids)
        print(f"  {animal}+{obj}: {len(ids)} imgs (cand {n_cand})", flush=True)
    print(f"[gen] wrote {total} images -> {out_dir}")


if __name__ == "__main__":
    main()
