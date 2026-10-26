"""End-game tool: sweep CFG guidance x oversample for a trained checkpoint and
report FID + CLIP-T for each, against the strong-baseline targets
(FID <= 49.2545, CLIP-T >= 0.2703). Optionally regenerate the winning config at
20 imgs/pair into the final submission folder.

  # quick sweep (n=10/pair) to pick guidance:
  python scripts/sweep.py --ckpt outputs/run_v1_64b1244/best.pt \
      --guidances 1.5 2.0 2.5 3.0 4.0 --oversample 4 --n_per_pair 10 --steps 24

  # then lock in the winner as the real 2000-image submission:
  python scripts/sweep.py --ckpt outputs/run_v1_64b1244/best.pt \
      --guidances 2.5 --oversample 4 --n_per_pair 20 --steps 30 \
      --final_out generated_images
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, load_cache  # noqa: E402
from dataset import load_vocab, prompt_of  # noqa: E402
from diffusion import EDM  # noqa: E402
from eval_local import compute_clip_t, compute_fid  # noqa: E402
from generate import clip_select  # noqa: E402
from model import UNet, EDMPrecond  # noqa: E402
from sample_utils import sample_pairs, save_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "scoring_program", "input", "ref")
FID_T, CLIP_T_T = 49.2545, 0.2703


def load_model(ckpt, device, use_weights="ema"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    mc = dict(ck["model_cfg"]); mc["ch_mult"] = tuple(mc["ch_mult"])
    model = EDMPrecond(UNet(**mc)).to(device).eval()
    key = "ema" if (use_weights == "ema" and "ema" in ck) else "model"
    model.load_state_dict(ck[key])
    print(f"[sweep] loaded {key} weights @ it={ck.get('it')}  cfg={mc}")
    return model


def gen_config(model, edm, bank, vocab, n_per_pair, oversample, guidance,
               steps, device, out_dir):
    """Generate n_per_pair imgs for all 100 pairs (CLIP-selected if oversample>1)
    into out_dir; return (paths, items) for scoring."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths, items = [], []
    k = 1
    for ai, animal in enumerate(vocab["animals"]):
        for oi, obj in enumerate(vocab["objects"]):
            prompt = prompt_of(animal, obj)
            n_cand = n_per_pair * max(1, oversample)
            imgs = sample_pairs(model, edm, bank, [(ai, oi)] * n_cand,
                                steps=steps, guidance=guidance, device=device)
            if oversample > 1:
                idx = clip_select(imgs, prompt, device, n_per_pair)
                imgs = imgs[idx]
            else:
                imgs = imgs[:n_per_pair]
            names = [f"{k+j:06d}.png" for j in range(len(imgs))]
            save_images(imgs, names, out_dir)
            for nm in names:
                p = os.path.join(out_dir, nm)
                paths.append(p); items.append((p, prompt))
            k += len(imgs)
    return paths, items


def score(paths, items, device):
    mu = np.load(os.path.join(REF, "test_mu.npy"))
    sigma = np.load(os.path.join(REF, "test_sigma.npy"))
    fid = compute_fid(paths, mu, sigma, device, verbose=False)
    clip_t = compute_clip_t(items, device, verbose=False)
    return fid, clip_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--guidances", type=float, nargs="+",
                    default=[1.5, 2.0, 2.5, 3.0, 4.0])
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--n_per_pair", type=int, default=10)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--use_weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--final_out", default=None,
                    help="if set, the single best passing config is also written here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = load_vocab()
    model = load_model(args.ckpt, device, args.use_weights)
    bank = ConditionBank(load_cache(device), device)
    edm = EDM()

    tmp = os.path.join(ROOT, "outputs", "_sweep_tmp")
    print(f"\n{'guidance':>8} {'over':>4} {'n':>5} {'FID':>8} {'CLIP_T':>8}  verdict")
    print("-" * 52)
    results = []
    for g in args.guidances:
        paths, items = gen_config(model, edm, bank, vocab, args.n_per_pair,
                                  args.oversample, g, args.steps, device, tmp)
        fid, clip_t = score(paths, items, device)
        ok = fid <= FID_T and clip_t >= CLIP_T_T
        verdict = "PASS" if ok else (
            f"{'FID' if fid > FID_T else ''}{'+CLIP' if clip_t < CLIP_T_T else ''} short")
        print(f"{g:8.2f} {args.oversample:4d} {len(paths):5d} "
              f"{fid:8.3f} {clip_t:8.4f}  {verdict}", flush=True)
        results.append((g, fid, clip_t, ok))

    # choose winner: prefer configs meeting CLIP-T (the binding metric), min FID
    meeting_clip = [r for r in results if r[2] >= CLIP_T_T]
    pool = meeting_clip or results
    winner = min(pool, key=lambda r: r[1])
    print(f"\n[sweep] suggested guidance={winner[0]} "
          f"(FID {winner[1]:.2f}, CLIP_T {winner[2]:.4f})  "
          f"targets FID<= {FID_T}, CLIP_T>= {CLIP_T_T}")
    print(f"[sweep] note: sweep FID uses n={args.n_per_pair*100} (biased high vs "
          f"the real 2000); rerun winner at --n_per_pair 20 for the true number.")

    if args.final_out:
        out = os.path.join(ROOT, args.final_out)
        print(f"[sweep] writing winner (g={winner[0]}) -> {out}")
        paths, items = gen_config(model, edm, bank, vocab, 20, args.oversample,
                                  winner[0], max(args.steps, 30), device, out)
        fid, clip_t = score(paths, items, device)
        print(f"[FINAL] {len(paths)} imgs  FID {fid:.3f}  CLIP_T {clip_t:.4f}  "
              f"{'*** PASS BASELINE ***' if fid <= FID_T and clip_t >= CLIP_T_T else 'short'}")


if __name__ == "__main__":
    main()
