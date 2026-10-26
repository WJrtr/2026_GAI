"""
Local evaluation harness for HW6 — faithful to the official Codabench scorer.

FID  : IMPORTANT — the provided scoring_program/score.py uses torchvision
       inception_v3, but the official test_mu.npy / test_sigma.npy were verified
       (see scripts/probe_ptfid.py) to come from the CANONICAL pytorch-fid
       InceptionV3 (real ref images -> FID 20.2 with pytorch-fid vs 333 with
       torchvision). We therefore use pytorch-fid's InceptionV3 so that our
       local FID matches the leaderboard. Feed native 64x64; it upscales to 299
       internally (resize_input=True, normalize_input=True).
CLIP-T: open_clip ViT-B-32-quickgelu / openai, cosine(image_emb, text_emb),
        averaged. Matches score.py's clip_score_image_text (fake image + prompt
        only; no real reference images needed).

Usage:
  python scripts/eval_local.py --images_dir scoring_program/input/res \
      --labels_csv data/generate.csv \
      --ref_dir scoring_program/input/ref --scores fid clip_t
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ----------------------------- FID -----------------------------
class _FIDDataset(Dataset):
    """Feed native-resolution [0,1] tensors; pytorch-fid InceptionV3 handles
    resize to 299 + normalization to [-1,1] internally."""

    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


_INCEPTION = None


def _inception(device):
    global _INCEPTION
    if _INCEPTION is None:
        from pytorch_fid.inception import InceptionV3
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        _INCEPTION = InceptionV3([block], resize_input=True,
                                 normalize_input=True).to(device).eval()
    return _INCEPTION


def _calc_fid(mu1, sig1, mu2, sig2):
    covmean, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sig1 + sig2 - 2 * covmean))


@torch.no_grad()
def compute_fid(image_paths, ref_mu, ref_sigma, device, batch_size=64,
                num_workers=0, verbose=True):
    dl = DataLoader(_FIDDataset(image_paths), batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
    model = _inception(device)
    feats = []
    for i, batch in enumerate(dl):
        f = model(batch.to(device))[0].squeeze(-1).squeeze(-1)
        feats.append(f.cpu())
        if verbose and i % 10 == 0:
            print(f"  [fid] inception batch {i}/{len(dl)}", end="\r")
    feats = torch.cat(feats, 0).numpy()
    mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
    if verbose:
        print()
    return _calc_fid(ref_mu, ref_sigma, mu, sigma)


# ----------------------------- CLIP-T -----------------------------
class _CLIPDataset(Dataset):
    def __init__(self, items, preprocess, tokenizer):
        self.items = items
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, prompt = self.items[idx]
        img = self.preprocess(Image.open(path).convert("RGB"))
        tok = self.tokenizer([prompt])[0]
        return img, tok


@torch.no_grad()
def compute_clip_t(items, device, batch_size=64, num_workers=4, verbose=True):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model = model.to(device).eval()
    dl = DataLoader(_CLIPDataset(items, preprocess, tokenizer),
                    batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    scores = []
    for i, (imgs, toks) in enumerate(dl):
        imgs, toks = imgs.to(device), toks.to(device)
        ie = model.encode_image(imgs)
        te = model.encode_text(toks)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        scores.append((ie * te).sum(-1).cpu())
        if verbose and i % 10 == 0:
            print(f"  [clip] batch {i}/{len(dl)}", end="\r")
    if verbose:
        print()
    return float(torch.cat(scores).mean())


# ----------------------------- helpers -----------------------------
def load_labels(labels_csv):
    """Return {basename: prompt}. Builds prompt if column absent."""
    out = {}
    with open(labels_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row["id"]
            prompt = row.get("prompt") or f"a {row['animal']} and a {row['object']}"
            out[name] = prompt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_csv", default="data/generate.csv")
    ap.add_argument("--ref_dir", default="scoring_program/input/ref")
    ap.add_argument("--scores", nargs="+", default=["fid", "clip_t"],
                    choices=["fid", "clip_t"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--expect", type=int, default=None,
                    help="assert this many PNGs are present")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(args.images_dir, "**", "*.png"),
                             recursive=True))
    if args.expect is not None and len(paths) != args.expect:
        raise SystemExit(f"Expected {args.expect} PNGs, found {len(paths)}")
    print(f"[eval] {len(paths)} images in {args.images_dir}")

    result = {}
    if "fid" in args.scores:
        mu = np.load(os.path.join(args.ref_dir, "test_mu.npy"))
        sigma = np.load(os.path.join(args.ref_dir, "test_sigma.npy"))
        result["FID"] = compute_fid(paths, mu, sigma, device,
                                    args.batch_size, args.num_workers)
        print(f"[eval] FID    = {result['FID']:.4f}")
    if "clip_t" in args.scores:
        labels = load_labels(args.labels_csv)
        items = [(p, labels[os.path.basename(p)]) for p in paths
                 if os.path.basename(p) in labels]
        if len(items) != len(paths):
            print(f"[warn] {len(paths)-len(items)} images had no label row")
        result["CLIP_T"] = compute_clip_t(items, device,
                                          args.batch_size, args.num_workers)
        print(f"[eval] CLIP_T = {result['CLIP_T']:.6f}")

    print("[eval] result:", result)
    return result


if __name__ == "__main__":
    main()
