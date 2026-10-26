"""
D0 sanity check: verify the local eval harness end-to-end and calibrate the
FID/CLIP-T range we are aiming inside of.

  - REAL baseline: sample 20 real train images per (animal,object) pair -> 2000
    images that mimic the generation target distribution. This is the FID FLOOR
    (best we could hope for) and the CLIP-T ceiling-ish for this data.
  - NOISE baseline: 2000 random-noise images -> FID CEILING / CLIP-T floor.

If the harness is correct we expect: real FID small, noise FID huge;
real CLIP-T ~0.27-0.32, noise CLIP-T ~0.1.
"""
import csv
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_vocab, prompt_of  # noqa: E402
from eval_local import compute_clip_t, compute_fid  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "outputs", "_real_baseline")
NOISE_DIR = os.path.join(ROOT, "outputs", "_noise")
REF_DIR = os.path.join(ROOT, "scoring_program", "input", "ref")
SEED = 0


def build_real_baseline():
    os.makedirs(REAL_DIR, exist_ok=True)
    by_pair = {}
    with open(os.path.join(ROOT, "data", "train.csv"),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            by_pair.setdefault((r["animal"], r["object"]), []).append(r["id"])
    rng = random.Random(SEED)
    items = []
    idx = 1
    for (a, o), ids in sorted(by_pair.items()):
        pick = rng.sample(ids, min(20, len(ids)))
        # if a pair has <20 train imgs, sample with replacement to reach 20
        while len(pick) < 20:
            pick.append(rng.choice(ids))
        for src in pick:
            name = f"{idx:06d}.png"
            Image.open(os.path.join(ROOT, "data", "trainset", src)).convert(
                "RGB").save(os.path.join(REAL_DIR, name))
            items.append((os.path.join(REAL_DIR, name), prompt_of(a, o)))
            idx += 1
    return items


def build_noise_baseline(n=2000):
    os.makedirs(NOISE_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    items = []
    vocab = load_vocab()
    pairs = [(a, o) for a in vocab["animals"] for o in vocab["objects"]]
    for i in range(n):
        arr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        name = f"{i+1:06d}.png"
        Image.fromarray(arr).save(os.path.join(NOISE_DIR, name))
        a, o = pairs[i % len(pairs)]
        items.append((os.path.join(NOISE_DIR, name), prompt_of(a, o)))
    return items


def eval_set(name, items, device):
    paths = [p for p, _ in items]
    mu = np.load(os.path.join(REF_DIR, "test_mu.npy"))
    sigma = np.load(os.path.join(REF_DIR, "test_sigma.npy"))
    fid = compute_fid(paths, mu, sigma, device, verbose=False)
    clip_t = compute_clip_t(items, device, verbose=False)
    print(f"[{name}]  FID = {fid:8.4f}   CLIP_T = {clip_t:.6f}   (n={len(paths)})")
    return fid, clip_t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("building REAL baseline (20 real imgs / pair = 2000) ...")
    real = build_real_baseline()
    print("building NOISE baseline (2000) ...")
    noise = build_noise_baseline()
    print("scoring (first run downloads inception + CLIP weights) ...")
    eval_set("REAL ", real, device)
    eval_set("NOISE", noise, device)
    print("\nTargets to beat: FID < 40 (baseline4=49.25), CLIP_T >= 0.270")


if __name__ == "__main__":
    main()
