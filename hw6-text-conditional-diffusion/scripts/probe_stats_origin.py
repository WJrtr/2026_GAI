"""Determine the ORIGIN of the official test_mu/test_sigma.

Compute (mu,sigma) over the local ref/test/ images (3000, sampled from trainset)
using score.py's EXACT recipe, and compare to the provided official stats.

If they match (cosine~1, FID~0): official stats == torchvision features of these
images -> our harness is correct and real images must score ~0; any high number
is a content mismatch.
If they DON'T match: official stats were produced by a different inception
checkpoint / pipeline (e.g. the standard pt_inception FID model) and our local
FID is NOT comparable to the leaderboard.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch import nn
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

REF = "scoring_program/input/ref"
mu = np.load(os.path.join(REF, "test_mu.npy"))
sig = np.load(os.path.join(REF, "test_sigma.npy"))
dev = "cuda"
tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


@torch.no_grad()
def stats(paths):
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                     transform_input=False)
    m.fc = nn.Identity()
    m.eval().to(dev)
    out, buf = [], []
    for i, p in enumerate(paths):
        buf.append(tf(Image.open(p).convert("RGB")))
        if len(buf) == 100 or i == len(paths) - 1:
            out.append(m(torch.stack(buf).to(dev)).cpu())
            buf = []
    f = torch.cat(out).numpy()
    return f.mean(0), np.cov(f, rowvar=False), f


def fid(m2, s2):
    cov, _ = linalg.sqrtm(sig @ s2, disp=False)
    cov = cov.real if np.iscomplexobj(cov) else cov
    d = mu - m2
    return float(d @ d + np.trace(sig + s2 - 2 * cov))


def main():
    paths = sorted(glob.glob(os.path.join(REF, "test", "*.png")))
    print(f"ref/test images: {len(paths)}")
    m2, s2, _ = stats(paths)
    cos = float(m2 @ mu / (np.linalg.norm(m2) * np.linalg.norm(mu)))
    print(f"cosine(ref/test mu, official mu) = {cos:.6f}")
    print(f"||ref/test mu - official mu||     = {np.linalg.norm(m2-mu):.4f}")
    print(f"FID(ref/test vs official)         = {fid(m2, s2):.4f}")
    print(f"official trace={np.trace(sig):.2f}  reftest trace={np.trace(s2):.2f}")


if __name__ == "__main__":
    main()
