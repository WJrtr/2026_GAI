"""Reverse-engineer how the official test_mu/test_sigma were preprocessed.

We feed REAL train images (which must be close to the test distribution) through
inception under several preprocessing variants and pick the one giving the
LOWEST FID against the official stats. That variant is the official recipe.
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


def fid(m2, s2):
    cov, _ = linalg.sqrtm(sig @ s2, disp=False)
    cov = cov.real if np.iscomplexobj(cov) else cov
    d = mu - m2
    return float(d @ d + np.trace(sig + s2 - 2 * cov))


@torch.no_grad()
def feats(paths, resize, normalize, ti, antialias=True):
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                     transform_input=ti)
    m.fc = nn.Identity()
    m.eval().to(dev)
    tf_list = [transforms.Resize((resize, resize), antialias=antialias),
               transforms.ToTensor()]
    if normalize is not None:
        tf_list.append(transforms.Normalize(*normalize))
    tf = transforms.Compose(tf_list)
    out = []
    buf = []
    for i, p in enumerate(paths):
        buf.append(tf(Image.open(p).convert("RGB")))
        if len(buf) == 100 or i == len(paths) - 1:
            x = torch.stack(buf).to(dev)
            out.append(m(x).cpu())
            buf = []
    del m
    f = torch.cat(out).numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def main():
    paths = sorted(glob.glob("outputs/_real_baseline/*.png"))
    print(f"{len(paths)} real images\n")
    N5 = ([0.5] * 3, [0.5] * 3)
    NIMG = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    variants = [
        ("A score.py Norm0.5 ti=F res299", 299, N5, False, True),
        ("B [0,1] ti=F res299", 299, None, False, True),
        ("C [0,1] ti=T res299", 299, None, True, True),
        ("D ImageNet norm ti=F res299", 299, NIMG, False, True),
        ("E Norm0.5 ti=F res299 NOaa", 299, N5, False, False),
        ("F [0,1] ti=T res299 NOaa", 299, None, True, False),
    ]
    results = []
    for name, res, nrm, ti, aa in variants:
        m2, s2 = feats(paths, res, nrm, ti, aa)
        score = fid(m2, s2)
        results.append((name, score))
        print(f"{name:34s} FID = {score:9.3f}", flush=True)
    results.sort(key=lambda x: x[1])
    print("\nBEST:", results[0])


if __name__ == "__main__":
    main()
