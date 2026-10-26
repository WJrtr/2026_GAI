"""Test whether official test_mu/sigma come from the canonical pytorch-fid
InceptionV3 (the standard FID model). If cosine~1 and FID~0 on the ref/test
real images, we've found the correct scorer pipeline."""
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pytorch_fid.inception import InceptionV3

REF = "scoring_program/input/ref"
mu = np.load(os.path.join(REF, "test_mu.npy"))
sig = np.load(os.path.join(REF, "test_sigma.npy"))
dev = "cuda"


class DS(Dataset):
    def __init__(self, paths, resize):
        self.paths = paths
        t = [transforms.Resize((resize, resize)), transforms.ToTensor()] \
            if resize else [transforms.ToTensor()]
        self.tf = transforms.Compose(t)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def feats(paths, resize):
    # pytorch-fid InceptionV3: input [0,1], resizes to 299 internally,
    # normalizes to [-1,1]; block idx 3 -> 2048-dim
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block], resize_input=True, normalize_input=True).to(dev).eval()
    dl = DataLoader(DS(paths, resize), batch_size=50, num_workers=0)
    out = []
    for b in dl:
        f = model(b.to(dev))[0].squeeze(-1).squeeze(-1)
        out.append(f.cpu())
    f = torch.cat(out).numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def fid(m2, s2):
    cov, _ = linalg.sqrtm(sig @ s2, disp=False)
    cov = cov.real if np.iscomplexobj(cov) else cov
    d = mu - m2
    return float(d @ d + np.trace(sig + s2 - 2 * cov))


def main():
    paths = sorted(glob.glob(os.path.join(REF, "test", "*.png")))
    print(f"ref/test images: {len(paths)}")
    for resize in (None, 64, 299):
        # resize=None feeds native 64x64; pytorch-fid then upscales to 299
        m2, s2 = feats(paths, resize)
        cos = float(m2 @ mu / (np.linalg.norm(m2) * np.linalg.norm(mu)))
        print(f"[pre-resize={str(resize):4s}] cosine={cos:.6f}  "
              f"FID(ref/test vs official)={fid(m2, s2):.4f}", flush=True)


if __name__ == "__main__":
    main()
