"""Shared sampling helpers used by both train.py (proxy eval) and generate.py."""
import os

import torch
from PIL import Image


@torch.no_grad()
def sample_pairs(model, edm, bank, pair_list, steps=30, guidance=2.0,
                 device="cuda", batch=250, amp=True):
    """pair_list: list of (a_idx, o_idx). Returns uint8 tensor [N,3,64,64]."""
    model.eval()
    out = []
    for i in range(0, len(pair_list), batch):
        chunk = pair_list[i:i + batch]
        a = torch.tensor([p[0] for p in chunk], device=device, dtype=torch.long)
        o = torch.tensor([p[1] for p in chunk], device=device, dtype=torch.long)
        cond = bank.get(a, o)
        uncond = bank.null(len(chunk)) if guidance != 1.0 else None
        ctx = torch.autocast(device, dtype=torch.bfloat16) if amp \
            else torch.autocast(device, enabled=False)
        with ctx:
            x = edm.sample(model, (len(chunk), 3, 64, 64), cond, uncond,
                           steps=steps, guidance=guidance, device=device)
        x = (x.float().clamp(-1, 1) + 1) * 127.5
        out.append(x.round().to(torch.uint8).cpu())
    return torch.cat(out)


def save_images(imgs_uint8, names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(imgs_uint8, names):
        arr = img.permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(os.path.join(out_dir, name))


def save_grid(imgs_uint8, path, ncol=10):
    n = len(imgs_uint8)
    nrow = (n + ncol - 1) // ncol
    H = W = 64
    sheet = Image.new("RGB", (ncol * W, nrow * W), "white")
    for i, img in enumerate(imgs_uint8):
        arr = img.permute(1, 2, 0).numpy()
        sheet.paste(Image.fromarray(arr), ((i % ncol) * W, (i // ncol) * W))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)
