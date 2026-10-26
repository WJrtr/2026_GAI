"""Diagnose the memory hotspot: measure peak VRAM for one train step under
(1) current code, (2) no cross-attn mask, (3) gradient checkpointing."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, load_cache  # noqa: E402
from diffusion import EDM  # noqa: E402
import model as M  # noqa: E402

dev = "cuda"
edm = EDM()
bank = ConditionBank(load_cache(dev), dev)
B = 32


def one_step(net, opt):
    x = torch.randn(B, 3, 64, 64, device=dev)
    a = torch.randint(0, 10, (B,), device=dev)
    o = torch.randint(0, 10, (B,), device=dev)
    cond = bank.get(a, o, torch.rand(B, device=dev) < 0.1)
    with torch.autocast(dev, dtype=torch.bfloat16):
        loss = edm.loss(net, x, cond)
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)


def measure(tag):
    net = M.EDMPrecond(M.UNet(base=128, ch_mult=(1, 2, 2, 2),
                              num_res_blocks=3)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    one_step(net, opt)  # warmup/alloc
    torch.cuda.reset_peak_memory_stats()
    one_step(net, opt)
    torch.cuda.synchronize()
    print(f"{tag:30s} peak {torch.cuda.max_memory_allocated()/1e9:5.2f} GB "
          f"(B={B})", flush=True)
    del net, opt
    torch.cuda.empty_cache()


# (1) current
measure("1) current (bool mask)")

# (2) monkeypatch AttnBlock.forward to skip cross-attn mask
_orig = M.AttnBlock.forward
def _nomask(self, x, ctx, ctx_keep):
    return _orig(self, x, ctx, None)
M.AttnBlock.forward = _nomask
measure("2) no cross-attn mask")
M.AttnBlock.forward = _orig
