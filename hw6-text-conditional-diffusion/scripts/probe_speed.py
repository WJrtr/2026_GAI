"""Measure real 59M model training throughput + peak VRAM at several batch sizes
to pick the final training config for the RTX 4080 SUPER (16GB)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, load_cache  # noqa: E402
from diffusion import EDM  # noqa: E402
from model import UNet, EDMPrecond, count_params  # noqa: E402

dev = "cuda"
edm = EDM()
bank = ConditionBank(load_cache(dev), dev)
model = EDMPrecond(UNet(base=128, ch_mult=(1, 2, 2, 2), num_res_blocks=3,
                        use_checkpoint=True)).to(dev)
model.train()
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
print(f"model {count_params(model)/1e6:.2f}M params")

for B in (96, 128, 192, 256):
    try:
        torch.cuda.reset_peak_memory_stats()
        for i in range(12):
            x = torch.randn(B, 3, 64, 64, device=dev)
            a = torch.randint(0, 10, (B,), device=dev)
            o = torch.randint(0, 10, (B,), device=dev)
            drop = torch.rand(B, device=dev) < 0.1
            cond = bank.get(a, o, drop)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = edm.loss(model, x, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if i == 2:
                torch.cuda.synchronize(); t0 = time.time()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"B={B:4d}  {B/dt:7.0f} img/s  {dt*1000:6.1f} ms/it  "
              f"peak {peak:5.2f} GB", flush=True)
    except RuntimeError as e:
        print(f"B={B:4d}  OOM/error: {str(e)[:60]}")
        torch.cuda.empty_cache()

# sampling speed (Heun, CFG) for the full 2000-image generation estimate
model.eval()
from sample_utils import sample_pairs  # noqa: E402
for steps in (24, 30):
    torch.cuda.synchronize(); t0 = time.time()
    sample_pairs(model, edm, bank, [(0, 0)] * 250, steps=steps,
                 guidance=2.0, device=dev, batch=250)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"sample steps={steps}: {250/dt:.0f} img/s -> 2000 imgs ~ {2000/(250/dt):.0f}s")
