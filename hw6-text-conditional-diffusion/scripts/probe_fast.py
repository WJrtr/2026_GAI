"""Find a FAST V1 config: measure params / peak VRAM / img-s at batch 128 with
NO gradient checkpointing, for several lean architectures."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, load_cache  # noqa: E402
from diffusion import EDM  # noqa: E402
from model import UNet, EDMPrecond, count_params  # noqa: E402

dev = "cuda"
# Cap usable VRAM so over-allocation raises OOM (caught below) instead of
# silently spilling into shared system RAM (which just hangs at ~15s/step).
torch.cuda.set_per_process_memory_fraction(0.90)
edm = EDM()
bank = ConditionBank(load_cache(dev), dev)
B = 128
OUT = open(os.path.join(os.path.dirname(__file__), "..", "outputs",
                        "_fast_bench.txt"), "w")

configs = [
    ("64 (1,2,4,4) nrb2 cross=F", dict(base=64, ch_mult=(1, 2, 4, 4), num_res_blocks=2, use_cross_attn=False)),
    ("64 (1,2,4,4) nrb2 cross=T", dict(base=64, ch_mult=(1, 2, 4, 4), num_res_blocks=2, use_cross_attn=True)),
    ("96 (1,2,2,2) nrb2 cross=F", dict(base=96, ch_mult=(1, 2, 2, 2), num_res_blocks=2, use_cross_attn=False)),
    ("96 (1,2,3,4) nrb2 cross=F", dict(base=96, ch_mult=(1, 2, 3, 4), num_res_blocks=2, use_cross_attn=False)),
    ("128 (1,2,2,2) nrb2 cross=F", dict(base=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, use_cross_attn=False)),
    ("128 (1,2,2,2) nrb2 cross=T", dict(base=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2, use_cross_attn=True)),
]


def bench(cfg):
    net = EDMPrecond(UNet(**cfg)).to(dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)

    def step():
        x = torch.randn(B, 3, 64, 64, device=dev)
        a = torch.randint(0, 10, (B,), device=dev)
        o = torch.randint(0, 10, (B,), device=dev)
        cond = bank.get(a, o, torch.rand(B, device=dev) < 0.1)
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss = edm.loss(net, x, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    try:
        for i in range(6):
            step()
            if i == 2:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        peak = torch.cuda.max_memory_allocated() / 1e9
        n = count_params(net) / 1e6
        ips = B / dt
        eta25k = 25000 * dt / 3600
        return n, ips, peak, eta25k, None
    except RuntimeError as e:
        return count_params(net) / 1e6, 0, 0, 0, str(e)[:40]
    finally:
        del net, opt
        torch.cuda.empty_cache()


def emit(s):
    print(s, flush=True)
    OUT.write(s + "\n"); OUT.flush()


emit(f"{'config':28s} {'params':>8s} {'img/s':>7s} {'peakGB':>7s} {'25k_eta_h':>9s}")
for name, cfg in configs:
    n, ips, peak, eta, err = bench(cfg)
    if err:
        emit(f"{name:28s} {n:7.1f}M   OOM/err: {err}")
    else:
        emit(f"{name:28s} {n:7.1f}M {ips:7.0f} {peak:7.2f} {eta:9.1f}")
OUT.close()
