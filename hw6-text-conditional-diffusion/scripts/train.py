"""From-scratch EDM training loop for HW6 Brainrot 64x64 conditional generation.

  python scripts/train.py --iters 150000 --batch 128 --lr 2e-4
  python scripts/train.py --smoke      # quick end-to-end smoke test

Checkpoints + sample grids + proxy metrics go to outputs/run_<name>/.
"""
import argparse
import copy
import os
import sys
import time
from collections import Counter

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.rich import tqdm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, load_cache  # noqa: E402
from dataset import BrainrotDataset, load_vocab  # noqa: E402
from diffusion import EDM  # noqa: E402
from model import UNet, EDMPrecond, count_params  # noqa: E402
from sample_utils import sample_pairs, save_grid, save_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        # warmup ramp so EMA tracks the model early instead of lagging ~10k steps
        d = min(self.decay, (1 + step) / (10 + step))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def make_loader(ds, batch, balanced, workers):
    if balanced:
        pair = [r["animal"] + "|" + r["object"] for r in ds.rows]
        freq = Counter(pair)
        w = torch.tensor([1.0 / freq[p] for p in pair])
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler,
                          num_workers=workers, pin_memory=True, drop_last=True,
                          persistent_workers=workers > 0)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                      pin_memory=True, drop_last=True, persistent_workers=workers > 0)


def proxy_eval(shadow, edm, bank, n_per_pair, steps, guidance, device, tmp_dir):
    vocab = load_vocab()
    pairs = [(a, o) for a in range(10) for o in range(10)]
    pair_list, names = [], []
    k = 1
    for (a, o) in pairs:
        for _ in range(n_per_pair):
            pair_list.append((a, o))
            names.append(f"{k:06d}.png")
            k += 1
    imgs = sample_pairs(shadow, edm, bank, pair_list, steps, guidance, device)
    save_images(imgs, names, tmp_dir)
    from eval_local import compute_fid, compute_clip_t
    from dataset import prompt_of
    paths = [os.path.join(tmp_dir, n) for n in names]
    mu = np.load(os.path.join(ROOT, "scoring_program/input/ref/test_mu.npy"))
    sigma = np.load(os.path.join(ROOT, "scoring_program/input/ref/test_sigma.npy"))
    fid = compute_fid(paths, mu, sigma, device, verbose=False)
    items = [(p, prompt_of(vocab["animals"][a], vocab["objects"][o]))
             for p, (a, o) in zip(paths, pair_list)]
    clip_t = compute_clip_t(items, device, verbose=False)
    return fid, clip_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--preset", choices=["v1", "v2"], default="v1",
                    help="v1=fast 27.8M class-cond (~2.5h); v2=59M +CLIP cross-attn")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--ch_mult", type=int, nargs="+", default=None)
    ap.add_argument("--num_res_blocks", type=int, default=None)
    ap.add_argument("--use_cross", dest="use_cross", action="store_true", default=None)
    ap.add_argument("--no_cross", dest="use_cross", action="store_false")
    ap.add_argument("--checkpoint", dest="checkpoint", action="store_true", default=None)
    ap.add_argument("--no_checkpoint", dest="checkpoint", action="store_false")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--cfg_dropout", type=float, default=0.1)
    ap.add_argument("--balanced", action="store_true", default=True)
    ap.add_argument("--no_balanced", dest="balanced", action="store_false")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--eval_n_per_pair", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=24)
    ap.add_argument("--eval_guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # ---- resolve preset (explicit flags override preset defaults) ----
    PRESETS = {
        "v1": dict(base=64, ch_mult=[1, 2, 4, 4], num_res_blocks=2,
                   use_cross=False, checkpoint=False, iters=30000),
        "v2": dict(base=128, ch_mult=[1, 2, 2, 2], num_res_blocks=3,
                   use_cross=True, checkpoint=True, iters=80000),
    }
    p = PRESETS[args.preset]
    for k in ("base", "ch_mult", "num_res_blocks", "use_cross", "checkpoint", "iters"):
        if getattr(args, k) is None:
            setattr(args, k, p[k])
    if args.name is None:
        args.name = f"{args.preset}_{args.base}b{''.join(map(str, args.ch_mult))}"

    if args.smoke:
        args.base, args.ch_mult, args.num_res_blocks = 48, [1, 2, 2], 1
        args.use_cross, args.checkpoint = True, False
        args.iters, args.batch, args.workers = 60, 16, 0
        args.log_every, args.sample_every, args.eval_every = 10, 30, 60
        args.ckpt_every, args.eval_n_per_pair, args.eval_steps = 60, 1, 8

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(ROOT, "outputs", f"run_{args.name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cfg] {vars(args)}")

    ds = BrainrotDataset(root=os.path.join(ROOT, "data/trainset"),
                         csv_path=os.path.join(ROOT, "data/train.csv"),
                         hflip=True)
    loader = make_loader(ds, args.batch, args.balanced, args.workers)
    bank = ConditionBank(load_cache(device), device)
    edm = EDM()

    model_cfg = dict(base=args.base, ch_mult=tuple(args.ch_mult),
                     num_res_blocks=args.num_res_blocks, dropout=args.dropout,
                     use_cross_attn=args.use_cross)
    model = EDMPrecond(UNet(use_checkpoint=args.checkpoint, **model_cfg)).to(device)
    print(f"[model] {count_params(model)/1e6:.2f}M params  "
          f"cross_attn={args.use_cross} checkpoint={args.checkpoint}")
    ema = EMA(model, args.ema)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_it = 0
    best_fid = float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_it = ck["it"]
        best_fid = ck.get("best_fid", best_fid)
        print(f"[resume] from {args.resume} @ it={start_it}")

    def cycle(dl):
        while True:
            for b in dl:
                yield b
    data = cycle(loader)

    model.train()
    t0 = time.time()
    run_loss = 0.0
    pbar = tqdm(range(start_it + 1, args.iters + 1), total=args.iters,
                initial=start_it, desc="train")
    for it in pbar:
        x, a, o, _ = next(data)
        x = x.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        o = o.to(device, non_blocking=True)
        drop = torch.rand(x.shape[0], device=device) < args.cfg_dropout
        cond = bank.get(a, o, drop)
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = edm.loss(model, x, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.update(model, it)
        run_loss += loss.item()

        if it % args.log_every == 0:
            ips = args.log_every * args.batch / (time.time() - t0)
            pbar.set_postfix(loss=f"{run_loss/args.log_every:.4f}",
                             img_s=f"{ips:.0f}", best_fid=f"{best_fid:.1f}")
            run_loss = 0.0
            t0 = time.time()

        if it % args.sample_every == 0:
            torch.cuda.empty_cache()
            pl = [(a_, o_) for a_ in range(10) for o_ in range(10)]
            # raw model -> immediate learning signal (no EMA warmup lag)
            imgs = sample_pairs(model, edm, bank, pl, steps=args.eval_steps,
                                guidance=args.eval_guidance, device=device)
            save_grid(imgs, os.path.join(out_dir, f"sample_{it:07d}_raw.png"))
            imgs = sample_pairs(ema.shadow, edm, bank, pl, steps=args.eval_steps,
                                guidance=args.eval_guidance, device=device)
            save_grid(imgs, os.path.join(out_dir, f"sample_{it:07d}_ema.png"))
            model.train()
            torch.cuda.empty_cache()

        if it % args.eval_every == 0:
            torch.cuda.empty_cache()
            fid, clip_t = proxy_eval(
                ema.shadow, edm, bank, args.eval_n_per_pair, args.eval_steps,
                args.eval_guidance, device, os.path.join(out_dir, "_proxy"))
            print(f"[eval it {it}] proxy FID(n={args.eval_n_per_pair*100}) "
                  f"{fid:.3f}  CLIP_T {clip_t:.4f}", flush=True)
            with open(os.path.join(out_dir, "metrics.csv"), "a") as f:
                f.write(f"{it},{fid:.4f},{clip_t:.6f}\n")
            if fid < best_fid:
                best_fid = fid
                torch.save({"model": model.state_dict(),
                            "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                            "it": it, "best_fid": best_fid, "args": vars(args),
                            "model_cfg": model_cfg},
                           os.path.join(out_dir, "best.pt"))
                print(f"[eval it {it}] new best proxy FID {best_fid:.3f} -> best.pt")
            model.train()
            torch.cuda.empty_cache()

        if it % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(),
                        "ema": ema.shadow.state_dict(), "opt": opt.state_dict(),
                        "it": it, "best_fid": best_fid, "args": vars(args),
                        "model_cfg": model_cfg},
                       os.path.join(out_dir, "last.pt"))

    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                "opt": opt.state_dict(), "it": args.iters, "best_fid": best_fid,
                "args": vars(args), "model_cfg": model_cfg},
               os.path.join(out_dir, "last.pt"))
    print(f"[done] best proxy FID {best_fid:.3f}  ckpts in {out_dir}")


if __name__ == "__main__":
    main()
