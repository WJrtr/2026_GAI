"""Unit tests for model / diffusion / conditioning. Run BEFORE training.

  python scripts/test_model.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_cond import ConditionBank, build_cache  # noqa: E402
from diffusion import EDM  # noqa: E402
from model import NULL_CLASS, UNet, EDMPrecond, count_params  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  ok: {name}")


def small_net():
    return EDMPrecond(UNet(base=32, ch_mult=(1, 2, 2), num_res_blocks=1,
                           attn_res=(16, 8), img_res=64)).to(DEV)


def fake_cond(B):
    return dict(
        a_idx=torch.randint(0, 10, (B,), device=DEV),
        o_idx=torch.randint(0, 10, (B,), device=DEV),
        ctx_seq=torch.randn(B, 77, 512, device=DEV),
        ctx_pooled=torch.randn(B, 512, device=DEV),
        ctx_pad_mask=torch.zeros(B, 77, dtype=torch.bool, device=DEV),
    )


def test_shapes():
    net = small_net()
    B = 4
    x = torch.randn(B, 3, 64, 64, device=DEV)
    sigma = torch.rand(B, device=DEV) + 0.1
    out = net(x, sigma, **fake_cond(B))
    check("EDMPrecond output shape", out.shape == (B, 3, 64, 64))
    check("EDMPrecond finite", torch.isfinite(out).all().item())


def test_pad_mask_changes_output():
    # NOTE: proj1/proj2 are zero-init by design (attn starts as identity), so an
    # untrained full net ignores attention. Test the mechanism on AttnBlock with
    # a non-zero output projection.
    from model import AttnBlock
    blk = AttnBlock(ch=32, ctx_dim=64, n_heads=4).to(DEV).eval()
    torch.nn.init.normal_(blk.proj2.weight, std=0.2)
    B = 2
    x = torch.randn(B, 32, 8, 8, device=DEV)
    ctx = torch.randn(B, 6, 64, device=DEV)
    keep_all = torch.ones(B, 6, dtype=torch.bool, device=DEV)
    keep_some = keep_all.clone(); keep_some[:, 3:] = False
    with torch.no_grad():
        o1 = blk(x, ctx, keep_all)
        o2 = blk(x, ctx, keep_some)
    check("cross-attn pad mask changes output", not torch.allclose(o1, o2))


def test_loss_backward():
    net = small_net()
    edm = EDM()
    x = torch.randn(8, 3, 64, 64, device=DEV)
    loss = edm.loss(net, x, fake_cond(8))
    loss.backward()
    g = [p.grad for p in net.parameters() if p.grad is not None]
    check("loss is finite scalar", torch.isfinite(loss).item() and loss.ndim == 0)
    check("gradients flow", len(g) > 0 and all(torch.isfinite(gi).all() for gi in g))


def test_cfg_sampler():
    net = small_net().eval()
    edm = EDM()
    B = 4
    cond = fake_cond(B)
    uncond = fake_cond(B)
    img = edm.sample(net, (B, 3, 64, 64), cond, uncond, steps=6,
                     guidance=2.5, device=DEV)
    check("sampler output shape", img.shape == (B, 3, 64, 64))
    check("sampler output in [-1,1]", img.min() >= -1.001 and img.max() <= 1.001)
    check("sampler output finite", torch.isfinite(img).all().item())


def test_condition_bank():
    cache = build_cache(DEV, save=True)
    bank = ConditionBank(cache, DEV)
    B = 6
    a = torch.randint(0, 10, (B,), device=DEV)
    o = torch.randint(0, 10, (B,), device=DEV)
    drop = torch.zeros(B, dtype=torch.bool, device=DEV)
    drop[:2] = True
    c = bank.get(a, o, drop)
    check("bank seq shape", c["ctx_seq"].shape == (B, 77, 512))
    check("bank pooled shape", c["ctx_pooled"].shape == (B, 512))
    check("dropped -> NULL_CLASS", (c["a_idx"][:2] == NULL_CLASS).all().item()
          and (c["a_idx"][2:] == a[2:]).all().item())
    check("dropped seq == null_seq",
          torch.allclose(c["ctx_seq"][0], bank.null_seq))
    nz = bank.null(3)
    check("bank.null shapes", nz["ctx_seq"].shape == (3, 77, 512)
          and (nz["a_idx"] == NULL_CLASS).all().item())


def test_real_config_params():
    net = EDMPrecond(UNet(base=128, ch_mult=(1, 2, 2, 2), num_res_blocks=3))
    n = count_params(net) / 1e6
    check(f"real config ~59M (got {n:.2f}M, <62.68M baseline)", 55 < n < 62.68)


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in [test_shapes, test_pad_mask_changes_output, test_loss_backward,
               test_cfg_sampler, test_condition_bank, test_real_config_params]:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
