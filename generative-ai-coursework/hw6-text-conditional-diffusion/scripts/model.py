"""From-scratch conditional UNet + EDM preconditioning for 64x64 generation.

Conditioning (decided in PLAN.md):
  - learned animal/object class embeddings + CLIP pooled  -> AdaGN (global)
  - CLIP text token sequence                              -> cross-attention
  - classifier-free guidance via a null class index (10) + empty-string CLIP feats

EDM preconditioning (Karras et al. 2022, "Elucidating the Design Space of
Diffusion-Based Generative Models"): the network F is wrapped so that
  D(x; sigma) = c_skip*x + c_out * F(c_in*x, c_noise, cond).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NULL_CLASS = 10  # extra index in each 11-way embedding = "unconditional"


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def gn(ch):
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch, eps=1e-6)


class ResBlock(nn.Module):
    """ADM-style residual block with AdaGN (scale/shift from cond emb)."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.in_norm = gn(in_ch)
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.out_norm = gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.in_conv(F.silu(self.in_norm(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_conv(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Spatial self-attention, optionally + cross-attention to CLIP text tokens."""

    def __init__(self, ch, ctx_dim, n_heads=8, use_cross=True):
        super().__init__()
        self.n_heads = n_heads
        self.use_cross = use_cross
        self.norm1 = gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj1 = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj1.weight); nn.init.zeros_(self.proj1.bias)
        if use_cross:
            self.norm2 = gn(ch)
            self.q = nn.Conv2d(ch, ch, 1)
            self.kv = nn.Linear(ctx_dim, ch * 2)
            self.proj2 = nn.Conv2d(ch, ch, 1)
            nn.init.zeros_(self.proj2.weight); nn.init.zeros_(self.proj2.bias)

    def _heads(self, t, B, L):  # [B, L, C] -> [B, h, L, d]
        return t.view(B, L, self.n_heads, -1).transpose(1, 2)

    def forward(self, x, ctx, ctx_keep):
        B, C, H, W = x.shape
        L = H * W
        # --- self-attention ---
        h = self.norm1(x)
        q, k, v = self.qkv(h).reshape(B, 3 * C, L).chunk(3, dim=1)
        q = self._heads(q.transpose(1, 2), B, L)
        k = self._heads(k.transpose(1, 2), B, L)
        v = self._heads(v.transpose(1, 2), B, L)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, C, H, W)
        x = x + self.proj1(o)
        if not self.use_cross:
            return x
        # --- cross-attention to text tokens ---
        h = self.norm2(x)
        qc = self._heads(self.q(h).reshape(B, C, L).transpose(1, 2), B, L)
        kc, vc = self.kv(ctx).chunk(2, dim=-1)              # [B, T, C]
        T = ctx.shape[1]
        kc = self._heads(kc, B, T)
        vc = self._heads(vc, B, T)
        attn_mask = ctx_keep[:, None, None, :] if ctx_keep is not None else None
        oc = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=attn_mask)
        oc = oc.transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj2(oc)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, base=128, ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_res=(16, 8), img_res=64, ctx_dim=512, clip_dim=512,
                 n_classes=10, dropout=0.1, n_heads=8, use_checkpoint=False,
                 use_cross_attn=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_cross_attn = use_cross_attn
        emb_dim = base * 4
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.base = base
        self.animal_emb = nn.Embedding(n_classes + 1, emb_dim)
        self.object_emb = nn.Embedding(n_classes + 1, emb_dim)
        self.pooled_proj = nn.Linear(clip_dim, emb_dim)
        self.ctx_proj = nn.Linear(clip_dim, ctx_dim) if use_cross_attn else None

        def mk_attn(ch):
            return AttnBlock(ch, ctx_dim, n_heads, use_cross=use_cross_attn)

        chs = [base * m for m in ch_mult]
        attn_levels = {i for i, _ in enumerate(ch_mult)
                       if img_res // (2 ** i) in attn_res}

        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        self.downs = nn.ModuleList()
        skip_chs = [base]
        cur = base
        for lvl, ch in enumerate(chs):
            for _ in range(num_res_blocks):
                blk = nn.ModuleDict({"res": ResBlock(cur, ch, emb_dim, dropout)})
                if lvl in attn_levels:
                    blk["attn"] = mk_attn(ch)
                cur = ch
                self.downs.append(blk)
                skip_chs.append(cur)
            if lvl != len(chs) - 1:
                self.downs.append(nn.ModuleDict({"down": Down(cur)}))
                skip_chs.append(cur)

        self.mid_res1 = ResBlock(cur, cur, emb_dim, dropout)
        self.mid_attn = mk_attn(cur)
        self.mid_res2 = ResBlock(cur, cur, emb_dim, dropout)

        self.ups = nn.ModuleList()
        for lvl in reversed(range(len(chs))):
            ch = chs[lvl]
            for _ in range(num_res_blocks + 1):
                blk = nn.ModuleDict(
                    {"res": ResBlock(cur + skip_chs.pop(), ch, emb_dim, dropout)})
                if lvl in attn_levels:
                    blk["attn"] = mk_attn(ch)
                cur = ch
                self.ups.append(blk)
            if lvl != 0:
                self.ups.append(nn.ModuleDict({"up": Up(cur)}))

        self.out_norm = gn(cur)
        self.out_conv = nn.Conv2d(cur, 3, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def _res(self, res, h, emb):
        if self.use_checkpoint and self.training:
            return checkpoint(res, h, emb, use_reentrant=False)
        return res(h, emb)

    def _attn(self, attn, h, ctx, ctx_keep):
        if self.use_checkpoint and self.training:
            return checkpoint(attn, h, ctx, ctx_keep, use_reentrant=False)
        return attn(h, ctx, ctx_keep)

    def forward(self, x, c_noise, a_idx, o_idx, ctx_seq, ctx_pooled, ctx_pad_mask):
        emb = self.time_mlp(timestep_embedding(c_noise, self.base))
        emb = emb + self.animal_emb(a_idx) + self.object_emb(o_idx)
        emb = emb + self.pooled_proj(ctx_pooled)
        if self.use_cross_attn:
            ctx = self.ctx_proj(ctx_seq)
            ctx_keep = (~ctx_pad_mask) if ctx_pad_mask is not None else None
        else:
            ctx = ctx_keep = None

        h = self.in_conv(x)
        skips = [h]
        for blk in self.downs:
            if "down" in blk:
                h = blk["down"](h)
            else:
                h = self._res(blk["res"], h, emb)
                if "attn" in blk:
                    h = self._attn(blk["attn"], h, ctx, ctx_keep)
            skips.append(h)
        h = self._res(self.mid_res1, h, emb)
        h = self._attn(self.mid_attn, h, ctx, ctx_keep)
        h = self._res(self.mid_res2, h, emb)
        for blk in self.ups:
            if "up" in blk:
                h = blk["up"](h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = self._res(blk["res"], h, emb)
                if "attn" in blk:
                    h = self._attn(blk["attn"], h, ctx, ctx_keep)
        return self.out_conv(F.silu(self.out_norm(h)))


class EDMPrecond(nn.Module):
    """EDM preconditioning wrapper around the UNet."""

    def __init__(self, unet, sigma_data=0.5):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def forward(self, x, sigma, a_idx, o_idx, ctx_seq, ctx_pooled, ctx_pad_mask):
        sigma = sigma.view(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        F_x = self.unet(c_in * x, c_noise, a_idx, o_idx,
                        ctx_seq, ctx_pooled, ctx_pad_mask)
        return c_skip * x + c_out * F_x


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    net = EDMPrecond(UNet())
    print(f"params: {count_params(net) / 1e6:.2f}M")
