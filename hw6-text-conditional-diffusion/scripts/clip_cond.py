"""Precompute frozen CLIP text features for the 100 (animal,object) prompts.

Allowed by the assignment: pretrained CLIP may be used as an auxiliary module
for feature extraction / conditioning. We extract features OFFLINE and cache
them, so CLIP never appears in the training graph (the generative model is still
trained from scratch). Uses the SAME text encoder as the scorer
(open_clip ViT-B-32-quickgelu / openai) for maximal CLIP-T alignment.

Cache contents (indexed by pair_id = a_idx*10 + o_idx):
  seq     : [100, 77, 512]  token-level features (ln_final output) for cross-attn
  pooled  : [100, 512]      projected pooled (CLIP text space) for global cond
  pad_mask: [100, 77] bool  True = padding token to ignore in cross-attn
  null_*  : empty-string ("") equivalents for classifier-free guidance
"""
import os

import torch

from dataset import load_vocab, prompt_of

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "clip_cache.pt")
CLIP_DIM = 512
SEQ_LEN = 77


@torch.no_grad()
def _encode(model, tokenizer, prompts, device):
    toks = tokenizer(prompts).to(device)
    x = model.token_embedding(toks)
    x = x + model.positional_embedding
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)                                  # [B,77,512]
    eot = toks.argmax(-1)                                  # EOT position
    pooled = x[torch.arange(x.shape[0]), eot] @ model.text_projection
    # pad mask: positions strictly after EOT are padding -> True (ignore)
    pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    pad_mask = pos > eot.unsqueeze(1)
    return x.float().cpu(), pooled.float().cpu(), pad_mask.cpu()


def build_cache(device="cuda", save=True):
    import open_clip
    vocab = load_vocab()
    prompts = [None] * 100
    for a in vocab["animals"]:
        for o in vocab["objects"]:
            pid = vocab["a2i"][a] * 10 + vocab["o2i"][o]
            prompts[pid] = prompt_of(a, o)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model = model.to(device).eval()
    seq, pooled, pad = _encode(model, tokenizer, prompts, device)
    nseq, npooled, npad = _encode(model, tokenizer, [""], device)
    cache = {
        "seq": seq, "pooled": pooled, "pad_mask": pad,
        "null_seq": nseq[0], "null_pooled": npooled[0], "null_pad_mask": npad[0],
        "prompts": prompts,
    }
    if save:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        torch.save(cache, CACHE)
        print(f"saved CLIP cache -> {CACHE}")
    return cache


def load_cache(device="cuda"):
    if not os.path.exists(CACHE):
        build_cache(device)
    c = torch.load(CACHE, map_location="cpu")
    return c


class ConditionBank:
    """Assemble per-sample conditioning tensors from the cached CLIP features,
    indexed by pair_id = animal_idx*10 + object_idx. Handles CFG dropout/null."""

    NULL_CLASS = 10

    def __init__(self, cache=None, device="cuda"):
        c = cache or load_cache(device)
        self.seq = c["seq"].to(device)            # [100,77,512]
        self.pooled = c["pooled"].to(device)      # [100,512]
        self.pad = c["pad_mask"].to(device)       # [100,77] bool
        self.null_seq = c["null_seq"].to(device)
        self.null_pooled = c["null_pooled"].to(device)
        self.null_pad = c["null_pad_mask"].to(device)
        self.device = device

    def get(self, a_idx, o_idx, drop=None):
        pid = a_idx * 10 + o_idx
        seq = self.seq[pid]
        pooled = self.pooled[pid]
        pad = self.pad[pid]
        ai, oi = a_idx.clone(), o_idx.clone()
        if drop is not None and drop.any():
            ai = ai.clone(); oi = oi.clone()
            seq = seq.clone(); pooled = pooled.clone(); pad = pad.clone()
            ai[drop] = self.NULL_CLASS
            oi[drop] = self.NULL_CLASS
            seq[drop] = self.null_seq
            pooled[drop] = self.null_pooled
            pad[drop] = self.null_pad
        return dict(a_idx=ai, o_idx=oi, ctx_seq=seq,
                    ctx_pooled=pooled, ctx_pad_mask=pad)

    def null(self, n):
        ai = torch.full((n,), self.NULL_CLASS, device=self.device, dtype=torch.long)
        return dict(a_idx=ai, o_idx=ai.clone(),
                    ctx_seq=self.null_seq.expand(n, -1, -1).contiguous(),
                    ctx_pooled=self.null_pooled.expand(n, -1).contiguous(),
                    ctx_pad_mask=self.null_pad.expand(n, -1).contiguous())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = build_cache(dev)
    print("seq", c["seq"].shape, "pooled", c["pooled"].shape,
          "pad_mask", c["pad_mask"].shape)
    print("null_seq", c["null_seq"].shape, "null_pooled", c["null_pooled"].shape)
    print("pair0 prompt:", c["prompts"][0], "valid tokens:",
          int((~c["pad_mask"][0]).sum()))
