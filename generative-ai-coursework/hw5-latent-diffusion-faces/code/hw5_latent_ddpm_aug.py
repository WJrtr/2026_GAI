import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm


VAE_ID = "stabilityai/sd-vae-ft-mse"
LATENT_SIZE = 32
LATENT_CHANNELS = 4


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, image_size=256, limit_images=0):
        self.image_dir = Path(image_dir)
        self.paths = sorted(p for p in self.image_dir.glob("*.png") if p.is_file())
        if limit_images:
            self.paths = self.paths[:limit_images]
        if not self.paths:
            raise ValueError(f"No PNG images found in {self.image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), self.paths[idx].name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def unet_arch_config(arch):
    if arch == "base":
        return {
            "block_out_channels": (128, 256, 384, 512),
            "layers_per_block": 2,
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            "attention": "32x32 no attention, 16x16 and 8x8 attention",
        }
    if arch == "attn32":
        return {
            "block_out_channels": (128, 256, 384, 512),
            "layers_per_block": 2,
            "down_block_types": ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            "attention": "32x32, 16x16, and 8x8 attention",
        }
    if arch == "attn32_l3":
        return {
            "block_out_channels": (128, 256, 384, 512),
            "layers_per_block": 3,
            "down_block_types": ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            "attention": "32x32, 16x16, and 8x8 attention",
        }
    if arch == "attn32_l3_wide":
        return {
            "block_out_channels": (160, 320, 448, 576),
            "layers_per_block": 3,
            "down_block_types": ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            "attention": "32x32, 16x16, and 8x8 attention",
        }
    if arch == "attn32_l3_xwide":
        return {
            "block_out_channels": (192, 384, 512, 640),
            "layers_per_block": 3,
            "down_block_types": ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            "attention": "32x32, 16x16, and 8x8 attention",
        }
    if arch == "attn32_l3_xxwide":
        return {
            "block_out_channels": (224, 448, 576, 704),
            "layers_per_block": 3,
            "down_block_types": ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            "attention": "32x32, 16x16, and 8x8 attention",
        }
    if arch == "attn32_l3_xxwide_noattn32":
        # Same xxwide capacity, but NO attention at the 32x32 (highest) latent
        # resolution. The 32x32 self-attention (1024 tokens, O(N^2)) is the most
        # expensive layer; ADM/EDM typically do not attend at the highest
        # resolution, and dropping it both speeds training and tends to reduce
        # overfitting on small datasets. Attention kept at 16x16 and 8x8.
        return {
            "block_out_channels": (224, 448, 576, 704),
            "layers_per_block": 3,
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            "attention": "no 32x32 attention; 16x16 and 8x8 attention",
        }
    raise ValueError(f"Unknown U-Net architecture: {arch}")


def unet_build_kwargs(arch="base", dropout=0.0, resnet_time_scale_shift="default", attention_head_dim=8):
    config = unet_arch_config(arch)
    return dict(
        sample_size=LATENT_SIZE,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=config["layers_per_block"],
        block_out_channels=config["block_out_channels"],
        down_block_types=config["down_block_types"],
        up_block_types=config["up_block_types"],
        attention_head_dim=attention_head_dim,
        norm_num_groups=32,
        dropout=dropout,
        resnet_time_scale_shift=resnet_time_scale_shift,
    )


def build_unet(arch="base", dropout=0.0, resnet_time_scale_shift="default", attention_head_dim=8):
    # resnet_time_scale_shift="scale_shift" turns on AdaGN-style FiLM timestep
    # injection (ADM showed removing AdaGN cost ~2 FID). dropout adds small-data
    # regularization. Both default to the previous behavior ("default", 0.0).
    return UNet2DModel(**unet_build_kwargs(arch, dropout, resnet_time_scale_shift, attention_head_dim))


# ===========================================================================
# Non-leaking geometric augmentation with augmentation-label conditioning.
#
# WHY: with only ~5.5k images, geometric augmentation (small rotate/translate/
# scale) adds data variety. But unlike horizontal flip (a symmetry of faces),
# rotate/translate/scale are NOT symmetries of an ALIGNED face set: training on
# them naively makes the model GENERATE misaligned faces, which raises FID
# against the aligned reference (this is "augmentation leakage"). The fix
# (Karras et al., EDM) is to feed the applied transform as a conditioning
# vector ("aug-label") so the model attributes the augmentation to the label,
# then set the label to ZERO at inference -> the model generates canonical,
# aligned faces while still having benefited from the extra variety.
#
# HOW (this implementation): the aug-label is a 5-D vector [rot, tx, ty,
# log_sx, log_sy], normalized so the all-zero vector == identity (no aug). It
# is projected by a small MLP to the U-Net's time-embedding dimension and added
# to the time embedding via the U-Net's class-conditioning path
# (class_embed_type="identity"). The MLP's last layer is zero-initialized so
# conditioning starts as a no-op (stable).
#
# CRITICAL — augmentation is done in PIXEL space at cache-build time (transform
# the image, THEN VAE-encode), exactly like the proven horizontal-flip cache.
# This guarantees every augmented latent is a real, on-manifold encoding of a
# real (transformed) image. Doing affine directly on latents would be WRONG for
# rotation/scale (conv VAEs are not rotation/scale equivariant), producing
# off-manifold "fake" latents. The cache stores, per latent, its aug-label;
# originals and flips get the zero label. At inference the label is zero, so
# generation targets the canonical (un-augmented) distribution.
# ===========================================================================

AUG_DIM = 5


def _unet_forward(model, sample, timestep, aug_label=None):
    # Routes the aug-label only when present, so the plain (non-aug) U-Net is
    # called with exactly its original signature.
    if aug_label is None:
        return model(sample, timestep).sample
    return model(sample, timestep, aug_label=aug_label).sample


class AugUNet(nn.Module):
    """UNet2DModel + a small aug-label MLP injected via the identity class path."""

    def __init__(self, base_kwargs, aug_dim=AUG_DIM):
        super().__init__()
        self.unet = UNet2DModel(**base_kwargs, class_embed_type="identity")
        if self.unet.class_embedding is None:
            raise RuntimeError(
                "This diffusers version did not create an identity class embedding "
                "for UNet2DModel(class_embed_type='identity'); aug-label conditioning "
                "cannot be wired this way. Report the diffusers version."
            )
        time_embed_dim = self.unet.time_embedding.linear_2.out_features
        self.aug_dim = aug_dim
        self.aug_mlp = nn.Sequential(
            nn.Linear(aug_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # zero-init the last layer: aug conditioning starts as an exact no-op.
        nn.init.zeros_(self.aug_mlp[-1].weight)
        nn.init.zeros_(self.aug_mlp[-1].bias)

    @property
    def config(self):
        return self.unet.config

    def forward(self, sample, timestep, aug_label=None):
        if aug_label is None:
            aug_label = sample.new_zeros((sample.shape[0], self.aug_dim))
        aug_emb = self.aug_mlp(aug_label.to(self.aug_mlp[0].weight.dtype))
        return self.unet(sample, timestep, class_labels=aug_emb)

    def save_pretrained(self, save_directory, safe_serialization=True):
        self.unet.save_pretrained(save_directory, safe_serialization=safe_serialization)
        torch.save(
            {"aug_dim": self.aug_dim, "state_dict": self.aug_mlp.state_dict()},
            Path(save_directory) / "aug_mlp.pt",
        )

    @classmethod
    def from_pretrained_dir(cls, checkpoint_dir, device, dtype):
        unet = UNet2DModel.from_pretrained(checkpoint_dir)
        meta = torch.load(Path(checkpoint_dir) / "aug_mlp.pt", map_location="cpu")
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.unet = unet
        obj.aug_dim = int(meta["aug_dim"])
        time_embed_dim = unet.time_embedding.linear_2.out_features
        obj.aug_mlp = nn.Sequential(
            nn.Linear(obj.aug_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        obj.aug_mlp.load_state_dict(meta["state_dict"])
        return obj.to(device=device, dtype=dtype)


def pixel_space_affine(pixel_values, args):
    """Random affine applied in PIXEL space (used to build the augmented cache).

    Every sample is augmented (these are the augmented copies). Returns
    (aug_pixels, aug_label) where aug_label is [B, AUG_DIM] normalized to ~[-1, 1].
    The affine math runs in fp32 for grid_sample stability; the caller casts the
    result back to the VAE dtype before encoding.
    """
    b = pixel_values.shape[0]
    device = pixel_values.device
    rot_max = math.radians(args.aug_rot_deg)
    rot = (torch.rand(b, device=device) * 2 - 1) * rot_max
    tx = (torch.rand(b, device=device) * 2 - 1) * args.aug_trans
    ty = (torch.rand(b, device=device) * 2 - 1) * args.aug_trans
    log_sx = (torch.rand(b, device=device) * 2 - 1) * args.aug_scale
    log_sy = (torch.rand(b, device=device) * 2 - 1) * args.aug_scale
    sx = torch.exp(log_sx)
    sy = torch.exp(log_sy)
    cos = torch.cos(rot)
    sin = torch.sin(rot)
    theta = torch.zeros(b, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0] = cos * sx
    theta[:, 0, 1] = -sin * sy
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * sx
    theta[:, 1, 1] = cos * sy
    theta[:, 1, 2] = ty
    src = pixel_values.float()
    grid = F.affine_grid(theta, list(src.shape), align_corners=False)
    aug = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    aug_label = torch.stack(
        [
            rot / max(rot_max, 1e-6),
            tx / max(args.aug_trans, 1e-6),
            ty / max(args.aug_trans, 1e-6),
            log_sx / max(args.aug_scale, 1e-6),
            log_sy / max(args.aug_scale, 1e-6),
        ],
        dim=1,
    )
    return aug, aug_label


def build_scheduler(prediction_type="epsilon"):
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type=prediction_type,
    )


def build_sampling_scheduler(name, prediction_type="epsilon", timestep_spacing="leading", dpm_use_karras_sigmas=False):
    common_kwargs = dict(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type=prediction_type,
    )
    if name == "ddpm":
        return DDPMScheduler(clip_sample=False, timestep_spacing=timestep_spacing, **common_kwargs)
    if name == "ddim":
        return DDIMScheduler(clip_sample=False, timestep_spacing=timestep_spacing, **common_kwargs)
    if name == "dpmpp":
        if timestep_spacing != "leading":
            raise ValueError("Custom timestep spacing is only implemented for DDPM/DDIM samplers.")
        return DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            use_karras_sigmas=dpm_use_karras_sigmas,
            **common_kwargs,
        )
    raise ValueError(f"Unknown sampler: {name}")


def build_karras_timesteps(scheduler, num_steps, rho):
    if num_steps < 2:
        raise ValueError("Karras timestep schedule requires at least 2 steps.")

    alphas = scheduler.alphas_cumprod.float().cpu().numpy()
    sigmas = np.sqrt((1.0 - alphas) / alphas)
    sigma_min = float(sigmas[0])
    sigma_max = float(sigmas[-1])
    ramp = np.linspace(0.0, 1.0, num_steps)
    target_sigmas = (sigma_max ** (1.0 / rho) + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    nearest = np.abs(sigmas[:, None] - target_sigmas[None, :]).argmin(axis=0)

    timesteps = []
    previous = scheduler.config.num_train_timesteps
    for timestep in nearest:
        timestep = int(timestep)
        if timestep < previous:
            timesteps.append(timestep)
            previous = timestep
    if timesteps[-1] != 0:
        timesteps.append(0)
    if len(timesteps) < 2:
        raise ValueError("Karras timestep schedule collapsed to fewer than 2 unique steps.")
    return timesteps


# ===========================================================================
# EDM (Karras et al. 2022, "Elucidating the Design Space...") reformulation.
#
# IMPORTANT: these pieces are mutually dependent and MUST be used together.
# Preconditioning (c_skip/c_out/c_in/c_noise), lognormal-sigma training noise,
# the EDM loss weight, the rho sigma-schedule, and the Heun sampler form ONE
# coherent method. Bolting any single piece onto a VP/DDPM epsilon model (as in
# the prior lognormal-sigma and Karras-remap experiments) regresses; that is a
# wiring problem, not evidence that EDM is ineffective.
#
# Everything operates in the SAME per-channel-normalized latent space as the
# DDPM path. sigma_data should be measured on the normalized latents (~1.0).
# Formulas verified against NVlabs/edm (training/networks.py EDMPrecond,
# training/loss.py EDMLoss, generate.py edm_sampler).
# ===========================================================================


def edm_preconditioning(sigma, sigma_data):
    sigma = sigma.float()
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def edm_denoise(unet, x, sigma, sigma_data, aug_label=None):
    """D_theta(x; sigma) = c_skip*x + c_out * F_theta(c_in*x ; c_noise).

    `unet` is the raw network F_theta. `sigma` is a per-sample tensor (or scalar)
    of NOISE LEVELS (not DDPM timestep indices). c_noise is fed where the U-Net
    expects its timestep; the positional time-embedding accepts these floats.
    """
    if sigma.ndim == 0:
        sigma = sigma.reshape(1).expand(x.shape[0])
    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, sigma_data)
    c_skip = c_skip.view(-1, 1, 1, 1).to(x.dtype)
    c_out = c_out.view(-1, 1, 1, 1).to(x.dtype)
    c_in = c_in.view(-1, 1, 1, 1).to(x.dtype)
    net_dtype = next(unet.parameters()).dtype
    f = _unet_forward(unet, (c_in * x).to(net_dtype), c_noise.to(net_dtype), aug_label)
    return c_skip * x + c_out * f.to(x.dtype)


def edm_sigma_schedule(num_steps, sigma_min, sigma_max, rho, device):
    # rho-schedule from sigma_max down to sigma_min, with a trailing 0 (t_N = 0).
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    inv_rho = 1.0 / rho
    sigmas = (
        sigma_max ** inv_rho
        + step_indices / max(num_steps - 1, 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)
    ) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)])


@torch.no_grad()
def edm_sample(
    unet,
    init_noise,
    sigma_data,
    num_steps,
    sigma_min,
    sigma_max,
    rho,
    s_churn=0.0,
    s_min=0.0,
    s_max=float("inf"),
    s_noise=1.0,
    aug_label=None,
):
    """EDM 2nd-order Heun sampler (deterministic when s_churn == 0, otherwise the
    EDM stochastic sampler). Returns final clean latents in normalized space.
    NFE ~= 2*num_steps - 1. The init noise is scaled internally by sigma_max,
    so the DDPM `noise_scale` trick does NOT apply here."""
    device = init_noise.device
    t_steps = edm_sigma_schedule(num_steps, sigma_min, sigma_max, rho, device)
    x = init_noise.double() * t_steps[0]
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        gamma = 0.0
        if s_churn > 0 and s_min <= float(t_cur) <= s_max:
            gamma = min(s_churn / num_steps, math.sqrt(2.0) - 1.0)
        t_hat = t_cur * (1.0 + gamma)
        x_hat = x
        if gamma > 0:
            x_hat = x + torch.sqrt(t_hat ** 2 - t_cur ** 2) * (s_noise * torch.randn_like(x))
        denoised = edm_denoise(unet, x_hat.float(), t_hat, sigma_data, aug_label).double()
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if float(t_next) > 0:  # 2nd-order Heun correction (skip at the final step)
            denoised = edm_denoise(unet, x_next.float(), t_next, sigma_data, aug_label).double()
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * 0.5 * (d_cur + d_prime)
        x = x_next
    return x.float()


def load_vae(device, dtype):
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae


@torch.no_grad()
def encode_latents(args):
    want_aug = bool(getattr(args, "aug_label", False)) and int(getattr(args, "aug_copies", 0)) > 0
    cache_path = Path(args.latent_cache)
    if cache_path.exists() and not args.rebuild_cache:
        data = torch.load(cache_path, map_location="cpu")
        latents = data["latents"]
        aug_labels = data.get("aug_labels", None)
        extra = f" (+aug_labels {tuple(aug_labels.shape)})" if aug_labels is not None else ""
        print(f"Loaded latent cache {cache_path} with shape {tuple(latents.shape)}{extra}")
        if bool(getattr(args, "aug_label", False)) and aug_labels is None:
            raise RuntimeError(
                f"--aug_label is set but cache {cache_path} has no aug-labels. "
                f"Point --latent_cache at a new path or pass --rebuild_cache to build the augmented cache."
            )
        if bool(getattr(args, "aug_label", False)):
            expected_aug = {
                "aug_copies": int(args.aug_copies),
                "cache_hflip": bool(args.cache_hflip),
                "image_size": int(args.image_size),
                "latent_mode": args.latent_mode,
                "aug_params": {
                    "rot_deg": float(args.aug_rot_deg),
                    "trans": float(args.aug_trans),
                    "scale": float(args.aug_scale),
                },
            }
            actual_aug = {
                "aug_copies": int(data.get("aug_copies", -1)),
                "cache_hflip": bool(data.get("cache_hflip", False)),
                "image_size": int(data.get("image_size", -1)),
                "latent_mode": data.get("latent_mode", None),
                "aug_params": data.get("aug_params", {}),
            }
            if actual_aug != expected_aug:
                raise RuntimeError(
                    f"Aug cache metadata mismatch for {cache_path}.\n"
                    f"Expected: {expected_aug}\n"
                    f"Actual:   {actual_aug}\n"
                    "Use a new --latent_cache path or pass --rebuild_cache."
                )
        elif aug_labels is not None:
            raise RuntimeError(
                f"Cache {cache_path} contains aug-labels, but --aug_label is off. "
                "Use --aug_label, or point --latent_cache at a non-aug cache to avoid leaking geometry augmentation."
            )
        return latents, aug_labels

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical VAE latent encoding.")

    dtype = torch.float16 if args.amp else torch.float32
    vae = load_vae(device, dtype)
    dataset = ImageFolderDataset(args.image_dir, image_size=args.image_size, limit_images=args.limit_images)
    dataloader = DataLoader(dataset, batch_size=args.encode_batch_size, shuffle=False, num_workers=args.num_workers)

    def _encode(px):
        posterior = vae.encode(px).latent_dist
        v = posterior.mean if args.latent_mode == "mean" else posterior.sample()
        return (v * vae.config.scaling_factor).cpu().to(torch.float16)

    all_latents = []
    all_labels = []
    all_names = []
    for pixel_values, names in tqdm(dataloader, desc="Encoding VAE latents"):
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        b = pixel_values.shape[0]
        all_latents.append(_encode(pixel_values))
        all_names.extend(names)
        if want_aug:
            all_labels.append(torch.zeros(b, AUG_DIM))
        if args.cache_hflip:
            flipped_values = torch.flip(pixel_values, dims=[3])
            all_latents.append(_encode(flipped_values))
            all_names.extend([f"hflip_{name}" for name in names])
            if want_aug:
                all_labels.append(torch.zeros(b, AUG_DIM))
        if want_aug:
            for k in range(int(args.aug_copies)):
                aug_pixels, aug_label = pixel_space_affine(pixel_values, args)
                all_latents.append(_encode(aug_pixels.to(dtype)))
                all_names.extend([f"aug{k}_{name}" for name in names])
                all_labels.append(aug_label.detach().cpu().to(torch.float32))

    latents = torch.cat(all_latents, dim=0).contiguous()
    aug_labels = torch.cat(all_labels, dim=0).contiguous() if want_aug else None
    payload = {
        "latents": latents,
        "image_names": all_names,
        "vae": VAE_ID,
        "scaling_factor": float(vae.config.scaling_factor),
        "image_size": args.image_size,
        "cache_hflip": args.cache_hflip,
        "latent_mode": args.latent_mode,
    }
    if aug_labels is not None:
        payload["aug_labels"] = aug_labels
        payload["aug_copies"] = int(args.aug_copies)
        payload["aug_params"] = {
            "rot_deg": args.aug_rot_deg,
            "trans": args.aug_trans,
            "scale": args.aug_scale,
        }
    torch.save(payload, cache_path)
    extra = f" (+aug_labels {tuple(aug_labels.shape)})" if aug_labels is not None else ""
    print(f"Saved latent cache {cache_path} with shape {tuple(latents.shape)}{extra}")
    return latents, aug_labels


def init_ema(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }


@torch.no_grad()
def update_ema(model, ema_state, decay):
    model_state = model.state_dict()
    for name, value in ema_state.items():
        value.mul_(decay).add_(model_state[name].detach(), alpha=1.0 - decay)


def ema_decay_for_step(base_decay, step):
    warmup_decay = (1 + step) / (10 + step)
    return min(base_decay, warmup_decay)


def lr_for_step(args, step):
    if args.lr_schedule == "constant":
        return args.lr
    if args.lr_schedule == "cosine":
        if args.max_steps <= 1:
            return args.lr_final
        progress = min(max(step / (args.max_steps - 1), 0.0), 1.0)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr_final + (args.lr - args.lr_final) * scale
    raise ValueError(f"Unknown LR schedule: {args.lr_schedule}")


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_unet_checkpoint(model, ema_state, output_dir, name):
    checkpoint_dir = Path(output_dir) / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    current_state = model.state_dict()
    backup = {key: value.detach().clone() for key, value in current_state.items() if key in ema_state}
    merged = dict(current_state)
    merged.update(ema_state)
    model.load_state_dict(merged, strict=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    current_state = model.state_dict()
    current_state.update(backup)
    model.load_state_dict(current_state, strict=True)
    return checkpoint_dir


def write_train_config(args, output_dir):
    config = {key: value for key, value in vars(args).items() if key != "func"}
    unet_config = unet_arch_config(args.unet_arch)
    config["vae"] = VAE_ID
    config["latent_shape"] = [LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE]
    config["unet"] = {
        "arch": args.unet_arch,
        "block_out_channels": list(unet_config["block_out_channels"]),
        "layers_per_block": unet_config["layers_per_block"],
        "attention": unet_config["attention"],
        "attention_head_dim": args.attention_head_dim,
    }
    path = Path(output_dir) / "train_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def normalize_latents_if_requested(latents, args, output_dir):
    if not args.normalize_latents:
        return latents

    mean = latents.float().mean(dim=(0, 2, 3), keepdim=True)
    std = latents.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    stats = {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "source_cache": args.latent_cache,
    }
    stats_path = Path(output_dir) / "latent_stats.pt"
    torch.save(stats, stats_path)
    print(f"Saved latent normalization stats to {stats_path}")
    return ((latents.float() - mean) / std).to(torch.float16)


def loss_weights_for_timesteps(args, alphas_cumprod, timesteps, prediction_type):
    if args.loss_weighting == "none":
        return None

    alpha = alphas_cumprod[timesteps].float()
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    if args.loss_weighting == "min_snr":
        clipped = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma))
        denom = snr + 1.0 if prediction_type == "v_prediction" else snr
        return clipped / denom.clamp_min(1e-8)
    if args.loss_weighting == "p2":
        return (args.p2_k + snr).pow(-args.p2_gamma)
    raise ValueError(f"Unknown loss weighting: {args.loss_weighting}")


def sample_training_timesteps(args, train_sigmas, batch_size, device):
    if args.timestep_sampling == "uniform":
        return torch.randint(
            0,
            train_sigmas.shape[0],
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
    if args.timestep_sampling == "lognormal_sigma":
        targets = torch.exp(torch.randn(batch_size, device=device) * args.lognormal_std + args.lognormal_mean)
        targets = targets.clamp(train_sigmas[0], train_sigmas[-1])
        upper = torch.searchsorted(train_sigmas, targets).clamp(max=train_sigmas.shape[0] - 1)
        lower = (upper - 1).clamp(min=0)
        lower_diff = (targets - train_sigmas[lower]).abs()
        upper_diff = (targets - train_sigmas[upper]).abs()
        return torch.where(upper_diff < lower_diff, upper, lower).long()
    raise ValueError(f"Unknown timestep sampling: {args.timestep_sampling}")


def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical training.")
    if args.input_perturb_gamma < 0:
        raise ValueError("--input_perturb_gamma must be >= 0.")
    if args.input_perturb_gamma > 0 and args.objective != "ddpm":
        raise ValueError("--input_perturb_gamma is only implemented for --objective ddpm.")
    if args.aug_train_prob is not None and not (0.0 <= args.aug_train_prob <= 1.0):
        raise ValueError("--aug_train_prob must be between 0 and 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_train_config(args, output_dir)

    use_aug = bool(getattr(args, "aug_label", False))
    latents, aug_labels = encode_latents(args)
    latents = normalize_latents_if_requested(latents, args, output_dir)
    if use_aug:
        if aug_labels is None:
            raise RuntimeError("--aug_label requires --aug_copies >= 1 (so the cache contains augmented copies).")
        if aug_labels.shape[0] != latents.shape[0]:
            raise RuntimeError(
                f"aug_labels ({aug_labels.shape[0]}) and latents ({latents.shape[0]}) length mismatch; rebuild the cache."
            )
        dataset = TensorDataset(latents, aug_labels.to(torch.float32))
    else:
        if args.aug_train_prob is not None:
            raise ValueError("--aug_train_prob requires --aug_label.")
        dataset = TensorDataset(latents)
    sampler = None
    if use_aug and args.aug_train_prob is not None:
        label_energy = aug_labels.float().abs().sum(dim=1)
        canonical_mask = label_energy == 0
        aug_mask = ~canonical_mask
        canonical_count = int(canonical_mask.sum().item())
        aug_count = int(aug_mask.sum().item())
        if canonical_count == 0 or aug_count == 0:
            raise RuntimeError(
                "--aug_train_prob requires both zero-label canonical samples and nonzero augmented samples."
            )
        weights = torch.empty(len(aug_labels), dtype=torch.double)
        weights[canonical_mask] = (1.0 - args.aug_train_prob) / canonical_count
        weights[aug_mask] = args.aug_train_prob / aug_count
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(aug_labels),
            replacement=True,
            generator=sampler_generator,
        )
        print(
            f"[aug-label] weighted training sampler ON: nonzero aug-label target probability "
            f"{args.aug_train_prob:.3f} ({canonical_count} canonical, {aug_count} augmented cache rows)."
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    use_aug = bool(getattr(args, "aug_label", False))  # (defined above; kept for locality)
    if args.init_checkpoint:
        if use_aug:
            raise RuntimeError("--aug_label is not supported together with --init_checkpoint (train fresh).")
        unet = UNet2DModel.from_pretrained(args.init_checkpoint).to(device)
        print(f"Initialized U-Net from {args.init_checkpoint}")
    elif use_aug:
        unet = AugUNet(
            unet_build_kwargs(
                args.unet_arch,
                args.dropout,
                args.resnet_time_scale_shift,
                args.attention_head_dim,
            ),
            aug_dim=AUG_DIM,
        ).to(device)
        print(
            f"[aug-label] pixel-space geometric augmentation ON (copies={args.aug_copies}/image, "
            f"rot=±{args.aug_rot_deg}deg, trans=±{args.aug_trans}, log-scale=±{args.aug_scale}); "
            f"aug-label is zeroed at inference."
        )
    else:
        unet = build_unet(
            args.unet_arch,
            dropout=args.dropout,
            resnet_time_scale_shift=args.resnet_time_scale_shift,
            attention_head_dim=args.attention_head_dim,
        ).to(device)
    unet.train()
    # Optional speed levers. forward_unet shares parameters with unet, so EMA,
    # checkpoint saving, and state_dict keep operating on the eager `unet`.
    forward_unet = torch.compile(unet) if getattr(args, "compile", False) else unet
    scheduler = build_scheduler(args.prediction_type)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    train_sigmas = torch.sqrt((1.0 - alphas_cumprod.float()) / alphas_cumprod.float().clamp_min(1e-8))
    if args.input_perturb_gamma > 0:
        print(
            f"[input-perturbation] ON: gamma={args.input_perturb_gamma:.4f}; "
            "extra noise is applied only to the DDPM model input, not to the training target."
        )
    edm_sigma_data = None
    if args.objective == "edm":
        if args.edm_sigma_data and args.edm_sigma_data > 0:
            edm_sigma_data = float(args.edm_sigma_data)
            source = "user-provided"
        else:
            edm_sigma_data = float(latents.float().std().item())
            source = "measured on normalized latents"
        edm_meta = {
            "sigma_data": edm_sigma_data,
            "P_mean": args.edm_p_mean,
            "P_std": args.edm_p_std,
            "sigma_min": args.edm_sigma_min,
            "sigma_max": args.edm_sigma_max,
            "rho": args.edm_rho,
        }
        with (output_dir / "edm_meta.json").open("w", encoding="utf-8") as f:
            json.dump(edm_meta, f, indent=2)
        print(
            f"[EDM] sigma_data={edm_sigma_data:.5f} ({source}). "
            f"When generating, pass: --sampler edm --edm_sigma_data {edm_sigma_data:.5f}"
        )
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema_state = init_ema(unet)

    log_path = output_dir / "train_log.csv"
    log_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["step", "loss", "lr", "seconds"])

        step = 0
        loss_window = []
        start_time = time.time()
        progress = tqdm(total=args.max_steps, desc="Training latent DDPM")
        while step < args.max_steps:
            for batch in dataloader:
                current_lr = lr_for_step(args, step)
                set_optimizer_lr(optimizer, current_lr)
                if use_aug:
                    batch_latents, aug_label = batch
                    aug_label = aug_label.to(device=device, dtype=torch.float32, non_blocking=True)
                else:
                    (batch_latents,) = batch
                    aug_label = None
                batch_latents = batch_latents.to(device=device, dtype=torch.float32, non_blocking=True)
                target_latents = batch_latents  # augmentation is already baked into the cached latents (pixel space)

                if args.objective == "ddpm":
                    noise = torch.randn_like(target_latents)
                    timesteps = sample_training_timesteps(args, train_sigmas, target_latents.shape[0], device)
                    input_noise = noise
                    if args.input_perturb_gamma > 0:
                        input_noise = noise + args.input_perturb_gamma * torch.randn_like(noise)
                    noisy_latents = scheduler.add_noise(target_latents, input_noise, timesteps)
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        model_pred = _unet_forward(forward_unet, noisy_latents, timesteps, aug_label)
                        if args.prediction_type == "v_prediction":
                            target = scheduler.get_velocity(target_latents, noise, timesteps)
                        else:
                            target = noise
                        mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        weights = loss_weights_for_timesteps(args, alphas_cumprod, timesteps, args.prediction_type)
                        if weights is not None:
                            mse = mse * weights.view(-1, 1, 1, 1)
                        loss = mse.mean()
                else:  # edm
                    rnd_normal = torch.randn(target_latents.shape[0], device=device)
                    sigma = torch.exp(rnd_normal * args.edm_p_std + args.edm_p_mean)
                    sigma_b = sigma.view(-1, 1, 1, 1)
                    noised = target_latents + torch.randn_like(target_latents) * sigma_b
                    weight = (sigma ** 2 + edm_sigma_data ** 2) / (sigma * edm_sigma_data) ** 2
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        denoised = edm_denoise(forward_unet, noised, sigma, edm_sigma_data, aug_label)
                        sq_err = (denoised.float() - target_latents.float()) ** 2
                        loss = (weight.view(-1, 1, 1, 1) * sq_err).mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                step += 1
                decay = ema_decay_for_step(args.ema_decay, step)
                update_ema(unet, ema_state, decay)
                loss_window.append(float(loss.detach().cpu()))

                if step % args.log_steps == 0:
                    mean_loss = sum(loss_window[-args.log_steps:]) / min(len(loss_window), args.log_steps)
                    writer.writerow([step, f"{mean_loss:.6f}", f"{current_lr:.8g}", f"{time.time() - start_time:.1f}"])
                    f.flush()
                    progress.set_postfix(loss=f"{mean_loss:.4f}")

                if step % args.save_steps == 0:
                    ckpt = save_unet_checkpoint(unet, ema_state, output_dir, f"unet_step_{step:06d}")
                    print(f"Saved checkpoint {ckpt}")

                progress.update(1)
                if step >= args.max_steps:
                    break
        progress.close()

    final_dir = save_unet_checkpoint(unet, ema_state, output_dir, "unet_final")
    print(f"Saved final checkpoint {final_dir}")


@torch.no_grad()
def generate(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for practical generation.")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not args.overwrite:
            raise RuntimeError(f"{output_dir} is not empty; pass --overwrite only after confirming it is safe.")
        if existing and args.overwrite:
            for item in existing:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.amp else torch.float32
    vae = load_vae(device, dtype)
    use_aug = (Path(args.checkpoint_dir) / "aug_mlp.pt").exists()
    if use_aug:
        unet = AugUNet.from_pretrained_dir(args.checkpoint_dir, device, dtype)
        print("[aug-label] aug-conditioned checkpoint detected; generating with zero aug-label (canonical).")
    else:
        unet = UNet2DModel.from_pretrained(args.checkpoint_dir).to(device=device, dtype=dtype)
    unet.requires_grad_(False)
    unet.eval()

    forward_unet = torch.compile(unet) if getattr(args, "compile", False) else unet
    use_edm = args.sampler == "edm"

    scheduler = None
    karras_timesteps = None
    if not use_edm:
        if args.timestep_schedule == "karras" and args.sampler != "ddpm":
            raise RuntimeError("Karras timestep schedule is currently implemented only for DDPM sampling.")
        timestep_spacing = "leading" if args.timestep_schedule == "karras" else args.timestep_schedule
        scheduler = build_sampling_scheduler(
            args.sampler,
            args.prediction_type,
            timestep_spacing,
            args.dpm_use_karras_sigmas,
        )
        if args.timestep_schedule == "karras":
            karras_timesteps = build_karras_timesteps(scheduler, args.sampling_steps, args.karras_rho)
            print(
                f"Using Karras-style timestep schedule with {len(karras_timesteps)} unique steps "
                f"(requested {args.sampling_steps}, rho={args.karras_rho:g})"
            )
    else:
        print(
            f"[EDM] Heun sampling: steps={args.sampling_steps} (NFE~{2 * args.sampling_steps - 1}), "
            f"sigma_data={args.edm_sigma_data:g}, sigma=[{args.edm_sigma_min:g},{args.edm_sigma_max:g}], "
            f"rho={args.edm_rho:g}, S_churn={args.edm_s_churn:g}. (--noise_scale is ignored for EDM.)"
        )

    eps_scale_end = args.eps_scale if args.eps_scale_end < 0 else args.eps_scale_end
    apply_eps_scale = (not use_edm) and args.prediction_type == "epsilon" and (
        args.eps_scale != 1.0 or eps_scale_end != 1.0
    )
    if apply_eps_scale:
        print(f"[eps-scaling] dividing predicted noise by {args.eps_scale:g} -> {eps_scale_end:g} across steps.")

    to_pil = transforms.ToPILImage()
    latent_mean = None
    latent_std = None
    if args.latent_stats:
        stats = torch.load(args.latent_stats, map_location="cpu")
        latent_mean = stats["mean"].to(device=device, dtype=dtype)
        latent_std = stats["std"].to(device=device, dtype=dtype)
        print(f"Loaded latent normalization stats from {args.latent_stats}")

    sample_idx = 0
    total_batches = math.ceil(args.num_samples / args.batch_size)
    for _ in tqdm(range(total_batches), desc="Generating"):
        current_batch = min(args.batch_size, args.num_samples - sample_idx)

        if use_edm:
            init_noise = torch.randn(
                (current_batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                device=device,
                dtype=dtype,
            )
            edm_aug_label = (
                torch.zeros((current_batch, getattr(unet, "aug_dim", AUG_DIM)), device=device, dtype=dtype)
                if use_aug
                else None
            )
            latents = edm_sample(
                forward_unet,
                init_noise,
                args.edm_sigma_data,
                args.sampling_steps,
                args.edm_sigma_min,
                args.edm_sigma_max,
                args.edm_rho,
                s_churn=args.edm_s_churn,
                s_min=args.edm_s_min,
                s_max=args.edm_s_max,
                s_noise=args.edm_s_noise,
                aug_label=edm_aug_label,
            ).to(dtype)
        else:
            latents = torch.randn(
                (current_batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                device=device,
                dtype=dtype,
            ) * args.noise_scale

            if karras_timesteps is not None:
                scheduler.set_timesteps(timesteps=karras_timesteps, device=device)
            else:
                scheduler.set_timesteps(args.sampling_steps, device=device)
            timesteps = scheduler.timesteps
            num_t = len(timesteps)
            ddpm_aug_label = (
                torch.zeros((current_batch, getattr(unet, "aug_dim", AUG_DIM)), device=device, dtype=dtype)
                if use_aug
                else None
            )
            for j, timestep in enumerate(timesteps):
                model_pred = _unet_forward(forward_unet, latents, timestep, ddpm_aug_label)
                if apply_eps_scale:
                    frac = j / (num_t - 1) if num_t > 1 else 0.0
                    s = args.eps_scale + (eps_scale_end - args.eps_scale) * frac
                    model_pred = model_pred / s
                if isinstance(scheduler, DDIMScheduler):
                    latents = scheduler.step(model_pred, timestep, latents, eta=args.eta).prev_sample
                else:
                    latents = scheduler.step(model_pred, timestep, latents).prev_sample

        if latent_mean is not None and latent_std is not None:
            latents = latents * latent_std + latent_mean

        decoded_latents = latents.to(dtype) / vae.config.scaling_factor
        images = vae.decode(decoded_latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2

        for image in images:
            image_path = output_dir / f"{sample_idx:04d}.png"
            to_pil(image.float().cpu()).save(image_path)
            sample_idx += 1

    print(f"Saved {sample_idx} PNG images to {output_dir}")


def validate(args):
    output_dir = Path(args.output_dir)
    paths = sorted(output_dir.glob("*.png"))
    non_png = [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() != ".png"] if output_dir.exists() else []
    errors = []
    if len(paths) != args.num_samples:
        errors.append(f"expected {args.num_samples} PNG files, found {len(paths)}")
    if non_png:
        errors.append(f"found non-PNG files: {[p.name for p in non_png[:5]]}")

    for path in tqdm(paths, desc="Validating PNGs"):
        with Image.open(path) as image:
            if image.size != (args.image_size, args.image_size):
                errors.append(f"{path.name} has size {image.size}")
                break
            if image.mode != "RGB":
                errors.append(f"{path.name} has mode {image.mode}")
                break

    if errors:
        raise RuntimeError("; ".join(errors))
    print(f"Validation OK: {len(paths)} RGB PNG files at {args.image_size}x{args.image_size}")


def add_common_args(parser):
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)


def main():
    parser = argparse.ArgumentParser(description="Legal latent DDPM baseline for GAI HW5.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--image_dir", default="public_data/images")
    train_parser.add_argument("--image_size", type=int, default=256)
    train_parser.add_argument("--latent_cache", default="outputs/cache/sd_vae_latents_256.pt")
    train_parser.add_argument("--rebuild_cache", action="store_true")
    train_parser.add_argument("--cache_hflip", action="store_true")
    train_parser.add_argument("--latent_mode", choices=["sample", "mean"], default="sample")
    train_parser.add_argument("--normalize_latents", action="store_true")
    train_parser.add_argument("--limit_images", type=int, default=0)
    train_parser.add_argument("--output_dir", default="outputs/latent_ddpm_v1")
    train_parser.add_argument("--init_checkpoint", default="")
    train_parser.add_argument(
        "--unet_arch",
        choices=[
            "base",
            "attn32",
            "attn32_l3",
            "attn32_l3_wide",
            "attn32_l3_xwide",
            "attn32_l3_xxwide",
            "attn32_l3_xxwide_noattn32",
        ],
        default="base",
    )
    train_parser.add_argument("--objective", choices=["ddpm", "edm"], default="ddpm")
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument(
        "--resnet_time_scale_shift", choices=["default", "scale_shift"], default="default"
    )
    train_parser.add_argument(
        "--attention_head_dim",
        type=int,
        default=8,
        help="Attention head dimension for UNet2DModel; default 8 preserves previous runs.",
    )
    train_parser.add_argument("--compile", action="store_true")
    train_parser.add_argument("--edm_sigma_data", type=float, default=0.0,
                              help="EDM sigma_data; 0 = auto-measure std of normalized latents.")
    train_parser.add_argument("--edm_p_mean", type=float, default=-1.2)
    train_parser.add_argument("--edm_p_std", type=float, default=1.2)
    train_parser.add_argument("--edm_sigma_min", type=float, default=0.002)
    train_parser.add_argument("--edm_sigma_max", type=float, default=80.0)
    train_parser.add_argument("--edm_rho", type=float, default=7.0)
    train_parser.add_argument("--aug_label", action="store_true",
                              help="Enable non-leaking pixel-space geometric augmentation with aug-label conditioning.")
    train_parser.add_argument("--aug_copies", type=int, default=2,
                              help="Number of pixel-space geometric augmentations per image baked into the cache "
                                   "(in addition to the original and optional hflip, which carry the zero label).")
    train_parser.add_argument("--aug_rot_deg", type=float, default=10.0,
                              help="Max rotation magnitude in degrees.")
    train_parser.add_argument("--aug_trans", type=float, default=0.1,
                              help="Max translation as a fraction of size (normalized coords).")
    train_parser.add_argument("--aug_scale", type=float, default=0.1,
                              help="Max abs log-scale (e.g. 0.1 ~ +-10.5% zoom).")
    train_parser.add_argument("--aug_train_prob", type=float, default=None,
                              help="When --aug_label is enabled, sample nonzero aug-label rows with this probability "
                                   "during training while keeping the cache unchanged. Default uses the cache's natural ratio.")
    train_parser.add_argument("--input_perturb_gamma", type=float, default=0.0,
                              help="DDPM input perturbation strength: add gamma*N(0,I) to the input noise only; "
                                   "the denoising target remains the original noise.")
    train_parser.add_argument("--max_steps", type=int, default=30000)
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--encode_batch_size", type=int, default=32)
    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    train_parser.add_argument("--lr_final", type=float, default=1e-5)
    train_parser.add_argument("--prediction_type", choices=["epsilon", "v_prediction"], default="epsilon")
    train_parser.add_argument("--loss_weighting", choices=["none", "min_snr", "p2"], default="none")
    train_parser.add_argument("--min_snr_gamma", type=float, default=5.0)
    train_parser.add_argument("--p2_gamma", type=float, default=0.5)
    train_parser.add_argument("--p2_k", type=float, default=1.0)
    train_parser.add_argument("--timestep_sampling", choices=["uniform", "lognormal_sigma"], default="uniform")
    train_parser.add_argument("--lognormal_mean", type=float, default=-1.2)
    train_parser.add_argument("--lognormal_std", type=float, default=1.2)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--ema_decay", type=float, default=0.9999)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--log_steps", type=int, default=50)
    train_parser.add_argument("--save_steps", type=int, default=5000)
    train_parser.set_defaults(func=train)

    gen_parser = subparsers.add_parser("generate")
    add_common_args(gen_parser)
    gen_parser.add_argument("--checkpoint_dir", default="outputs/latent_ddpm_v1/checkpoints/unet_final")
    gen_parser.add_argument("--output_dir", default="scoring_program/input/res")
    gen_parser.add_argument("--num_samples", type=int, default=3000)
    gen_parser.add_argument("--batch_size", type=int, default=32)
    gen_parser.add_argument("--sampling_steps", type=int, default=250)
    gen_parser.add_argument("--sampler", choices=["ddpm", "ddim", "dpmpp", "edm"], default="ddpm")
    gen_parser.add_argument("--noise_scale", type=float, default=1.0)
    gen_parser.add_argument("--compile", action="store_true")
    gen_parser.add_argument("--eps_scale", type=float, default=1.0,
                            help="Epsilon-scaling factor; predicted noise is divided by this. 1.0 = off. "
                                 "Only applies to epsilon-prediction DDPM/DDIM/DPM++ sampling.")
    gen_parser.add_argument("--eps_scale_end", type=float, default=-1.0,
                            help="If >=0, linearly ramp eps_scale to this value across steps; "
                                 "-1 = constant eps_scale.")
    gen_parser.add_argument("--edm_sigma_data", type=float, default=1.0,
                            help="Must match the value printed at EDM training time.")
    gen_parser.add_argument("--edm_sigma_min", type=float, default=0.002)
    gen_parser.add_argument("--edm_sigma_max", type=float, default=80.0)
    gen_parser.add_argument("--edm_rho", type=float, default=7.0)
    gen_parser.add_argument("--edm_s_churn", type=float, default=0.0,
                            help=">0 enables the EDM stochastic sampler.")
    gen_parser.add_argument("--edm_s_min", type=float, default=0.0)
    gen_parser.add_argument("--edm_s_max", type=float, default=float("inf"))
    gen_parser.add_argument("--edm_s_noise", type=float, default=1.0)
    gen_parser.add_argument("--prediction_type", choices=["epsilon", "v_prediction"], default="epsilon")
    gen_parser.add_argument("--eta", type=float, default=0.0)
    gen_parser.add_argument("--timestep_schedule", choices=["leading", "linspace", "trailing", "karras"], default="leading")
    gen_parser.add_argument("--karras_rho", type=float, default=7.0)
    gen_parser.add_argument("--dpm_use_karras_sigmas", action="store_true")
    gen_parser.add_argument("--latent_stats", default="")
    gen_parser.add_argument("--overwrite", action="store_true")
    gen_parser.set_defaults(func=generate)

    val_parser = subparsers.add_parser("validate")
    val_parser.add_argument("--output_dir", default="scoring_program/input/res")
    val_parser.add_argument("--num_samples", type=int, default=3000)
    val_parser.add_argument("--image_size", type=int, default=256)
    val_parser.set_defaults(func=validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
