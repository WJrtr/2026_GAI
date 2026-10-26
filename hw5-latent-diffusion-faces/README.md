# HW5 — Latent Diffusion Face Generation (from scratch)

## Assignment

Train an image generative model **without any pretrained generator** (only the provided Stable-Diffusion VAE may be used) on a face dataset, then generate 3,000 images at 256×256. Graded by FID against a held-out set.

> Summarized from the course assignment spec (NYCU Generative AI, Spring 2026). The original spec PDFs are not redistributed.


A latent DDPM U-Net trained from scratch on Stable-Diffusion VAE latents to generate 256×256 face images. **Local FID: 9.84.**

![samples](results/samples_grid.png)

Run all commands from this directory:

```powershell
cd hw5-latent-diffusion-faces
```

All paths below are relative to this folder.

## Model Weights

The model weights are too large to include directly, so they are provided via a Google Drive link. See [`model_weight.md`](model_weight.md) for the download link and details.

🔗 **[Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1JuKmyQX8ypXjQWz3ts1CswNq369iBUZR?usp=sharing)**

After downloading, place the files under `model/` so the paths below resolve correctly.

## Best Result

- Checkpoint: `model/unet_step_027000`
- Latent normalization stats: `model/latent_stats.pt`
- Train config: `model/train_config.json`
- Sampler: DDPM
- Sampling steps: 250
- Noise scale: 0.892
- Seed: 20260528
- Local FID: 9.843272882006055

## Package Contents

- `results/samples_grid.png`: 32 samples from the 3000 submitted images (the full set is omitted for size)
- `model/`: `unet_step_027000`, `latent_stats.pt`, `train_config.json` (weights provided via the Google Drive link above — see [`model_weight.md`](model_weight.md))
- `code/`: final training/generation script, `hw5_latent_ddpm_aug.py`

## Inference

To regenerate the submitted images, run:

```powershell
python code/hw5_latent_ddpm_aug.py generate --sampler ddpm --checkpoint_dir model/unet_step_027000 --latent_stats model/latent_stats.pt --sampling_steps 250 --noise_scale 0.892 --seed 20260528 --num_samples 3000 --batch_size 64 --output_dir results_reproduced --overwrite
```

Validate the regenerated folder:

```powershell
python code/hw5_latent_ddpm_aug.py validate --output_dir results_reproduced --num_samples 3000 --image_size 256
```

## Training

To retrain from scratch, place the official training PNGs under `public_data/images/` relative to this directory, then run:

```powershell
python code/hw5_latent_ddpm_aug.py train --objective ddpm --normalize_latents --image_dir public_data/images --latent_cache outputs/cache/sd_vae_latents_256_aug_hflip.pt --cache_hflip --aug_label --aug_copies 2 --aug_rot_deg 3.0 --aug_trans 0.05 --aug_scale 0.03 --aug_train_prob 0.15 --input_perturb_gamma 0.1 --unet_arch attn32_l3_xxwide --attention_head_dim 32 --dropout 0.05 --ema_decay 0.9995 --lr 1e-4 --weight_decay 0.0001 --batch_size 64 --max_steps 30000 --save_steps 1000 --seed 20260525 --output_dir outputs/latent_ddpm_v36_retrain
```

After training, use the generated checkpoint and latent stats in the same inference command pattern:

```powershell
python code/hw5_latent_ddpm_aug.py generate --sampler ddpm --checkpoint_dir outputs/latent_ddpm_v36_retrain/checkpoints/unet_step_027000 --latent_stats outputs/latent_ddpm_v36_retrain/latent_stats.pt --sampling_steps 250 --noise_scale 0.892 --seed 20260528 --num_samples 3000 --batch_size 64 --output_dir outputs/retrain_results --overwrite
```

## Notes

- The generated images are RGB PNG files at 256x256.
- The code uses the allowed Stable Diffusion VAE workflow (`stabilityai/sd-vae-ft-mse`) for latent encoding/decoding.
- No pretrained U-Net, CLIP, face recognition model, or external identity model is included.
