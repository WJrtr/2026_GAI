# Model Weights

Since the model weights are too large to include directly in this submission,
they are provided via the following Google Drive link:

🔗 **[Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1JuKmyQX8ypXjQWz3ts1CswNq369iBUZR?usp=sharing)**

## Contents

The shared folder contains the trained checkpoint and supporting files:

| File | Description |
| --- | --- |
| `diffusion_pytorch_model.safetensors` | U-Net diffusion model weights (~985 MB) |
| `aug_mlp.pt` | Augmentation conditioning MLP weights |
| `config.json` | Model architecture / configuration |
| `latent_stats.pt` | Latent space normalization statistics |
| `train_config.json` | Training configuration |

## Local Paths

After downloading and placing the files into the `hw5_112511086/` directory,
they should be located at:

- Model checkpoint: `model/unet_step_027000`
- Latent normalization stats: `model/latent_stats.pt`
- Training configuration: `model/train_config.json`

## Checkpoint Info

- Checkpoint: `unet_step_027000`
- Model branch: `latent_ddpm_v36_pixaug_hflip_ip010_augp015_drop005_head32_xxwide_30k`
- Local FID: 9.843272882006055

## Usage

Use the inference command in `README.md` from the `hw5_112511086/` directory.
