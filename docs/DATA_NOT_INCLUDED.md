# Data And Large Files Not Included

This handoff package intentionally excludes large files.

## Not Included

```text
datasets/
Stable Diffusion 2.1 model files
Stable Diffusion 2.1 VAE files
*.pth
*.pth.tar
checkpoints
refiner checkpoints
raw generated images
attacked outputs
train pairs
LPIPS image folders
large logs or temporary outputs
```

## Why They Are Not Included

The goal of this package is to be small enough to download and share. Large files should be stored separately on lab storage, cloud storage, or external drives.

## Required External Resources

See:

```text
99_MISC_DOCS/EXTERNAL_RESOURCES.md
```

Important local reference paths from the original machine:

```bash
/home/jeffrey/projects/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked
/home/jeffrey/projects/VAE_backups/sd21_original_vae_20260604
/home/jeffrey/projects/Test/DiffW-main/stable-diffusion-2-1-base
```

These paths may need to be recreated or replaced on a new machine.

## For Dataset Generalization

The recommended first task uses external datasets such as Kodak24, BSD100 / BSDS subset, DIV2K validation, and one additional natural-image dataset. These datasets are not included in this package.

