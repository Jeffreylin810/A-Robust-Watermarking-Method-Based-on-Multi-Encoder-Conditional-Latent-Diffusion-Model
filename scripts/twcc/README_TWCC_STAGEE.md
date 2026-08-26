# TWCC StageE EXP53 Full From-Scratch Run

This folder is for running the final StageE setting on a TWCC development container.

Recommended TWCC choices for V100:

- Image type: PyTorch
- Image tag: avoid PyTorch/CUDA 13 images on V100 because they may drop `sm_70` support
- Tested path: `pytorch-25.02-py3:latest` plus the bundled conda environment installer
- Instance type: `cm.super` for the first full replication
- GPU count: 1
- Shared memory: prefer the option with shared memory (`cm.super` shows 30 GB)

The primary run is EXP53 / seed61 because the original result is a strict-pass final StageE setting.

## Expected Inputs

Set these paths inside the TWCC container:

```bash
export TWCC_O256_ROOT=/path/to/O256-PAPER-github-backup
export TWCC_FORMAL_DATASET_ROOT=/path/to/experiment_input_folder/CoCo_256_paper_final/train10000_finaltest1000_locked
export TWCC_MODEL_PATH=/path/to/stable-diffusion-2-1-base
export TWCC_EXP_ROOT=/path/to/output/53_formal_o256_full_from_scratch_stageE_seed61_twcc
export TWCC_PYTHON=python
```

`TWCC_FORMAL_DATASET_ROOT` must contain:

```text
data/CoCo/train/gt
data/CoCo/val/gt
```

`TWCC_MODEL_PATH` should point to a Diffusers-compatible SD2.1 VAE path. The safest layout is:

```text
stable-diffusion-2-1-base/vae/
```

The full SD2.1 pipeline is not required by the O256 StageE scripts, but the VAE must be loadable by `diffusers.AutoencoderKL.from_pretrained`.

## Preflight

Run this before starting a full paid run:

```bash
cd "$TWCC_O256_ROOT"
bash scripts/twcc/install_twcc_o256_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate o256-watermark
export TWCC_PYTHON="$(which python)"
bash scripts/twcc/collect_twcc_env.sh | tee twcc_env_$(date +%Y%m%d_%H%M%S).log
bash scripts/twcc/run_twcc_exp53_full_from_scratch_stageE_seed61.sh --check-only
```

## Full Run

Use `tmux` or `screen` so the process survives browser or SSH disconnects:

```bash
cd "$TWCC_O256_ROOT"
tmux new -s stagee_exp53
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate o256-watermark
export TWCC_PYTHON="$(which python)"
bash scripts/twcc/run_twcc_exp53_full_from_scratch_stageE_seed61.sh 2>&1 | tee "$TWCC_EXP_ROOT/twcc_exp53_full_run.log"
```

Detach from tmux with `Ctrl-b`, then `d`.

Reattach:

```bash
tmux attach -t stagee_exp53
```

## What The Script Runs

1. identity20k from scratch
2. base30 nongeom-light10k
3. base50 nongeom-light +20k
4. base55 weakattack +5k
5. base60 dropout-focus +5k
6. base65 dropout-margin +5k
7. base65 finaltest1000 metrics and 11 non-geometric attacks
8. train10000 pair generation from base65
9. RGB refiner StageA/B/B+/C/D/E
10. final RGB StageE finaltest1000 metrics and 11 non-geometric attacks

Queue or container startup waiting time should be reported separately from model training wall-clock time.
