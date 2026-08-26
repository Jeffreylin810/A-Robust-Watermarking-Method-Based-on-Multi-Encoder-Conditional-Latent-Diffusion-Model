# Experiment 10: Pixel Bridge Schedule Ablation

Classification: exploratory architecture ablation.

Question:

```text
Must Pixel Bridge be used during the first 20k message-learning stage,
or is it sufficient to enable Pixel Bridge only during the final 5k
non-geometric attack fine-tuning stage?
```

Controlled comparison:

```text
Existing control:
Pixel Bridge + Identity 20k
-> Pixel Bridge + nongeom-light 5k

New branch:
latent-direct message recovery 20k
-> Pixel Bridge + nongeom-light 5k
```

Both branches use the same:

```text
COCO train10000 / unseen eval1000
256x256
message length 32
batch size 2
seed 61
loss weights
20k + 5k optimizer schedule
final 11 non-geometric evaluation attacks
```

Run:

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER
tmux new -s exp10_bridge_schedule
bash scripts/run_exp10_pixel_bridge_schedule_ablation.sh
```

Detach from tmux:

```text
Ctrl-b, then d
```

Interpretation:

```text
If the new final branch matches the existing control:
Pixel Bridge may be limited to the 5k attack-aware stage, simplifying training.

If latent-direct 20k has high training BA but low saved-image Clean BA:
the VAE/RGB round-trip creates a real train-deployment gap.

If the final 5k cannot recover the gap:
Pixel Bridge is necessary during the initial message-learning stage.
```
