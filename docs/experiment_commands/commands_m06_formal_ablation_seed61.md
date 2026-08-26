# M06：Formal Stage-wise Ablation（Seed61）

## 目的

這一步屬於 **F 類正式實驗**，目的不是再找 recipe，而是把目前已完成的 formal artifacts
整理成一張可以放進第四章的正式 ablation 表。

固定條件：

- formal base checkpoint：`Result/CoCo_ddpm_m01_formal_seed61_identity20k_then_nongeomlight10k_final.pth.tar`
- locked split：`train10000_finaltest1000_locked`
- config：`configs/CoCo_finaltest1000_ours256_msg32_nongeomlight_base_seed61_formal.yml`
- seed：`61`

比較對象：

1. `O256 base`（重用 `M03`）
2. `Stage A`
3. `Stage B`
4. `Stage B+`
5. `Full module`（重用 `M03`）

## 腳本

先做 preflight：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m06_formal_ablation_seed61.sh --check-only
```

正式執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_m06_formal_ablation_seed61.sh
```

## 主要輸出

Summary：

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/02_ablation/formal_stagewise_seed61/summary.md
```

中間輸出：

- `results/refinement/m06_formal_seed61_stageA_finaltest1000_apply`
- `results/refinement/m06_formal_seed61_stageB_finaltest1000_apply`
- `results/refinement/m06_formal_seed61_stageBP_finaltest1000_apply`
- `results/attacked_Result/m06_formal_seed61_stageA_finaltest1000_nongeom`
- `results/attacked_Result/m06_formal_seed61_stageB_finaltest1000_nongeom`
- `results/attacked_Result/m06_formal_seed61_stageBP_finaltest1000_nongeom`

## 預期用途

這一步完成後，可在第四章放一張正式 ablation 表，說明：

- Stage A：偏 image-priority
- Stage B：加入較強 message preservation
- Stage B+：再強化 clean message preservation
- Full：再加入 targeted weak non-geometric message preservation

也就是從 formal base 到 final module 的 trade-off 演化過程。
