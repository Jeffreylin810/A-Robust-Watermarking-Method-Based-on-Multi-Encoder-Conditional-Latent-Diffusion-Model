# Experiment 8: 正式速度與成本評估

本實驗不修改模型、loss 或模組參數。目的為分開量測：

1. 公平 DiffW-like 與 O256 base 的訓練時間。
2. DiffW-like、O256 base、模組與 O256+模組的 compute-only inference。
3. 模組所需的一次性 preparation cost。

比較對象固定為：

```text
DiffW-like: 256x256 / msg32 / train10000 / 20k steps / seed61
O256 base: 256x256 / msg32 / train10000 / identity20k + nongeom-light5k / seed61
Module: refine10000 targeted Stage C-500 / seed61
```

## 1. 執行前檢查

這個指令只檢查路徑、GPU 與 checkpoint，不會啟動訓練：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp8_formal_speed_cost.sh --check-only
```

正式計時必須等 GPU 沒有其他訓練或評估程序時再執行。若偵測到其他 CUDA
compute process，主腳本會停止，不會產生受干擾的速度結果。

## 2. 正式執行

目前公平 DiffW-like checkpoint 已清除，因此第一次正式執行會先重訓約
2.5 小時，再完成兩套 inference benchmark。重訓後只保留具名 final
checkpoint，step-1 snapshot 會自動移除：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp8_formal_speed_cost.sh \
  2>&1 | tee logs/exp8_formal_speed_cost_master.log
```

若 DiffW-like checkpoint 已經完成，只想重新跑 benchmark：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_exp8_formal_speed_cost.sh --skip-train \
  2>&1 | tee logs/exp8_formal_speed_cost_rerun.log
```

## 3. 最終輸出

```text
results/experiment_summaries/exp8_formal_speed_cost_seed61/summary.md
```

原始 benchmark JSON：

```text
results/experiment_summaries/exp8_formal_speed_cost_seed61/raw/diffw_like_benchmark.json
results/experiment_summaries/exp8_formal_speed_cost_seed61/raw/o256_module_benchmark.json
```

## 4. 計時邊界

Compute-only inference：

```text
包含模型原生 sampling 路徑。
O256 包含 frozen VAE encode/decode。
O256+模組再包含 RGB refinement forward。
包含原生 sampling 程式本身的 CPU/GPU intermediate tensor 搬移。
不包含 dataset I/O、PNG 寫檔、PSNR、SSIM、attack 與 BA 評估。
```

一次性成本：

```text
Base training
產生 refine10000 pairs
Module Stage A / B / B+ / targeted Stage C
```

正式報告時不可把 base training、完整 preparation 與 inference 三種時間混成
同一個「速度」數字。
