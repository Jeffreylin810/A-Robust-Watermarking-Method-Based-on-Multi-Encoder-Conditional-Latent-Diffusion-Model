# Formal Inference Sanity (seed61)

這一步是 **F 類正式補強實驗**，用途是確認：

> `O256 base` 與 `O256 + RGB refinement module`
> 在 benchmark 中看起來的微小差異，
> 是否只是量測波動，而不是實際加速或減速。

它不會重訓模型，也不會碰 `finaltest1000` 的正式結果。

## 0. Preflight

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_formal_inference_sanity_seed61.sh --check-only
```

## 1. 正式執行

請在 **GPU idle** 時執行：

```bash
cd /home/jeffrey/projects/diffw_comparison_experiment_5_27/O256-PAPER

bash scripts/run_formal_inference_sanity_seed61.sh
```

## 2. 產出位置

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_sanity_seed61/summary.md
```

raw JSON：

```text
/home/jeffrey/projects/實驗結果/16_mainline_formal_experiments/03_efficiency/formal_inference_sanity_seed61/o256_formal_inference_sanity.json
```

## 3. 這份實驗要怎麼用

這份 sanity 的用途是支撐第四章的這種說法：

- `module overhead is negligible relative to base sampling`

而不是支撐這種說法：

- `module makes inference faster`

也就是說，它是用來**避免誤解 benchmark noise**，不是用來宣稱模組會加速。  
