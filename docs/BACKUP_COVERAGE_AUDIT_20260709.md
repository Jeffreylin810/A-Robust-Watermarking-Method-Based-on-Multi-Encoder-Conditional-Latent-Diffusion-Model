# Backup Coverage Audit（2026-07-09）

本文件檢查 GitHub backup 是否足以快速恢復研究。重點不是保存 raw outputs，而是保存：

1. 程式本體
2. 可重跑的 script/config
3. 重大節點 summary
4. 研究脈絡文檔

## 程式與設定覆蓋

| 類別 | 狀態 |
|---|---|
| O256 diffusion/model code | 已備份：`models/`, root `*.py` |
| attack/noise code | 已備份：`noise_layers/`, `Noise.py` |
| RGB refiner code | 已備份：`models/restoration.py`, `scripts/train_rgb_refinement.py`, `scripts/apply_rgb_refinement.py` |
| summary / diagnosis / audit code | 已備份：`scripts/` |
| formal / supplementary configs | 已備份：`configs/` |
| environment | 已備份：`environment.yml` |

## 重大節點摘要覆蓋

| 節點 | 摘要 | Runner / analyzer |
|---|---|---|
| mainline method comparison | `results_summaries/01_method_comparison` | root eval/train scripts + configs |
| efficiency / deployment | `results_summaries/03_efficiency`, `actual10k_deployment`, `repeatability_sanity` | `scripts/run_actual10k_end_to_end_deployment_benchmark_seed61_20260628.sh`, related benchmark scripts |
| latent vs pixel | `results_summaries/07_latent_vs_pixel` | `scripts/setup_supplementary_latent_vs_pixel_experiment.py` |
| exp16 | `results_summaries/exp16` | `scripts/run_o256_base_continuation_ba_push_plus5k_plus10k_seed61_20260629.sh` |
| exp17 | `results_summaries/exp17` | `scripts/run_o256_rgb_refiner_refresh_plus10k_bundle_seed61_20260630.sh` |
| exp18 | `results_summaries/exp18` | `scripts/run_o256_base_plus15k_then_refresh500_seed61_20260630.sh` |
| exp19 | `results_summaries/exp19` | `scripts/run_o256_rgb_refiner_refresh_plus15k_refresh1000_seed61_20260630.sh` |
| exp20 | `results_summaries/exp20` | `scripts/run_o256_rgb_refiner_refresh_plus15k_refresh1500_seed61_20260630.sh` |
| exp21 | `results_summaries/exp21` | `scripts/run_o256_base_plus20k_then_refresh500_seed61_20260630.sh` |
| exp22 | `results_summaries/exp22` | `scripts/run_o256_rgb_refiner_refresh_plus20k_refresh1500_seed61_20260630.sh` |
| exp23 | `results_summaries/exp23` | `scripts/run_o256_base_plus20k_weakattack_plus5k_seed61_20260630.sh` |
| exp24 | `results_summaries/exp24` | `scripts/run_o256_rgb_refiner_refresh_plus20k_weaktargeted1000_seed61_20260630.sh` |
| exp25 | `results_summaries/exp25_message`, `exp25_bitplane` | message-preserving scripts + `scripts/run_o256_bitplane_weakbit_diagnosis_20260701.sh` |
| exp26 | `results_summaries/exp26` | `scripts/run_o256_bitweighted_refiner_repair_plus20k_seed61_20260701.sh` |
| exp27 | `results_summaries/exp27` | `scripts/run_o256_base_bitweighted_weakattack_plus5k_seed61_20260701.sh` |
| exp28 | `results_summaries/exp28` | `scripts/run_o256_attack_schedule_dropout_focus_from_A_seed61_20260701.sh` |
| exp29 | `results_summaries/exp29` | `scripts/run_o256_exp28_quality_recovery_refresh500_seed61_20260702.sh` |
| exp30 | `results_summaries/exp30` | `scripts/run_o256_exp28_conservative_refresh250_seed61_20260702.sh` |
| exp31 | `results_summaries/exp31` | `scripts/run_o256_exp28_message_preserving_attackaware_refresh500_seed61_20260702.sh` |
| exp32 | `results_summaries/exp32` | `scripts/run_o256_exp28_dropout_specific_refresh500_seed61_20260702.sh` |
| exp33 | `results_summaries/exp33` | `scripts/run_o256_exp28_dropout_balanced_refresh500_seed61_20260702.sh` |
| exp34 | `results_summaries/exp34` | `scripts/run_o256_exp34_full_candidate_validation_seed61_20260703.sh` |
| exp35 | `results_summaries/exp35` | `scripts/run_o256_exp35_dropout_compensation_refresh150_seed61_20260703.sh` |
| exp36 | `results_summaries/exp36` | `scripts/run_o256_exp36_native_dropout_proxy_refresh500_seed61_20260703.sh` |
| exp37 | `results_summaries/exp37` | `scripts/run_o256_exp37_base_dropout_margin_plus1epoch_seed61_20260703.sh` |
| exp38 | `results_summaries/exp38` | `scripts/run_o256_exp38_base_dropout_margin_seed62_validation_20260704.sh` |
| exp39 | 無獨立 summary | `scripts/run_o256_exp39_locked_continuation_validation_seed61_20260704.sh`；此節點後續由 exp40/41 from-scratch formal 取代 |
| exp40 | `results_summaries/exp40` | `scripts/run_o256_exp40_full_from_scratch_formal_seed61_20260704.sh` |
| exp41 | `results_summaries/exp41` | `scripts/run_o256_exp41_full_from_scratch_formal_seed62_20260705.sh` |
| exp42 | `results_summaries/exp42` | `scripts/run_o256_exp42A_dropout_floor_safety_margin_probe_seed62_20260705.sh` |
| exp43 | `results_summaries/exp43` | `scripts/run_o256_exp43_dropout_floor_error_diagnosis_seed62_20260705.sh` |
| exp44 | `results_summaries/exp44` | `scripts/run_o256_exp44_train_hardcase_dropout_replay_seed62_20260705.sh` |
| exp45 | `results_summaries/exp45` | `scripts/run_o256_exp45_dropout_floor_delta_diagnosis_seed62_20260706.sh` |
| exp46 | `results_summaries/exp46` | `scripts/run_o256_exp46_residual_frequency_diagnosis_seed62_20260706.sh` |
| exp47 | `results_summaries/exp47` | `scripts/run_o256_exp47_guarded_refiner_probe_seed62_20260706.sh` |
| exp48 | `results_summaries/exp48` | `scripts/run_o256_exp48_bitplane_refiner_damage_tracking_20260708.sh`, `scripts/analyze_o256_exp48_bitplane_refiner_damage.py` |
| exp49 | `results_summaries/exp49` | `scripts/run_o256_exp49_evaluation_integrity_audit_20260708.sh`, `scripts/analyze_o256_exp49_evaluation_integrity_audit.py` |
| exp50 | `results_summaries/exp50` | `scripts/run_o256_exp50_decoderaware_refiner_probe_seed62_20260708.sh` |
| exp51 | `results_summaries/exp51` | `scripts/run_o256_exp51_decoderaware_stageE_crossseed_probe_20260708.sh` |
| exp52 | `results_summaries/exp52` | `scripts/run_o256_exp52_full_from_scratch_stageE_seed62_20260708.sh` |
| exp53 | `results_summaries/exp53` | `scripts/run_o256_exp53_full_from_scratch_stageE_seed61_20260709.sh` |

## 不足之處

- GitHub 不保存 checkpoint；如果要 checkpoint-level continuation，需要另外從本機/雲端硬碟保存 checkpoint，或重新跑 from-scratch protocol。
- GitHub 不保存 dataset 與 VAE 權重；恢復研究時要重新放置 dataset 與 SD2.1 VAE。
- 這是研究備份版，不是最終公開版。公開版應另外整理，移除本機絕對路徑並保留最終 protocol。

