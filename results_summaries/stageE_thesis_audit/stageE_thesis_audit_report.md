# StageE Thesis Audit Report v2

## 1. Executive summary

- exp52 / exp53 / exp54 all have complete formal summary CSV and per-attack BA CSV.
- exp52 and exp53 pass locked StageE strict gates and can serve as final main results.
- exp54 is complete but is a seed63 near-miss: PSNR 33.4948 and Dropout/Min BA 89.98 miss the strict gates, while Clean BA 99.94 and Nongeom Avg BA 98.35 remain strong.
- StageE training cost logs are complete for exp52/53/54.
- StageE-specific compute-only timing is missing; StageE end-to-end apply timing is available from partial formal logs.

## 2. Can Chapter 4 replace old O256-RGB results with exp52 / exp53?

Yes. exp52 and exp53 are full-from-scratch formal reruns on finaltest1000, 32-bit messages, 256x256 images, and both pass the strict gates.

## 3. exp52 / exp53 是否足以作為 final main results

Yes, with conservative wording and with exp54 presented as supplemental near-miss validation rather than a third pass.

## 4. exp54 是否完成

Yes. It completed summary/per-attack evaluation, but does not pass strict PSNR and Min BA gates.

## 5. 是否需要修改第三章方法公式

Minor formula update recommended for RGB refiner objective. Core O256 latent watermark formula does not need to be rewritten.

## 6. 是否需要修改第三章文字

Yes. Add a short description of decoder-aware StageE refiner training, frozen decoder message loss, and dropout proxy attack-message branch.

## 7. 是否需要修改摘要

Yes, if StageE becomes the final result. Update claims to mention improved formal trade-off, not全面 superiority.

## 8. 是否需要修改第五章

Yes. State that StageE improves PSNR / Avg BA / worst-case BA balance, but seed63 near-miss indicates remaining seed-dependent boundary.

## 9. 4.3 是否能保留 per-attack 表

Yes. The StageE per-attack table is complete for exp52/53/54 and should be kept.

## 10. 4.5 是否能更新正式成本表

Yes for preparation cost if timing scope is stated. For compute-only inference, StageE-specific benchmark is missing, so keep it conservative or use only end-to-end apply timing.

## 11. 哪些表格可以直接放入第四章

- `thesis_ready_tables_stageE.md`
- `stageE_final_metrics.csv`
- `stageE_per_attack_ba.csv`
- `formal_attack_name_mapping.csv`
- `stageE_training_cost_summary.csv`
- `stageE_end_to_end_inference.csv`
- `diffw_like_comparison_summary.csv`

## 12. 哪些結果只能作為補充分析

- exp54 seed63 near-miss validation.
- exp48-exp51 diagnosis/continuation probes.
- DiffW-like sf0.5/sf0.75/singlevar/epoch7/epoch12 long-training variants.
- Old O256-RGB compute-only timing baseline when discussing StageE compute-only cost.

## 13. 哪些結論不能寫

- Do not claim O256 StageE全面優於 DiffW-like.
- Do not call exp54 a strict-pass seed.
- Do not treat continuation probes as full-from-scratch formal results.
- Do not mix non-finaltest1000 results into final main tables.
- Do not include Clean BA in Nongeom Avg BA.
- Do not state StageE-specific compute-only timing is complete.

## 14. 仍需人工確認的項目

- Whether to put exp54 in the main result table as a near-miss row or move it to supplemental validation.
- Whether the advisor wants a dedicated StageE compute-only benchmark.
- Whether Chapter 3 should present StageE as a final training protocol only, or as part of the final method definition.

## Conservative final claim

O256 StageE 在本文正式協定下達成較穩定的 PSNR / Nongeom Avg BA / worst-case BA trade-off；DiffW-like 在長訓練下可能取得更高 PSNR，但 worst-case BA 仍可能不足。
