# O256 exp51 decoder-aware StageE cross-seed probe summary

## Main Table

| Seed | Variant | Kind | PSNR | Clean BA | Avg BA | Min BA | Dropout BA | dPSNR | dMin BA | dDropout BA | Train s | Setting |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed61 | exp40_seed61_formal_stageD_reference | reference | 33.9372 | 99.96 | 98.77 | 91.01 | 91.01 | 0.0000 | 0.00 | 0.00 | N/A | src=formal StageD, msg=1.4, atk_msg=0.45, qgate=33.0/70.0 |
| seed61 | exp51_seed61_stageE_decoderaware_refresh | stageE_probe | 33.9337 | 99.96 | 98.79 | 91.17 | 91.17 | -0.0035 | 0.16 | 0.16 | 264.02 | src=exp40_stageD_dropout_balanced500, msg=1.5, atk_msg=0.65, qgate=33.8/90.0 |
| seed62 | exp41_seed62_formal_stageD_reference | reference | 34.0293 | 99.98 | 98.68 | 89.94 | 89.94 | 0.0000 | 0.00 | 0.00 | N/A | src=formal StageD, msg=1.4, atk_msg=0.45, qgate=33.0/70.0 |
| seed62 | exp51_seed62_stageE_decoderaware_refresh | stageE_probe | 33.9524 | 99.98 | 98.72 | 90.16 | 90.16 | -0.0769 | 0.22 | 0.22 | 260.55 | src=exp41_stageD_dropout_balanced500, msg=1.5, atk_msg=0.65, qgate=33.8/90.0 |

## Probe Interpretation

- `exp51_seed61_stageE_decoderaware_refresh` passes the cross-seed StageE probe threshold: PSNR `33.9337` (-0.0035), Min BA `91.17` (+0.16), Dropout BA `91.17` (+0.16).
- `exp51_seed62_stageE_decoderaware_refresh` passes the cross-seed StageE probe threshold: PSNR `33.9524` (-0.0769), Min BA `90.16` (+0.22), Dropout BA `90.16` (+0.22).

## Per-Attack BA


### exp51_seed61_stageE_decoderaware_refresh

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 91.17 |
| Paper_GaussianBlur_sigma2 | 99.98 |
| Paper_GaussianBlur_sigma3 | 99.98 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 97.83 |
| Paper_GaussianNoise_sigma03_var009 | 99.59 |
| Paper_JPEG_Q30 | 99.82 |
| Paper_JPEG_Q40 | 99.88 |
| Paper_JPEG_Q50 | 99.95 |
| Paper_MedianFilter_w5 | 99.98 |
| Paper_SaltPepper_r02 | 98.50 |

### exp51_seed62_stageE_decoderaware_refresh

| Attack | BA |
|---|---:|
| Paper_Dropout_r03 | 90.16 |
| Paper_GaussianBlur_sigma2 | 99.99 |
| Paper_GaussianBlur_sigma3 | 99.98 |
| Paper_GaussianBlur_sigma4 | 99.98 |
| Paper_GaussianNoise_arg03 | 97.92 |
| Paper_GaussianNoise_sigma03_var009 | 99.63 |
| Paper_JPEG_Q30 | 99.81 |
| Paper_JPEG_Q40 | 99.90 |
| Paper_JPEG_Q50 | 99.94 |
| Paper_MedianFilter_w5 | 99.98 |
| Paper_SaltPepper_r02 | 98.61 |

## Notes

- exp51 is still a continuation probe, not a full from-scratch formal result.
- It tests whether the exp50B decoder-aware StageE rule is stable across both seed61 and seed62 formal from-scratch checkpoints.
- If both seeds pass with small PSNR loss, the next step is a full from-scratch formal rerun that includes this StageE in the locked protocol.
