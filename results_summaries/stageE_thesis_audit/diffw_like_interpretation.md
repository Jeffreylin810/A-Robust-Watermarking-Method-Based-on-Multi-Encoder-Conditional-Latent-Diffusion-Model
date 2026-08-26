# DiffW-like Interpretation

- The original formal DiffW-like result remains the proper formal main baseline because it is the locked finaltest1000 formal run.
- DiffW-like epoch7, epoch12/60k, sf0.5/sf0.75, and singlevar runs are supplemental comparisons. They should not replace the formal baseline in the main table.
- The epoch12/60k run shows that DiffW-like can reach much higher PSNR, but its worst-case BA drops to 68.18 under `Paper_GaussianNoise_arg03`.
- It is conservative and accurate to write: DiffW-like long training can obtain higher PSNR, but worst-case BA remains insufficient; O256 StageE targets a more balanced PSNR / Nongeom Avg BA / worst-case BA trade-off.
- Do not write that O256 StageE全面優於 DiffW-like.
