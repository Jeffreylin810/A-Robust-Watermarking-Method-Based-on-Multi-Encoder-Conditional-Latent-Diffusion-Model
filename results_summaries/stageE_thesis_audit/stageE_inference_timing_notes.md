# StageE Inference Timing Notes

- StageE-specific compute-only benchmark was not found. The compute-only CSV therefore includes old DiffW-like/O256-Base/O256-RGB formal timing baselines and explicit missing rows for StageE.
- StageE end-to-end timing is available as partial formal logs: base finaltest1000 apply plus StageE refiner apply.
- These end-to-end logs include I/O and output writing but exclude PSNR/SSIM/LPIPS/BA/attack evaluation.
- StageE refiner architecture is the same ResidualRefiner family as O256-RGB, but because StageE-specific compute-only timing was not rerun, the old O256-RGB compute-only timing should be described only as a baseline/reference.
- Chapter 4 Table 11/12 can be updated with StageE end-to-end timing if labeled as partial formal apply timing. Avoid claiming a completed StageE pure compute benchmark.
