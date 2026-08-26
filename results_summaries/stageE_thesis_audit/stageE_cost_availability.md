# StageE Cost Availability

- exp52 / exp53 / exp54 all have formal training cost logs for base training, pair generation, and refiner training.
- The StageE preparation-cost table can be updated in Chapter 4 if the thesis clearly states the timing scope.
- Timing scope: `/usr/bin/time -p real`; includes training logging and checkpoint saving overhead; excludes final formal attacks, BA, PSNR, SSIM, and LPIPS.
- GPU model was not printed in StageE training logs. Do not hard-code a GPU model for StageE training cost unless another source is added.
- Existing O256-RGB / DiffW-like cost tables may be retained as historical or baseline timing, but should not be used as a substitute for StageE training cost.
- For Section 4.5, use exp52/53/54 preparation costs directly and keep the claim conservative: StageE has a multi-hour one-time preparation cost, while inference remains much faster than DiffW-like in the available end-to-end timing.
