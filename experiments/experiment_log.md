# ORION experiment log

No experiments or metric results have been recorded yet.

For each run, record date, code revision, dataset and source-level split, image resolution, mask semantics and generator, model/configuration, optimizer and training budget, hardware/software, random seed, evaluation sample count, metric definitions, results, and observed failure cases.

| Run ID | Status | Notes |
| E1-Telea-BMW | EXPERIMENTAL | User-run, 2026-11-09 UTC; image `D:\projects\bmw.jpg` resized to 288×512; rectangular mask at (72,128), 96×170; soft edge 12 px; seed 42; Telea radius 3. Binary and soft masks both mark 16,320 pixels; metrics are identical (MAE 0.024901, masked MAE 0.224991, PSNR 19.5486 dB, global SSIM 0.838271). Expected because Telea thresholds the mask at >0. This is not evidence about learned soft-mask benefits. |

| CP-01 scaffold | IMPLEMENTED, NOT VERIFIED HERE | Initial synthetic rectangular mask pipeline and Telea demo added. CNN architecture is untrained; no learned results. |
