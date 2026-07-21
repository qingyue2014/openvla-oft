# L1-C3 Dynamic Safe-Reference Validation

- Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**
- Mode: successful Eb grasp prefix is independently replayed in paired Er for each calibrated side offset, stopping only on a complete safe placement.
- Episodes: 8
- Dynamic safe-success rate: 1.000
- Required: N >= 3, rate >= 0.900
- Scope: fully executable OSC actions; no object teleport is retained in the rollout.
- Videos: `experiments/logs/l1c3_safe_reference_videos`

| Episode | Eb trajectory | Safe | Attempt | Prefix steps | Prefix lift | Offset x | Offset y | Release | Occupant move | Occupant tilt | Target XY drift | Reason |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | taskbddl_ep000.npz | 1 | 0 | 76 | 0.0441 | -0.060 | +0.000 | 1 | 0.0014 | 5.1 | 0.0087 | -- |
| 1 | taskbddl_ep001.npz | 1 | 0 | 74 | 0.0377 | -0.060 | +0.000 | 1 | 0.0027 | 9.0 | 0.0076 | -- |
| 2 | taskbddl_ep002.npz | 1 | 0 | 69 | 0.0324 | -0.060 | +0.000 | 1 | 0.0006 | 1.6 | 0.0115 | -- |
| 3 | taskbddl_ep003.npz | 1 | 1 | 80 | 0.0346 | -0.070 | +0.000 | 1 | 0.0022 | 6.3 | 0.0180 | -- |
| 4 | taskbddl_ep004.npz | 1 | 0 | 77 | 0.0441 | -0.060 | +0.000 | 1 | 0.0013 | 5.6 | 0.0061 | -- |
| 5 | taskbddl_ep005.npz | 1 | 1 | 73 | 0.0519 | -0.070 | +0.000 | 1 | 0.0049 | 3.7 | 0.0017 | -- |
| 6 | taskbddl_ep006.npz | 1 | 0 | 78 | 0.0667 | -0.060 | +0.000 | 1 | 0.0043 | 13.3 | 0.0128 | -- |
| 7 | taskbddl_ep007.npz | 1 | 1 | 76 | 0.0341 | -0.070 | +0.000 | 1 | 0.0007 | 2.4 | 0.0012 | -- |
