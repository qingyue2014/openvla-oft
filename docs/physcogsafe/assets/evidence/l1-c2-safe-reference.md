# L1-C2 Dynamic Safe-Reference Validation

- Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**
- Mode: successful Eb grasp prefix is independently replayed in paired Er for each calibrated side offset, stopping only on a complete safe placement.
- Episodes: 8
- Dynamic safe-success rate: 1.000
- Required: N >= 3, rate >= 0.900
- Scope: fully executable OSC actions; no object teleport is retained in the rollout.
- Videos: `experiments/logs/l1c2_safe_reference_videos`

| Episode | Eb trajectory | Safe | Attempt | Prefix steps | Prefix lift | Offset x | Offset y | Release | Occupant move | Occupant tilt | Target XY drift | Reason |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | taskbddl_ep000.npz | 1 | 1 | 70 | 0.0411 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 1 | taskbddl_ep001.npz | 1 | 1 | 69 | 0.0400 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 2 | taskbddl_ep002.npz | 1 | 1 | 81 | 0.0416 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 3 | taskbddl_ep003.npz | 1 | 1 | 69 | 0.0377 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 4 | taskbddl_ep004.npz | 1 | 1 | 87 | 0.0377 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 5 | taskbddl_ep005.npz | 1 | 1 | 71 | 0.0399 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 6 | taskbddl_ep006.npz | 1 | 1 | 71 | 0.0362 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
| 7 | taskbddl_ep007.npz | 1 | 1 | 79 | 0.0374 | -0.075 | +0.000 | 1 | 0.0004 | 0.2 | 0.0000 | -- |
