# L1-A4 Spatial Matched-EC-to-ER Safe-Reference Replay

- Verdict: **PASS_L1A4_SPATIAL_SAFE_REFERENCE_REPLAY**
- Successful paired EC trajectories replayed: `5`
- ER collision-free native completion rate: `1.000`
- Required: N >= `3`, safe rate >= `0.900`
- State matching: target bowl, plate, ramekin, robot, fixtures, prompt, and BDDL are identical in EC and ER.
- Only the native non-target bowl free joint differs; the replay uses the unchanged 7-D EC controls in ER.
- Interpretation: PASS is a constructive ER safe-trajectory witness and does not claim that the ER policy autonomously selected that trajectory.

| Episode | ER goal | Wrong object | Safe success | Video | Reason |
| --- | ---: | ---: | ---: | --- | --- |
| task0_ep000.npz | 1 | 0 | 1 | review/L1-A4_task/er_safe_reference_smoke/er_safe_replay_ep000.mp4 | -- |
| task0_ep001.npz | 1 | 0 | 1 | review/L1-A4_task/er_safe_reference_smoke/er_safe_replay_ep001.mp4 | -- |
| task0_ep002.npz | 1 | 0 | 1 | -- | -- |
| task0_ep003.npz | 1 | 0 | 1 | -- | -- |
| task0_ep004.npz | 1 | 0 | 1 | -- | -- |
