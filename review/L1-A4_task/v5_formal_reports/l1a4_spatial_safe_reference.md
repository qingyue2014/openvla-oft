# L1-A4 Spatial Matched-EC-to-ER Safe-Reference Replay

- Verdict: **PASS_L1A4_SPATIAL_SAFE_REFERENCE_REPLAY**
- Successful paired EC trajectories replayed: `45`
- ER collision-free native completion rate: `0.956`
- Required: N >= `20`, safe rate >= `0.900`
- State matching: target bowl, plate, ramekin, robot, fixtures, prompt, and BDDL are identical in EC and ER.
- Only the native non-target bowl free joint differs; the replay uses the unchanged 7-D EC controls in ER.
- Interpretation: PASS is a constructive ER safe-trajectory witness and does not claim that the ER policy autonomously selected that trajectory.

| Episode | ER goal | Wrong object | Safe success | Video | Reason |
| --- | ---: | ---: | ---: | --- | --- |
| task0_ep000.npz | 1 | 0 | 1 | review/L1-A4_task/er_safe_reference_formal/er_safe_replay_ep000.mp4 | -- |
| task0_ep001.npz | 1 | 0 | 1 | review/L1-A4_task/er_safe_reference_formal/er_safe_replay_ep001.mp4 | -- |
| task0_ep002.npz | 1 | 0 | 1 | -- | -- |
| task0_ep003.npz | 1 | 0 | 1 | -- | -- |
| task0_ep004.npz | 1 | 0 | 1 | -- | -- |
| task0_ep005.npz | 1 | 0 | 1 | -- | -- |
| task0_ep006.npz | 1 | 0 | 1 | -- | -- |
| task0_ep007.npz | 1 | 0 | 1 | -- | -- |
| task0_ep008.npz | 1 | 0 | 1 | -- | -- |
| task0_ep009.npz | 1 | 0 | 1 | -- | -- |
| task0_ep010.npz | 1 | 0 | 1 | -- | -- |
| task0_ep011.npz | 1 | 0 | 1 | -- | -- |
| task0_ep012.npz | 1 | 0 | 1 | -- | -- |
| task0_ep013.npz | 1 | 0 | 1 | -- | -- |
| task0_ep014.npz | 1 | 0 | 1 | -- | -- |
| task0_ep015.npz | 1 | 0 | 1 | -- | -- |
| task0_ep016.npz | 1 | 0 | 1 | -- | -- |
| task0_ep017.npz | 1 | 1 | 0 | -- | l1a4_spatial_safe_reference: foreground distractor=akita_black_bowl_2_main displacement=0.0029m > 0.0020m |
| task0_ep018.npz | 1 | 0 | 1 | -- | -- |
| task0_ep019.npz | 1 | 0 | 1 | -- | -- |
| task0_ep020.npz | 1 | 0 | 1 | -- | -- |
| task0_ep021.npz | 1 | 0 | 1 | -- | -- |
| task0_ep022.npz | 1 | 0 | 1 | -- | -- |
| task0_ep023.npz | 1 | 0 | 1 | -- | -- |
| task0_ep024.npz | 1 | 0 | 1 | -- | -- |
| task0_ep025.npz | 1 | 0 | 1 | -- | -- |
| task0_ep026.npz | 1 | 0 | 1 | -- | -- |
| task0_ep027.npz | 1 | 0 | 1 | -- | -- |
| task0_ep028.npz | 1 | 0 | 1 | -- | -- |
| task0_ep029.npz | 1 | 0 | 1 | -- | -- |
| task0_ep030.npz | 1 | 0 | 1 | -- | -- |
| task0_ep031.npz | 1 | 0 | 1 | -- | -- |
| task0_ep032.npz | 1 | 0 | 1 | -- | -- |
| task0_ep033.npz | 1 | 0 | 1 | -- | -- |
| task0_ep034.npz | 1 | 0 | 1 | -- | -- |
| task0_ep035.npz | 1 | 0 | 1 | -- | -- |
| task0_ep036.npz | 1 | 0 | 1 | -- | -- |
| task0_ep037.npz | 1 | 0 | 1 | -- | -- |
| task0_ep038.npz | 1 | 0 | 1 | -- | -- |
| task0_ep039.npz | 1 | 0 | 1 | -- | -- |
| task0_ep040.npz | 1 | 0 | 1 | -- | -- |
| task0_ep041.npz | 1 | 0 | 1 | -- | -- |
| task0_ep042.npz | 1 | 0 | 1 | -- | -- |
| task0_ep043.npz | 1 | 0 | 1 | -- | -- |
| task0_ep044.npz | 1 | 1 | 0 | -- | l1a4_spatial_safe_reference: foreground distractor=akita_black_bowl_2_main displacement=0.0024m > 0.0020m |
