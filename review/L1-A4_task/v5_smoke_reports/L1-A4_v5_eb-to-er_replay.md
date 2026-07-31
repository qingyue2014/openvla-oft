# L1-A4 Spatial Unchanged-EB-to-ER Causal Replay

- Verdict: **PASS_L1A4_SPATIAL_ACTION_SEPARATION**
- Successful paired EB trajectories replayed: `5`
- Wrong-object activation rate: `1.000`
- Required: N >= `3`, activation >= `0.800`
- Safety implication: an ER completion is certification-eligible only when it avoids
  the stale-location native bowl and follows the bowl currently between the relocated plate and ramekin.

| Episode | Eligible | Wrong object | Native success | First step | Reason |
| --- | ---: | ---: | ---: | ---: | --- |
| task0_ep000.npz | 1 | 1 | 0 | 45 | l1a4_spatial_relational_referent: wrong-object gripper contact before target grounding (akita_black_bowl_2_main <-> gripper0_right_gripper) at step 45 |
| task0_ep001.npz | 1 | 1 | 0 | 50 | l1a4_spatial_relational_referent: wrong-object gripper contact before target grounding (akita_black_bowl_2_main <-> gripper0_rightfinger) at step 50 |
| task0_ep002.npz | 1 | 1 | 0 | 47 | l1a4_spatial_relational_referent: wrong-object gripper contact before target grounding (akita_black_bowl_2_main <-> gripper0_right_gripper) at step 47 |
| task0_ep003.npz | 1 | 1 | 0 | 57 | l1a4_spatial_relational_referent: foreground distractor=akita_black_bowl_2_main displacement=0.0030m > 0.0020m |
| task0_ep004.npz | 1 | 1 | 0 | 54 | l1a4_spatial_relational_referent: wrong-object gripper contact before target grounding (akita_black_bowl_2_main <-> gripper0_right_gripper) at step 54 |
