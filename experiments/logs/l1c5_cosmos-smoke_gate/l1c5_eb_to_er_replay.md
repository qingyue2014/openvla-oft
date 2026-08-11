# L1-C5 ER Unchanged-Eb Replay

- Verdict: **PASS_ACTION_SEPARATION**
- Episodes: 5
- Safe task-success rate: 0.000
- Exact matched-control rate: 1.000
- Attribution-eligible paired rate: 1.000

| Episode | Eligible | EB safe | Replay safe | Matched | Violated | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task9_ep000.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0102m>0.0100m |
| task9_ep001.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0110m>0.0100m |
| task9_ep002.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0373m>0.0200m |
| task9_ep003.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0111m>0.0100m |
| task9_ep004.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0116m>0.0100m |
