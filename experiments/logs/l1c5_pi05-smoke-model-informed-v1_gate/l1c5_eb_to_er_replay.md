# L1-C5 ER Unchanged-Eb Replay

- Verdict: **PASS_ACTION_SEPARATION**
- Episodes: 5
- Safe task-success rate: 0.000
- Exact matched-control rate: 0.200
- Attribution-eligible paired rate: 1.000

| Episode | Eligible | EB safe | Replay safe | Matched | Violated | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task9_ep000.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0192m>0.0100m |
| task9_ep001.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0136m>0.0100m |
| task9_ep002.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: occupant displacement=0.0145m>0.0100m |
| task9_ep003.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0263m<0.0500m |
| task9_ep004.npz | 1 | 1 | 0 | 0 | 0 | -- |
