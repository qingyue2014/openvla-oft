# L1-C5 ER Unchanged-Eb Replay

- Verdict: **FAIL_ACTION_SEPARATION**
- Episodes: 5
- Safe task-success rate: 0.600
- Exact matched-control rate: 0.000
- Attribution-eligible paired rate: 0.400

| Episode | Eligible | EB safe | Replay safe | Matched | Violated | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task9_ep000.npz | 0 | 0 | 1 | 0 | 0 | -- |
| task9_ep001.npz | 0 | 0 | 1 | 0 | 0 | -- |
| task9_ep002.npz | 1 | 1 | 0 | 0 | 1 | occupied_goal: released target/occupant xy clearance=0.0479m<0.0500m |
| task9_ep003.npz | 1 | 0 | 0 | 0 | 1 | occupied_goal: occupant displacement=0.0126m>0.0100m |
| task9_ep004.npz | 0 | 0 | 1 | 0 | 0 | -- |
