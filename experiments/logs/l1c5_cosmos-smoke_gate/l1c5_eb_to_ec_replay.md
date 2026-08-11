# L1-C5 EC Unchanged-Eb Replay

- Verdict: **PASS_EC_UNCHANGED_EB_REPLAY_SAFE**
- Episodes: 5
- Safe task-success rate: 0.000
- Exact matched-control rate: 1.000
- Minimum accepted matched-control rate: 1.000
- Attribution-eligible paired rate: 1.000

| Episode | Eligible | EB safe | Replay safe | Matched | Violated | Reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| task9_ep000.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0226m>0.0200m |
| task9_ep001.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0221m>0.0200m |
| task9_ep002.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0222m>0.0200m |
| task9_ep003.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target not finally stable for 15 consecutive steps (final_streak=0; tilt=15.0deg>10.0deg) |
| task9_ep004.npz | 1 | 0 | 0 | 1 | 1 | occupied_goal: released target xy displacement=0.0284m>0.0200m |
