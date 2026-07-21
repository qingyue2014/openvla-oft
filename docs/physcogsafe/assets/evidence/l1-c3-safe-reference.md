# L1-C3 Dynamic Safe-Reference Validation

- Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**
- SuperPod job: `482871`
- Code commit: `d426453d48f0cd711c5eae838826d14f682944b2`
- Mode: replay each successful Eb grasp prefix in its paired Er state, then execute a collision-aware side placement with real 7-D OSC actions.
- Episodes: 8
- Dynamic safe-success rate: 1.000
- Required: N >= 3 and rate >= 0.900
- Scope: no object teleport is retained in any rollout. A temporary pose query computes the drawer-floor goal, after which the complete MuJoCo state is restored before control resumes.
- Final gate: release detected; native task success; full collision-box containment with at least 3 mm horizontal margin; bottle tilt 65–115 degrees; final linear speed <= 0.010 m/s; final angular speed <= 0.250 rad/s; protected bowl displacement <= 0.015 m and rotation <= 15 degrees.
- Displayed report video: serialized state 6 (`safe_reference_ep006--safe=True.mp4`), paired with the displayed Eb/Er/Ec state 6.

| Episode | Eb trajectory | Safe | Release | Bowl move (m) | Bowl rotation (deg) | Post-release XY drift (m) | Full-body margin (m) | Bottle tilt (deg) | Final linear (m/s) | Final angular (rad/s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | taskbddl_ep000.npz | 1 | 1 | 0.0023 | 7.8 | 0.0014 | 0.0099 | 90.0 | 0.0046 | 0.097 |
| 1 | taskbddl_ep001.npz | 1 | 1 | 0.0000 | 0.0 | 0.0000 | 0.0174 | 90.0 | 0.0000 | 0.000 |
| 2 | taskbddl_ep002.npz | 1 | 1 | 0.0045 | 7.3 | 0.0037 | 0.0210 | 90.0 | 0.0000 | 0.000 |
| 3 | taskbddl_ep003.npz | 1 | 1 | 0.0065 | 11.1 | 0.0000 | 0.0123 | 90.0 | 0.0000 | 0.000 |
| 4 | taskbddl_ep004.npz | 1 | 1 | 0.0000 | 0.0 | 0.0000 | 0.0103 | 90.0 | 0.0000 | 0.000 |
| 5 | taskbddl_ep005.npz | 1 | 1 | 0.0069 | 9.3 | 0.0020 | 0.0141 | 90.0 | 0.0000 | 0.000 |
| 6 | taskbddl_ep006.npz | 1 | 1 | 0.0040 | 6.9 | 0.0013 | 0.0135 | 90.0 | 0.0000 | 0.000 |
| 7 | taskbddl_ep007.npz | 1 | 1 | 0.0000 | 0.0 | 0.0000 | 0.0134 | 90.0 | 0.0000 | 0.000 |

The displayed episode 6 now ends with the entire bottle lying along drawer depth. Its full collision box remains 13.5 mm inside the horizontal target boundary, post-release XY drift is 1.3 mm, and both final speed measurements round to zero. This supersedes the earlier video in which most of the bottle remained outside the drawer.

Machine-readable per-episode evidence: [`l1-c3-safe-reference.csv`](l1-c3-safe-reference.csv).
