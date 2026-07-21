# L1-A2 Dynamic Safe-Reference Validation

- Verdict: **FAIL_DYNAMIC_SAFE_REFERENCE**
- Episodes: 8
- Dynamic safe-success rate: 0.000
- Required rate: 0.900
- Scope: executable OSC bowl-to-plate sequence in Er occlusion states,
  not teleport-only physics. Proves the occluder does not physically
  block the instructed grasp/place, so Er policy failures are
  attributable to perception.

| Episode | Safe success | Place XY (m) | Place gap (m) | Occluder moved (m) | Failure stage | Best error (m) | Final error (m) | Reason |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | 0 | 0.0978 | 0.0671 | 0.0001 | move_above_plate | 0.1003 | 0.0960 | waypoint_timeout |
| 1 | 0 | 0.0048 | -0.1008 | 5.8814 | -- | nan | nan | -- |
| 2 | 0 | 0.0062 | -0.1007 | 0.0000 | -- | nan | nan | -- |
| 3 | 0 | 0.2354 | -0.1088 | 0.0002 | -- | nan | nan | -- |
| 4 | 0 | 0.0058 | -0.1007 | 0.0000 | -- | nan | nan | -- |
| 5 | 0 | 0.0244 | 0.0749 | 0.0000 | move_above_plate | 0.0301 | 0.0272 | waypoint_timeout |
| 6 | 0 | 0.0066 | -0.1011 | 0.0018 | -- | nan | nan | -- |
| 7 | 0 | 0.0038 | -0.1008 | 0.0000 | -- | nan | nan | -- |
