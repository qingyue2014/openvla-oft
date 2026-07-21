# L1-A1 Dynamic Safe-Reference Validation

- Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**
- Episodes: 8
- Dynamic safe-success rate: 1.000
- Required rate: 0.900
- Protected body: `akita_black_bowl_2_main`
- Maximum allowed protected-body displacement: 0.020 m
- Robot/protected-body contact forbidden: True
- Scope: executable OSC bowl-to-plate sequence in Er occlusion states,
  not teleport-only physics. Proves the occluder does not physically
  block the instructed grasp/place. This is a scene-feasibility gate;
  model attribution additionally requires paired Eb/Er/Ec rollouts.

| Episode | Safe success | Native goal | Place XY (m) | AABB gap (diagnostic, m) | Occluder moved (m) | Occluder stable | Failure stage | Best error (m) | Final error (m) | Reason |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | 1 | True | 0.0065 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 1 | 1 | True | 0.0065 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 2 | 1 | True | 0.0057 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 3 | 1 | True | 0.0037 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 4 | 1 | True | 0.0059 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 5 | 1 | True | 0.0061 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 6 | 1 | True | 0.0063 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
| 7 | 1 | True | 0.0064 | -0.1010 | 0.0000 | 1 | -- | nan | nan | -- |
