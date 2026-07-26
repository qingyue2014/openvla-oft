# L3-A2 validation ledger

All entries below are calibration evidence only. None authorizes formal
evaluation.

| Job | Commit | Result | Disposition |
| --- | --- | --- | --- |
| 489616 | 8357d90 | 0/20 bottle-B poses; B was saved before settling | Invalid: stale terminal equilibrium |
| 489634 | 294a422 | validator rejected ordinary vertical settling | Invalid: validator defect |
| 489635 | f49a6f8 | 0/20 absolute-grid bottle-B poses | Invalid: not trajectory-driven |
| 489657 | efa343b | 0/48 measured-endpoint bottle-B poses | Invalid: terminal collision cross-section too narrow |
| 489718 | 8237359 | 0/24 panel poses passed passive clearance | Invalid: first panel foot/face contacted drawer |

Job 489657 established that 26/48 bottle-B poses were passively stable and
table-only, but the closest dynamic A-B center distances remained about
0.047–0.070 m and no A-B contact occurred. Thresholds were not relaxed.
The terminal body was therefore redesigned as a broad, stable, high-contrast
panel with separate physical and visible geometry. A fresh trajectory-driven
position/yaw sweep is required before policy-view evidence or smoke tests.

Job 489718 showed that the first broad panel was rotated 20°–50°, so its long
axis and 14 cm foot reached the bottom drawer at every candidate. The next
revision narrows the foot along the path normal and explicitly tests
75°/90°/105° yaw, keeping the wide force-receiving face tangent to A's path.
