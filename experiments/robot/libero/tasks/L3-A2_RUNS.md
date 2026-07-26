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
| 489727 | 71b00ab | 1/24 panel poses passed passive clearance; 0 cascades | Invalid: sole clear pose began Er in A-B contact |
| 489736 | a40c033 | 18/24 passed Eb/Ec clearance; 0 cascades | Invalid: all clear poses began Er in A-B contact |

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

Job 489727 found one table-only Eb/Ec pose at `(0.135, 0.075, 90°)`, but it
began Er in A-B contact. The next clearance sweep shifts centers another
1.5–3.5 cm away from the cabinet (negative world y) and narrows the yaw band
to 85°/90°/95°, seeking a no-contact gap that A crosses only after release.

Job 489736 established a useful null-stability seed: at
`(0.115, 0.060, 95°)`, A-disabled residual response was only 0.013 mm and
0.069°, but Er still began in A-B contact. The next revision lowers the panel
to 84 mm, concentrates mass in a denser 22 mm-half-width foot, shifts y
outward to the 0.045–0.060 m band, and makes Er reset clearance a prefilter.
