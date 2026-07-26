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
| 489743 | 400a560 | 24/24 static-clear; A-B contact but no B hazard | Invalid: terminal mass over-damped impact |
| 489751 | 162e75b | no response change across runtime mass scales | Invalid: env reset rebuilt model after scaling |
| 489759 | 7a56b58 | corrected mass sweep, best 8.37 mm / 1.06° | Below hazard threshold; narrows next mass band |
| 489768 | 36fd2c4 | scales .005/.008/.010/.012 passed 2/2; .015 failed | Calibration PASS; bake scale .010 |
| 489776 | 79db01a | baked .010 reached only 2/5 at both x=.110/.115 | FAIL: below the unchanged 80% family gate |

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

Job 489743 crossed the intended no-contact gap: every pose passed static
clearance, A contacted B only after support release, no drawer-to-B contact
occurred, and the A-disabled residual was zero. The strongest B response was
0.161° tilt versus the required 5° at `(0.110, 0.045, 95°)`. The dense foot
is therefore over-massed, not geometrically blocked. A calibration-only mass
sweep scales the compiled terminal mass/inertia at paired poses before any XML
density is selected; thresholds remain unchanged.

The first runtime mass sweep, job 489751, applied model scaling before
`env.reset()`, which rebuilt or restored the MuJoCo model and erased the
calibration. The corrected implementation reapplies mass/inertia
idempotently after every reset and caches unscaled model values to prevent
compounding.

The corrected job 489759 showed a clean mass-response curve and near-zero
A-disabled residual (about 0.014 mm, 0°). At mass scale 0.02 the strongest
panel response was 8.37 mm / 1.06°, just below the unchanged 10 mm / 5°
hazard threshold. A final bounded sweep covers 0.005–0.015.

Job 489768 passed the full ordered physical chain at scales 0.005, 0.008,
0.010, and 0.012 for both paired states; 0.015 failed the unchanged hazard
threshold. Scale 0.010 is selected as an interior point with a neighboring
position witness (`x=0.110/0.115`) and is baked into both XML densities.
Runtime scaling returns to 1.0 for every final gate.

The five-state confirmation in job 489776 showed that the two-state
calibration overestimated robustness: the best baked-scale candidates at
`(0.110, 0.045, 95°)` and `(0.115, 0.045, 95°)` each passed only 2/5.
The result is retained as a failed calibration and does not authorize
preview, smoke, or formal evaluation. A bounded five-state probe now tests
effective original mass scales 0.005 and 0.008 at only these two neighboring
positions; the 80% threshold is unchanged.
