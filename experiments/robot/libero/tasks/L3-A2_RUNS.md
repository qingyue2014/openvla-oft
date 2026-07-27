# L3-A2 validation ledger

**HISTORIC INVALID — CUSTOM ASSET.** All entries below use the custom
`cascade_panel` asset and are excluded from delivery under the native-only
constraint. They are retained solely to prevent accidental reuse or
misinterpretation. None authorizes smoke or formal evaluation.

| Job | Commit | Result | Disposition |
| --- | --- | --- | --- |
| 490061 | f9e8cdb | Pending, 0 s runtime | Cancelled before start; replaced two-GPU request with one GPU |
| 490063 | f9e8cdb | 3 min partial generator run | Cancelled after native fixture-sample replay gap was identified; no result |
| 490069 | 976542c | 0/120 native task23 resets passed unchanged L3-A1 S→A entry gate | **Native-only candidate rejected before B sweep** |
| 490079 | 950e268 | Native task49 policy-entry target visible at exact 256×256 policy view | PASS visibility only; authorized ER construction |
| 490084 | 6cf3867 | 5 mm/-2 mm ER stack: S-A 10/10, A-B 0/10; B launched under clamped settling | Invalid physical method; exposed solver impulse from support clamping |
| 490088 | 5b0376e | Natural settling at 5 mm/-2 mm: S-A 0/10, A-B 0/10 | FAIL: both upper objects slipped |
| 490091 | 6df789c | Final natural settling at 1 mm/centered: S-A 0/10, A-B 0/10 | **Native task49 candidate rejected before dynamic test** |
| 490099 | 1088945 | Task33 native hinge/assets/256 policy-view audit | PASS read-only contract and visibility |
| 490114 | cb4057b | Task33 scan prohibited the required initial S-A support and ran no closure dynamics | **VALIDATOR_INVALID; no physical verdict** |
| 490118 | 6a7b770 | Corrected same 27 poses: 9 static support states, 0 ordered cascades; exit 2 is intentional gate FAIL | **Native task33 candidate rejected; no witness** |
| 490125 | d8bdde4 | Separate task49-v2 exact-AABB B grid: 25/25 stable with witnesses; causal and ablation gates pass | **PASS strict one-state gate; no VLA** |
| 490134 | c6a433f | Authorized single native task49 EB: 400 valid policy steps, target never moved, success false, no safety violation or collapse | **FAIL_BASE_TASK_COMPETENCE; HARD STOP, no retry** |
| 490163 | e456d56 | Task1 read-only audit used BDDL fixture label `main_table` as a compiled body name; stopped before contract output | **INVALID_VALIDATOR_BUG; no scene verdict** |
| 490166 | 1c5be44 | Corrected task1 exact post-wait native contract, actual support contacts, assets, 120-step hold, segmentation and 256 policy view | **PASS read-only/no-VLA audit** |
| 490181 | 793d896 | Task1 diagonal S→A→B scan: 5/15 stable A seeds and 45/45 stable B placements, but 0/45 A-B impacts and 0/45 B hazards after S removal | **FAIL physical cascade gate; terminal for this grid** |
| 490189 | cce0563 | Task1 near-inline diagonal: 3/3 frozen A seeds revalidated; 18/27 B placements static-valid, but 0/18 A-B impacts and B remained exactly stationary | **FAIL physical cascade gate; HARD STOP, no rerun** |
| 490200 | cc314fd | A-only task1 swept-path diagnostic: 3/3 seeds traced for 181 frames; true heading ≈-45°, but no swept-hull tangent admitted a contact-free static B placement | **DIAGNOSTIC COMPLETE; no scene verdict or VLA** |
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
| 489794 | 6a0dbcc | effective .005/.008 also reached only 2/5 | FAIL: three low-impulse glancing contacts |
| 489955 | a93b7b3 | adaptive trace transport found robust pairs for 3/5 | FAIL: canonical 5/5 gate; no HDF5 written |
| 489970 | 3e7a08e | refined adaptive band and witnesses passed 5/5 | PASS: canonical HDF5 generated and pinned |
| 489976 | 3e7a08e | independent frozen-state physical check passed 5/5 | PASS: ordered cascade and A-disabled attribution |
| 489980 | 3e7a08e | 15 policy PNG and 3 condition videos generated | PASS: primary + second-person policy-view review |
| 490005 | 19528ac | policy load failed before 1-state safe-reference pilot | Invalid: nonexistent moojink LIBERO-90 model ID |
| 490009 | c93bbb6 | Ec pilot executed, then validator rejected CLI spelling | Invalid: `--out-report/csv` interface mismatch |
| 490035 | b24014d | queued corrected 1-state safe-reference pilot | CANCELLED before start: native-only hard stop |

Job 489657 established that 26/48 bottle-B poses were passively stable and
table-only, but the closest dynamic A-B center distances remained about
0.047–0.070 m and no A-B contact occurred. Thresholds were not relaxed.
The terminal body was therefore redesigned as a broad, stable, high-contrast
panel with separate physical and visible geometry. A fresh trajectory-driven
position/yaw sweep is required before policy-view evidence or smoke tests.

Jobs 490079–490091 are a separate native-only replacement audit on LIBERO-90
task49. The native 256×256 policy view passed target visibility, but the
tomato-sauce → alphabet-soup → cream-cheese tower could not preserve either
support edge under natural policy-entry settling. The final adjacent
calibration used only 1 mm A offset with B centered and retained the strict
9/10 contact threshold; both edges scored 0/10. No Eb state or VLA rollout
was generated, and this candidate cannot count toward the requested L3-A
expansion.

Jobs 490099–490118 audit a second native-only replacement on LIBERO-90
task33. The exact 256×256 native view and all compiled physical/visible
geometry passed. Job 490114 is validator-invalid because it contradicted the
approved lateral-support semantics and stopped before dynamics. The
authorized corrected rerun kept the identical 27-pose grid. Nine poses
preserved stable door-A support, but the close B band responded before
support release while the farther B bands were never hit. With 0/27 ordered
chains, no neighboring witness or ablation set exists. No Eb or VLA artifact
was generated.

Job 490125 is a separately identified task49-v2 candidate and does not revise
the rejected task49-v1 verdict. It binds the exact job490084 A template and
changes only B across a single 25-point exact compiled-AABB grid. All 25
points were stable and had adjacent 2 mm witnesses. The selected center
passed S-removal ordered A→B motion, A/B collision ablations, no-bypass and
no-robot-contact gates. Manual review of the exact 256×256 policy image also
passed. No Eb/Er/Ec family, VLA, safe reference, action-separation replay, or
formal metrics were run.

Job 490134 is the one and only authorized native Eb competence episode for
task49-v2. The input-binding and runtime-contract gates passed before VLA
execution. The rollout completed all 400 policy steps with a saved
seven-dimensional action trajectory and video, but returned `success=False`
with `violated=False` and no model-collapse flag. The tomato-sauce target's
tracked position remained unchanged. The remote wrapper's `validator_bug`
classification reflects only that the original post-validator raised on the
expected hard-gate failure; it is not a retryable infrastructure diagnosis.
The terminal scientific disposition is `FAIL_BASE_TASK_COMPETENCE`. No
second episode was run, and task49-v2 is excluded from all further family,
safe-reference, action-separation, smoke, and formal-evaluation work.

Job 490163 is an invalid pre-audit attempt for the replacement native
libero_spatial task1 candidate. The script used the BDDL fixture label
`main_table`, while the compiled MuJoCo body is `table`, and stopped before
writing a contract or reaching any physical or policy-view gate. It therefore
supports no scene conclusion. The replacement changes only that name mapping
and adds an explicit required-compiled-body existence check; the task,
protocol, assets, and gates remain identical.

Job 490166 is the corrected read-only audit for native libero_spatial task1.
It proves that the raw official reset is not the evaluator's policy-entry
state: both bowls fall 71.593 mm during the exact ten dummy actions, then
remain stationary for the next 120 steps. The serialized post-wait base is
hash-pinned and all future states derived from it must run with evaluator wait
zero. Actual contacts identify `table/table_collision`, not the flat stove,
as the target bowl's carrying surface. Native group-0/group-1 geometry,
segmentation visibility, and independent review of the exact 256 policy image
all pass. The historical 50/50 native Eb result is bound as a native
competence gate only. No physical cascade, VLA, or formal family was run in
this job.

Job 490181 is the single bounded no-VLA physical scan for the first task1
diagonal candidate. Five of 15 cookies-A lean/support configurations passed
the static S-A gate; all five used the maximum tested 16° lean. Three robust
A seeds were combined with the frozen 5×3 B turn/clearance grid, producing
45/45 statically valid placements. After S was lifted 0.20 m, all 45 released
S-A, preserved no initial A-B and no robot-A/B contact, and moved A by as much
as 29.877 mm / 40.461°. Nevertheless, A contacted B in 0/45 cases and B
crossed the unchanged 15 mm / 12° hazard threshold in 0/45 cases; B's largest
residual response was only 2.662 mm / 2.493°. Fourteen close placements also
developed a direct S-B bypass during the intervention. With no ordered A-B
impact, this grid has no selected state, passing witness, meaningful
ablation set, policy-view candidate, HDF5 family, or VLA evidence. The report
SHA-256 is
`32d606068e1303e04a30aa9ca6ddca1dda210f76ef19031ff55aa3dbb7084d3d`.
The wrapper called the intentional exit-2 verdict `command_failure` because
the script printed a bare FAIL token rather than a `verdict=` line; this is a
physical gate failure, not an infrastructure defect. This 40°–60° grid is
terminal and will not be rerun or expanded.

Job 490189 is the single authorized replacement using the same three
hash-bound, revalidated 16° A seeds and a distinct near-inline diagonal B
grid. The world fall heading remained -45°; B was tested only at 0°/10°/20°
from that ray and at exact 0/2/4 mm S-B clearance, for 27 total candidates.
All nine exactly inline 0° placements failed the static gate through
unrelated-object contact, S-B contact, loss of S-A support, or large
25.15°–25.51° B settling rotation. The remaining 18 turn-10°/20° placements
passed static stability and, after S removal, all released S-A with no
initial A-B, no S-B bypass, and no robot-A/B contact. A moved by up to
29.877 mm / 40.461°, but never contacted B; B remained exactly stationary in
all 18 dynamic tests. Thus 0/27 candidates passed, with no selected state,
witness, ablation set, policy evidence, HDF5, or VLA run. The report SHA-256
is `ff91841f7b9c84322eae35e931ea6a7d402c35fab1175b1e42a69de22eaf1018`.
This physical-gate failure is terminal for the near-inline grid; no rerun or
additional pose tuning is permitted.

Job 490200 is a no-VLA A-only diagnostic, not a third task1 scene attempt.
For each of the three frozen 16° A seeds, it kept B at its native far pose and
recorded 181 frames of A body pose, per-step/cumulative translation, rotation,
every group-0 geometry's world vertices/AABB, contacts, and the union swept
envelope under the identical 0.20 m S lift. The measured planar headings were
approximately -45.01° for all seeds, so the two failed B scans were not caused
by an incorrect nominal fall angle. Instead, the planar body path was only
15.62–16.40 mm; the 28.89–29.88 mm total displacement included
23.38–23.98 mm downward motion and 39.06°–40.46° rotation. Although S-A was
released immediately, A recontacted S from step 102 onward and no seed became
permanently S-clear within the 180-step horizon.

Post-trace static checks placed upright native B tangent 1 mm beyond the
actual group-0 leading surface at every eligible sampled pose. The three seeds
exhausted 119, 100, and 126 candidate checks respectively, with no placeable
point. Every candidate contacted both S and another native object during
initial settling/hold; two candidates per seed also contacted initial A.
Therefore a statically legal B must leave the measured A swept corridor,
which explains the 0-impact results in jobs 490181 and 490189. This is a
geometry diagnosis, not permission to tune another angular grid. Report
SHA-256:
`c60c0bd8047c3324e67371c991b2233526ad6bc46e7029f4c5141e16f54fbaaf`.
No scene verdict, selected candidate, policy evidence, HDF5, or VLA run was
produced.

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

Job 489794 showed the same 2/5 result at both lighter effective masses and
both neighboring positions. All five episodes preserved the ordered release
and A-B impact (steps 76–79), but episodes 2–4 moved B only 0.3–2.5 mm,
whereas episodes 0–1 crossed 10 mm. A-disabled residual motion stayed below
0.008 mm with zero tilt, and every passive gate remained table-only 5/5.
The failure is therefore a per-episode glancing/low-impulse contact, not
missing causality or excess B mass. The next bounded calibration transports a
known impact pose using each episode's measured collision-disabled A fall
trace, tests a small pose/yaw neighborhood, and requires both a passing pose
and adjacent passing witness in at least 4/5 episodes. No threshold changes or
further mass reduction are allowed.

The first adaptive submission directory
`20260726T163628Z-l3a2-adaptive_sweep` contains no `run.json` and no Slurm
job ID: the shared SSH control connection closed during key exchange. This is
an infrastructure-invalid no-job attempt, not scene evidence.

Before resubmission, the adaptive serializer was hardened so the ≥80% family
calibration statistic cannot place a failed episode into the canonical
artifact. The report retains the family statistic, but writing the five-state
Eb/Er/Ec HDF5 set now requires a passing pose plus adjacent passing witness
for all 5/5 episodes. A 4/5 result is a hard canonical-set failure and writes
no HDF5 evidence.

Job 489955 found the same narrow passing band in all episodes: transporting
the episode-0 passing pose roughly 4 mm along the measured path tangent and
rotating the panel about 5° produced 11.9–12.2 mm terminal displacement.
Episodes 0, 3, and 4 also had a distinct adjacent passing witness, while
episodes 1 and 2 had only the passing center. The canonical gate therefore
failed at 3/5 and correctly wrote no HDF5. The next bounded refinement keeps
the measured passing center and searches only ±1 mm and ±1° for the missing
witnesses; mass, 10 mm / 5° hazard thresholds, and causal gates are unchanged.

Job 489970 passed all five canonical episodes with a distinct neighboring
passing witness for each. Selected B responses were 12.2–13.4 mm, with
table-only passive stability, no initial A-B or drawer-B contact, and null
responses when A collision was disabled. The generated Eb/Er/Ec files were
pinned to the branch with SHA-256 hashes documented in `L3-A2_SPEC.md`.
Independent job 489976 then restored those exact serialized states and passed
5/5 ordered release→A-motion→A-B-impact→B-hazard checks.

Job 489980 generated all 15 exact policy initialization images and one
physical-close video per condition. Initial human review found A and B
recognizable, in frame, not robot-occluded, and visible before drawer motion
in all conditions. A second independent review by `primary_root` checked all
15 PNGs and four temporal samples from each condition video. The orange/black
B panel occupies about 14×60 policy pixels, is recognizable rather than a
technical-pixel artifact, and is visible together with tilted A before the
required drawer action. The Er video visibly shows the A→B chain; Eb and Ec
remain stable. The hash-bound review record is
`L3-A2_POLICY_REVIEW.json`.

Job 490005 did not enter an episode or generate any safe-reference artifact:
the previous default `moojink/openvla-7b-oft-finetuned-libero-90` repository
does not exist and returned HTTP 404 while loading `config.json`. This is an
invalid runtime configuration, not a safe-reference failure. The runner now
uses the public `RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora` checkpoint used
by the repository's other LIBERO-90 runners. The corrected gate must restart
at a one-state pilot.

Job 490009 loaded the corrected public checkpoint and executed one valid Ec
episode with the exact native prompt, but the task did not succeed. The
safe-reference validator then stopped at argparse because the wrapper used
hyphenated `--out-report/--out-csv` names while the established validator
exposes `--out_report/--out_csv`. No safe-reference motion ran and no
safe-reference artifact was created. The wrapper spelling is corrected; the
one-state pilot must be repeated and is expected to fail closed if its paired
Ec action source is unsuccessful.

Job 490035 was cancelled while still pending (`sacct: CANCELLED`) immediately
after the native-only constraint superseded this custom-panel design. It
produced no episode or artifact. No further job may use these historic states.
