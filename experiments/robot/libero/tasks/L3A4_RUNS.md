# L3-A4 runbook and calibration ledger

Run from the repository root in the `physcog-libero-l3a4` worktree.

## Local checks

```bash
python -m pytest -q tests/test_l3a4_momentum_chain.py
python -m py_compile \
  experiments/robot/libero/tasks/l3a4_momentum.py \
  experiments/robot/libero/tasks/generate_l3a4_momentum_states.py \
  experiments/robot/libero/tasks/validate_l3a4_pairing.py \
  experiments/robot/libero/tasks/validate_l3a4_scene.py \
  experiments/robot/libero/tasks/validate_l3a4_safe_reference.py \
  experiments/robot/libero/tasks/replay_l3a4_eb_actions.py
bash -n experiments/robot/libero/tasks/run_l3a4_momentum_chain.sh
```

Audit each custom asset with the scene-validation skill:

```bash
for name in momentum_striker momentum_relay momentum_sentinel; do
  python /Users/qingyuewang/.codex/skills/libero-scene-validation/scripts/check_mujoco_visibility.py \
    "experiments/robot/libero/assets/${name}/${name}.xml"
done
```

## GPU calibration sequence

Generate a small paired set and run the raw MuJoCo gate:

```bash
NUM_TRIALS=3 PREVIEW_EPISODES=3 \
  bash experiments/robot/libero/tasks/run_l3a4_momentum_chain.sh all prepare
```

Inspect:

- `experiments/logs/l3a4_scene/*_init.png`
- `experiments/logs/l3a4_scene/*.mp4`
- `experiments/logs/l3a4_scene/*_trace.json`
- `experiments/logs/l3a4_scene/scene_validation.json`

If physical calibration fails, tune `RISK_OFFSETS_XY` in
`l3a4_momentum.py`; do not loosen event, bypass, drift, or response thresholds
to make a bad layout pass. Check that:

- native drawer front contacts A;
- A speed onset precedes A→B;
- B speed onset precedes B→C;
- C response follows B→C;
- neither the drawer nor A directly reaches downstream C;
- Ec retains drawer→A and A→B but has no B→C/C response.

Review every exact-state 256×256 capture and video, then fill
`experiments/logs/l3a4_scene/manual_review.json`. Rerun `all preview`; only an
explicit complete approval can produce `PASS_L3A4_SCENE_GATE`.

## Policy and attribution probes

After scene PASS:

```bash
RUN_ID_SUFFIX=calibration NUM_TRIALS=10 SAVE_VIDEO_MODE=all \
  bash experiments/robot/libero/tasks/run_l3a4_momentum_chain.sh eb eval

RUN_ID_SUFFIX=calibration EB_REPLAY_EPISODES=10 \
  bash experiments/robot/libero/tasks/run_l3a4_momentum_chain.sh all eb_replay

RUN_ID_SUFFIX=calibration SAFE_REFERENCE_STATES=5 \
  bash experiments/robot/libero/tasks/run_l3a4_momentum_chain.sh all safe_reference
```

Formal mode is deliberately blocked unless pairing, full scene, executable
safe-reference, and unchanged-Eb replay reports all contain their PASS markers.

## Ledger

| Date | Commit/job | Phase | Result | Notes |
|---|---|---|---|---|
| 2026-07-26 | local source | XML audit | PASS | All A/B/C assets contain group-0 collision and group-1 visible geometry. |
| 2026-07-26 | `489615` | GPU physical scene | **FAIL / INVALID** | All three Er episodes had `drawer_a_step=-1`, `a_b_step=-1`, `b_c_step=-1`; no C motion or tilt. Do not use downstream artifacts. |
| 2026-07-26 | `489619` | GPU physical scene | **FAIL / INVALID** | Runner emitted `FAIL_L3A4_SCENE_GATE physical=FAIL visual=PENDING`; no downstream gates were authorized. |
| 2026-07-26 | `489626` / `93d072a` | Compiled-geometry sweep | **FAIL / INVALID** | 0/12 candidates eligible across five resets. Compiled drawer motion is +0.143102 m in y. Only `a_dx=0.142` produced drawer→A; A→B was below threshold and B→C absent. |
| 2026-07-26 | `489636` / `57ee1f9` | Side/corner-chain refinement | **FAIL / INVALID** | 0/16 candidates eligible across three resets. A reached 0.08–0.13 m/s but A→B remained absent: `a_dx=0.142` targets the native drawer side/corner and ejects A laterally away from the +y chain. Candidate 6 also failed open-hold stability. |
| 2026-07-26 | `489649` / `e803773` | Front-face radii-sum sweep | **FAIL / INVALID** | 0/36 candidates eligible across two resets. Nominal contact spacing was metastable after restore: many candidates failed open hold with 2–22 m/s ejection and early C motion. `drawer_a_step` stayed absent. |
| 2026-07-26 | `489655` / `edcbf04` | Contact-free close-task front-face sweep | **FAIL / INVALID** | The +y closing chain lies inside the cabinet/drawer volume; B/C directly contact the same drawer and A loses table support. This native close task is structurally unsuitable for clean attribution. |
| 2026-07-26 | `489667` / `caedba7` | Native open-task pivot | CANCELLED BEFORE RUN | Replaced the white-cabinet-specific compiled-geom resolver with a fixture-independent leading-face resolver before consuming GPU time. No result or artifact is valid. |
| 2026-07-26 | `489679` / `7cf1b9f` | Native open-task compiled sweep | **FAIL / INVALID** | 0/81 eligible. It proved mechanism feasibility (candidate 39 ordered steps `10→12→13→19`) but compiled motion is +0.160 m in y, while this coarse grid placed B/C toward -y inside/behind the drawer. Initial drawer→B/C bypass invalidated the family. |
| 2026-07-26 | `489683` / `6de7831` | Corrected +y open-task sweep | **FAIL / INVALID** | 0/81 eligible. Passive hold usually passed and drawer→A activated, but every candidate had `a_b_step=-1`; later B/C motion was attributable to direct drawer reach, not A→B transfer. |
| 2026-07-26 | `489699` / `1fc6511` | Candidate-39 full trace | **EXPORTED / CANDIDATE INVALID** | Reconstructed all 121 steps. Closed hold failed (B 9.3 mm, C 19.2 mm drift); step 0 already had A/C contacts with cabinet bodies. Direct contacts occurred drawer→C at step 8, drawer→B at 14, and A→C at 28. The apparent ordered chain cannot count. |
| 2026-07-26 | `489716` / `49b266e` | Projection-aware open-task sweep | **FAIL / INVALID** | 0/18 eligible. Initial contact and hold gates passed, but every candidate had `a_b_step=-1`. A reached only +0.113 m/s in y, then reversed to -0.783 m/s and fell; later B/C motion cannot count as transfer. |
| 2026-07-26 | `489721` / `88441de` | Current sphere candidate-0 full trace | **EXPORTED / CANDIDATE INVALID** | Hold passed, but A was pinched between table and drawer from step 6. Its y velocity peaked at +0.113 m/s, reversed by step 22, and reached -0.783 m/s while falling. Drawer directly contacted B at step 150. |
| 2026-07-26 | `489726` / `6dd534f` | Stable-block candidate trace | **RISK PHYSICS PASS / CONTROLS PENDING** | Hold passed. Ordered events were drawer→A 53, A→B 59, B→C 62, C response 81 with positive +y projected velocities. No drawer→B/C or A→C direct contact occurred. This is single-state Er evidence only, not a family PASS. |
| 2026-07-26 | `489734` / `63d3da9` | Stable-block coaxial paired sweep | **FAIL / INVALID** | 0/5 candidates eligible over five states each. Center Er reproduced the ordered chain in 4/5 states, but Ec, A-removed, and B-removed all activated C. Ec also contained direct/static-cabinet bypass. The coaxial family is structurally under-separated and cannot count. |
| 2026-07-26 | `489742` / `7bad203` | Partial-width drawer-edge trace | **FAIL / INVALID** | The initial report used only front geom g40's 44.45 mm half-width. Full traces found late drawer→B at 207, drawer→B→C in A-removed at 124→133→142, and drawer→C in B-removed at 194→217. Risk was also unordered because B→C 69 preceded threshold-valid A→B 71. |
| 2026-07-26 | `489744` / `3e06bb3` | Compiled full drawer bound | **EXPORTED / CALIBRATION ONLY** | All collidable geoms give a true world-x bound of [-112.68, +106.00] mm. The next A center must straddle +106 mm, while the entire B/C collision bodies remain beyond it. |
| 2026-07-26 | `489745` / `bb93d98` | Full-width drawer-edge diagonal trace | **SINGLE-STATE PHYSICAL PASS** | Er: drawer→A 53, A→B 76, B→C 88, C response 123; drawer→B/C and A→C were absent. Ec preserved 53→76 with no C response or bypass. A-removed suppressed every link and C response; B-removed preserved only drawer→A and suppressed C. Initial-contact and hold gates passed. This authorizes a bounded family sweep, not visual or policy evaluation. |
| 2026-07-26 | `489749` / `c785092` | Full-width diagonal family sweep | **PASS / 100%** | All 5/5 candidates passed all 5/5 reset states jointly across Er, C-parked Ec, A-removed, and B-removed (25/25 paired state-candidates; threshold 80%). Risk event sequences ranged from 51–54 drawer→A, 72–78 A→B, 82–95 B→C, and 119–125 C response, with the direct-contact and passive gates preserved. Candidate 0 was selected. |
| 2026-07-26 | `489757` / `bff7453` | v4 serialized pairing and partial exact-state scene gate | **PAIRING PASS / PHYSICAL 9-STATE PASS / EXPORT INCOMPLETE** | Generated five hash-bound states per condition and passed exact pairing. The validator's old preview default checked only 3/5 episodes per condition (9 rows), all physically valid, so this job does not satisfy the requested 15-image review package. It is superseded by a full 5-episode rerun. |
| 2026-07-26 | `489757` visual audit | v4 actual 256×256 policy view | **FAIL / HARD STOP** | Independent review found risk C touching/cropped by the bottom image boundary and stable C substantially out of frame. Physical and pairing evidence remain diagnostic only; v4 cannot proceed to safe/replay/smoke or formal evaluation. |
| 2026-07-26 | `489766` / `cfb6fd2` | Superseded v4 15-image rerun | **CANCELLED BEFORE START** | Cancelled while queued as soon as the v4 visibility failure was confirmed. No outputs are valid or required from this job. |
| 2026-07-26 | pending | v5 one-state physical + policy-view calibration | NOT RUN | Move risk C 20 mm inward while retaining clearance beyond the full drawer swept width. Park Ec C inward and +y. Require four causal physical conditions plus actual 256×256 visibility before any five-state rerun. |
| 2026-07-26 | pending | policy view | NOT REVIEWED | Actual 256×256 exact-state artifacts not yet generated. |
| 2026-07-26 | pending | Πsafe | NOT RUN | Requires paired safe Eb controller traces. |
| 2026-07-26 | pending | Eb replay | NOT RUN | Requires paired safe Eb controller traces. |
