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
| 2026-07-26 | pending | Current candidate-0 full trace | NOT RUN | Export all 241 steps for A dy=.142, AB=.056, BC=.038 to identify the exact contact geometry and instant causing A's -y/downward reversal. |
| 2026-07-26 | pending | policy view | NOT REVIEWED | Actual 256×256 exact-state artifacts not yet generated. |
| 2026-07-26 | pending | Πsafe | NOT RUN | Requires paired safe Eb controller traces. |
| 2026-07-26 | pending | Eb replay | NOT RUN | Requires paired safe Eb controller traces. |
