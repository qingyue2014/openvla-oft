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
| 2026-07-26 | pending | GPU scene | NOT RUN | Candidate offsets require live MuJoCo calibration. |
| 2026-07-26 | pending | policy view | NOT REVIEWED | Actual 256×256 exact-state artifacts not yet generated. |
| 2026-07-26 | pending | Πsafe | NOT RUN | Requires paired safe Eb controller traces. |
| 2026-07-26 | pending | Eb replay | NOT RUN | Requires paired safe Eb controller traces. |

