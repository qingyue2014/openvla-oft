# L3-A4 exact-native goal task-4 dynamic failure audit

## Scope and frozen identity

This audit covers the exact-native `libero_goal` zero-based task 4 candidate
with the unchanged policy prompt `put the bowl on top of the cabinet` and
official goal `(On akita_black_bowl_1 wooden_cabinet_1_top_side)`.

- untouched BDDL SHA-256:
  `2ffba859a154f50c3c99ffb3420743fa5aa65c70bf7d4cd26f5bc81d07be5713`;
- policy-entry base SHA-256:
  `93f35fe43d2a955274004a71332d41b83fb25508472f4f2b307f5b3bc80bfe9e`;
- selected state SHA-256:
  `4bc9d9b1e5c01f60b9f0b54a68450f95a71e958fd7aae01d7f0b1e738ff722c5`;
- adjacent witness state SHA-256:
  `be1d1ea777c5aef0a8891779acdadd9def9d993533a9fe2633af65cb916e7385`.

No physical parameter, serialized state, trajectory, threshold, duration, or
distance was tuned during the runs in this report.

## Ordered validation record

### Witness preflight: job 490331

Commit `25a6d13` ran only the witness policy-view preflight and exited with
`PASS_L3A4_WITNESS_POLICY_PREFLIGHT_PENDING_MANUAL_RGB` and
`dynamic_started=false`.

The exact 256x256 agent-view segmentation counts for
S/A/B/plate/cabinet were 472/233/393/1347/9285. The RGB SHA-256 was
`b77b79315e62b84761eed31c8f230be53f4c252c38feb065d5e74631c8c5a7bc`.
Manual review passed the recognizable layout.

### Invalid approval-validator run: job 490345

Commit `4ecf1c8` rerendered the exact same state on a different compute node.
The five segmentation PNGs were byte-identical to job 490331, and their
pixel counts and dimensions were unchanged. The renderer-dependent RGB PNG
SHA changed to
`47ea7d165ba4e0b0ca31ebadd198a2fec8bcb3358189ba4ecce1e934db88c29a`.

The original approval validator incorrectly treated the RGB byte hash as
decisive, rejected the approval, and exited with
`FAIL_L3A4_GOAL_TASK4_DYNAMIC_NOT_COMPLETED`. Its remote classification was
`validator_bug`, and `dynamic_started=false`.

Job 490345 is invalid and provides no dynamic, causal-control, or
safe-reference evidence. It is not a scene failure.

The validator-only correction in commit `9a129f8` bound approval to the exact
state and all five segmentation hashes, pixel counts, and dimensions. It
retained the RGB SHA only as a non-decisive diagnostic. Regression tests
verify that RGB equality is not part of the approval predicate.

### Valid dynamic hard stop: job 490366

Commit `9a129f8` passed the corrected deterministic approval gate, set
`dynamic_started=true`, and ran the unchanged frozen protocol. The raw
verdict was **FAIL_L3A4_GOAL_TASK4_FIXED_DYNAMIC_SAFE_REFERENCE**.

Robot-grasp preflight failed in both selected and witness states. After the
seat phase, the robot contacted A but did not retain robot/S contact. The
risk tests therefore used the predeclared and explicitly labelled
`kinematic_object_calibration` fallback. The required actual-OSC safe
reference failed at grasp and did not execute transport or call the official
goal success gate.

| Gate | Raw result | Decisive evidence |
|---|---:|---|
| Selected risk | FAIL | no A/B contact or impulse; A motion and B response both at step 1; precontact B 41.99 mm/98.06 degrees; carry drift 67.95 mm/10.23 degrees; bypass gate failed |
| Witness risk | FAIL | strict A step 1 → A/B step 4 → B step 5 sequence and 0.0025404 N s impulse, but carry drift 67.84 mm/10.69 degrees exceeded 3 mm/3 degrees |
| S-fixed control | PASS | B motion 0.00453 mm; negligible tilt; no A/B contact or bypass |
| A-disabled control | PASS | B motion 0.00453 mm; negligible tilt; no A/B contact or bypass |
| A-frozen control | FAIL | B moved 5.434 mm and tilted 3.832 degrees |
| Actual-OSC safe reference | FAIL | grasp failed before transport; official goal was not checked |

The job generated 256x256 policy-view videos for the selected and witness
kinematic risk calibrations and all three controls. It produced no
safe-reference video because grasp failed before transport. No VLA, HDF5,
formal evaluation, or action replay ran.

## Final classification

The layout remains a valid static candidate under the earlier exact-state
audit, but it is **not a validated dynamic L3-A extension**. Job 490366 is
the decisive dynamic failure. In accordance with the fixed hard-stop rule,
there was no tuning and no further run.

Raw fetched evidence is stored locally under:

- `.physcog-agent/runs/20260727T055223Z-l3a4-goal_task4_witness_preflight/`;
- `.physcog-agent/runs/20260727T060018Z-l3a4-goal_task4_fixed_dynamic/`;
- `.physcog-agent/runs/20260727T060843Z-l3a4-goal_task4_fixed_dynamic/`.
