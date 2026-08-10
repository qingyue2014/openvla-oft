# L1-B3 Task-4 Outcome V2 frozen pi0.5 handoff

These six artifacts are the exact EB/ER/EC scene handoff produced by Superpod
job `512800` at commit `9a6516b184cd6695f502a17fd7cb011eaf8a9a75`.
They are preserved for pi0.5 and subsequent Cosmos evaluation without scene
regeneration, recalibration, or outcome-conditioned modification.

The archived native-preflight record captures the state at the source job. Its
then-current OpenVLA follow-up wording is historical metadata; the current
preregistration supersedes model order with `pi0.5 -> Cosmos` and treats
OpenVLA-OFT only as immutable development/calibration provenance. The archived
record itself remains byte-for-byte unchanged so its registered SHA-256 stays
auditable.

Run the simulator-free verifier before staging any file:

```bash
python experiments/robot/libero/tasks/verify_l1b3_task4_pi05_handoff.py
```

The handoff is not human-approved and does not authorize formal evaluation.
