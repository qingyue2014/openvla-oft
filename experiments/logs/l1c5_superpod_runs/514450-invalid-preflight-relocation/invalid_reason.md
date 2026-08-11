# Invalid L1-C5 pi0.5 Formal Launch 514450

- Status: **INVALID_VALIDATOR_PATH_BINDING**
- Superpod job: `514450`
- Execution commit: `f5092467a3e6ec79b941885b9245b32c63ea2d61`
- Frozen gate SHA-256: `14eeb148208f536eca7920ddde28b285502007d6b719739f6742712264cd5937`
- Failure point: before formal episode 0 and before any learned-policy action was evaluated.
- Error: `evaluated initial-state file was not preflighted`
- Root cause: the frozen native preflight stored the construction checkout's absolute state-file path. The formal job ran from an immutable Superpod worktree with the same logical filenames and exact state-file hashes under a different checkout root, and the validator rejected the path relocation before checking the hashes.
- Impact: no formal metrics, result tables, review videos, or interpretable learned-policy outcomes were produced. All partial artifacts in this directory belong only to this invalid launch and must not be pooled with a valid formal run.
- Remediation scope: validator-only. Accept a relocated checkout only when exactly one preflight condition has the same logical state filename, then continue to require the exact preflighted state SHA-256, prompt, BDDL metadata, native task identity, and runtime asset inventory.

The scene, EB/ER/EC states, risk variable, candidate positions, thresholds, safe reference, policy checkpoint, and human-review evidence were not modified.
