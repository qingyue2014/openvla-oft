# Invalid L1-C5 Cosmos Smoke Run 514621

- Status: **INVALID_BEFORE_LEARNED_POLICY_OUTCOME**
- Superpod job: `514621`
- Bound commit: `78a6b49b60c98be5fa02ff3f483824f963d7c76f`
- Frozen gate SHA-256: `14eeb148208f536eca7920ddde28b285502007d6b719739f6742712264cd5937`
- First affected condition: EB, episode 0, before the first policy action
- Hard-stop reason: the compiled evaluator asset-inventory hash differed from the frozen native-task inventory hash.
- Root cause: the in-process Cosmos runtime selected its bundled simulator stack (`robosuite 1.4.0`, MuJoCo `3.3.2`, NumPy `2.2.6`) instead of the frozen evaluator stack (`robosuite 1.4.1`, MuJoCo `3.9.0`, NumPy `1.26.4`).

No learned-policy result, safety rate, task-success rate, attribution result, or review video from this run is valid. The mismatch was not waived and the inventory comparator was not relaxed. A replacement run must isolate Cosmos inference from simulation so the evaluator remains on the same frozen simulator stack used by pi0.5.
