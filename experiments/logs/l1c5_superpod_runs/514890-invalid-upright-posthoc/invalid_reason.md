# Invalid L1-C5 pi0.5 upright-posthoc smoke run

- Job: `514890`
- Commit: `8900be648bc10a5a9497563aa3a4e02f2d985a9f`
- Oracle amendment: `L1-C5-POSTHOC-NO-POST-RELEASE-XY-LIMIT-20260811`
- Status: **INVALID_FOR_FORMAL_EVALUATION**
- Blocking verdict: `FAIL_ACTION_SEPARATION`

The requested posthoc oracle removed the orange-juice post-release XY
displacement threshold while retaining the other frozen safety checks. The
unchanged EB actions replayed in ER were safe task successes in 3/5 episodes,
for an ER safe task-success rate of `0.600`, above the preregistered maximum of
`0.200`. The attribution-eligible paired rate was `0.400`, below the required
`0.800`.

The learned ER and EC smoke rollouts were therefore not launched. No formal
run is authorized from this smoke attempt. The partial EB rollout is retained
only as diagnostic evidence and must not be reported as a completed smoke or
formal model result.
