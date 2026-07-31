# INVALID — L1-A4 Spatial v4 Formal Evidence

This entire directory is invalid and retired.

- Retired intervention:
  `l1a4_spatial_native_near_adaptive_v4`
- Retired run: `20260730T065043Z-l1a4s-formal_pi05`
- Retired Slurm job: `498131`
- Scope: every metric, table, report, attribution result, trajectory,
  downloaded artifact, video reference, and HTML entry derived from this run.

## Reason

The exact evaluator reset and 10-step no-op wait were replayed after the run.
In ER and EC the native target bowl contacted native `cookies_1` during the
wait and tipped in place. All 45 ER and all 45 EC target bowls were tilted
`15.57–30.99°` at the first policy-visible frame (median `23.69°`).

The retired validator checked translation drift but not post-wait
orientation, support, velocity, or stepwise contact. Its PASS verdict
therefore did not certify a physically valid initial policy state.

Do not interpret, aggregate, publish, or restore these files as evidence.
