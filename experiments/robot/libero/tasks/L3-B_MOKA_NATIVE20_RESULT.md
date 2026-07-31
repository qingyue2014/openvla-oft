# L3-B moka native-20 result

The preregistered `l3b_moka_native20_v1` capability screen **failed**.

- Raw native LIBERO success: 11/20 (55%).
- Stable native success: 5/20 (25%).
- Locked requirement: at least 12/20 (60%).
- Stable-success state indices: 3, 5, 7, 17, and 18.
- Raw successes rejected for terminal tilt: states 0, 1, 9, 10, 11,
  and 13.
- Result: `near_far_authorized=false`; neither partial-history condition ran.

All 20 native initialization states and all 60 paired-condition saved-state
replays passed the native-only physical and policy-view gates before the policy
job. The policy job completed all trials and then exited non-zero as required
by the failed capability gate.

The first preparation attempt exposed a renderer-portability issue in the old
image replay threshold. Before any policy rollout, an all-60-state diagnostic
showed 60/60 physical PASS, maximum mean pixel error 0.161/255, and agent-view
p99 error 4--5/255. The renderer gate was transparently calibrated from
`mean <= 1, p99 <= 3` to `mean <= 1, p99 <= 8`; no task, state, physical, or
policy-outcome threshold changed.

Audit bindings:

- preregistration SHA-256:
  `26f97649aa51ff689ee3c221d0679436b994ea335b45adda146b64960edf1356`;
- capability report SHA-256:
  `6b44fd1db40491df056c589f0be52445e2573c28927ee41ccac3f6c68007413b`;
  and
- result-binding SHA-256:
  `dfd06195b67de243b5db49c629d605e4e31e460e324ca09a3560145b71299ad4`.

This result does not estimate an order effect. It rejects the current moka
candidate for L3-B evaluation with this checkpoint because native two-pot task
competence is below the preregistered gate.
