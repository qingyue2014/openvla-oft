# L3-B moka sequence v5 result

Status: **Ec capability control failed closed; Er was not run.**

- Source commit: `e7fa9847a6fbeb1f68a095e0bde91d1a91dd2984`.
- Slurm evaluation job: `500075`.
- Exact native task: `libero_10`, task 8.
- Exact prompt: `put both moka pots on the stove`.
- Ec report SHA-256:
  `84273adbed8c648ff700e10a10691b38f2a7852da604c278f2d42e77d4e75d49`.
- Ec native-goal successes: `5/5`.
- Ec strict terminal-stable successes: `2/5` (required `3/5`).
- Failed terminal maximum pot2 tilts: `10.71599`, `1.07478`, and
  `7.68537` degrees.
- The preplaced pot1 displacement was effectively zero in every episode.
- Er outcomes used during v5 development: none.

The v5 landing-axis change removed the earlier pot-to-pot dynamic
interference: π0.5 manipulated pot2 and reached the unchanged native goal in
all five Ec episodes without disturbing pot1. The result still fails because
three completed pot2 placements exceeded the locked 1.0-degree receptacle
tilt limit.

v6 therefore keeps the v5 geometry, roles, prompt, task, assets, and all
physical thresholds unchanged. It replaces the five-state
native-success-conditioned pool with the first 20 official states in native
file order and retains the same 60% Ec gate (`12/20`). The 20-state pool and
gate were locked before any v6 outcome, and Er remains blocked until Ec
passes.

Jobs `500071` and `500072` ended before any policy episode because their
runtime cache or CUDA module environment was incomplete. They are
infrastructure failures, not experiment evidence.
