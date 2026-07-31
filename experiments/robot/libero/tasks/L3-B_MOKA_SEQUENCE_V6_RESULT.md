# L3-B moka sequence v6 result

Status: **the preregistered Ec capability control failed closed; Er was not
run.**

- Source commit:
  `ef1a5b980dc256c5e5035736f0914b69475d7c53`.
- Slurm evaluation job: `500095`.
- Exact native task: `libero_10`, task 8.
- Exact prompt: `put both moka pots on the stove`.
- v6 design preregistration SHA-256:
  `75172021209dc277ad49458822882789978d5d4e721627224f2da9b0b7550aa2`.
- Ec report SHA-256:
  `cbbc1f0a81958e149a0e91a9ec6ca27aec6fcd09e7218b3c286e80243c93b099`.
- Ec native-goal successes: `18/20`.
- Ec strict terminal-stable successes: `10/20` (required `12/20`).
- Strict-stable official state indices:
  `0, 5, 7, 8, 10, 11, 12, 15, 16, 17`.
- Er outcomes used during v6 or v7 preregistration: none.

The v6 result remains a failure. The 1.0-degree receptacle tilt limit,
terminal-window drift limit, and `12/20` threshold were not relaxed, and the
strict-stable subset is not used to reclassify v6 as a pass.

Safe controller v4 independently passed all 20 v6 states. It changed only the
controller's open-gripper approach by starting from the vertical handle-entry
portion of the previously registered native trajectory reference. All 20
real-action trajectories completed the unchanged native goal without moving
the preplaced pot; ten success videos were retained to respect the local
per-category limit. This confirms environment solvability but does not repair
the failed v6 policy control.

v7 is a new conditional diagnostic, locked before any Er rollout. It asks:
among states where the frozen v6 Ec trajectory already proves that π0.5 can
make a strict-stable pot2 placement, does putting prior progress at the
policy's default landing side reduce Er completion or induce repair? The v7
pool contains all and only those ten frozen Ec successes, binds every source
trajectory by SHA-256, forbids a new Ec rollout, and runs Er on the identical
official state indices. Its estimates are explicitly conditional and must not
be reported as an unconditional LIBERO success rate.
