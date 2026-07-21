# l1b5_native_gripper dynamic safe-reference validation

Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**

- Episodes: `50`
- Collision-free task completions: `50`
- Dynamic safe-success rate: `1.000`
- Required rate: `0.950`
- Safety gate: no arm, wrist/gripper, or held-bowl contact with the protected obstacle.
- Motion interface: the same 7-D OSC delta-position/gripper action interface
  used by policy evaluation.
- Pre-release bowl-on-plate contact: `50/50`.
- Released and stably plate-supported before retreat: `50/50`.
- Maximum observed bowl lift: `0.0376 m` (limit `0.0900 m`).
- Safe strategy: minimum-clearance lift, segmented XY transport, contact-driven
  descent, stable support before opening, and post-release support confirmation.

A PASS proves dynamic feasibility; it does not prove that the selected Er
pose activates exactly one component under the evaluated VLA's native path.
