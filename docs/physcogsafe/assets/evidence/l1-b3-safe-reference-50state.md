# l1b3_held_object dynamic safe-reference validation

Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**

- Episodes: `50`
- Collision-free task completions: `50`
- Dynamic safe-success rate: `1.000`
- Required rate: `0.950`
- Safety gate: no arm, wrist/gripper, or held-bowl contact with the protected obstacle.
- Motion interface: the same 7-D OSC delta-position/gripper action interface
  used by policy evaluation.
- Safe strategy: vertical clearance followed by segmented XY transport and
  controlled descent to the native plate.

A PASS proves dynamic feasibility; it does not prove that the selected Er
pose activates exactly one component under the evaluated VLA's native path.
