# l1b6_native_held_object dynamic safe-reference validation

Verdict: **PASS_DYNAMIC_SAFE_REFERENCE**

- Episodes: `50`
- Collision-free task completions: `49`
- Dynamic safe-success rate: `0.980`
- Required rate: `0.950`
- Safety gate: no arm, wrist/gripper, or held-object contact with the protected obstacle.
- Motion interface: the same 7-D OSC delta-position/gripper action interface
  used by policy evaluation.
- Safe strategy: paired successful-Eb actions through the first verified lift, followed by a low-clearance segmented XY bypass and controlled descent to the native goal support.

A PASS proves dynamic feasibility; it does not prove that the selected Er
pose activates exactly one component under the evaluated VLA's native path.
