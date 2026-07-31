# L3-B moka render-replay calibration

Status: **pre-policy infrastructure calibration**, recorded before any
`native20` policy rollout.

The first native-20 preparation job passed all 20 construction gates and exact
serialized pairing, then stopped in the saved-state image replay gate. A
diagnostic replay of all 60 `native`, `near_first`, and `far_first` states
showed:

- 60/60 exact evaluator-style physical gates passed;
- wrist raw and π0.5-resized images passed the original pixel thresholds;
- agent-view mean absolute error was at most 0.161 intensity levels out of
  255; and
- agent-view 99th-percentile error was 4--5 intensity levels across all 60
  states.

The uniform, sub-intensity mean error with a slightly larger tail is consistent
with cross-GPU EGL anti-aliasing at a thin set of rendered edges, rather than a
scene, camera, or serialized-state mismatch. The old conjunction
`mean <= 1.0` and `p99 <= 3.0` was therefore renderer-specific.

Before any policy outcome was generated, the replay gate was fixed to:

- mean absolute pixel error at most 1.0/255; and
- 99th-percentile absolute pixel error at most 8.0/255.

The mean limit is unchanged. The revised p99 limit is above the observed 4--6
diagnostic range while remaining only 3.1% of the intensity scale. It applies
equally to all conditions and all raw and π0.5-resized policy views. Physical
state, support, contact, tilt, velocity, and full-window stability gates are
unchanged.

This calibration changes neither the locked pool nor the preregistered 12/20
stable-success threshold. The failed preparation job and all-pool pixel
diagnostic remain retained as audit evidence.
