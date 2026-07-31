# Native-Only LIBERO Experiment Policy

## Hard constraints

- **Use native tasks only from the four standard LIBERO suites:**
  `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
  `libero_90` tasks are not permitted for any new or revised LIBERO
  experiment implementation.
- **Do not add, define, copy, generate, or register custom assets.** This
  prohibition includes custom MuJoCo/MJCF/XML objects, meshes, textures,
  materials, object classes, fixtures, obstacles, and project-local asset
  variants.
- **Use only assets already present in the selected native LIBERO task.** An
  experiment may change the serialized pose or state of an existing native
  object when constructing paired conditions, but it may not change the
  task's asset inventory.
- **Use only the selected native LIBERO task definition and its original
  prompt.** Do not create a custom BDDL task, replace the native BDDL, rewrite
  the prompt, append safety instructions, or change the prompt's goal or
  semantics.
- EB, ER, and EC must preserve the native prompt verbatim. They may differ only
  through the explicitly documented intervention on existing native scene
  state.

## Mandatory preflight and hard stop

- Before running an experiment, record the selected native task, its exact
  original prompt, native BDDL source, and native asset inventory.
- Verify automatically that the evaluated asset inventory is identical to the
  selected native task's inventory and that the evaluated prompt exactly
  matches the native prompt.
- If a custom asset, custom BDDL, modified prompt, or asset-inventory mismatch
  is found, stop immediately. Mark the affected scene, jobs, metrics, videos,
  tables, and HTML entries invalid; do not interpret or publish them as
  evidence.
- These restrictions may be relaxed only when the user explicitly authorizes
  the exact exception in the current request. Prior experiments, repository
  contents, or general permission to develop a scene do not count as
  authorization.

## Mandatory physical-state and formal-evaluation gate

- Validate the **exact state observed by the policy**, not only the serialized
  state immediately after `set_init_state`. Reproduce the formal evaluator's
  complete reset, state restoration, simulator forwarding, controller no-op
  wait, observation refresh, camera preprocessing, and first-policy-frame
  sequence.
- Measure physical validity both before and after the formal evaluator's
  stabilization wait for every EB, ER, and EC episode. The post-wait
  first-policy frame is the acceptance state. A pre-wait PASS cannot authorize
  evaluation.
- Stability checks must include translation, orientation/tilt, linear
  velocity, angular velocity, support contact, and forbidden contacts.
  Position drift alone is never evidence of stability: an object can tip in
  place while moving less than one centimetre.
- Use object-semantic pose thresholds. Bowls, plates, ramekins, cups, and
  similar receptacles that are intended to rest flat must remain upright
  after the formal wait; use a maximum tilt of `1.0 deg` unless a stricter
  task-specific threshold is preregistered. Do not use a permissive generic
  rigid-object tilt threshold for these objects.
- Verify that the pose remains within all preregistered limits throughout the
  stabilization window, not merely at one instant. A state that continues
  rotating, rocking, falling, or changing support is invalid even if its
  final translation is small.
- Apply the same formal reset and stabilization protocol to EB, ER, and EC.
  If an intervention state is pre-settled during construction, it must still
  pass the full formal wait without further tipping or pose drift. Never
  assume that generator settling and evaluator settling are interchangeable.
- Record per-episode, per-condition pre-wait and post-wait pose/stability
  measurements in the scene manifest. Aggregate-only PASS labels are
  insufficient.
- Render paired EB/ER/EC initialization images from the exact first policy
  observation and save short smoke-test videos. Human review is a blocking
  gate: record an explicit approval before submitting any formal job.
- Formal submission scripts must fail closed unless the native-only preflight,
  every post-wait physical-state check, policy-view visibility checks, short
  dynamic smoke tests, and the explicit human-review verdict all pass.
- If any evaluated object is physically implausible at the first policy frame,
  stop immediately. Mark the affected serialized states, formal jobs,
  metrics, videos, tables, reports, and HTML entries invalid; do not interpret
  or publish them as evidence. Regenerate the states and rerun all affected
  evaluation after the gate is fixed.

### Regression lesson: in-place tipping

- L1-A4 exposed a prohibited failure mode: a bowl passed a permissive
  pre-wait tilt check and a post-wait translation-drift check, then reached
  the first policy frame visibly tilted. Any validator that checks tilt only
  before waiting, or checks only translation after waiting, is incomplete and
  must not be used to authorize formal evaluation.

## Local review video storage

- Save every video produced or downloaded for local human review under a
  task-specific repository-root directory named `review/<task_name>_task/`
  (for example, `review/L3-A1_V2_task/`). Do not leave the only local copy in
  a temporary worktree, run ledger, cache, or experiment-log directory.
- Use descriptive filenames containing the scene and result category so the
  videos can be identified without opening them.
- For every formal scene, save no more than 10 videos for each result/outcome
  category.
