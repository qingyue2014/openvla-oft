# L3-B3 — Closed-microwave precondition before mug insertion

Status: **v5 scene states and all derived jobs are invalid: both mugs fell
roughly seven centimetres during the formal evaluator wait. V6 keeps the moved
target geometry but serializes native-physics-supported mug heights before the
wait and applies strict throughout-window stability gates. V6 state generation,
human review, smoke, and formal evaluation remain pending and unauthorized.**

## 1. Claim and unchanged native task

L3-B3 tests whether a policy recognizes that a closed microwave must be
opened before placing the target mug inside. The selected task is the only
matching task in the four permitted standard suites:

- suite/task: `libero_10`, task id `9`;
- BDDL: `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl`;
- exact prompt: `put the yellow and white mug in the microwave and close it`;
- exact goal: `(And (In white_yellow_mug_1 microwave_1_heating_region) (Close microwave_1))`;
- native fixtures: `kitchen_table:kitchen_table`, `microwave_1:microwave`;
- native objects: `porcelain_mug_1:porcelain_mug`,
  `white_yellow_mug_1:white_yellow_mug`.

The prompt and goal are byte-for-byte native. A valid Er completion must show
`Close=True -> False`, then `In=False -> True`, then `Close=False -> True`.
The sequence trace is diagnostic; LIBERO's native goal remains the task-success
definition.

## 2. V6 paired conditions

| Arm | Initial state | Required behavior |
|---|---|---|
| Eb / `native` | native open microwave, target outside | insert and close |
| Er / `closed_microwave` | same scene with hinge closed | open, insert, reclose |
| Ec / `open_control` | same scene with hinge fully open | insert and close |
| Safe | exact Er replay | robot-only open, insert, reclose |

All arms use the same native BDDL, prompt, goal, robot, camera, fixtures,
objects, registered classes, and unmodified LIBERO asset files. V6 applies one
common native-state construction to all three arms: the target free-joint
world `x/y` is `[-0.30, -0.15]` m, and native MuJoCo physics determines the
already-supported free-joint `z` for both mugs before serialization. After
that common change, only the preregistered microwave hinge qpos/qvel may differ
across Eb, Er, and Ec.

The replacement design is bound by
`l3b3_microwave_v6_design_prereg.json`. It requires both mugs to have table
support throughout the ten-step formal wait, with at most `3 mm` translation,
`0.015 m/s` linear speed, `0.05 rad/s` angular speed, and `1.0 deg` tilt.
Passing v6 results have not yet been generated.

## 3. Failed predecessors

- V1 retained the official mug position. Opening the door swept the target
  from roughly `[-0.0196, -0.0194]` to `[0.0980, -0.1679]` m and left it at
  `82.657 deg` tilt.
- V2 moved the target to `[0.22, -0.25]` m, outside the robot's usable
  post-opening workspace.
- V3 used `[0.05, -0.15]` m; the controller stopped `47.44 mm` short.
- V4 used `[-0.25, -0.10]` m; no compiled no-contact grasp corridor existed.

Each predecessor has a repository invalidation record. None of its states,
videos, metrics, or gates may be reused as v6 evidence. V5 is additionally
invalidated by `l3b3_microwave_v5_state_invalidation.json`: job `504111`
revealed unsupported pre-wait mugs and therefore its 0/3 π0.5 result is not
interpretable or publishable.

## 4. V5 Safe-controller outcome

Revision 3 uses a vertical native-handle grasp at target-local
`[0.035, 0.0, 0.072]` m. Exact-Er calibration passes:

- closure-final target tilt: `0.285 deg`;
- post-lift target tilt: `0.934 deg`;
- post-lift follow error: `0.906 mm`;
- two-finger target contact retained;
- no microwave or distractor contact during the grasp prefix.

The complete insertion does not pass. After replacing a conservative
mesh/capsule bound with exact compiled convex-mesh geometry, the remaining
failure is physical: the held gripper or mug reaches a native microwave wall
or front-frame geom before any candidate can satisfy native-In, floor support,
positive gripper/mug clearance, and positive closing-door clearance together.

Calibration-only branches were retained, not promoted to Safe evidence:

- 65-degree held yaw stays upright but does not clear the entrance;
- slow 90-degree held yaw stays upright but worsens the right-wall conflict;
- heating-frame lateral offsets clear one wall but leave palm/front-frame
  contact, and large offsets eventually collide with the opposite wall;
- centered rim grasps exceed `1.0 deg` or fail to retain contact;
- empty-gripper pre-yaw grasps either have an invalid descent or exceed
  `1.0 deg` at closure;
- higher or farther-out handle grasps either exceed `1.0 deg` or still touch
  the front frame.

The hash-bound controller outcome is
`l3b3_microwave_v5_handle_grasp_controller_v3_invalidation.json`. Its scope is
the legacy controller and derived Safe evidence. Independently, the v5 state
invalidation now invalidates the serialized Eb/Er/Ec states and every derived
job, metric, video, table, report, and HTML entry.

## 5. Mandatory gates and authorization state

Formal submission remains fail-closed until a new controller revision passes:

1. exact Er reset and formal stabilization;
2. robot-only open, grasp, transport, release, retreat, and reclose;
3. positive compiled continuous sweeps and live forbidden-contact checks;
4. native task success and full post-action stability;
5. exact OpenVLA policy-view certification and smoke runs;
6. explicit hash-bound human review under `review/L3-B3_task/`.

No formal OpenVLA-OFT job has been authorized. Therefore the mandatory
post-formal pi0.5/Cosmos cascade has not started.

## 6. Entry points

```bash
bash experiments/robot/libero/tasks/run_l3b3_microwave_precondition.sh design
bash experiments/robot/libero/tasks/run_l3b3_microwave_precondition.sh check
bash experiments/robot/libero/tasks/run_l3b3_microwave_precondition.sh policy_views
bash experiments/robot/libero/tasks/run_l3b3_microwave_precondition.sh safe_reference
```

`design` validates the v6 preregistration. `prepare` regenerates v6 states and
must pass the strict support/stability gates before `check` can pass.
`safe_reference` verifies the legacy controller-revision-3 invalidation hash
and exits non-zero. `smoke` and `formal` cannot proceed without a future,
separately preregistered Safe PASS.

L3-B3 shares the native source task with L3-B2. They are not independent scenes
and must not be pooled or counted twice in coverage or headline totals.
