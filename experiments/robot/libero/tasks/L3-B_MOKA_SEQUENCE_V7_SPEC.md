# L3-B moka sequence v7

Status: **implementation and preregistration are locked before any Er policy
rollout. Formal evaluation remains blocked on all gates and explicit human
review.**

## Cognitive variable

This scene tests **future-reachability-aware subgoal ordering**. Given that
π0.5 has already shown it can place moka pot2 stably in a matched control
state, does putting valid prior progress at the policy's default landing side
make it preserve, repair, or fail the remaining subgoal?

This is not a collision task. It uses `safety_oracle=none`; the oracle is the
unchanged native task success plus full-window physical stability and measured
repair of the already-completed pot.

## Native-only task lock

- Suite/task: `libero_10`, task 8.
- Prompt, verbatim: `put both moka pots on the stove`.
- BDDL:
  `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`.
- Movable inventory: native `moka_pot_1` and `moka_pot_2` only.
- Fixture inventory: native `kitchen_table` and `flat_stove_1` only.
- No custom asset, BDDL, prompt, predicate, or collision oracle is permitted.

Eb is the bit-exact native state. Er preplaces pot1 at the native cook site's
`(+x,+y)` diagonal side, aligned with π0.5's default pot2 landing direction.
Ec preplaces the same pot1 at the opposite `(-x,-y)` side. Both conditions
leave the same native pot2 and preserve the prompt, asset inventory, fixture
poses, and all other serialized state. The slot separation remains 0.145 m,
with at least 0.0237 m coordinate margin inside the native cook site.

Each generated Eb/Er/Ec state must pass the evaluator's exact reset,
restoration, 10-step no-op wait, observation refresh, first-policy RGB
preprocessing, and 100-step stability hold. Both moka pots must remain within
1.0 degree, retain the intended native support, and have no initial pot contact.

## Why v7 is separate from the failed v6 experiment

The preregistered v6 Ec control reached the native goal in 18/20 episodes but
only 10/20 were terminal-stable, below the locked 12/20 gate. v6 therefore
failed, and Er was not run. Its result is frozen in
`L3-B_MOKA_SEQUENCE_V6_RESULT.md`; no threshold has been relaxed and v6 is not
reclassified.

v7 asks a narrower, explicitly conditional question. Its state pool is all and
only the v6 Ec episodes that already satisfied both native success and the
complete terminal-stability gate:

`[0, 5, 7, 8, 10, 11, 12, 15, 16, 17]`.

The selection rule, source Ec report hash, source commit, and all ten Ec
trajectory hashes are locked in `l3b_moka_v7_design_prereg.json`. No Er
outcome was observed before this lock. A new Ec rollout is prohibited; the
runner revalidates the exact frozen v6 report and trajectory hashes before it
can start Er.

This conditioning is not an independent v7 estimate of Ec performance. The
valid claim is only:

> Conditional on π0.5 having demonstrated a strict-stable single-pot Ec
> completion in these native states, does the Er placement of prior progress
> cause an order-dependent completion or repair deficit?

It must not be reported as an unconditional LIBERO success rate.

## Contrasts and gates

Er runs on all and only the ten selected official state indices. The paired
summarizer rejects any mismatch between Er's runtime-recorded official indices
and the frozen Ec indices.

- Candidate order effect: `Ec - Er` strict-stable success-rate gap at least
  0.4, or `Er - Ec` repair-then-complete rate gap at least 0.4.
- Strict success: native goal true and both pots remain within 1.0 degree and
  0.003 m terminal drift throughout the final 30 recorded settle samples.
- Safe: real OSC actions from every exact Er initial state must complete the
  unchanged native goal without pot-to-pot contact or material displacement of
  the preplaced pot.

Safe controller v4 uses the registered native π0.5 grasp reference beginning
at its vertical handle-entry keyframe. This avoids the open-gripper orientation
sweep that contacted the preplaced pot in one of the expanded v6 states.
Controller v4 passed all 20 v6 states; v7 nevertheless reruns Safe on its exact
ten-state Er bundle and records every trajectory and report. At most ten Safe
success videos are saved.

## Commands

```bash
bash experiments/robot/libero/tasks/run_l3b_moka_order.sh prepare
bash experiments/robot/libero/tasks/run_l3b_moka_order.sh safe_reference
FROZEN_EC_SOURCE_REPORT=/exact/v6/report.json \
EC_TRAJECTORY_DIR=/exact/frozen/v6/ec/trajectories \
  bash experiments/robot/libero/tasks/run_l3b_moka_order_pi05.sh er_smoke
```

`ec_capability` and combined `smoke` modes fail closed in v7 because a new Ec
rollout would no longer be the preregistered frozen control evidence.

All first-policy images, Safe trajectories/videos, policy videos, per-episode
metrics, and binding manifests are stored under the task-specific
`review/L3-B_moka_order_task/` tree. Formal mode remains blocked until every
native-only, physical, visibility, dynamic, Safe, hash-binding, paired-smoke,
and explicit human-review gate passes.
