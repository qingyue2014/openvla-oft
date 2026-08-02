# L3-B — Predicate-order rollback and repair

## Claim

This native-only probe asks whether a policy can recognize that an apparently
completed subgoal is unusable because it was completed in the wrong order,
temporarily undo it, complete its prerequisite, and then redo it. It is an
embodied non-monotonic planning test. It is **not a memory-of-past-actions
test**: the policy observes the current Markov state, not an action-history
token stream.

The primary diagnostic is an event trace in Er, not an Er–Ec success-rate gap:

- drawer `Close`: `True → False → True`;
- bowl `In`: `False → True` between the opening and reclosing events;
- the unmodified native goal succeeds after this ordered repair.

## Locked native task

- Suite: `libero_10`
- Task ID: `3`
- BDDL: `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl`
- Prompt, verbatim: `put the black bowl in the bottom drawer of the cabinet and close it`
- Goal: `(And (Close white_cabinet_1_bottom_region) (In akita_black_bowl_1 white_cabinet_1_bottom_region))`
- Fixed official-state pool: indices `0..19`

No BDDL, prompt, asset, object inventory, texture, XML, or task semantics are
changed. Only serialized state of an already-present native body or joint is
intervened on.

## Conditions

| Scene | Initial predicates | Serialized intervention | Required behavior |
|---|---|---|---|
| Eb / native | `Close=False`, `In=False` | none; bit-exact official state | insert bowl, then close |
| Er / premature close | `Close=True`, `In=False` | bottom drawer scalar joint only | reopen, insert, reclose |
| Ec / prerequisite done | `Close=False`, `In=True` | black-bowl free joint only | close |
| Safe | starts from exact Er | none after restore | scripted real 7-D OSC open, insert, close reference |

Er and Ec are intentionally different current progress states. Their raw
success rates are therefore **workload-asymmetric**: Er has more remaining
motor work than Ec. Consequently, raw `SR(Ec)-SR(Er)` is descriptive only.
The primary L3-B measure is the fraction of Er episodes that exhibit the full
ordered rollback-and-repair trace. EB and Ec report native and close-only
capability separately.

## Why this is not L1-B, L2, or L3-A

- It is not L1-B swept-volume awareness: no protected obstacle, collision
  oracle, or collision-derived label is used.
- It is not merely L2 local state recognition: seeing a closed drawer is
  insufficient; Er requires deliberately reducing predicate progress before
  later restoring it.
- It is not L3-A downstream consequence prediction: no separate dependent
  object is harmed by an otherwise goal-directed action. The diagnostic is
  subgoal rollback caused by order constraints.

## Physical and visual gates

- Exact evaluator reset, fixture replay, `set_init_state`, forward/update,
  ten controller no-ops, refreshed observations, and first policy frame.
- Per-episode pre-wait, first-policy, full-window, and 100-step post-wait hold
  records for Eb/Er/Ec.
- Black bowl remains upright within `1.0°` throughout every gate.
- Translation, linear/angular speed, drawer-joint drift, support contact, and
  forbidden robot/object contacts are checked independently.
- Official native objects are allowed their known ten-step spawn drop from
  `z=0.97 m`; strict post-wait hold thresholds reject continued motion.
- Agent and wrist views are saved in raw 256 and π0.5 224 preprocessing. The
  intended Eb→Er and Eb→Ec changes must be visible in agent-view policy RGB.
- No custom-asset XML audit is applicable because the inventory is native.

## Metrics and outcome taxonomy

Primary:

`Er full_ordered_repair_rate = count(full ordered trace AND stable native task success) / N`

Stage metrics:

1. rollback recognition (`Close True→False`);
2. insertion after rollback (`In False→True`);
3. reclose after insertion (`Close False→True`);
4. stable native task success.

Failures are separated into `no_rollback`, `rollback_without_insertion`,
`insertion_without_reclose`, and `ordered_trace_without_native_success`.

## Formal authorization

Formal submission is fail-closed until all native-only, pairing, exact-runtime,
physical, policy-view, and short dynamic smoke gates pass and an explicit human
approval JSON hash-binds the reviewed videos and smoke report. At most ten
local videos are retained per result category under
`review/L3-B_bowl_order_task/`.
