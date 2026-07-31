# L3-B moka sequence v3

Status: **Eb/Er/Ec preparation, the Safe real-action controller, and smoke
execution are implemented fail-closed. Formal evaluation remains blocked on
the complete gates and hash-bound human review.**

## Cognitive variable

This scene tests **future-reachability-aware subgoal ordering**: after one
valid subgoal has been completed, can the policy finish the remaining subgoal
without needlessly disturbing prior progress?

It is not a collision task and uses `safety_oracle=none`. The evidence is the
history-sensitive difference in native task completion and repair behavior
when the same already-completed object occupies two different parts of the
shared native goal region.

## Native-only task lock

- Suite/task: `libero_10`, task 8.
- Prompt, verbatim: `put both moka pots on the stove`.
- BDDL:
  `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`.
- Movable inventory: native `moka_pot_1` and `moka_pot_2` only.
- Fixture inventory: native `kitchen_table` and `flat_stove_1` only.
- No custom asset, custom BDDL, prompt change, slot predicate, or collision
  oracle is permitted.

## Four scenes

| Scene | Internal condition | Initial state | Required policy work |
| --- | --- | --- | --- |
| Eb | `native` | Bit-exact official native state | Place both native moka pots |
| Er | `near_first` | Only pot1's native free-joint pose/velocity is changed; pot1 is stable in the near part of the native cook region | Place pot2 while preserving or safely repairing prior progress |
| Ec | `far_first` | Only the same pot1 free-joint pose/velocity is changed; pot1 is stable in the far part of the same cook region | Place the same pot2 |
| Safe | real-action reference from Er | Exact Er reset and evaluator wait | A scripted 7-D OSC rollout places pot2 and reaches the unchanged native goal |

Near and far are state-construction coordinates separated diagonally by
0.105 m. Both centers remain inside the native stove cook site. The separation
was selected because the earlier 0.08 m calibration produced native
moka-to-moka contact for the successful handle orientation. These coordinates
are not task predicates and are never exposed in the prompt.

The Er/Ec pairing validator requires both states to:

1. share the exact same official base state and fixed-fixture replay;
2. modify the exact same pot1 qpos/qvel slices and nothing else;
3. leave pot2 as the remaining object;
4. bind pot1 to distinct near/far coordinates from the same native cook site;
5. pass the complete evaluator reset, 10-step wait, first-policy observation,
   and 100-step stability hold; and
6. keep both moka pots upright within 1.0 degree with correct native support
   and no initial pot-to-pot contact.

## Capability-conditioned state pool

The matched v3 run uses official state indices `3, 5, 7, 17, 18`. They are
all and only the `success && terminal_stable` episodes from the previously
frozen 20-state native screen. The deterministic selection is locked in
`l3b_moka_v3_design_prereg.json`, including the source preregistration,
capability-report, and failed-v2 Ec-report hashes. It was not selected from
any Er/Ec outcome.

This makes the v3 claim explicitly conditional: it asks about subgoal ordering
where π0.5 already demonstrated stable native task competence. It does not
estimate unconditional native success. Every HDF5 episode retains its original
official state index, and generation refuses a different count, order, or
substitution.

The v2 matched design preplaced pot2 and left pot1. Its preregistered Ec run
failed at 0/5 strict stable successes: trajectory/video review found three
pot1 grasp topples, one missed pot1 grasp, and one raw completion that tipped
the preplaced pot2. Er was never run. Before any v3 rollout, v3 therefore
locked a role swap—preplace pot1 and leave the identical native pot2—without
changing slots, thresholds, task, prompt, or inventory.

## Capability and interpretation

Eb is an official-layout baseline and is reported descriptively. Its
whole-task success rate does not reject the paired experiment because Eb
requires two placements, whereas Er and Ec require the same single pot2
placement.

Ec is the matched capability control. Smoke execution therefore runs:

1. Eb for descriptive context;
2. Ec and its preregistered stable-success threshold;
3. Er only if Ec passes; and
4. the primary `Ec - Er` stable-success contrast plus the `Er - Ec` repair
   contrast.

If Ec fails, the result is
`INCONCLUSIVE_EC_CAPABILITY_CONTROL_FAIL`. A positive L3-B candidate requires
Ec to exceed Er by at least 0.4 in stable success rate or Er to exceed Ec by
at least 0.4 in repair-then-complete rate. These smoke thresholds do not
authorize formal reporting.

Safe is an environment-solvability control, not a fifth formal condition. It
must use only `env.step` actions after the exact Er state is restored. Its
grasp is a compact set of pose keyframes distilled from a successful native
π0.5 task-8 trajectory and bound in the report to that source trajectory's
SHA-256. The controller approaches and sweeps the handle while open, closes
during the lift, transports above the preplaced pot, releases, withdraws
laterally from the handle, and then retreats.

Safe passes only when the native goal is true, no pot-to-pot contact occurs,
the final object has no robot contact, the preplaced pot moves at most 0.01 m,
the placed pot finishes within 0.04 m XY of the preregistered far coordinate,
both pots retain stove support, and both remain within the 1.0-degree and
0.003 m terminal-window limits throughout a 100-step settle window.
`safe_reference` runs this gate for every state in the selected pool
and writes a hash-bound batch manifest; the default five-state run therefore
stores five Safe-success videos, below the per-category limit of ten.

## Commands

```bash
bash experiments/robot/libero/tasks/run_l3b_moka_order.sh prepare
bash experiments/robot/libero/tasks/run_l3b_moka_order.sh safe_reference
bash experiments/robot/libero/tasks/run_l3b_moka_order_pi05.sh smoke
```

All first-policy images, Safe trajectory/video, smoke videos, and gate
manifests are stored under `review/L3-B_moka_order_task/`. Formal mode remains
fail-closed until the physical and visibility gates, Safe, Ec capability
control, smoke, and an explicit hash-bound human review all pass.

## Frozen v1 evidence

The completed `l3b_moka_native20_v1` result (raw 11/20, stable 5/20) remains a
valid report about whole-task native performance. It used the older
different-pot near/far pairing and cannot authorize or reject this v3 matched
single-placement contrast. The v1 source is frozen at commit `6883655`; the
current v1 runner refuses to overwrite its artifacts. The failed v2 Ec control
is frozen at commit `313e5de` and report SHA-256
`5147bb5a88efcce34a4284676a2686024c9dee10a5f099eec526ea6a9e341a1f`.
