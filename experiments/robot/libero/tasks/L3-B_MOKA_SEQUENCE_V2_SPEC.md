# L3-B moka sequence v2

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
| Er | `near_first` | Only pot2's native free-joint pose/velocity is changed; pot2 is stable in the near part of the native cook region | Place pot1 while preserving or safely repairing prior progress |
| Ec | `far_first` | Only the same pot2 free-joint pose/velocity is changed; pot2 is stable in the far part of the same cook region | Place the same pot1 |
| Safe | real-action reference from Er | Exact Er reset and evaluator wait | A scripted 7-D OSC rollout places pot1 and reaches the unchanged native goal |

Near and far are state-construction coordinates separated diagonally by
0.105 m. Both centers remain inside the native stove cook site. The separation
was selected because the earlier 0.08 m calibration produced native
moka-to-moka contact for the successful handle orientation. These coordinates
are not task predicates and are never exposed in the prompt.

The Er/Ec pairing validator requires both states to:

1. share the exact same official base state and fixed-fixture replay;
2. modify the exact same pot2 qpos/qvel slices and nothing else;
3. leave pot1 as the remaining object;
4. bind pot2 to distinct near/far coordinates from the same native cook site;
5. pass the complete evaluator reset, 10-step wait, first-policy observation,
   and 100-step stability hold; and
6. keep both moka pots upright within 1.0 degree with correct native support
   and no initial pot-to-pot contact.

## Capability and interpretation

Eb is an official-layout baseline and is reported descriptively. Its
whole-task success rate does not reject the paired experiment because Eb
requires two placements, whereas Er and Ec require the same single pot1
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
0.003 m terminal-window limits for at least 30 samples of a 100-step settle
window.

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
different-pot near/far pairing and cannot authorize or reject this v2 matched
single-placement contrast. The v1 source is frozen at commit `6883655`; the
current v1 runner refuses to overwrite its artifacts.
