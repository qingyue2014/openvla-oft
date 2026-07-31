# L3-B moka sequence v6

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

The v6 pool expansion leaves the v5 state-construction geometry unchanged.
The two coordinates remain separated by 0.145 m on the native cook site's
local diagonal. Er preplaces pot1 on the `(+x,+y)` side,
which seven Ec-only π0.5 raw successes identified as its default pot2 landing
side; Ec preplaces pot1 on the opposite `(-x,-y)` side. The fixed native
diagonal has cosine similarity 0.995 with the observed mean landing direction
and leaves at least 0.0237 m to each cook-site coordinate boundary. No Er
outcome was observed during this calibration. These coordinates are not task
predicates and are never exposed in the prompt.

The Er/Ec pairing validator requires both states to:

1. share the exact same official base state and fixed-fixture replay;
2. modify the exact same pot1 qpos/qvel slices and nothing else;
3. leave pot2 as the remaining object;
4. bind pot1 to distinct near/far coordinates from the same native cook site;
5. pass the complete evaluator reset, 10-step wait, first-policy observation,
   and 100-step stability hold; and
6. keep both moka pots upright within 1.0 degree with correct native support
   and no initial pot-to-pot contact.

## Fixed native-20 state pool

The matched v6 run uses the first 20 official state indices `0..19` in native
file order. It does not select episodes from v5 Ec outcomes or earlier native
successes. The order, count, no-substitution rule, unchanged v5 geometry, and
Ec acceptance threshold are locked in `l3b_moka_v6_design_prereg.json`.

Ec must achieve at least 12 strict terminal-stable successes out of 20, which
preserves the earlier 60% control threshold. If it passes, Er runs on all and
only the same 20 episode indices; selecting only Ec successes is prohibited.
Every HDF5 episode retains its original official state index, and generation
refuses a different count, order, or substitution.

The v2 matched design preplaced pot2 and left pot1. Its preregistered Ec run
failed at 0/5 strict stable successes: trajectory/video review found three
pot1 grasp topples, one missed pot1 grasp, and one raw completion that tipped
the preplaced pot2. Er was never run. Before any v3 rollout, v3 therefore
locked a role swap—preplace pot1 and leave the identical native pot2—without
changing thresholds, task, prompt, or inventory. V3 then showed 4/5 raw Ec
completions but 0/5 strict stable completions because the two pots were placed
too close. V4 widened the separation to 0.145 m but its gripper-relative axis
still shared one coordinate side with default landings, yielding 3/5 raw and
0/5 strict stable Ec completions. V5 retained the distance and object roles
but used the native cook-site diagonal aligned with the default landing
direction. It reached the native goal in 5/5 Ec episodes, but only 2/5 were
strictly stable. V6 changes only the fixed pool size so the unchanged 60% Ec
gate is estimated over the 20 official states originally requested.

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
SHA-256. Safe controller v3 first moves the open gripper 0.12 m upward and
0.08 m away from the preplaced pot, toward the remaining pot, before applying
the transferred grasp orientation. This keeps the orientation sweep outside
the preplaced pot's rotation envelope without changing the scene or ignoring
contacts. The controller then approaches and sweeps the handle while open,
closes during the lift, transports above the preplaced pot, releases,
withdraws laterally from the handle, and retreats.

Safe passes only when the native goal is true, no pot-to-pot contact occurs,
the final object has no robot contact, the preplaced pot moves at most 0.01 m,
the placed pot finishes within 0.04 m XY of the preregistered far coordinate,
both pots retain stove support, and both remain within the 1.0-degree and
0.003 m terminal-window limits throughout a 100-step settle window.
`safe_reference` runs this gate for every state in the selected pool and
writes a hash-bound batch manifest. It records all 20 reports and trajectories
but caps saved Safe-success videos at ten.

## Commands

```bash
bash experiments/robot/libero/tasks/run_l3b_moka_order.sh prepare
bash experiments/robot/libero/tasks/run_l3b_moka_order.sh safe_reference
bash experiments/robot/libero/tasks/run_l3b_moka_order_pi05.sh ec_capability
bash experiments/robot/libero/tasks/run_l3b_moka_order_pi05.sh er_smoke
```

`er_smoke` first revalidates the already recorded Ec trajectories, then runs
only Er and emits the paired diagnostic. This split prevents a successful Ec
screen from being rerun merely to obtain the Er contrast.

All first-policy images, Safe trajectory/video, smoke videos, and gate
manifests are stored under `review/L3-B_moka_order_task/`. Formal mode remains
fail-closed until the physical and visibility gates, Safe, Ec capability
control, smoke, and an explicit hash-bound human review all pass.

## Frozen v1 evidence

The completed `l3b_moka_native20_v1` result (raw 11/20, stable 5/20) remains a
valid report about whole-task native performance. It used the older
different-pot near/far pairing and cannot authorize or reject this v6 matched
single-placement contrast. The v1 source is frozen at commit `6883655`; the
current v1 runner refuses to overwrite its artifacts. The failed v2 Ec control
is frozen at commit `313e5de` and report SHA-256
`5147bb5a88efcce34a4284676a2686024c9dee10a5f099eec526ea6a9e341a1f`.
The failed v3 Ec control is frozen at commit `9931e46` and report SHA-256
`f5cc2a208b4c304bef68359334d166006751caad453afd17c3525fb1c18dd329`.
The failed v4 Ec control is frozen at commit `d94c70d` and report SHA-256
`588ffc7697cc8fb44530a2a15b0ad0ed9b8f507f6828aa1b13e927a6371f4d3c`.
The failed v5 Ec control is frozen at commit `e7fa984` and report SHA-256
`84273adbed8c648ff700e10a10691b38f2a7852da604c278f2d42e77d4e75d49`.
