# L3-B moka native-20 capability preregistration

Preregistration ID: `l3b_moka_native20_v1`

Status: **locked before the native-20 policy run**

Scope: native capability screening only; this does not authorize `near_first`,
`far_first`, or formal EB/ER/EC evaluation.

## Fixed native task

- Suite/task: `libero_10`, task 8.
- Original prompt, verbatim: `put both moka pots on the stove`.
- Original BDDL:
  `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl`.
- Native objects: `moka_pot_1`, `moka_pot_2`.
- Native fixtures: `kitchen_table`, `flat_stove_1`.
- No custom asset, BDDL, prompt, goal, inventory, or collision oracle is
  permitted.

The official OpenPI `pi05_libero` checkpoint is fixed at
`gs://openpi-assets/checkpoints/pi05_libero`, with seed 42 and replanning every
5 actions.

## Fixed pool

Use exactly official native initialization indices 0 through 19, in that
order, from
`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.pruned_init`. No failed,
invisible, unstable, or otherwise inconvenient state may be replaced.

Indices 0 through 4 were already evaluated in the earlier five-state
capability screen, which produced one stable success. Indices 5 through 19 are
new. Consequently, this run is a declared pool extension and must not be
reported as an independent blinded 20-state replication.

## Fixed physical and capability gates

Every state must pass the exact evaluator reset, serialized-state restoration,
simulator forward, 10 controller no-op steps, refreshed observations, and
first-policy-frame checks. Pre-wait and full-window post-wait measurements must
cover position, tilt, linear and angular velocity, support, forbidden contacts,
and policy-view visibility.

A trial counts as a stable native success only if:

1. the unchanged native LIBERO goal succeeds;
2. at least 30 settle samples are recorded for each moka pot;
3. each pot's maximum tilt over the terminal window is at most 1.0 degree; and
4. each pot's maximum terminal-window translation drift is at most 0.003 m.

Raw LIBERO success is reported separately and cannot substitute for stable
success. The capability gate passes at **12 or more stable successes out of
20** (60%). If it fails, no `near_first` or `far_first` policy run is
authorized.

At most five episode pairs (agent and wrist views) are retained for each
result category, giving no more than ten videos per category. Human review of
the exact first policy images and smoke videos remains blocking for any later
formal promotion.

The machine-readable source of truth is
`l3b_moka_native20_prereg.json`; the dedicated runner refuses an uncommitted or
modified copy and binds its SHA-256 into the capability report.
