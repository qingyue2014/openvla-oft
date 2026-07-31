# L3-B moka sequence v7 result

Status: **the conditional paired smoke diagnostic passed and found a
preregistered candidate order effect. Formal promotion remains blocked on
explicit human video review.**

## Locked identity

- Source commit:
  `2aad83f7a6120738fd3cc364d7f319e7eff11802`.
- Slurm job: `500127` (`COMPLETED`, exit `0`).
- Exact native task: `libero_10`, task 8.
- Exact prompt: `put both moka pots on the stove`.
- v7 preregistration SHA-256:
  `9ec88a31e3f61d2ff26d67c9578c42675d9e34637c149a754f3fade09f9ade93`.
- Frozen v6 Ec source report SHA-256:
  `cbbc1f0a81958e149a0e91a9ec6ca27aec6fcd09e7218b3c286e80243c93b099`.
- v7 frozen-Ec binding SHA-256:
  `0c96ee1c746ee5901f0454ddae9a910fd2e6a38207c10d3e128e898f1435a374`.
- Safe batch SHA-256:
  `c8211e5aaa5b1515a6c2e4a86b189ed94dcb976b435047351a74a01b9628fa55`.
- Paired smoke report SHA-256:
  `b7acf8a7c130734b6b61b2faf56cbeb6f51f9983162f8e54ff9e0d0d6b343425`.

The official state indices were
`[0, 5, 7, 8, 10, 11, 12, 15, 16, 17]`. They are all and only the strict
terminal-stable Ec successes in the frozen v6 screen. No new Ec rollout was
run, no Er result participated in selection, and the failed v6 result was not
reclassified.

## Mandatory gates

- Tests: `19 passed`, `3 skipped` before artifact generation.
- Initial physical gates: `30/30` Eb/Er/Ec episodes passed.
- Exact serialized pairing: passed.
- Saved-state evaluator replay and actual first-policy RGB comparison: passed.
- Native BDDL, verbatim prompt, and compiled asset inventory preflight:
  passed separately for Eb, Er, and Ec.
- Remote Safe controller v4: `10/10`.
- Safe maximum preplaced-pot displacement:
  `2.53e-15 m`.
- Safe maximum target XY error: `0.00565 m`.
- Safe minimum grasp lift: `0.11240 m`.

Formal authorization remains false because human review has not yet been
recorded.

## Paired outcome

| Measure | Ec, frozen capability evidence | Er, first run |
| --- | ---: | ---: |
| Strict terminal-stable success | 10/10 | 0/10 |
| Raw native-goal success | 10/10 | 4/10 |
| Target pot activated | 10/10 | 10/10 |
| Direct strict completion | 10/10 | 0/10 |
| Strict repair-then-complete | 0/10 | 0/10 |
| Repair attempt, incomplete | 0/10 | 6/10 |
| Raw completion, terminal-unstable | 0/10 | 4/10 |

The preregistered `Ec - Er` strict-stable success-rate contrast is `1.0`,
above the `0.4` candidate threshold. The paired diagnostic verdict is
`PASS_L3B_MOKA_POLICY_SMOKE_DIAGNOSTIC`, with candidate status
`POTENTIAL_ORDER_EFFECT_REQUIRES_FORMAL_REVIEW`.

Every Er episode moved pot2 by at least the locked target-activation threshold
and also displaced the preplaced pot1 beyond the `0.015 m` direct-completion
threshold. Six episodes attempted repair without completing the native goal.
The other four triggered the native success predicate but failed the strict
terminal gate because at least one moka pot exceeded 1.0 degree; three of
those raw successes left a pot severely tilted or side-resting. The result is
therefore not explained by total inaction, and the strict gate prevents
physically implausible native-predicate successes from inflating Er.

## Interpretation limit

This is evidence for a **conditional order/future-reachability diagnostic**:
among states where π0.5 already demonstrated a stable single-pot Ec
completion, placing prior progress at its default landing side induced repair
behavior and eliminated strict Er completion.

It is not an unconditional LIBERO task success estimate. Geometry is part of
the intervention, so the defensible cognitive interpretation is that the
policy failed to plan or execute the reordering needed to preserve or repair
prior progress in a still-solvable native scene. Safe proves reachability; it
does not by itself distinguish high-level planning failure from the policy's
low-level difficulty executing the required repair. Human video review is the
next blocking gate.
