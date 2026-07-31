# L3-B moka sequence v4 result

V4 retained the v3 pot1-first role swap and widened the symmetric slot
separation from 0.105 m to 0.145 m on the gripper-relative axis.

Ec raw native-goal completion was 3/5 but strict stable completion remained
0/5. One raw completion left pot2 upright and preplaced pot1 essentially
upright, but pot2 itself was tilted 7.26 degrees in another; the remaining raw
completions tipped pot1 or both pots. Er was therefore not run and no order
effect was estimated.

Across seven raw Ec successes from v3 and v4, π0.5's mean pot2 landing
direction was `[0.6363, 0.7714]`, while the gripper-relative slot axis had the
opposite x sign. This motivated preregistering the fixed native cook-site
`(+x,+y)` diagonal before v5.

Audit bindings:

- source commit: `d94c70d`;
- v4 design preregistration SHA-256:
  `bac40860150d4880adfadfa32986d0a15e37dfccbead471faafece6f12e97e7e`;
- Ec capability report SHA-256:
  `588ffc7697cc8fb44530a2a15b0ad0ed9b8f507f6828aa1b13e927a6371f4d3c`.

No threshold, native task, prompt, asset, or BDDL exception was used.
