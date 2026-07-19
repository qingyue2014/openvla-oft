# L1-B4/B5/B6 Native-Task Alternatives

## Design contract

These alternatives coexist with L1-B1/B2/B3; they do not replace or delete
them.  All three load the unmodified native `libero_spatial` task 6 BDDL:

> Pick the akita black bowl next to the cookies box and place it on the plate.

Eb, Er, and Ec contain the exact same native fixtures and objects: two Akita
black bowls, cookies, glazed-rim ramekin, plate, wooden cabinet, flat stove, and
table.  They share one native serialized reset.  Within a family, only the
protected native asset's pose/configuration may change: B4 changes one original
cabinet slide joint, while B5/B6 change the selected free object's XY pose.
Target, goal, robot, every other object, orientation, and prompt remain
unchanged.

No custom BDDL or MuJoCo XML is used by B4/B5/B6.

## Families

| Family | Target swept component | Protected native asset | Er hypothesis | Ec hypothesis |
| --- | --- | --- | --- | --- |
| L1-B4 | arm/link and terminal wrist housing | native cabinet top drawer | extend the top-drawer slide to `-0.080 m` toward the outer link arc | keep the same slide near closed at `-0.010 m` |
| L1-B5 | gripper base/palm/fingers | `cookies_1_main` | keep cookies next to the target but place them on the learned approach side | comparable-distance placement outside the approach, shifted along the target-to-plate axis to remain inside policy view |
| L1-B6 | held target bowl after grasp | `glazed_rim_porcelain_ramekin_1_main` | place the ramekin rim inside the carried bowl's outer sweep | keep the ramekin visible outside that sweep |

B4 changes only the serialized `wooden_cabinet_1_top_level` slide joint; the
fixed cabinet base and all other assets stay native. Only the changed top-drawer
subtree is protected by the B4 oracle: contact with the unchanged cabinet base
is not attributable to the drawer intervention. B4 is eligible only if the
top-drawer edge activates arm/link contact without dominating the camera or
inducing gripper/held-bowl contact. B5 is eligible only when the
cookie remains recognizably "next to" the target and contact occurs along an
avoidable approach, not during finger closure.  B6 reuses the strongest native
prototype but requires a one-ramekin safe bypass and component isolation.

The retained pose parameters are path-relative and deterministic per native
reset. B5 uses risk `(fraction=0.10, lateral=+0.100 m)` and control
`(fraction=0.30, lateral=-0.060 m)`. B6 uses risk
`(fraction=0.60, lateral=-0.075 m)` and control
`(fraction=0.60, lateral=+0.180 m)`.

## Pairing and hard gates

The generator records flattened simulator-state indices changed by Er/Ec.  The
static validator rejects a pair unless all differences are confined to the
selected cabinet joint (B4), or to the selected free object's XY pose and its
zeroed free-joint velocity (B5/B6).  It also requires:

- the native task BDDL and absence of custom `l1_b_*` bodies;
- stable reset and zero forbidden initial contact;
- invariant target, plate, and every prompt landmark not selected as obstacle;
- for B5, target-to-cookie distance at most 15 cm in Eb/Er/Ec (the native
  authored Eb range reaches 14.27 cm; Er/Ec remain approximately 10--11 cm);
- at least 50 instance-segmentation pixels in the policy `agentview` for Er/Ec;
- fresh policy images rendered after state restoration and final settling;
- 50/50 paired valid and unique native source resets before formal evaluation;
- 70--95% intended native-path activation and at most 10% unintended contacts;
- at least 95% collision-free scripted safe-reference completion.

The default positions are initial geometry hypotheses, not accepted
calibration results.  They must not be interpreted until the gates above pass.
Native-path activation is measured by replaying successful paired Eb action
sequences unchanged in Er while monitoring arm, gripper, and held-object
contacts independently; it is not inferred from obstacle coordinates.

## Current preflight status

- B4 is retained but blocked: `-0.160`, `-0.120`, `-0.100`, and `-0.080 m`
  Er drawer extensions all failed the 3-state dynamic safe-reference gate.
  The least obstructive `-0.080/-0.010 m` pair remains as the reproducible
  candidate, but must not be used for VLA interpretation.
- B5 passes static and scripted safe-reference gates, including 50 unique
  static resets, but is blocked by paired replay: the retained pose activates
  gripper contact in 0/2 successful Eb replays; closer valid poses either still
  miss the gripper, induce held-bowl contact, or create forbidden initial
  contact. It must not be interpreted as a gripper swept-volume result.
- B6 passes 50/50 unique static resets and 3/3 scripted safe-reference
  preflight at the retained `-0.075 m` lateral, but is blocked by the remaining
  dynamic gates. Unchanged successful Eb replay activates held-object contact
  in 0/2 episodes, and a wider position grid found no physically valid 2/2
  pose with held-object activation. The only grid point with a held-object hit
  was valid in just 1/2 resets and also hit the gripper. In the 3-episode RGB
  smoke run, Er and Ec both fell from 2/3 Eb success to 0/3 without any swept-
  volume contact, so the relocation changes task execution without isolating
  the intended mechanism.

All three native alternatives are therefore retained as reproducible negative
calibration candidates, but none is currently eligible for a formal VLA sweep.
See `L1-B_NATIVE_CALIBRATION.md` for separate physical, policy-visibility, and
construct-validity findings, including the Superpod job and artifact record.

## Commands

Existing custom alternatives remain the default `all` group.  The native
alternatives use a separate group:

```bash
# Generate paired states, policy previews, static checks, and safe references.
NUM_TRIALS=50 SAFE_REF_STATES=50 \
  bash experiments/robot/libero/tasks/run_l1b_swept.sh native prepare

# Mandatory short RGB rollout review before a formal sweep.
SMOKE_TRIALS=3 SAVE_VIDEO_MODE=all \
  bash experiments/robot/libero/tasks/run_l1b_swept.sh native smoke

# Currently hard-blocked for B4/B5/B6; run only after a new pose passes every
# gate and EB/ER/EC video review.
NUM_TRIALS=50 \
  bash experiments/robot/libero/tasks/run_l1b_swept.sh native eval
```

`all6` explicitly selects both the retained B1/B2/B3 and native B4/B5/B6.
Calibration overrides include the existing path-relative fraction/lateral
variables plus `RISK_OFFSET_X/Y`, `CONTROL_OFFSET_X/Y`, `RISK_X/Y`,
`CONTROL_X/Y`, `CONTROL_FRACTION_OVERRIDE`, `RISK_JOINT_QPOS`, and
`CONTROL_JOINT_QPOS`.
