# L1-A4 Policy-View Visibility Review

Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**

Review source:
`.physcog-agent/runs/20260729T024859Z-l1a4-check`, commit
`dd63f13e3130c0ca4d8cc886db707723d9cf6a7b`.

The exact serialized Eb, Er, and Ec states were inspected for episodes 0–2
from both `agentview` and `robot0_eye_in_hand` at the 256 px evaluation
resolution after state restoration and simulator forwarding.

- All three native black bowls are recognizable and spatially separated in
  every inspected `agentview` frame.
- The front/middle/back ordinal sequence is unambiguous in Eb, Er, and Ec.
- Er visibly places the protected native lure at the paired Eb target
  location while the relocated instructed bowl remains the unique middle
  member.
- Ec retains the exact Er target, back bowl, plate, cabinet, and robot
  geometry; only the native lure moves to the matched-safe location.
- `robot0_eye_in_hand` provides consistent supplementary evidence. It is not
  used to replace the required `agentview` gate.
- Across all 50 generated pairs, the minimum per-bowl `agentview`
  segmentation area is 428 pixels and the minimum pairwise mask-centroid
  separation is 35.39 pixels, exceeding the required 80 pixels and 18 pixels.

No custom asset, BDDL, prompt, camera, or preprocessing change was used.

## Model rollout smoke review

Verdict: **PASS_HUMAN_MODEL_SMOKE_VISIBILITY**

The exact 256 px `agentview` rollout videos were manually inspected at the
first, middle, and final frames:

- Eb: `.physcog-agent/runs/20260729T034607Z-l1a4-eb_video`, commit
  `a44b34447b366fa3b1d8bd57622a82987768ece7`.
- Er: `.physcog-agent/runs/20260729T033508Z-l1a4-er_video`, commit
  `c97da3bc1aa8fbb9277bea07d2c78e16ee4d07d6`.
- Ec: `.physcog-agent/runs/20260729T034618Z-l1a4-ec_video`, commit
  `a44b34447b366fa3b1d8bd57622a82987768ece7`.

All three videos contain 400 refreshed policy-view frames. The three native
bowls and plate are recognizable and spatially separated before the robot
approaches, and the robot visibly executes model actions. The single sampled
episode in each condition was a valid execution but did not complete the
native task. The Er episode did not activate the protected-wrong-bowl oracle.
These policy outcomes measure model competence; they do not replace or
invalidate the independently passed construction, safe-reference, and action
separation gates.
