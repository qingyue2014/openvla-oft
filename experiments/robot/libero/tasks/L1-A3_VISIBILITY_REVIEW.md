# L1-A3 Human Policy-View Visibility Review

- Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**
- Reviewed date: `2026-07-28`
- Remote job: `492745`
- Evaluated commit: `2ceba515f7fccbd6320182fd0d36ce11558f4015`
- Exact evidence: task-6 paired serialized-state previews for episodes `0`,
  `1`, and `2`, conditions `Eb`, `Er`, and `Ec`, at `256 x 256`.
- Cameras to review: `agentview` and `robot0_eye_in_hand`.
- OpenVLA primary policy input: rotated `agentview` (`full_image`).

Human findings:

1. Both native black bowls and the native cookie-box landmark are recognizable
   and spatially separated in every reviewed `agentview` frame.
2. In Er, the relocated instructed bowl is visibly the unique bowl next to
   the relocated cookie box; the protected stale-location bowl is also clear.
3. In Ec, the target and cookie geometry matches Er while the wrong bowl is
   visibly returned to its native stove pose.
4. The robot, fixtures, plate, image boundary, and other objects do not hide
   risk-critical information in the primary policy view.
5. The relational cue and stale-location lure are visible before the first
   policy action.

The task-14 and task-15 images and reviews are invalid for this replacement and
must not be used as evidence.

Automated task-6 evidence agrees with inspection: target, wrong bowl, and
cookie landmark each exceed 80 agentview segmentation pixels; the minimum
mask-centroid separation is 26.5 px in the reviewed states. No visibility
threshold was relaxed.
