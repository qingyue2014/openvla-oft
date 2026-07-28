# L1-A3 Human Policy-View Visibility Review

- Verdict: **PENDING_NEW_TASK6_POLICY_VIEW_REVIEW**
- Reviewed date: pending
- Remote job: pending
- Evaluated commit: pending
- Exact evidence: task-6 paired serialized-state previews pending.
- Cameras to review: `agentview` and `robot0_eye_in_hand`.
- OpenVLA primary policy input: rotated `agentview` (`full_image`).

Required findings:

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
