# L1-A3 Human Policy-View Visibility Review

- Verdict: **PENDING_FIXED_POSE_POLICY_VIEW_REVIEW**
- Reviewed date: `2026-07-28`
- Remote job: pending
- Evaluated commit: pending
- Exact evidence: fixed-pose task-6 paired previews pending.
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

The earlier variable-offset task-6 images are invalid for the fixed-pose
replacement and must not be used as evidence.
