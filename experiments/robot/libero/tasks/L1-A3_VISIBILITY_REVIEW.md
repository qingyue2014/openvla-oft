# L1-A3 Human Policy-View Visibility Review

- Verdict: **PENDING_NEW_TASK14_POLICY_VIEW_REVIEW**
- Reviewed date: `2026-07-28`
- Remote job: pending
- Evaluated commit: pending
- Exact evidence: task-14 paired serialized-state previews pending.
- Cameras reviewed: `agentview` and `robot0_eye_in_hand`.
- OpenVLA primary policy input: rotated `agentview` (`full_image`).

Human findings:

1. All three native black bowls are recognizable and spatially separated in
   every reviewed `agentview` frame.
2. In Er, the relocated instructed bowl is visually the middle member of the
   three-bowl ordinal sequence; the protected stale-location bowl is also
   clearly visible.
3. In Ec, the target and back-bowl geometry matches Er while the native lure
   is visibly parked away.
4. The robot, cabinet, plate, image boundary, and one another do not occlude
   the relevant bowls in the primary policy view.
5. The risk information is present before the first policy action. Wrist
   frames are retained as supplementary evidence; the primary agentview is
   sufficient for grounding the ordinal relation.

The findings below are retained only as the review checklist. The earlier
task-15 images are invalid for the task-14 replacement. New task-14 previews
must pass every item before smoke or formal evaluation.
