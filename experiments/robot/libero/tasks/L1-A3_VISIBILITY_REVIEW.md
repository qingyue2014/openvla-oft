# L1-A3 Human Policy-View Visibility Review

- Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**
- Reviewed date: `2026-07-28`
- Remote job: `492729`
- Evaluated commit: `309e7a5a779240a5469b60f981c25ccc9e9f60ee`
- Exact evidence: task-14 paired serialized-state previews for episodes `0`,
  `1`, and `2`, conditions `Eb`, `Er`, and `Ec`, at `256 x 256`.
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

Automated task-14 evidence in `l1a3_pairing.json` agrees with inspection: each
bowl has at least 80 agentview segmentation pixels and pairwise mask-centroid
separation is at least 18 px. No visibility threshold was relaxed. The earlier
task-15 images and review are not used as evidence for this verdict.
