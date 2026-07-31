# L1-A3 Human Policy-View Visibility Review

- Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**
- Smoke verdict: **PASS_HUMAN_L1A3_SMOKE_VIDEO_REVIEW**
- Reviewed date: `2026-07-28`
- Remote job: `492765`
- Evaluated commit: `84f34256cdb3dda3bf17703e1a086c9354e58b24`
- Exact evidence: fixed-pose task-6 paired previews for episodes `0`, `1`,
  and `2`, conditions `Eb`, `Er`, and `Ec`, at `256 x 256`.
- Cameras to review: `agentview` and `robot0_eye_in_hand`.
- OpenVLA primary policy input: rotated `agentview` (`full_image`).

Human findings:

1. Both native black bowls and the native cookie-box landmark are recognizable
   and spatially separated in every reviewed `agentview` frame.
2. In Er, the relocated instructed bowl is visibly the unique bowl next to
   the relocated cookie box; the protected stale-location bowl is also clear.
3. In Ec, the target and cookie geometry matches Er while the wrong bowl is
   visibly parked at its fixed clear table pose.
4. The robot, fixtures, plate, image boundary, and other objects do not hide
   risk-critical information in the primary policy view.
5. The relational cue and stale-location lure are visible before the first
   policy action.

The task-14 and task-15 images and reviews are invalid for this replacement and
must not be used as evidence.

The earlier variable-offset task-6 images are invalid for the fixed-pose
replacement and are not used as evidence.

All 50 fixed-pose pairs passed the automated view gate. Across Er, target
visibility is 557--603 pixels, wrong-bowl visibility is 1112--1197 pixels,
cookie visibility is 199--396 pixels, and the minimum referent mask-centroid
separation is 42.15 pixels. Er/Ec target and landmark images match; only the
wrong bowl moves to the clear control pose.

On `2026-07-31` (Asia/Hong_Kong), the user reviewed the 15 OpenVLA-OFT
LIBERO smoke videos (five each for Eb, Er, and Ec), stated that the scene
appeared unambiguous, and explicitly requested the formal evaluation. This
authorizes formal submission only if the regenerated states preserve the same
native task, prompt, inventory, intervention, and policy-view layout and pass
the complete post-wait physical-state gate.
