# L1-A4 Spatial Policy-View Visibility Review

Verdict: **PENDING_REVIEW_NATIVE_RELATION_PAIRING**

## Audited experiment

- The prior PASS applied only to the retired fixed-layout candidate.
- The translated-native-relation paired states introduced after the
  fixed-layout EC capability failure require fresh exact-state preview
  generation and human review before any model evaluation.
- Native suite/task: `libero_spatial`, task `0`
- Native BDDL:
  `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl`
- Exact benchmark prompt:
  `pick up the black bowl between the plate and the ramekin and place it on the plate`
No translated-native-relation rollout is authorized until this file is
replaced with a commit-bound passing human policy-view review.
