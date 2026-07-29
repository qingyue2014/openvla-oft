# L1-A4 Spatial Policy-View Visibility Review

Verdict: **NEEDS_HUMAN_POLICY_VIEW_VISIBILITY_REVIEW**

The PASS below applies only to commit `b0ee2006`. Pair construction now uses
poses sampled from other native initial states, so a new full-pool policy-view
review is required before further model rollout.

## Audited experiment

- Remote run: `20260729T084717Z-l1a4s-check`
- Commit: `b0ee2006baaadf07bdda3d4024e420f590e48496`
- Native suite/task: `libero_spatial`, task `0`
- Native BDDL:
  `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl`
- Exact benchmark prompt:
  `pick up the black bowl between the plate and the ramekin and place it on the plate`
- Paired-state verdict: `PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE`
- Accepted paired states: 45
- Rejected native indices: 31, 34, 40, 45, and 48. In each rejected
  state the native robot configuration displaced the moved target during
  settling, so the strict between-relation gate failed. Rejected states are
  not included in any condition.

## Human review

The exact post-settle serialized states for episodes 0, 1, and 2 were
regenerated through the LIBERO environment wrapper and inspected in both RGB
streams consumed by the policy:

- `agentview`: EB shows both native black bowls, the plate, and the ramekin.
  ER clearly shows the moved target bowl between the moved plate and ramekin,
  while the second bowl remains at the paired EB target location. EC preserves
  the same target/plate/ramekin layout and moves only the second bowl away.
- `robot0_eye_in_hand`: the near-field objects are sharp and unoccluded. Its
  crop does not always contain every landmark at once, but the simultaneously
  supplied `agentview` fully resolves the complete ordinal relation and both
  candidate bowls.
- No audited frame is blank, corrupted, dominated by robot occlusion, or
  dependent on an oracle-only camera.

The first three episodes vary in native EB placement; the ER/EC intervention
remains visually distinguishable and semantically identical across all three.

## Automatic visibility and stability evidence

Across all 45 accepted pairs:

- ER minimum `agentview` pixels:
  target bowl 1090, second bowl 763, plate 1536, ramekin 568.
- EC minimum `agentview` pixels:
  target bowl 1090, second bowl 1490, plate 1536, ramekin 568.
- Minimum referent centroid separation: 35.9095 pixels in both ER and EC
  (gate: 12 pixels).
- Maximum stale-location pairing error: 0.00003781 m.
- Maximum settle drift: 0.00078924 m.
- Maximum policy-wait drift: 0.00000466 m.
- Maximum unallowed ER/EC or EB/ER joint-state difference: 0.

This review authorizes model evaluation only for the validated 45-state pool
at the commit and task identity above. It does not claim task success or safe
adaptation by any model.
