# L1-A4 Spatial Policy-View Visibility Review

Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**

Intervention ID: `l1a4_spatial_native_near_adaptive_v4`

## Audited state pool

- Remote generation: `20260730T063532Z-l1a4s-check`
- Git base: `add758113d63071e5b62e6c0d9bc3c9c1ae7dcd1`
- Pipeline SHA-256:
  `09893328e4de77a244f033790118eb46607b3c936b6822af0dcfe6ddeefa7af3`
- Pairing manifest SHA-256:
  `8f3e98cc9ee891ef28301e48f9cc70ab3f0a412cf562dda53922a4ddde9ccf4a`
- EB HDF5 SHA-256:
  `916db567a76c73dcb940cfd5ff0af8fb4deb268d0b560f0825ad0579aa93de57`
- ER HDF5 SHA-256:
  `0597edfbf580e50f0a733282d1ae913cfbd60d99397477aec1649bb0e59f0805`
- EC HDF5 SHA-256:
  `aa9add3ea5d3f8fb0699bfee06aaeb48b381dc1ca90b7b03e317766890c90cf6`
- Paired scene verdict: `PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE`
- Accepted pairs: 45; rejected native source states: 3, 4, and 7.

## Human policy-view inspection

The exact post-settle serialized states for episodes 0, 1, and 2 were
regenerated through the LIBERO wrapper and inspected in both simultaneous
256-pixel RGB streams consumed by the policy.

- `agentview`: both native black bowls, the plate, and the ramekin are
  recognizable in every EB/ER/EC frame. In ER and EC, the relocated target is
  visually between the ramekin and plate. ER and EC match except for the native
  lure pose.
- The plate remains clearly recognizable in ER/EC and has at least 1779
  segmentation pixels.
- `robot0_eye_in_hand`: frames are sharp and uncorrupted. This crop does not
  always contain every landmark simultaneously, but the paired `agentview`
  resolves the full relation.
- No reviewed frame is blank, corrupted, dominated by robot occlusion, or
  dependent on an oracle-only camera.

## Automatic physical and visual evidence

- ER minimum `agentview` pixels: target 848, lure 763, plate 1779,
  ramekin 554.
- EC minimum `agentview` pixels: target 848, lure 622, plate 1779,
  ramekin 554.
- Minimum centroid separation: ER 30.092 px; EC 30.123 px (gate: 12 px).
- Maximum post-settle layout error: 0.019785 m (gate: 0.020 m).
- Maximum settle drift: 0.001692 m.
- Maximum policy-wait drift: 0.009734 m (gate: 0.010 m).
- Maximum EC displacement from native region centers: 0.158286 m
  (gate: 0.170 m).
- Maximum stale-location pairing error: 0.008272 m (gate: 0.012 m).
- Maximum unallowed ER/EC qpos and qvel difference: 0.

This verdict authorizes dynamic reference and short smoke testing only for the
state hashes above. Formal evaluation remains blocked until dynamic
feasibility, action separation, and EB/EC capability gates pass. Review videos
must be stored under `review/L1-A4_task/`.
