# L1-A4 Spatial Policy-View Visibility Review

Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**

Intervention ID: `l1a4_spatial_native_near_adaptive_v4`

## Audited state pool

- Local generation: `2026-07-30 13:34 +08:00`
- Git base: `cb491ef630b3cd8f432141e189f7a8ecbef94e51`
- Pipeline SHA-256:
  `e20c8f93dd0255576f14eff96dbb0c0916115612215115a21cfdd9e36939fee6`
- Pairing manifest SHA-256:
  `d8c71d1e608ace5a2b9777d51a980bbe8d00b598b8abad08555b3f5727941f31`
- EB HDF5 SHA-256:
  `9d170876b29c0ac95d8f3deb712897a0c2a2f8b4e12abf0c332550cd0a04976b`
- ER HDF5 SHA-256:
  `968c9a43cd9070b19693dafa0b380f168e18b0cd70b25372dc46a76166087552`
- EC HDF5 SHA-256:
  `b20bb09802cc95354da3017434ed08878e60e3442c9357da5e4c22c1fe719412`
- Paired scene verdict: `PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE`
- Accepted pairs: 45; rejected native source states: 5.

## Human policy-view inspection

The exact post-settle serialized states for episodes 0, 1, and 2 were
regenerated through the LIBERO wrapper and inspected in both simultaneous
256-pixel RGB streams consumed by the policy.

- `agentview`: both native black bowls, the plate, and the ramekin are
  recognizable in every EB/ER/EC frame. In ER and EC, the relocated target is
  visually between the ramekin and plate. ER and EC match except for the native
  lure pose.
- The plate remains clearly recognizable in ER/EC and has at least 1774
  segmentation pixels.
- `robot0_eye_in_hand`: frames are sharp and uncorrupted. This crop does not
  always contain every landmark simultaneously, but the paired `agentview`
  resolves the full relation.
- No reviewed frame is blank, corrupted, dominated by robot occlusion, or
  dependent on an oracle-only camera.

## Automatic physical and visual evidence

- ER minimum `agentview` pixels: target 874, lure 769, plate 1774,
  ramekin 554.
- EC minimum `agentview` pixels: target 874, lure 626, plate 1774,
  ramekin 557.
- Minimum centroid separation: ER 29.496 px; EC 30.510 px (gate: 12 px).
- Maximum post-settle layout error: 0.019928 m (gate: 0.020 m).
- Maximum settle drift: 0.002507 m.
- Maximum policy-wait drift: 0.009994 m (gate: 0.010 m).
- Maximum EC displacement from native region centers: 0.156934 m
  (gate: 0.170 m).
- Maximum stale-location pairing error: 0.009735 m (gate: 0.012 m).
- Maximum unallowed ER/EC qpos and qvel difference: 0.

This verdict authorizes dynamic reference and short smoke testing only for the
state hashes above. Formal evaluation remains blocked until dynamic
feasibility, action separation, and EB/EC capability gates pass. Review videos
must be stored under `review/L1-A4_task/`.
