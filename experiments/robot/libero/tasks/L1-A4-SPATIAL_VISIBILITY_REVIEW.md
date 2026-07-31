# L1-A4 Spatial Policy-View Visibility Review

Verdict: **PASS_HUMAN_POLICY_VIEW_VISIBILITY**

Intervention ID: `l1a4_spatial_native_flat_postwait_v5`

## Audited v5 state pool

- Accepted pairs: 45
- Rejected native source states: 2, 3, and 5
- Pairing manifest SHA-256:
  `889f564430320b6136cadde5dd36e11a919c924ae1bdab51b9fb60eceaa5cd95`
- EB HDF5 SHA-256:
  `c98c4bc1770fd49eca206bb47ecee5dbe4a228b6df59cc0861071baab58f26f9`
- ER HDF5 SHA-256:
  `7557ab34112cc9f03b85f8425c189792ee8dcda4f3ae4d94507c71b88d62da9b`
- EC HDF5 SHA-256:
  `d10c7b002d32762165d47fa1ecab36725241dd9cccba1ff9a89d2760b364b086`
- Native-only preflight: `PASS_L1A4_SPATIAL_NATIVE_ONLY_PREFLIGHT`
- Paired-scene gate: `PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE`
- Post-wait physical gate:
  `PASS_L1A4_SPATIAL_POSTWAIT_PHYSICAL_GATE`

## Materials awaiting human approval

- Exact first-policy `agentview` frames for EB/ER/EC episodes 0–2:
  `review/L1-A4_task/v5_initial_policy_frames/`
- Post-wait physical-stability videos for EB/ER/EC episodes 0–1:
  `review/L1-A4_task/v5_physical_stability/`
- Physical-review manifest:
  `review/L1-A4_task/v5_physical_stability/L1-A4_v5_physical-review_manifest.json`

Every image is captured after the exact formal 10-step no-op wait. Every
video starts on that same first policy-visible frame and records 30 additional
no-op steps. Across the six videos, the maximum recorded receptacle tilt is
`0.003349°`, maximum final drift is `2.646e-7 m`, and forbidden contacts are
zero.

The smoke gate required confirmation that both native bowls, the plate, and
the ramekin are recognizable, the intended relation is visible, and all
receptacles appear physically flat. That confirmation is recorded below.

## Human approval

- Approved by the user in the active Codex session on `2026-07-31`
  (Asia/Hong_Kong).
- User verdict: the current v5 frames have no visual problem.
- Scope: the exact v5 state hashes and intervention ID recorded above.
- The bowls, plate, and ramekin are recognizable; the relational layout is
  visible; all receptacles appear physically flat.

This approval authorizes the short learned-policy smoke test only. Formal
evaluation remains blocked until the resulting EB/ER/EC smoke videos and
dynamic gates are reviewed.

## Retired review

The prior v4 approval is revoked. Its exact evaluator replay showed target
bowl tipping in all 45 ER and all 45 EC scenes. The v4 state hashes, videos,
metrics, and approval cannot authorize v5 or any future run.
