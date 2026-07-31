# L1-A3 Human Review

- Active version: `milk_near_target_v1`
- Policy-view verdict: **PENDING_HUMAN_POLICY_VIEW_VISIBILITY**
- Smoke-video verdict: **PENDING_HUMAN_L1A3_SMOKE_VIDEO_REVIEW**
- Automated paired-scene verdict: **PASS_L1A3_PAIRED_SCENE_GATE**
- Check job / commit: `499746` / `ee1f09ec78684ef54ec02f7a5f1af714ce2e90c3`
- Reviewed files:
  `review/L1-A3_task/milk_near_target_v1/policy_view_preview/`

All 50 exact native-state pairs passed the automated physical and policy-view
gates. Er used the preregistered 9 cm radius in every pair. Across the formal
state set:

- Er milk: 558–583 px; cream cheese: 151–164 px; basket: 2928–3104 px.
- Er minimum critical-mask centroid separation: 26.90–27.72 px.
- Ec milk: 558–581 px; cream cheese: 189–202 px; basket: 2928–3104 px.
- Ec minimum critical-mask centroid separation: 25.49–26.27 px.
- Er milk occlusion relative to Eb: 0%; Ec: 0–1.75%.
- Maximum first-policy orientation change from the serialized native pose:
  0.38°.
- Maximum five-step confirmation translation: approximately `1.1e-9` m.
- State purity outside the cream-cheese free joint: exactly `0`.

Required review:

1. Inspect the exact first-policy-observation Eb/Er/Ec `agentview` and
   `robot0_eye_in_hand` PNGs under `l1a3_preview/`.
2. Confirm all native objects visually retain their native resting orientation
   and are floor-supported.
3. Confirm milk, cream cheese, and basket are recognizable in `agentview`.
4. Confirm Er places cream cheese on the pickup approach side of milk while Ec
   moves it to a clearly safer matched-radius side.
5. After smoke, inspect videos under
   `review/L1-A3_task/milk_near_target_v1/smoke/`.

The former bowl/cookie L1-A3 review and smoke approvals belong to a replaced
scene and are invalid for this version. They must not authorize any new
rollout or formal evaluation.
