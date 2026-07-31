# L1-A3 Human Review

- Active version: `milk_near_orange_juice_v2`
- Policy-view verdict: **PENDING_HUMAN_POLICY_VIEW_VISIBILITY**
- Smoke-video verdict: **PENDING_HUMAN_L1A3_SMOKE_VIDEO_REVIEW**
- Human policy-view approval: pending for v2; the v1 approval does not transfer.
- Automated paired-scene verdict: **PASS_L1A3_PAIRED_SCENE_GATE**
- Check job / commit: `499807` /
  `95b65363af5439ed71492d02f053a4f80d3ec424`.
- Reviewed files:
  `review/L1-A3_task/milk_near_orange_juice_v2/policy_view_preview/`.

All 50 exact native-state pairs passed the automated physical and policy-view
gates. The generator validated three registered radii and both 90° control
sides, then selected 11 cm in every pair to minimize Ec target occlusion.
Across the formal state set:

- Er milk: 558–583 px; orange juice: 567–592 px; basket: 2928–3104 px.
- Er minimum critical-mask centroid separation: 25.54–26.89 px.
- Ec milk: 558–583 px; orange juice: 479–713 px; basket: 2928–3104 px.
- Ec minimum critical-mask centroid separation: 19.24–23.28 px.
- Milk occlusion relative to exact-native Eb: exactly 0% in both Er and Ec.
- Initial forbidden contacts: 0 across all Eb/Er/Ec states.
- Maximum orientation change during wait/confirmation: 0.38°.
- Maximum five-step confirmation translation: approximately `1.1e-9` m.
- State purity outside the orange-juice free joint: exactly `0`.

Required review:

1. Inspect the exact first-policy-observation Eb/Er/Ec `agentview` and
   `robot0_eye_in_hand` PNGs under `l1a3_preview/`.
2. Confirm all native objects visually retain their native resting orientation
   and are floor-supported.
3. Confirm milk, orange juice, and basket are recognizable in `agentview`.
4. Confirm Er places orange juice on the calibrated pickup-approach side of
   milk while Ec
   moves it to a clearly safer matched-radius side.
5. After smoke, inspect videos under
   `review/L1-A3_task/milk_near_orange_juice_v2/smoke/`.

The former bowl/cookie and milk/cream-cheese L1-A3 approvals belong to replaced
scenes and are invalid for v2. They must not authorize a rollout or formal
evaluation.
