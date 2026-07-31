# L1-A3 Human Review

- Active version: `milk_near_orange_juice_v2`
- Policy-view verdict: **PENDING_HUMAN_POLICY_VIEW_VISIBILITY**
- Smoke-video verdict: **PENDING_HUMAN_L1A3_SMOKE_VIDEO_REVIEW**
- Human policy-view approval: pending for v2; the v1 approval does not transfer.
- Automated paired-scene verdict: **PENDING_V2_PAIRED_SCENE_GATE**
- Check job / commit: pending.
- Reviewed files: pending under
  `review/L1-A3_task/milk_near_orange_juice_v2/policy_view_preview/`.

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
