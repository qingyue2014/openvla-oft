# L1-A3 Human Review

- Active version: `milk_near_target_v1`
- Policy-view verdict: **PENDING_HUMAN_POLICY_VIEW_VISIBILITY**
- Smoke-video verdict: **PENDING_HUMAN_L1A3_SMOKE_VIDEO_REVIEW**

Required review:

1. Inspect the exact first-policy-observation Eb/Er/Ec `agentview` and
   `robot0_eye_in_hand` PNGs under `l1a3_preview/`.
2. Confirm all native objects are physically upright and floor-supported.
3. Confirm milk, cream cheese, and basket are recognizable in `agentview`.
4. Confirm Er places cream cheese on the pickup approach side of milk while Ec
   moves it to a clearly safer matched-radius side.
5. After smoke, inspect videos under
   `review/L1-A3_task/milk_near_target_v1/smoke/`.

The former bowl/cookie L1-A3 review and smoke approvals belong to a replaced
scene and are invalid for this version. They must not authorize any new
rollout or formal evaluation.
