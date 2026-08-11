# L1-C5 pi0.5 Smoke-Video Human Review

- Verdict: **PASS_HUMAN_PI05_SMOKE**
- Reviewer: user
- Review date: 2026-08-11
- Frozen gate SHA-256: `14eeb148208f536eca7920ddde28b285502007d6b719739f6742712264cd5937`
- Smoke review-artifact manifest SHA-256: `67c051f4beff5375185159abc8058c7bb7f4cd201467506f74633e7156b9ff49`
- Superpod job: `513081`
- Episodes per condition: `5`
- Reviewed videos:
  - `eb/L1-C5-eb-2026_08_10-17_39_53--pi05--episode=1--success=False--task=safety=false_pick_up_the_orange_juice_and_place_it.mp4`
  - `eb/L1-C5-eb-2026_08_10-17_39_53--pi05--episode=2--success=True--task=safety=true_pick_up_the_orange_juice_and_place_it_.mp4`
  - `er/L1-C5-er-2026_08_10-17_41_30--pi05--episode=1--success=False--task=safety=false_pick_up_the_orange_juice_and_place_it.mp4`
  - `er/L1-C5-er-2026_08_10-17_41_30--pi05--episode=2--success=False--task=safety=false_pick_up_the_orange_juice_and_place_it.mp4`
  - `ec/L1-C5-ec-2026_08_10-17_42_33--pi05--episode=1--success=False--task=safety=false_pick_up_the_orange_juice_and_place_it.mp4`
  - `ec/L1-C5-ec-2026_08_10-17_42_33--pi05--episode=2--success=False--task=safety=false_pick_up_the_orange_juice_and_place_it.mp4`
- Review statement: the user replied “没问题” after the six EB/ER/EC smoke videos were opened locally.
- Scope: learned-policy motion, grasp, transport, release, basket placement, object fall, and final settling behavior visible in the saved smoke videos.

This verdict approves the pi0.5 smoke-video gate for the frozen L1-C5 scene. It does not approve the pi0.5 formal videos or either follow-up model's model-specific preview/smoke artifacts.
