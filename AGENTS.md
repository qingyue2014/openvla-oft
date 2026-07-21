# Agent Development Isolation

## Mandatory branch and worktree policy

- Before starting any task that will modify files, identify the development scenario from the user's request (for example `L1`, `L2`, or `L3`).
- Each scenario must use its own Git branch. Use the existing project convention `physcog-libero-<scenario>` with the scenario normalized to lowercase, for example `physcog-libero-l1` and `physcog-libero-l3`.
- Each concurrently developed scenario must also use its own Git worktree. A branch alone is not sufficient isolation when multiple agents share one checkout.
- Before editing, run `git worktree list` and `git status --short --branch` to verify the current branch and worktree.
- If the current worktree is already on the correct scenario branch, continue there. Otherwise, create or reuse a sibling worktree for that scenario and perform all reads, edits, builds, and tests for the task inside it.
- Suggested sibling worktree path: `../openvla-oft-<scenario>`, for example `../openvla-oft-l3`.
- To create a new scenario branch and worktree from the default branch, use `git worktree add -b physcog-libero-<scenario> ../openvla-oft-<scenario> main`. If the branch already exists, use `git worktree add ../openvla-oft-<scenario> physcog-libero-<scenario>`.
- Never switch a shared worktree to another scenario's branch and never edit another scenario's worktree.
- Do not move, delete, clean, stash, commit, or otherwise alter unrelated user changes or untracked files while setting up isolation.
- Tell the user which branch and worktree will be used before making development changes.
- Read-only tasks such as inspection, explanation, and review do not require creating a branch or worktree.

# LIBERO Scene and Experiment Validation

## Mandatory preflight for new or modified scenes

- Apply this section whenever a task creates or changes LIBERO BDDL layouts, MuJoCo XML assets, serialized initial states, safety obstacles, collision oracles, EB/ER/EC conditions, or evaluation camera settings.
- Before smoke tests or formal evaluation, audit every custom MuJoCo asset. Task-relevant objects must have both physical collision geometry and policy-camera-visible geometry. Under the project convention, collision geoms use `group="0"` and visible geoms use `group="1"`.
- A custom asset containing only `group="0"` collision geoms is invalid even if body poses and collision checks pass. Visual-only duplicate geoms should normally set `contype="0" conaffinity="0"` and use an opaque, high-contrast material or `rgba`.
- Restore the exact serialized initial states used by evaluation and render paired EB, ER, and EC initialization images after all state changes and simulator forwarding or settling. Do not save a stale observation captured before `set_init_state`, direct `qpos` edits, or the final `sim.step()` calls.
- Use the VLA's actual policy camera, resolution, crop, orientation, color conversion, and preprocessing path as the visibility gate. Debug cameras and high-resolution renders are supplementary diagnostics only.
- Manually inspect and record that each task-relevant obstacle is recognizable at policy resolution, is not hidden by the robot or fixtures, is within the image boundary, and becomes visible early enough for the policy to react.
- Save policy-view initialization images and at least one short rollout video for every EB/ER/EC condition before submitting a formal sweep.

## Independent validation gates

- Report physical validity and visual validity separately. A static or numerical `PASS` never proves policy-view visibility.
- Verify stable reset, absence of forbidden initial contacts, paired target/goal/landmark poses, intended collision-oracle activation and component attribution, and collision-free safe-reference feasibility.
- Require automated tests for custom asset visibility conventions and run them before submission. When available, use segmentation or object-ID rendering to measure visible pixels instead of relying only on body coordinates.
- Preserve the original task prompt and semantics across paired conditions. ER and EC should differ only by the intended risk placement, while EB remains a matched benign baseline.
- **Action-separation gate.** Before running the formal (full-trial) ER/EC sweep, verify that ER is not solvable by replaying the paired EB episode's own recorded action sequence unchanged in the ER initial state. If the unchanged-EB replay is already safe/successful for that episode, the layout does not separate "the model reacted to the risk" from "the native EB action happened to be fine anyway," and the episode cannot count toward SAR/UIR/attribution. Require this check to pass at the family level (paired eligibility rate over a documented minimum, e.g. >=80%) before spending compute on the full ER/EC trial budget, not only as a post-hoc filter applied after formal evaluation. See `experiments/robot/libero/tasks/replay_l1c1_eb_actions.py` (`--fail_on_invalid`) and `run_l1c1_task2.sh`'s `bowl_stack_recalibrate` / `bowl_stack_direction_sweep` / `bowl_stack_eval` modes on the `physcog-libero-l1c1` branch for the reference implementation: a cheap EB-only probe batch is rolled out first, its actions are replayed into candidate ER states, and the risk offset/direction is widened or resampled until the eligibility gate passes, before the expensive paired EB/ER/EC formal run starts.

## Hard-stop rule

- If a task-relevant object is collidable but absent or unrecognizable in policy RGB, stop and mark the scene calibration invalid. Do not start, continue, interpret, or publish formal evaluation results for that scene.
- If the unchanged-EB action-replay eligibility rate for a family falls below its documented minimum, stop and mark the risk layout as under-separated. Do not run or publish the formal ER/EC evaluation, and do not report SAR/UIR/OCR/NOR numbers computed on the sub-threshold episode set as if the family were attribution-ready; widen the risk perturbation (offset, direction, or occupant choice) and rerun the eligibility probe first.
- Fix the asset or layout, regenerate serialized states and previews, rerun all static and dynamic calibration gates, and rerun every affected experiment.
- When the issue is discovered after jobs have started, explicitly identify the affected job IDs and outputs as invalid so they cannot be mistaken for usable results.
