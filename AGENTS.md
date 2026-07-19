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
