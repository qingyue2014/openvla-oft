# Canonical Experiment Asset Policy

## No new assets

- Canonical PhysCog experiments must use only objects and fixtures already
  present in the selected native LIBERO task.
- Do not define, register, or insert a new BDDL object, MuJoCo XML asset,
  safety obstacle, visual duplicate, or replacement fixture unless the user
  explicitly authorizes that new asset in the current request.
- Moving an existing native object's pose is allowed only when the scenario
  contract permits it and all pairing, physics, visibility, and
  action-separation gates still pass.
- Before any formal run or promotion, compare the active BDDL/object inventory
  with the selected native LIBERO task. A new or substituted asset is a hard
  stop even if collision, visibility, safe-reference, replay, and rollout
  checks pass.
- Results produced with an unauthorized custom asset are diagnostic-only.
  Label their job IDs and artifacts invalid for canonical reporting; never map
  or publish them as a formal scenario.
- Historical custom-asset files may remain for provenance, but canonical
  runners, state generators, result mappings, tables, and HTML must not
  reference or pool them.
