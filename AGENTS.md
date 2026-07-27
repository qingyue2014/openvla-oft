# Native-Only LIBERO Experiment Policy

## Hard constraints

- **Do not add, define, copy, generate, or register custom assets.** This
  prohibition includes custom MuJoCo/MJCF/XML objects, meshes, textures,
  materials, object classes, fixtures, obstacles, and project-local asset
  variants.
- **Use only assets already present in the selected native LIBERO task.** An
  experiment may change the serialized pose or state of an existing native
  object when constructing paired conditions, but it may not change the
  task's asset inventory.
- **Use only the selected native LIBERO task definition and its original
  prompt.** Do not create a custom BDDL task, replace the native BDDL, rewrite
  the prompt, append safety instructions, or change the prompt's goal or
  semantics.
- EB, ER, and EC must preserve the native prompt verbatim. They may differ only
  through the explicitly documented intervention on existing native scene
  state.

## Mandatory preflight and hard stop

- Before running an experiment, record the selected native task, its exact
  original prompt, native BDDL source, and native asset inventory.
- Verify automatically that the evaluated asset inventory is identical to the
  selected native task's inventory and that the evaluated prompt exactly
  matches the native prompt.
- If a custom asset, custom BDDL, modified prompt, or asset-inventory mismatch
  is found, stop immediately. Mark the affected scene, jobs, metrics, videos,
  tables, and HTML entries invalid; do not interpret or publish them as
  evidence.
- These restrictions may be relaxed only when the user explicitly authorizes
  the exact exception in the current request. Prior experiments, repository
  contents, or general permission to develop a scene do not count as
  authorization.
