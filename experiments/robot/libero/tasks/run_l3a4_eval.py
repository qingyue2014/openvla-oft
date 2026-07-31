"""L3-A4 guarded entry point for the shared PhysCog LIBERO evaluator.

The shared evaluator intentionally supports only a small set of native-only
manifests.  This entry point adds the L3-A4 task identity, artifact binding,
and compiled microwave inventory checks without weakening those other gates.
"""

from __future__ import annotations

import os

from experiments.robot.libero import run_physcog_libero_l1_eval as evaluator
from experiments.robot.libero.tasks.validate_l3a4_native_preflight import (
    verify_evaluation_request,
    verify_runtime_asset_inventory,
)


MANIFEST_ENV = "L3A4_NATIVE_PREFLIGHT_MANIFEST"


def _install_guards() -> None:
    manifest = os.environ.get(MANIFEST_ENV, "")
    if not manifest:
        raise RuntimeError(
            f"{MANIFEST_ENV} is required; L3-A4 evaluation fails closed"
        )

    original_run = evaluator.run_task_with_safety
    original_get_env = evaluator.get_libero_env
    runtime_checked = False

    def guarded_get_env(*args, **kwargs):
        nonlocal runtime_checked
        env, description = original_get_env(*args, **kwargs)
        verify_runtime_asset_inventory(manifest, env.sim.model)
        runtime_checked = True
        return env, description

    def guarded_run(cfg, task_suite, task_id, *args, **kwargs):
        task = task_suite.get_task(task_id)
        verify_evaluation_request(
            manifest,
            task_suite_name=cfg.task_suite_name,
            task_id=task_id,
            task_language=task.language,
            task_bddl=task_suite.get_task_bddl_file_path(task_id),
            policy_prompt=cfg.task_description_override or task.language,
            initial_states_path=cfg.initial_states_path,
        )
        result = original_run(cfg, task_suite, task_id, *args, **kwargs)
        if not runtime_checked:
            raise RuntimeError("L3-A4 compiled runtime inventory was not checked")
        return result

    evaluator.get_libero_env = guarded_get_env
    evaluator.run_task_with_safety = guarded_run


def main() -> None:
    _install_guards()
    evaluator.eval_physcog_libero_l1()


if __name__ == "__main__":
    main()
