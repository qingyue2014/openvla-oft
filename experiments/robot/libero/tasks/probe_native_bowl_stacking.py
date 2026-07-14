"""List task metadata and MuJoCo bodies for native LIBERO-90 bowl stacking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_libero_importable() -> None:
    try:
        import libero  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[4]
    for candidate in (repo_root / "_deps" / "LIBERO", repo_root.parent / "LIBERO", repo_root.parent / "libero"):
        if (candidate / "libero").is_dir():
            sys.path.insert(0, str(candidate))
            return
    raise ModuleNotFoundError("Could not find LIBERO. Set LIBERO_ROOT or install LIBERO in the active environment.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    args = parser.parse_args()

    _ensure_libero_importable()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    expected = {
        16: ("akita_black_bowl_1_main", "akita_black_bowl_2_main"),
        17: ("akita_black_bowl_2_main", "akita_black_bowl_3_main"),
    }
    for task_id, (upper_body, lower_body) in expected.items():
        task = suite.get_task(task_id)
        bddl_path = suite.get_task_bddl_file_path(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=64,
            camera_widths=64,
            hard_reset=False,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        try:
            env.reset()
            body_names = {
                env.sim.model.body_id2name(i)
                for i in range(env.sim.model.nbody)
                if env.sim.model.body_id2name(i)
            }
            missing = [name for name in (upper_body, lower_body) if name not in body_names]
            print(f"task_id={task_id} language={task.language}")
            print(f"  bddl={bddl_path}")
            print(f"  upper={upper_body} lower={lower_body}")
            print(f"  body_mapping={'OK' if not missing else 'MISSING: ' + ','.join(missing)}")
            if missing:
                raise RuntimeError(f"Task {task_id} body mapping is invalid: {missing}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
