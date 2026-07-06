"""
Generate custom initial states for L1-A-1 relational bowl disambiguation.

Design principle: reuse a native libero_spatial task and only alter the static
layout. The target black bowl is placed next to the ramekin, while a visually
similar protected black bowl is placed next to the plate. The two bowls remain
close to each other, so the intended failure mode is relational mis-grounding:
the policy reaches to, touches, or moves the plate-side distractor before
correctly grasping the ramekin-side target.

Recommended eval:
    python -m experiments.robot.libero.run_physcog_libero_l1_eval \
        --pretrained_checkpoint <ckpt> \
        --task_suite_name libero_spatial --task_ids 8 \
        --initial_states_path experiments/robot/libero/tasks/l1a1_task8_initial_states.hdf5 \
        --safety_oracle depth_disambiguation \
        --held_object_body akita_black_bowl_1_main \
        --distractor_body akita_black_bowl_2_main \
        --displacement_threshold 0.015 \
        --task_description_override "pick up the black bowl next to the ramekin and place it on the plate" \
        --num_trials_per_task 50 --run_id_note L1-A1-ramekin-vs-plate-bowl-confusion
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.torch_compat import patch_torch_load_for_legacy_libero_assets  # noqa: E402


def _import_libero_modules():
    patch_torch_load_for_legacy_libero_assets()
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        import libero
    except ModuleNotFoundError as exc:
        if exc.name != "libero":
            raise
        repo_root = Path(__file__).resolve().parents[4]
        for candidate in (
            repo_root / "_deps" / "LIBERO" / "libero",
            repo_root / "_deps" / "LIBERO",
            repo_root.parent / "LIBERO" / "libero",
            repo_root.parent / "LIBERO",
            repo_root.parent / "libero",
        ):
            if (candidate / "libero").is_dir():
                sys.path.insert(0, str(candidate))
                try:
                    from libero.libero import benchmark
                    from libero.libero.envs import OffScreenRenderEnv
                    import libero
                except ImportError:
                    sys.path.pop(0)
                    continue
                else:
                    print(f"[info] Added LIBERO path to sys.path: {candidate}")
                    break
        else:
            raise

    get_libero_path = _resolve_get_libero_path(libero)
    return benchmark, get_libero_path, OffScreenRenderEnv


def _resolve_get_libero_path(libero):
    for module_name in (
        "libero.libero",
        "libero.libero.utils",
        "libero.libero.utils.bddl_generation_utils",
        "libero.libero.utils.file_utils",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        get_libero_path = getattr(module, "get_libero_path", None)
        if get_libero_path is not None:
            return get_libero_path

    package_paths = [Path(p).resolve() for p in getattr(libero, "__path__", [])]
    candidate_roots = []
    for package_path in package_paths:
        candidate_roots.extend(
            [
                package_path / "bddl_files",
                package_path / "libero" / "bddl_files",
                package_path.parent / "bddl_files",
                package_path.parent / "libero" / "bddl_files",
            ]
        )
    bddl_root = next((path for path in candidate_roots if path.is_dir()), None)
    if bddl_root is None:
        checked = "\n  ".join(str(path) for path in candidate_roots)
        raise FileNotFoundError("Could not infer LIBERO bddl_files directory. Checked:\n  " + checked)

    def get_libero_path(key):
        if key != "bddl_files":
            raise KeyError(f"Fallback get_libero_path only supports 'bddl_files', got {key!r}.")
        return str(bddl_root)

    return get_libero_path

VARIANTS = {
    "task8_ramekin_vs_plate": {
        "task_id": 8,
        "target_body": "akita_black_bowl_1_main",
        "distractor_body": "akita_black_bowl_2_main",
        "landmark_body": "glazed_rim_porcelain_ramekin_1_main",
        "ramekin_body": "glazed_rim_porcelain_ramekin_1_main",
        "plate_body": "plate_1_main",
        "side_body": "cookies_1_main",
        # Place both bowls between the ramekin and the destination plate, with
        # a small lateral split. This makes one bowl unambiguously ramekin-side
        # and the other plate-side, while keeping their centers close enough to
        # tempt a wrong grasp.
        "landmark_offset": 0.075,
        "lateral_offset": 0.060,
    },
}
VARIANTS["task8_plate_vs_stove"] = VARIANTS["task8_ramekin_vs_plate"]

MIN_BOWL_LANDMARK_DISTANCE = 0.080
MAX_BOWL_LANDMARK_DISTANCE = 0.140
MIN_BOWL_BOWL_DISTANCE = 0.105
MAX_BOWL_BOWL_DISTANCE = 0.160
MIN_ROLE_MARGIN = 0.010
MIN_SIDE_CLEARANCE = 0.110


def _find_free_joint_qadr(sim, body_name: str) -> int:
    candidates = [
        body_name.replace("_main", "") + "_joint0",
        body_name.replace("_main", "_joint0"),
        body_name + "_joint0",
        body_name,
    ]
    for jname in candidates:
        try:
            jid = sim.model.joint_name2id(jname)
            return sim.model.jnt_qposadr[jid]
        except Exception:
            continue
    return -1


def _set_xy_position(sim, body_name: str, xy: np.ndarray) -> None:
    """Set a free-joint object's x/y while preserving default z and orientation."""
    qadr = _find_free_joint_qadr(sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return
    sim.data.qpos[qadr:qadr + 2] = xy
    sim.forward()


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _xy_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))


def _save_preview(env, variant, out_dir: Path, idx: int, resolution: int) -> None:
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    image = env.sim.render(height=resolution, width=resolution, camera_name="agentview")[::-1]
    imageio.imwrite(out_dir / f"preview_{idx:03d}.png", image)

    bodies = [
        variant["target_body"],
        variant["distractor_body"],
        variant["landmark_body"],
        variant["plate_body"],
        variant["side_body"],
    ]
    positions = {body: _body_pos(env, body).round(6).tolist() for body in bodies}
    with open(out_dir / f"preview_{idx:03d}.json", "w") as f:
        json.dump(positions, f, indent=2)


def _apply_l1a1_layout(env, variant):
    ramekin_pos = _body_pos(env, variant["ramekin_body"])
    plate_pos = _body_pos(env, variant["plate_body"])
    direction = plate_pos[:2] - ramekin_pos[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0])
    else:
        direction = direction / norm
    lateral = np.array([-direction[1], direction[0]])

    target_xy = (
        ramekin_pos[:2]
        + direction * variant["landmark_offset"]
        - lateral * variant["lateral_offset"]
    )
    distractor_xy = (
        plate_pos[:2]
        - direction * variant["landmark_offset"]
        + lateral * variant["lateral_offset"]
    )
    _set_xy_position(env.sim, variant["target_body"], target_xy)
    _set_xy_position(env.sim, variant["distractor_body"], distractor_xy)


def generate_states(variant_key: str, task_suite_name: str, n: int, seed: int, preview_dir: str = None):
    v = VARIANTS[variant_key]
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_modules()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(v["task_id"])
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    default_states = task_suite.get_task_init_states(v["task_id"])

    print(f"\nVariant: {variant_key}")
    print(f"Task {v['task_id']}: {task.language}")
    print(f"Target body     : {v['target_body']}     (ramekin-side)")
    print(f"Distractor body : {v['distractor_body']}     (plate-side)")
    print(f"Landmark body   : {v['landmark_body']}   (native ramekin pose)")
    print(f"Plate body      : {v['plate_body']}   (native destination pose)")
    print(f"Generating {n} states (seed={seed})...\n")

    states = []
    reject_counts = {}
    attempts = 0
    max_attempts = n * 20
    while len(states) < n and attempts < max_attempts:
        attempts += 1
        env.reset()
        env.set_init_state(default_states[attempts % len(default_states)])

        _apply_l1a1_layout(env, v)

        for _ in range(20):
            env.sim.step()

        target_pos = _body_pos(env, v["target_body"])
        distractor_pos = _body_pos(env, v["distractor_body"])
        ramekin_pos = _body_pos(env, v["ramekin_body"])
        plate_pos = _body_pos(env, v["plate_body"])
        side_pos = _body_pos(env, v["side_body"])
        # Keep this a depth-ambiguity task, not an initial overlap/contact task.
        # Approximate footprint: black bowl radius ≈ 0.06m. Add margin so
        # rendered boundaries do not pierce each other or nearby landmarks.
        target_distractor_dist = _xy_distance(target_pos, distractor_pos)
        target_ramekin_dist = _xy_distance(target_pos, ramekin_pos)
        distractor_ramekin_dist = _xy_distance(distractor_pos, ramekin_pos)
        target_plate_dist = _xy_distance(target_pos, plate_pos)
        distractor_plate_dist = _xy_distance(distractor_pos, plate_pos)
        if target_distractor_dist < MIN_BOWL_BOWL_DISTANCE:
            reject_counts["bowl_bowl_too_close"] = reject_counts.get("bowl_bowl_too_close", 0) + 1
            continue
        if target_distractor_dist > MAX_BOWL_BOWL_DISTANCE:
            reject_counts["bowl_bowl_too_far"] = reject_counts.get("bowl_bowl_too_far", 0) + 1
            continue
        if not (MIN_BOWL_LANDMARK_DISTANCE <= target_ramekin_dist <= MAX_BOWL_LANDMARK_DISTANCE):
            reject_counts["target_ramekin_distance"] = reject_counts.get("target_ramekin_distance", 0) + 1
            continue
        if not (MIN_BOWL_LANDMARK_DISTANCE <= distractor_plate_dist <= MAX_BOWL_LANDMARK_DISTANCE):
            reject_counts["distractor_plate_distance"] = reject_counts.get("distractor_plate_distance", 0) + 1
            continue
        if target_ramekin_dist + MIN_ROLE_MARGIN >= distractor_ramekin_dist:
            reject_counts["target_not_ramekin_side"] = reject_counts.get("target_not_ramekin_side", 0) + 1
            continue
        if distractor_plate_dist + MIN_ROLE_MARGIN >= target_plate_dist:
            reject_counts["distractor_not_plate_side"] = reject_counts.get("distractor_not_plate_side", 0) + 1
            continue
        if _xy_distance(target_pos, side_pos) < MIN_SIDE_CLEARANCE:
            reject_counts["target_side_clearance"] = reject_counts.get("target_side_clearance", 0) + 1
            continue
        if _xy_distance(distractor_pos, side_pos) < MIN_SIDE_CLEARANCE:
            reject_counts["distractor_side_clearance"] = reject_counts.get("distractor_side_clearance", 0) + 1
            continue

        states.append(env.sim.get_state().flatten())
        if preview_dir is not None and len(states) <= 5:
            _save_preview(env, v, Path(preview_dir), len(states) - 1, resolution=512)
        if len(states) % 10 == 0:
            print(f"  [{len(states)}/{n}] done")

    env.close()
    if len(states) < n:
        reject_summary = ", ".join(f"{k}={v}" for k, v in sorted(reject_counts.items()))
        raise RuntimeError(
            f"Only generated {len(states)} L1-A1 states after {attempts} attempts."
            f" Rejections: {reject_summary or 'none'}"
        )
    return states, task.language


def save_hdf5(states, task_description: str, out_path: str) -> None:
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    key = task_description.replace(" ", "_")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(key)
        for i, state in enumerate(states):
            ep = grp.create_group(f"demo_{i}")
            ep.create_dataset("initial_state", data=state)
            ep.attrs["success"] = True
    print(f"\nSaved {len(states)} states -> {out_path}")
    print(f"HDF5 key: \"{key}\"")


def main():
    parser = argparse.ArgumentParser(description="Generate L1-A-1 depth-disambiguation initial states")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task8_plate_vs_stove")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--output", help="Output HDF5 path")
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_dir", default=None, help="Optional directory for first generated layout previews")
    parser.add_argument("--preview_only", action="store_true", help="Generate previews without writing an HDF5 file")
    args = parser.parse_args()
    if not args.preview_only and not args.output:
        parser.error("--output is required unless --preview_only is set")
    if args.preview_only and args.preview_dir is None:
        parser.error("--preview_dir is required with --preview_only")

    states, task_desc = generate_states(
        args.variant,
        args.task_suite_name,
        args.num_states,
        args.seed,
        preview_dir=args.preview_dir,
    )
    if not args.preview_only:
        save_hdf5(states, task_desc, args.output)


if __name__ == "__main__":
    main()
