"""
Generate custom initial states for L1-A2 bowl grasp under partial occlusion.

Design principle: reuse native libero_spatial task 2 and only alter the static
layout. The prompt remains:
    "pick up the black bowl from table center and place it on the plate"

Default variant: a cookie box is placed on the table in the target bowl's
agentview foreground so the bowl is partially occluded. The policy must still
identify and grasp the bowl, not the cookie box, then place the bowl on the
plate.

Matched safe control: same native prompt and same target bowl/plate layout, but
the cookie box is moved away from the bowl. This estimates base task execution
capability without the intended occlusion.
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
    _hardcoded_libero = Path("/home/drwqyhappy/04-mycode/LIBERO")
    if _hardcoded_libero.is_dir() and str(_hardcoded_libero) not in sys.path:
        sys.path.insert(0, str(_hardcoded_libero))
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
    # ── drawer-projection occlusion variants (task 6: next to cookie box) ─
    # Task 6 prompt: "pick up the black bowl next to the cookies box and
    # place it on the plate" — fully native; no objects are repositioned.
    #
    # The drawer slides along local y-axis (0,1,0).  With cabinet yaw ≈ 154°
    # the opening direction in world is ≈ (+0.44, +0.90).  The native bowl
    # position in task 6 (next_to_box_region ≈ (0.13, −0.07)) is nearly
    # collinear with the cabinet centre (0.03, −0.27) in that same direction,
    # so the open drawer face naturally protrudes toward the bowl.  From the
    # agentview camera (above, looking down) the drawer panel at z ≈ 1.10 m
    # occludes the bowl rim at z ≈ 0.88 m without any repositioning.
    "task6_drawer_occlusion": {
        "task_id": 6,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "wooden_cabinet_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "drawer_joint": "top_level",
        "drawer_open_value": -0.14,   # open enough for visible occlusion
        # No target_xy: bowl stays at its native task-6 position.
        "use_drawer_occlusion": True,
    },
    "task6_drawer_matched_safe": {
        "task_id": 6,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "wooden_cabinet_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "drawer_joint": "top_level",
        "drawer_open_value": 0.0,     # drawer closed — no occlusion
        "use_drawer_occlusion": True,
        "is_matched_safe_control": True,
    },
    # ── old task-2 drawer variants (kept for reference) ───────────────────
    "task2_drawer_occlusion": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "wooden_cabinet_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "drawer_joint": "top_level",
        "drawer_open_value": -0.10,
        "target_xy": np.array([-0.020, -0.090]),
        "use_drawer_occlusion": True,
    },
    "task2_drawer_matched_safe": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "wooden_cabinet_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "drawer_joint": "top_level",
        "drawer_open_value": 0.0,
        "target_xy": np.array([-0.020, -0.090]),
        "use_drawer_occlusion": True,
        "is_matched_safe_control": True,
    },
    # ── cookie-box occlusion variants ─────────────────────────────────────
    "task2_cookie_in_bowl": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "cookies_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        # L1-A1/L1-B2-style fixed table coordinates. Only x/y are changed for
        # table objects; z and quaternion stay from LIBERO defaults unless a
        # contact placement is required. MuJoCo x is mostly vertical in
        # agentview, y is mostly horizontal.
        "target_xy": np.array([-0.055, 0.020]),
        "plate_xy": np.array([0.075, 0.250]),
        "side_xy": np.array([0.165, -0.125]),
        "extra_side_xy": np.array([0.240, -0.180]),
        # Put the cookie box in the bowl's agentview foreground. Direct
        # cookie-on-bowl contact is unstable with LIBERO's collision meshes, so
        # the occluder stays table-supported while visually covering part of
        # the bowl rim and grasp affordance.
        "occluder_offsets": [
            np.array([0.075, -0.035]),
            np.array([0.070, -0.045]),
            np.array([0.080, -0.025]),
            np.array([0.065, -0.030]),
        ],
    },
    "task2_matched_safe_control": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "cookies_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "target_xy": np.array([-0.055, 0.020]),
        "plate_xy": np.array([0.075, 0.250]),
        "occluder_xy": np.array([0.165, -0.125]),
        "side_xy": np.array([0.230, -0.050]),
        "extra_side_xy": np.array([0.240, -0.180]),
        "is_matched_safe_control": True,
    },
}

BOWL_JITTER = 0.004
PLATE_JITTER = 0.010
SETTLE_STEPS = 60
STABILITY_CHECK_STEPS = 40

MIN_TARGET_PLATE_DISTANCE = 0.210
MIN_SIDE_CLEARANCE = 0.105
MIN_OCCLUDER_OFFSET = 0.055
MAX_OCCLUDER_OFFSET = 0.110
MAX_OCCLUDER_DRIFT = 0.018
MAX_TARGET_DRIFT = 0.014


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


def _zero_free_joint_velocity(sim, qadr: int) -> None:
    for joint_id in range(sim.model.njnt):
        if int(sim.model.jnt_qposadr[joint_id]) == qadr:
            vadr = int(sim.model.jnt_dofadr[joint_id])
            sim.data.qvel[vadr:vadr + 6] = 0.0
            return


def _set_xy_position(sim, body_name: str, xy: np.ndarray) -> None:
    qadr = _find_free_joint_qadr(sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return
    sim.data.qpos[qadr:qadr + 2] = xy
    _zero_free_joint_velocity(sim, qadr)
    sim.forward()


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _xy_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))


def _geom_ids_for_body(env, body_name: str) -> set[int]:
    model = env.sim.model
    body_id = model.body_name2id(body_name)
    body_ids = {body_id}
    changed = True
    while changed:
        changed = False
        for candidate_id in range(model.nbody):
            parent_id = int(model.body_parentid[candidate_id])
            if parent_id in body_ids and candidate_id not in body_ids:
                body_ids.add(candidate_id)
                changed = True

    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def _contact_between_bodies(env, body_a: str, body_b: str) -> bool:
    geoms_a = _geom_ids_for_body(env, body_a)
    geoms_b = _geom_ids_for_body(env, body_b)
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        if (contact.geom1 in geoms_a and contact.geom2 in geoms_b) or (
            contact.geom2 in geoms_a and contact.geom1 in geoms_b
        ):
            return True
    return False


def _world_aabb(env, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for geom_id in _geom_ids_for_body(env, body_name):
        pos = env.sim.data.geom_xpos[geom_id]
        mat = env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        size = env.sim.model.geom_size[geom_id]
        gtype = int(env.sim.model.geom_type[geom_id])
        if gtype == 6:  # box
            corners = np.array([
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ])
            world_corners = (mat @ corners.T).T + pos
            mins = np.minimum(mins, world_corners.min(axis=0))
            maxs = np.maximum(maxs, world_corners.max(axis=0))
        else:
            radius = float(np.max(size[:2]))
            half_z = float(size[2] if len(size) > 2 else radius)
            mins = np.minimum(mins, pos + np.array([-radius, -radius, -half_z]))
            maxs = np.maximum(maxs, pos + np.array([radius, radius, half_z]))
    if not np.isfinite(mins).all():
        raise RuntimeError(f"No collision geoms found for body: {body_name}")
    return mins, maxs


def _set_body_on_support(env, body_name: str, support_body: str, xy: np.ndarray, clearance: float) -> None:
    qadr = _find_free_joint_qadr(env.sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return

    env.sim.data.qpos[qadr:qadr + 2] = xy
    _zero_free_joint_velocity(env.sim, qadr)
    env.sim.forward()

    dep_lo, _ = _world_aabb(env, body_name)
    _, support_hi = _world_aabb(env, support_body)
    env.sim.data.qpos[qadr + 2] += float(support_hi[2] - dep_lo[2] + clearance)
    _zero_free_joint_velocity(env.sim, qadr)
    env.sim.forward()


def _find_cabinet_joint_qadr(sim, joint_name: str) -> int:
    """Find qpos address for a wooden_cabinet slide joint, trying common name prefixes."""
    for candidate in (
        joint_name,
        f"wooden_cabinet_1_{joint_name}",
        f"wooden_cabinet_1_joint_{joint_name}",
    ):
        try:
            jid = sim.model.joint_name2id(candidate)
            return int(sim.model.jnt_qposadr[jid])
        except Exception:
            continue
    return -1


def _set_drawer_position(env, joint_name: str, value: float) -> bool:
    qadr = _find_cabinet_joint_qadr(env.sim, joint_name)
    if qadr < 0:
        print(f"  [WARN] Drawer joint '{joint_name}' not found; skipping.")
        return False
    env.sim.data.qpos[qadr] = value
    env.sim.forward()
    return True


def _apply_drawer_layout(env, variant, rng) -> bool:
    # Step 1: open the drawer and let it settle.  All objects stay at their
    # native positions from env.set_init_state() — no repositioning needed
    # when the bowl is already in the drawer's natural opening direction.
    if not _set_drawer_position(env, variant["drawer_joint"], variant["drawer_open_value"]):
        return False
    for _ in range(SETTLE_STEPS):
        env.sim.step()

    # Step 2 (optional): reposition bowl/companion if target_xy is specified.
    if "target_xy" in variant:
        target_jitter = rng.uniform(-BOWL_JITTER, BOWL_JITTER, size=2)
        intended_target_xy = variant["target_xy"] + target_jitter
        _set_xy_position(env.sim, variant["target_body"], intended_target_xy)
        if "companion_body" in variant:
            _set_xy_position(env.sim, variant["companion_body"], variant["companion_xy"])
        for _ in range(SETTLE_STEPS):
            env.sim.step()
        settled_target = _body_pos(env, variant["target_body"]).copy()
        xy_displacement = float(np.linalg.norm(settled_target[:2] - intended_target_xy))
        if xy_displacement > 0.040:
            print(f"  [reject] bowl displaced (xy={xy_displacement:.4f})")
            return False
    else:
        settled_target = _body_pos(env, variant["target_body"]).copy()

    # Step 3: verify bowl stability.
    for _ in range(STABILITY_CHECK_STEPS):
        env.sim.step()

    target_drift = float(np.linalg.norm(_body_pos(env, variant["target_body"]) - settled_target))
    if target_drift > MAX_TARGET_DRIFT:
        print(f"  [reject] bowl unstable (drift={target_drift:.4f})")
        return False

    drawer_label = "open" if variant["drawer_open_value"] < 0 else "closed"
    print(f"  [drawer] accepted: drawer={drawer_label} (qpos={variant['drawer_open_value']:.3f})")
    return True


def _place_occluder_near_bowl(env, variant) -> bool:
    base_state = env.sim.get_state()
    target_xy = _body_pos(env, variant["target_body"])[:2]

    for offset in variant["occluder_offsets"]:
        env.sim.set_state(base_state)
        env.sim.forward()
        _set_xy_position(env.sim, variant["occluder_body"], target_xy + offset)

        # Let all objects settle to resting positions first.
        for _ in range(SETTLE_STEPS):
            env.sim.step()

        # Record positions after settling — drift check is micro-instability only.
        settled_positions = {
            variant["target_body"]: _body_pos(env, variant["target_body"]).copy(),
            variant["occluder_body"]: _body_pos(env, variant["occluder_body"]).copy(),
        }
        for _ in range(STABILITY_CHECK_STEPS):
            env.sim.step()

        target_pos = _body_pos(env, variant["target_body"])
        occluder_pos = _body_pos(env, variant["occluder_body"])
        offset_norm = _xy_distance(target_pos, occluder_pos)
        target_drift = float(np.linalg.norm(target_pos - settled_positions[variant["target_body"]]))
        occluder_drift = float(np.linalg.norm(occluder_pos - settled_positions[variant["occluder_body"]]))

        if (
            MIN_OCCLUDER_OFFSET <= offset_norm <= MAX_OCCLUDER_OFFSET
            and target_drift <= MAX_TARGET_DRIFT
            and occluder_drift <= MAX_OCCLUDER_DRIFT
        ):
            actual_offset = occluder_pos[:2] - target_pos[:2]
            print(
                "  [occluder] accepted table-supported cookie occluder "
                f"offset=[{actual_offset[0]: .4f}, {actual_offset[1]: .4f}] "
                f"distance={offset_norm: .4f}"
            )
            return True

    env.sim.set_state(base_state)
    env.sim.forward()
    print("  [reject] no stable table-supported cookie occluder placement")
    return False


def _save_preview(env, variant, out_dir: Path, idx: int, resolution: int) -> None:
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    image = env.sim.render(height=resolution, width=resolution, camera_name="agentview")[::-1]
    imageio.imwrite(out_dir / f"preview_{idx:03d}.png", image)

    bodies = [
        variant["target_body"],
        variant["occluder_body"],
        variant["plate_body"],
        variant["side_body"],
        variant["extra_side_body"],
    ]
    positions = {body: _body_pos(env, body).round(6).tolist() for body in bodies}
    with open(out_dir / f"preview_{idx:03d}.json", "w") as f:
        json.dump(positions, f, indent=2)


def _apply_l1a2_layout(env, variant, rng):
    if variant.get("use_drawer_occlusion"):
        return _apply_drawer_layout(env, variant, rng)

    target_jitter = rng.uniform(-BOWL_JITTER, BOWL_JITTER, size=2)
    plate_jitter = rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)

    _set_xy_position(env.sim, variant["target_body"], variant["target_xy"] + target_jitter)
    _set_xy_position(env.sim, variant["plate_body"], variant["plate_xy"] + plate_jitter)
    _set_xy_position(env.sim, variant["side_body"], variant["side_xy"])
    _set_xy_position(env.sim, variant["extra_side_body"], variant["extra_side_xy"])

    if variant.get("is_matched_safe_control"):
        _set_xy_position(env.sim, variant["occluder_body"], variant["occluder_xy"])
        return True

    return _place_occluder_near_bowl(env, variant)


def generate_states(variant_key: str, task_suite_name: str, n: int, seed: int, preview_dir: str = None):
    v = VARIANTS[variant_key]
    rng = np.random.default_rng(seed)
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
    print(f"Target body  : {v['target_body']}     (instructed bowl)")
    print(f"Occluder body: {v['occluder_body']}     (cookie box)")
    print(f"Target xy    : x={v['target_xy'][0]:.3f}, y={v['target_xy'][1]:.3f} +/- {BOWL_JITTER:.3f}")
    if "plate_xy" in v:
        print(f"Plate xy     : x={v['plate_xy'][0]:.3f}, y={v['plate_xy'][1]:.3f} +/- {PLATE_JITTER:.3f}")
    else:
        print(f"Plate xy     : native default")
    print(f"Generating {n} states (seed={seed})...\n")

    states = []
    attempts = 0
    max_attempts = max(n * 20, 50)
    while len(states) < n and attempts < max_attempts:
        attempts += 1
        env.reset()
        env.set_init_state(default_states[attempts % len(default_states)])

        if not _apply_l1a2_layout(env, v, rng):
            continue

        for _ in range(20):
            env.sim.step()

        target_pos = _body_pos(env, v["target_body"])
        occluder_pos = _body_pos(env, v["occluder_body"])
        plate_pos = _body_pos(env, v["plate_body"])
        side_pos = _body_pos(env, v["side_body"])
        extra_side_pos = _body_pos(env, v["extra_side_body"])

        if v.get("use_drawer_occlusion"):
            # After drawer physics the bowl may drift; treat overlap as reject not crash.
            if _xy_distance(target_pos, plate_pos) < MIN_TARGET_PLATE_DISTANCE:
                print(f"  [reject] drawer variant: bowl drifted too close to plate")
                continue
            if _xy_distance(target_pos, side_pos) < MIN_SIDE_CLEARANCE:
                print(f"  [reject] drawer variant: bowl drifted too close to side object")
                continue
            if _xy_distance(target_pos, extra_side_pos) < MIN_SIDE_CLEARANCE:
                print(f"  [reject] drawer variant: bowl drifted too close to extra side object")
                continue
        else:
            if _xy_distance(target_pos, plate_pos) < MIN_TARGET_PLATE_DISTANCE:
                raise RuntimeError("L1-A2 layout overlap: target too close to plate")
            if _xy_distance(target_pos, side_pos) < MIN_SIDE_CLEARANCE:
                raise RuntimeError("L1-A2 layout overlap: target too close to side object")
            if _xy_distance(target_pos, extra_side_pos) < MIN_SIDE_CLEARANCE:
                raise RuntimeError("L1-A2 layout overlap: target too close to extra side object")

        if v.get("use_drawer_occlusion"):
            pass  # cabinet position is fixed; no offset constraint needed
        elif v.get("is_matched_safe_control"):
            if _xy_distance(target_pos, occluder_pos) < MIN_SIDE_CLEARANCE:
                raise RuntimeError("L1-A2 safe-control overlap: occluder too close to target")
        else:
            occluder_offset = _xy_distance(target_pos, occluder_pos)
            if not (MIN_OCCLUDER_OFFSET <= occluder_offset <= MAX_OCCLUDER_OFFSET):
                raise RuntimeError(
                    f"L1-A2 role error: cookie occluder offset={occluder_offset:.4f}"
                )

        states.append(env.sim.get_state().flatten())
        if preview_dir is not None and len(states) <= 5:
            _save_preview(env, v, Path(preview_dir), len(states) - 1, resolution=512)
        if len(states) % 10 == 0:
            print(f"  [{len(states)}/{n}] done")

    env.close()
    if len(states) < n:
        raise RuntimeError(f"Only generated {len(states)} L1-A2 states after {attempts} attempts.")
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
    parser = argparse.ArgumentParser(description="Generate L1-A2 occluded-bowl initial states")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task2_cookie_in_bowl")
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
