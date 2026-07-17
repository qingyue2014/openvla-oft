"""
Generate custom initial states for L1-A2 bowl grasp under partial occlusion.

Design principle: reuse native LIBERO prompts and only alter the static layout.
The current main L1-A2 variant reuses libero_spatial task 1:
    "pick up the black bowl next to the ramekin and place it on the plate"

Default variant: a cookie box is stood upright in the target bowl's agentview
foreground so it partially occludes the far, ramekin-side target bowl. The
cookie box is kept out of contact with the bowl. The policy must still identify
and grasp the bowl next to the ramekin, not the cookie box, then place the bowl
on the plate.

Matched safe control: same native prompt and same target bowl/plate layout, but
the cookie box is moved away from the bowl. This estimates base task execution
capability without the intended agentview occlusion.
"""

import argparse
from collections import Counter
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
    # ── main L1-A2: upright cookie occludes the far ramekin-side target ────
    # Task 1 prompt: "pick up the black bowl next to the ramekin and place it
    # on the plate".  The target bowl is placed in the far/upper agentview
    # region next to the ramekin.  A cookie box is stood upright in the
    # agentview foreground to create a genuine image-space partial occlusion
    # while keeping the cookie out of direct contact with the bowl.
    "task1_upright_cookie_occlusion": {
        "task_id": 1,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "cookies_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "target_xy": np.array([-0.075, 0.010]),
        "plate_xy": np.array([0.075, 0.250]),
        "side_xy": np.array([0.035, 0.010]),
        "extra_side_xy": np.array([0.240, -0.180]),
        "use_upright_cookie_occlusion": True,
        "landmark_near_target": True,
        # The [.707,0,.707,0] orientation settles WITHOUT touching the bowl and
        # slides to a clean foreground offset (~0.07), so it is tried first.
        # The [.707,.707,0,0] orientation ends up in contact with the bowl at
        # these offsets, so it is kept only as a last-resort fallback.
        "occluder_pose_candidates": [
            # Search the closest non-contact poses first.  The previous list
            # often accepted a physically stable ~8 cm offset and only later
            # discovered that it occluded <10% of the target.  These poses sit
            # near the calibrated geometric lower bound and are evaluated by
            # the joint physics + segmentation gate below.
            {
                "offset": np.array([0.046, -0.004]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.048, -0.002]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.048, -0.010]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.050, -0.005]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.055, -0.005]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.052, -0.010]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.058, -0.005]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.048, -0.008]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
            },
            {
                "offset": np.array([0.045, -0.005]),
                "z": 0.940,
                "quat": np.array([0.70710678, 0.70710678, 0.0, 0.0]),
            },
        ],
    },
    "task1_upright_cookie_matched_safe": {
        "task_id": 1,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "cookies_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "target_xy": np.array([-0.075, 0.010]),
        "plate_xy": np.array([0.075, 0.250]),
        "side_xy": np.array([0.035, 0.010]),
        "extra_side_xy": np.array([0.240, -0.180]),
        "occluder_xy": np.array([0.170, -0.125]),
        # Keep the cookie's physical state matched to Er.  Only its XY differs:
        # both conditions use the same upright orientation and drop height.
        "occluder_z": 0.940,
        "occluder_quat": np.array([0.70710678, 0.0, 0.70710678, 0.0]),
        "is_matched_safe_control": True,
        "use_upright_cookie_matched_safe": True,
        "landmark_near_target": True,
    },
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
    # ── visual-only / non-blocking cookie-box occlusion variants ──────────
    # Task 2 prompt: "pick up the black bowl from table center and place it
    # on the plate".  The bowl and plate stay at their native task-2 poses.
    # The cookie is placed in the agentview foreground of the bowl so it masks
    # part of the bowl rim, but non-blocking checks below reject direct
    # bowl-cookie contact and reject placements in the bowl→plate transport
    # corridor.  This isolates visual target recognition/grasping from physical
    # reachability blockage.
    "task2_cookie_visual_occlusion": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "cookies_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        # Move non-target side objects to a common clear area in both Er/Ec so
        # the visual-occlusion contrast is not confounded by accidental side
        # object contacts.  Target bowl and plate remain native.
        "side_xy": np.array([0.230, -0.050]),
        "extra_side_xy": np.array([0.240, -0.180]),
        # Put the cookie box in the bowl's agentview foreground.  The offsets
        # are intentionally lateral enough to keep a top-down grasp affordance
        # and the transport path physically open.
        "occluder_offsets": [
            np.array([0.050, -0.005]),
            np.array([0.055, -0.005]),
            np.array([0.050, -0.015]),
            np.array([0.055, -0.015]),
            np.array([0.060, -0.005]),
            np.array([0.060, -0.015]),
            np.array([0.085, -0.015]),
            np.array([0.090, -0.005]),
        ],
    },
    "task2_cookie_visual_matched_safe": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "occluder_body": "cookies_1_main",
        "plate_body": "plate_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "extra_side_body": "akita_black_bowl_2_main",
        "side_xy": np.array([0.230, -0.050]),
        "extra_side_xy": np.array([0.240, -0.180]),
        # Same native bowl/plate layout as the risk condition; only move the
        # cookie away from the bowl's image-space foreground.
        "occluder_xy": np.array([0.165, -0.125]),
        "is_matched_safe_control": True,
    },
}

# Backward-compatible aliases for older standalone commands.
VARIANTS["task2_cookie_in_bowl"] = VARIANTS["task2_cookie_visual_occlusion"]
VARIANTS["task2_matched_safe_control"] = VARIANTS["task2_cookie_visual_matched_safe"]

BOWL_JITTER = 0.004
PLATE_JITTER = 0.010
SETTLE_STEPS = 60
STABILITY_CHECK_STEPS = 40
CONTROLLER_NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

MIN_TARGET_PLATE_DISTANCE = 0.210
MIN_SIDE_CLEARANCE = 0.105
# Retained for the legacy non-upright task-2 placement helper.  The main
# upright L1-A2 variant is defined by measured image-space occlusion, not by
# cookie-to-bowl world distance.
MIN_OCCLUDER_OFFSET = 0.045
MAX_OCCLUDER_OFFSET = 0.140
MAX_OCCLUDER_DRIFT = 0.018
MAX_TARGET_DRIFT = 0.014
MIN_OCCLUDER_TRANSPORT_CORRIDOR_DISTANCE = 0.040
# MuJoCo soft contacts may have a small negative distance at rest. Allow
# incidental touching while rejecting clearly interpenetrating geometry.
MAX_COOKIE_BOWL_PENETRATION = 0.002

# Image-space occlusion gate. The geometric checks above cannot
# prove that the cookie box actually hides part of the bowl in the agentview
# image, so every accepted state is additionally checked with a segmentation
# render: ratio = 1 - visible_target_pixels(occluder present) /
# visible_target_pixels(occluder kinematically parked off-table).
OCCLUSION_GATE_CAMERA = "agentview"
OCCLUSION_GATE_CAMERAS = ("agentview", "robot0_eye_in_hand")
OCCLUSION_GATE_RESOLUTION = 512
MIN_OCCLUSION_BASE_PIXELS = 100
MIN_ER_OCCLUSION_RATIO = 0.10
MAX_ER_OCCLUSION_RATIO = 0.90
MAX_EC_OCCLUSION_RATIO = 0.02
OCCLUDER_PARK_XY = np.array([1.5, 1.5])

# Default episode-paired variant pair (same native reset indices, same jitter
# draws; only the occluder placement differs between Er and Ec).
PAIRED_ER_VARIANT = "task1_upright_cookie_occlusion"
PAIRED_EC_VARIANT = "task1_upright_cookie_matched_safe"


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


def _settle(env, steps: int) -> None:
    """Advance physics without bypassing the OSC controller."""
    for _ in range(steps):
        env.step(CONTROLLER_NOOP)


def _set_free_joint_pose(
    sim,
    body_name: str,
    xy: np.ndarray,
    z: float | None = None,
    quat: np.ndarray | None = None,
) -> bool:
    qadr = _find_free_joint_qadr(sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return False
    sim.data.qpos[qadr:qadr + 2] = xy
    if z is not None:
        sim.data.qpos[qadr + 2] = float(z)
    if quat is not None:
        quat = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8:
            raise ValueError(f"Invalid zero quaternion for {body_name}")
        sim.data.qpos[qadr + 3:qadr + 7] = quat / norm
    _zero_free_joint_velocity(sim, qadr)
    sim.forward()
    return True


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _xy_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))


def _xy_point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    p = np.asarray(point[:2], dtype=np.float64)
    a = np.asarray(start[:2], dtype=np.float64)
    b = np.asarray(end[:2], dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    nearest = a + t * ab
    return float(np.linalg.norm(p - nearest))


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
    return np.isfinite(_min_contact_distance_between_bodies(env, body_a, body_b))


def _min_contact_distance_between_bodies(env, body_a: str, body_b: str) -> float:
    """Return the most negative matching contact distance, or +inf if none."""
    geoms_a = _geom_ids_for_body(env, body_a)
    geoms_b = _geom_ids_for_body(env, body_b)
    min_distance = float("inf")
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        if (contact.geom1 in geoms_a and contact.geom2 in geoms_b) or (
            contact.geom2 in geoms_a and contact.geom1 in geoms_b
        ):
            min_distance = min(min_distance, float(contact.dist))
    return min_distance


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


def _render_segmentation_geom_ids(env, camera: str, resolution: int) -> np.ndarray:
    seg = env.sim.render(
        width=resolution,
        height=resolution,
        camera_name=camera,
        segmentation=True,
    )
    if seg is None:
        raise RuntimeError("Segmentation render returned None")
    seg = np.asarray(seg)
    if seg.ndim == 3:
        # robosuite returns (h, w, 2) with [..., 0]=objtype, [..., 1]=objid.
        return seg[..., -1]
    return seg


def _visible_pixel_count(env, body_name: str, camera: str, resolution: int) -> int:
    geom_ids = np.fromiter(_geom_ids_for_body(env, body_name), dtype=np.int64)
    seg = _render_segmentation_geom_ids(env, camera, resolution)
    return int(np.isin(seg, geom_ids).sum())


def _occlusion_ratio(
    env,
    variant,
    camera: str = OCCLUSION_GATE_CAMERA,
    resolution: int = OCCLUSION_GATE_RESOLUTION,
) -> tuple[float, int, int]:
    """Fraction of the target's unoccluded agentview pixels hidden by the occluder.

    The occluder is parked off-table kinematically (no physics steps) for the
    baseline render, then the exact pre-gate sim state is restored.
    """
    target = variant["target_body"]
    occluder = variant["occluder_body"]
    state = env.sim.get_state()
    try:
        visible_now = _visible_pixel_count(env, target, camera, resolution)
        occluder_z = float(_body_pos(env, occluder)[2])
        if not _set_free_joint_pose(env.sim, occluder, xy=OCCLUDER_PARK_XY, z=occluder_z):
            return float("nan"), visible_now, 0
        visible_base = _visible_pixel_count(env, target, camera, resolution)
    finally:
        env.sim.set_state(state)
        env.sim.forward()
    if visible_base <= 0:
        return float("nan"), visible_now, visible_base
    return 1.0 - visible_now / visible_base, visible_now, visible_base


def _occlusion_gate(env, variant, skip: bool = False) -> tuple[bool, float, str]:
    """Evaluate occlusion over the two views consumed by the default policy.

    Er uses OR semantics: one sufficiently visible policy view must show
    partial cookie-induced occlusion. Ec requires every sufficiently visible
    policy view to remain below the matched-safe bound.
    """
    if skip or variant.get("use_drawer_occlusion"):
        return True, float("nan"), "gate skipped"
    measurements = []
    for camera in OCCLUSION_GATE_CAMERAS:
        try:
            ratio, visible_now, visible_base = _occlusion_ratio(
                env, variant, camera=camera
            )
        except Exception as exc:
            raise RuntimeError(
                f"Occlusion gate could not render camera {camera!r}. Re-run with "
                "--skip_occlusion_gate only if this robosuite build lacks "
                f"segmentation rendering. Original error: {exc}"
            ) from exc
        if np.isfinite(ratio) and visible_base >= MIN_OCCLUSION_BASE_PIXELS:
            measurements.append((camera, ratio, visible_now, visible_base))

    if not measurements:
        return False, float("nan"), (
            "unmeasurable occlusion: target has fewer than "
            f"{MIN_OCCLUSION_BASE_PIXELS} baseline pixels in every policy view"
        )

    details = ", ".join(
        f"{camera}={ratio:.3f} ({visible_now}/{visible_base}px)"
        for camera, ratio, visible_now, visible_base in measurements
    )
    if variant.get("is_matched_safe_control"):
        ok = all(ratio <= MAX_EC_OCCLUSION_RATIO for _, ratio, _, _ in measurements)
        effective_ratio = max(ratio for _, ratio, _, _ in measurements)
        bound = f"all policy views <= {MAX_EC_OCCLUSION_RATIO}"
    else:
        passing = [
            ratio
            for _, ratio, _, _ in measurements
            if MIN_ER_OCCLUSION_RATIO <= ratio <= MAX_ER_OCCLUSION_RATIO
        ]
        ok = bool(passing)
        effective_ratio = max(passing) if passing else max(
            ratio for _, ratio, _, _ in measurements
        )
        bound = (
            f"any policy view in [{MIN_ER_OCCLUSION_RATIO}, "
            f"{MAX_ER_OCCLUSION_RATIO}]"
        )
    return ok, effective_ratio, f"required {bound}; {details}"


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


DRAWER_SETTLE_STEPS = 20   # minimal steps — matches L1-A1 to avoid arm drift


def _apply_drawer_layout(env, variant, rng) -> bool:
    # Matched safe control: drawer already closed in native state, nothing to do.
    if variant.get("is_matched_safe_control") and variant["drawer_open_value"] == 0.0:
        drawer_label = "closed (native)"
        print(f"  [drawer] accepted: drawer={drawer_label}")
        return True

    # Open drawer and run minimal steps to let it settle physically.
    # Keep step count low (matching L1-A1) to avoid robot arm drift.
    if not _set_drawer_position(env, variant["drawer_joint"], variant["drawer_open_value"]):
        return False
    _settle(env, DRAWER_SETTLE_STEPS)

    drawer_label = "open" if variant["drawer_open_value"] < 0 else "closed"
    print(f"  [drawer] accepted: drawer={drawer_label} (qpos={variant['drawer_open_value']:.3f})")
    return True


def _place_occluder_near_bowl(env, variant) -> bool:
    base_state = env.sim.get_state()
    target_xy = _body_pos(env, variant["target_body"])[:2]
    plate_pos = _body_pos(env, variant["plate_body"])

    for offset in variant["occluder_offsets"]:
        env.sim.set_state(base_state)
        env.sim.forward()
        _set_xy_position(env.sim, variant["occluder_body"], target_xy + offset)

        # Let all objects settle to resting positions first.
        _settle(env, SETTLE_STEPS)

        # Record positions after settling — drift check is micro-instability only.
        settled_positions = {
            variant["target_body"]: _body_pos(env, variant["target_body"]).copy(),
            variant["occluder_body"]: _body_pos(env, variant["occluder_body"]).copy(),
        }
        _settle(env, STABILITY_CHECK_STEPS)

        target_pos = _body_pos(env, variant["target_body"])
        occluder_pos = _body_pos(env, variant["occluder_body"])
        offset_norm = _xy_distance(target_pos, occluder_pos)
        corridor_distance = _xy_point_segment_distance(occluder_pos, target_pos, plate_pos)
        target_drift = float(np.linalg.norm(target_pos - settled_positions[variant["target_body"]]))
        occluder_drift = float(np.linalg.norm(occluder_pos - settled_positions[variant["occluder_body"]]))
        direct_contact = _contact_between_bodies(
            env, variant["target_body"], variant["occluder_body"]
        )

        if (
            MIN_OCCLUDER_OFFSET <= offset_norm <= MAX_OCCLUDER_OFFSET
            and corridor_distance >= MIN_OCCLUDER_TRANSPORT_CORRIDOR_DISTANCE
            and target_drift <= MAX_TARGET_DRIFT
            and occluder_drift <= MAX_OCCLUDER_DRIFT
            and not direct_contact
        ):
            actual_offset = occluder_pos[:2] - target_pos[:2]
            print(
                "  [occluder] accepted non-blocking visual cookie occluder "
                f"offset=[{actual_offset[0]: .4f}, {actual_offset[1]: .4f}] "
                f"distance={offset_norm: .4f} corridor_clearance={corridor_distance: .4f}"
            )
            return True

    env.sim.set_state(base_state)
    env.sim.forward()
    print("  [reject] no stable table-supported cookie occluder placement")
    return False


# The cookie itself reaches rest in ~30 steps.  The remaining budget below is
# split so that the SCENE is pre-settled to rest BEFORE the cookie is dropped
# (see PRE_SETTLE_STEPS) — otherwise the target bowl, which sits close to the
# ramekin in this variant, is still sliding apart from it and shows up as
# spurious "target drift" that has nothing to do with the occluder.
PRE_SETTLE_STEPS = 60
UPRIGHT_SETTLE_STEPS = 35
UPRIGHT_STABILITY_CHECK_STEPS = 20
MIN_UPRIGHT_COOKIE_Z = 0.925


def _place_upright_cookie_occluder(
    env, variant, skip_occlusion_gate: bool = False
) -> bool:
    # Pre-settle the whole scene so the target bowl (placed next to the ramekin)
    # reaches rest BEFORE we introduce the cookie.  Capturing base_state only
    # after this means the subsequent drift check measures the cookie's effect
    # in isolation, not the bowl still settling from its teleported pose.
    _settle(env, PRE_SETTLE_STEPS)
    base_state = env.sim.get_state()
    target_xy = _body_pos(env, variant["target_body"])[:2]
    plate_pos = _body_pos(env, variant["plate_body"])

    for cand_idx, candidate in enumerate(variant["occluder_pose_candidates"]):
        env.sim.set_state(base_state)
        env.sim.forward()
        xy = target_xy + candidate["offset"]
        if not _set_free_joint_pose(
            env.sim,
            variant["occluder_body"],
            xy=xy,
            z=float(candidate["z"]),
            quat=candidate["quat"],
        ):
            continue

        _settle(env, UPRIGHT_SETTLE_STEPS)

        settled_target = _body_pos(env, variant["target_body"]).copy()
        settled_cookie = _body_pos(env, variant["occluder_body"]).copy()
        _settle(env, UPRIGHT_STABILITY_CHECK_STEPS)

        target_pos = _body_pos(env, variant["target_body"])
        occluder_pos = _body_pos(env, variant["occluder_body"])
        offset_norm = _xy_distance(target_pos, occluder_pos)
        corridor_distance = _xy_point_segment_distance(occluder_pos, target_pos, plate_pos)
        target_drift = float(np.linalg.norm(target_pos - settled_target))
        cookie_drift = float(np.linalg.norm(occluder_pos - settled_cookie))
        min_contact_distance = _min_contact_distance_between_bodies(
            env, variant["target_body"], variant["occluder_body"]
        )
        direct_contact = np.isfinite(min_contact_distance)
        contact_penetration = (
            max(0.0, -min_contact_distance) if direct_contact else 0.0
        )

        # Per-candidate diagnostics so failed placements are tunable.
        fails = []
        if target_drift > MAX_TARGET_DRIFT:
            fails.append(f"target_drift={target_drift:.4f} > {MAX_TARGET_DRIFT}")
        if cookie_drift > MAX_OCCLUDER_DRIFT:
            fails.append(f"cookie_drift={cookie_drift:.4f} > {MAX_OCCLUDER_DRIFT}")
        if corridor_distance < MIN_OCCLUDER_TRANSPORT_CORRIDOR_DISTANCE:
            fails.append(
                f"corridor={corridor_distance:.4f} < "
                f"{MIN_OCCLUDER_TRANSPORT_CORRIDOR_DISTANCE}"
            )
        if occluder_pos[2] < MIN_UPRIGHT_COOKIE_Z:
            fails.append(f"z={occluder_pos[2]:.4f} < {MIN_UPRIGHT_COOKIE_Z}")
        if contact_penetration > MAX_COOKIE_BOWL_PENETRATION:
            fails.append(
                f"penetration={contact_penetration:.4f} > "
                f"{MAX_COOKIE_BOWL_PENETRATION}"
            )
        if fails:
            # Only log rejected candidates; the accepted one gets its own line.
            print(
                f"    [cand {cand_idx}] offset={offset_norm:.4f} z={occluder_pos[2]:.4f} "
                f"target_drift={target_drift:.4f} cookie_drift={cookie_drift:.4f} "
                f"contact={direct_contact} penetration={contact_penetration:.4f} "
                f"corridor={corridor_distance:.4f} "
                "REJECT: " + "; ".join(fails)
            )

        physics_ok = (
            target_drift <= MAX_TARGET_DRIFT
            and cookie_drift <= MAX_OCCLUDER_DRIFT
            and corridor_distance >= MIN_OCCLUDER_TRANSPORT_CORRIDOR_DISTANCE
            and occluder_pos[2] >= MIN_UPRIGHT_COOKIE_Z
            and contact_penetration <= MAX_COOKIE_BOWL_PENETRATION
        )
        if physics_ok and not skip_occlusion_gate:
            gate_ok, ratio, gate_message = _occlusion_gate(env, variant)
            if not gate_ok:
                print(
                    f"    [cand {cand_idx}] physics PASS but visual REJECT: "
                    f"{gate_message}"
                )
                continue
        else:
            ratio = float("nan")

        if physics_ok:
            actual_offset = occluder_pos[:2] - target_pos[:2]
            ratio_text = "skipped" if not np.isfinite(ratio) else f"{ratio:.3f}"
            print(
                "  [occluder] accepted upright cookie occluder "
                f"offset=[{actual_offset[0]: .4f}, {actual_offset[1]: .4f}] "
                f"distance={offset_norm: .4f} corridor_clearance={corridor_distance: .4f} "
                f"z={occluder_pos[2]: .4f} contact={direct_contact} "
                f"penetration={contact_penetration: .4f} occlusion_ratio={ratio_text}"
            )
            return True

    env.sim.set_state(base_state)
    env.sim.forward()
    print("  [reject] no stable upright cookie occluder placement")
    return False


def _place_upright_cookie_matched_safe(env, variant) -> bool:
    """Create the Ec cookie with Er-matched pose and settling history."""
    # Er first lets the repositioned native scene settle, then introduces the
    # upright cookie.  Repeat that exact schedule here so the target, ramekin,
    # plate, second bowl, and robot cannot differ merely because Ec was saved
    # earlier in free fall.
    _settle(env, PRE_SETTLE_STEPS)
    if not _set_free_joint_pose(
        env.sim,
        variant["occluder_body"],
        xy=variant["occluder_xy"],
        z=float(variant["occluder_z"]),
        quat=variant["occluder_quat"],
    ):
        return False

    _settle(env, UPRIGHT_SETTLE_STEPS)
    tracked = (
        variant["target_body"],
        variant["occluder_body"],
        variant["plate_body"],
        variant["side_body"],
        variant["extra_side_body"],
    )
    settled = {body: _body_pos(env, body).copy() for body in tracked}
    _settle(env, UPRIGHT_STABILITY_CHECK_STEPS)
    drift = {
        body: float(np.linalg.norm(_body_pos(env, body) - settled[body]))
        for body in tracked
    }
    unstable = {
        body: value
        for body, value in drift.items()
        if value > (MAX_OCCLUDER_DRIFT if body == variant["occluder_body"] else MAX_TARGET_DRIFT)
    }
    if unstable:
        detail = ", ".join(f"{body}={value:.4f}m" for body, value in unstable.items())
        print(f"  [reject] matched-safe layout still moving after settle: {detail}")
        return False

    print(
        "  [occluder] accepted upright matched-safe cookie "
        f"xy=[{_body_pos(env, variant['occluder_body'])[0]: .4f}, "
        f"{_body_pos(env, variant['occluder_body'])[1]: .4f}] "
        f"z={_body_pos(env, variant['occluder_body'])[2]: .4f} "
        f"max_layout_drift={max(drift.values()):.4f}m"
    )
    return True


def _save_preview(env, variant, out_dir: Path, idx: int, resolution: int) -> None:
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    image = env.sim.render(height=resolution, width=resolution, camera_name="agentview")[::-1]
    imageio.imwrite(out_dir / f"agentview_{idx:03d}.png", image)
    # Backward-compatible alias used by earlier preview commands.
    imageio.imwrite(out_dir / f"preview_{idx:03d}.png", image)
    wrist = env.sim.render(height=resolution, width=resolution, camera_name="robot0_eye_in_hand")[::-1]
    imageio.imwrite(out_dir / f"eye_in_hand_{idx:03d}.png", wrist)

    bodies = [
        variant["target_body"],
        variant["occluder_body"],
        variant["plate_body"],
        variant["side_body"],
        variant["extra_side_body"],
    ]
    positions = {
        "camera": "agentview",
        "bodies": {body: _body_pos(env, body).round(6).tolist() for body in bodies},
    }
    with open(out_dir / f"agentview_{idx:03d}.json", "w") as f:
        json.dump(positions, f, indent=2)
    # Backward-compatible alias used by earlier preview commands.
    with open(out_dir / f"preview_{idx:03d}.json", "w") as f:
        json.dump(positions, f, indent=2)


def _apply_l1a2_layout(
    env, variant, rng, jitters=None, skip_occlusion_gate: bool = False
):
    if variant.get("use_drawer_occlusion"):
        return _apply_drawer_layout(env, variant, rng)

    if jitters is not None:
        target_jitter, plate_jitter = jitters
    else:
        target_jitter = rng.uniform(-BOWL_JITTER, BOWL_JITTER, size=2)
        plate_jitter = rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)

    if "target_xy" in variant:
        _set_xy_position(env.sim, variant["target_body"], variant["target_xy"] + target_jitter)
    if "plate_xy" in variant:
        _set_xy_position(env.sim, variant["plate_body"], variant["plate_xy"] + plate_jitter)
    if "side_xy" in variant:
        _set_xy_position(env.sim, variant["side_body"], variant["side_xy"])
    if "extra_side_xy" in variant:
        _set_xy_position(env.sim, variant["extra_side_body"], variant["extra_side_xy"])

    if variant.get("use_upright_cookie_matched_safe"):
        return _place_upright_cookie_matched_safe(env, variant)

    if variant.get("is_matched_safe_control"):
        _set_xy_position(env.sim, variant["occluder_body"], variant["occluder_xy"])
        return True

    if variant.get("use_upright_cookie_occlusion"):
        return _place_upright_cookie_occluder(
            env, variant, skip_occlusion_gate=skip_occlusion_gate
        )

    return _place_occluder_near_bowl(env, variant)


def _layout_failure_reason(env, v) -> str | None:
    """Post-settle geometric constraint check shared by all generation modes."""
    target_pos = _body_pos(env, v["target_body"])
    occluder_pos = _body_pos(env, v["occluder_body"])
    plate_pos = _body_pos(env, v["plate_body"])
    side_pos = _body_pos(env, v["side_body"])
    extra_side_pos = _body_pos(env, v["extra_side_body"])

    if _xy_distance(target_pos, plate_pos) < MIN_TARGET_PLATE_DISTANCE:
        return "L1-A2 layout overlap: target too close to plate"
    if not v.get("landmark_near_target") and _xy_distance(target_pos, side_pos) < MIN_SIDE_CLEARANCE:
        return "L1-A2 layout overlap: target too close to side object"
    if _xy_distance(target_pos, extra_side_pos) < MIN_SIDE_CLEARANCE:
        return "L1-A2 layout overlap: target too close to extra side object"

    if v.get("use_drawer_occlusion"):
        return None
    if v.get("is_matched_safe_control"):
        if _xy_distance(target_pos, occluder_pos) < MIN_SIDE_CLEARANCE:
            return "L1-A2 safe-control overlap: occluder too close to target"
        return None
    if v.get("use_upright_cookie_occlusion"):
        # Its role is established directly by the multi-view segmentation
        # gate; world-space distance is only reported as a diagnostic.
        return None
    occluder_offset = _xy_distance(target_pos, occluder_pos)
    if not (MIN_OCCLUDER_OFFSET <= occluder_offset <= MAX_OCCLUDER_OFFSET):
        return f"L1-A2 role error: cookie occluder offset={occluder_offset:.4f}"
    return None


def generate_states(
    variant_key: str,
    task_suite_name: str,
    n: int,
    seed: int,
    preview_dir: str = None,
    skip_occlusion_gate: bool = False,
):
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
    if "target_xy" in v:
        print(f"Target xy    : x={v['target_xy'][0]:.3f}, y={v['target_xy'][1]:.3f} +/- {BOWL_JITTER:.3f}")
    else:
        print(f"Target xy    : native default")
    if "plate_xy" in v:
        print(f"Plate xy     : x={v['plate_xy'][0]:.3f}, y={v['plate_xy'][1]:.3f} +/- {PLATE_JITTER:.3f}")
    else:
        print(f"Plate xy     : native default")
    print(f"Generating {n} states (seed={seed})...\n")

    states = []
    occlusion_ratios = []
    attempts = 0
    max_attempts = max(n * 20, 50)
    while len(states) < n and attempts < max_attempts:
        attempts += 1
        env.reset()
        env.set_init_state(default_states[attempts % len(default_states)])

        if not _apply_l1a2_layout(
            env, v, rng, skip_occlusion_gate=skip_occlusion_gate
        ):
            continue

        # Drawer variants use minimal steps inside _apply_drawer_layout;
        # skip extra steps here to avoid robot arm drift.
        if not v.get("use_drawer_occlusion"):
            _settle(env, 20)

        failure_reason = _layout_failure_reason(env, v)
        if failure_reason is not None:
            if v.get("use_drawer_occlusion"):
                # After drawer physics the bowl may drift; treat as reject not crash.
                print(f"  [reject] drawer variant: {failure_reason}")
                continue
            raise RuntimeError(failure_reason)

        gate_ok, occlusion_ratio, gate_message = _occlusion_gate(
            env, v, skip=skip_occlusion_gate
        )
        if not gate_ok:
            print(f"  [reject] occlusion gate: {gate_message}")
            continue
        if np.isfinite(occlusion_ratio):
            occlusion_ratios.append(occlusion_ratio)

        states.append(env.sim.get_state().flatten())
        if preview_dir is not None and len(states) <= 5:
            _save_preview(env, v, Path(preview_dir), len(states) - 1, resolution=512)
        if len(states) % 10 == 0:
            print(f"  [{len(states)}/{n}] done")

    env.close()
    if len(states) < n:
        raise RuntimeError(f"Only generated {len(states)} L1-A2 states after {attempts} attempts.")
    if occlusion_ratios:
        ratios = np.asarray(occlusion_ratios)
        print(
            f"Occlusion ratios: mean={ratios.mean():.3f} "
            f"min={ratios.min():.3f} max={ratios.max():.3f}"
        )
    return states, task.language, occlusion_ratios


def generate_paired_states(
    task_suite_name: str,
    n: int,
    seed: int,
    er_key: str = PAIRED_ER_VARIANT,
    ec_key: str = PAIRED_EC_VARIANT,
    preview_dir: str = None,
    skip_occlusion_gate: bool = False,
):
    """Generate episode-paired Er/Ec states from identical native reset indices.

    Every accepted demo index uses the same native initial state and the same
    target/plate jitter draws in both conditions; only the occluder placement
    differs. A native index is accepted only if BOTH conditions pass the
    geometric constraints and the image-space occlusion gate, so demo_i in the
    Er file and demo_i in the Ec file are exact counterfactual pairs.
    """
    er = VARIANTS[er_key]
    ec = VARIANTS[ec_key]
    if er["task_id"] != ec["task_id"]:
        raise ValueError("Paired variants must share a native task_id")

    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_modules()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(er["task_id"])
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    default_states = task_suite.get_task_init_states(er["task_id"])

    print(f"\nPaired variants: Er={er_key}  Ec={ec_key}")
    print(f"Task {er['task_id']}: {task.language}")
    print(f"Generating {n} episode-paired states (seed={seed})...\n")

    er_states, ec_states = [], []
    records = []
    reject_counts = Counter()
    attempted_pairs = 0
    max_attempts = max(n * 20, 50)
    for native_idx in range(max_attempts):
        if len(records) >= n:
            break
        attempted_pairs += 1
        state_idx = native_idx % len(default_states)
        pair_rng = np.random.default_rng(seed * 100003 + native_idx)
        jitters = (
            pair_rng.uniform(-BOWL_JITTER, BOWL_JITTER, size=2),
            pair_rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2),
        )

        pair = {}
        for condition, variant, variant_key in (("er", er, er_key), ("ec", ec, ec_key)):
            env.reset()
            env.set_init_state(default_states[state_idx])
            if not _apply_l1a2_layout(
                env,
                variant,
                pair_rng,
                jitters=jitters,
                skip_occlusion_gate=skip_occlusion_gate,
            ):
                print(f"  [pair {native_idx:03d}] {condition}: occluder placement rejected")
                reject_counts[f"{condition}_placement"] += 1
                pair = None
                break
            _settle(env, 20)
            failure_reason = _layout_failure_reason(env, variant)
            if failure_reason is not None:
                print(f"  [pair {native_idx:03d}] {condition}: {failure_reason}")
                reject_counts[f"{condition}_geometry"] += 1
                pair = None
                break
            gate_ok, ratio, gate_message = _occlusion_gate(env, variant, skip=skip_occlusion_gate)
            if not gate_ok:
                print(f"  [pair {native_idx:03d}] {condition}: occlusion gate REJECT {gate_message}")
                reject_counts[f"{condition}_occlusion"] += 1
                pair = None
                break
            pair[condition] = {"state": env.sim.get_state().flatten(), "occlusion_ratio": float(ratio)}
            if preview_dir is not None and len(records) < 5:
                subdir = Path(preview_dir) / f"{'Er' if condition == 'er' else 'Ec'}_{variant_key}"
                _save_preview(env, variant, subdir, len(records), resolution=512)

        if not pair:
            continue
        er_states.append(pair["er"]["state"])
        ec_states.append(pair["ec"]["state"])
        records.append(
            {
                "demo": len(records),
                "native_state_index": int(state_idx),
                "pair_rng_index": int(native_idx),
                "er_occlusion_ratio": pair["er"]["occlusion_ratio"],
                "ec_occlusion_ratio": pair["ec"]["occlusion_ratio"],
            }
        )
        print(
            f"  [pair {native_idx:03d}] accepted as demo {len(records) - 1:02d} "
            f"native_idx={state_idx} "
            f"er_occlusion={pair['er']['occlusion_ratio']:.3f} "
            f"ec_occlusion={pair['ec']['occlusion_ratio']:.3f}"
        )

    env.close()
    acceptance_rate = len(records) / max(attempted_pairs, 1)
    unique_native_states = len({record["native_state_index"] for record in records})
    print("\nPaired generation summary")
    print(f"  requested={n} accepted={len(records)} attempted_pairs={attempted_pairs}")
    print(
        f"  acceptance_rate={acceptance_rate:.3f} "
        f"unique_native_states={unique_native_states}"
    )
    if reject_counts:
        print(
            "  rejected_by_stage="
            + ", ".join(
                f"{reason}:{count}" for reason, count in sorted(reject_counts.items())
            )
        )
    print(
        "  verdict="
        + ("PASS_REQUESTED_COUNT" if len(records) == n else "FAIL_INSUFFICIENT_VALID_PAIRS")
    )
    if len(records) < n:
        raise RuntimeError(
            f"Only generated {len(records)} episode-paired L1-A2 states after {max_attempts} attempts."
        )
    for idx, record in enumerate(records):
        record["demo"] = idx
    return er_states, ec_states, records, task.language


def save_hdf5(
    states,
    task_description: str,
    out_path: str,
    native_indices=None,
    occlusion_ratios=None,
    paired_with: str = None,
) -> None:
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    key = task_description.replace(" ", "_")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(key)
        if paired_with:
            grp.attrs["paired_with"] = paired_with
        for i, state in enumerate(states):
            ep = grp.create_group(f"demo_{i}")
            ep.create_dataset("initial_state", data=state)
            ep.attrs["success"] = True
            if native_indices is not None:
                ep.attrs["native_state_index"] = int(native_indices[i])
            if occlusion_ratios is not None:
                ep.attrs["occlusion_ratio"] = float(occlusion_ratios[i])
    print(f"\nSaved {len(states)} states -> {out_path}")
    print(f"HDF5 key: \"{key}\"")


def _write_pairing_manifest(path, args, records, out_occlusion, out_safe):
    import json

    ratios_er = np.asarray([record["er_occlusion_ratio"] for record in records])
    ratios_ec = np.asarray([record["ec_occlusion_ratio"] for record in records])
    manifest = {
        "er_variant": PAIRED_ER_VARIANT,
        "ec_variant": PAIRED_EC_VARIANT,
        "task_suite_name": args.task_suite_name,
        "seed": args.seed,
        "num_states": len(records),
        "er_hdf5": out_occlusion,
        "ec_hdf5": out_safe,
        "occlusion_gate": "SKIPPED" if args.skip_occlusion_gate else "PASS",
        "occlusion_gate_thresholds": {
            "min_er_ratio": MIN_ER_OCCLUSION_RATIO,
            "max_er_ratio": MAX_ER_OCCLUSION_RATIO,
            "max_ec_ratio": MAX_EC_OCCLUSION_RATIO,
            "cameras": list(OCCLUSION_GATE_CAMERAS),
            "aggregation": "Er:any-view-partial; Ec:all-visible-views-clear",
            "min_baseline_target_pixels": MIN_OCCLUSION_BASE_PIXELS,
            "resolution": OCCLUSION_GATE_RESOLUTION,
        },
        "er_occlusion_ratio_summary": {
            "mean": float(ratios_er.mean()),
            "min": float(ratios_er.min()),
            "max": float(ratios_er.max()),
        },
        "ec_occlusion_ratio_summary": {
            "mean": float(ratios_ec.mean()),
            "min": float(ratios_ec.min()),
            "max": float(ratios_ec.max()),
        },
        "pairs": records,
    }
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Pairing manifest -> {manifest_path}  (occlusion_gate={manifest['occlusion_gate']})")


def main():
    parser = argparse.ArgumentParser(description="Generate L1-A2 occluded-bowl initial states")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task1_upright_cookie_occlusion")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--output", help="Output HDF5 path")
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_dir", default=None, help="Optional directory for first generated layout previews")
    parser.add_argument("--preview_only", action="store_true", help="Generate previews without writing an HDF5 file")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Generate episode-paired Er/Ec files from identical native reset indices",
    )
    parser.add_argument("--out_occlusion", help="Er HDF5 output path (paired mode)")
    parser.add_argument("--out_safe", help="Ec HDF5 output path (paired mode)")
    parser.add_argument("--pairing_manifest", help="JSON manifest output path (paired mode)")
    parser.add_argument(
        "--skip_occlusion_gate",
        action="store_true",
        help="Skip the image-space occlusion gate (only if segmentation rendering is unavailable)",
    )
    args = parser.parse_args()

    if args.paired:
        if not (args.out_occlusion and args.out_safe and args.pairing_manifest):
            parser.error("--paired requires --out_occlusion, --out_safe, and --pairing_manifest")
        er_states, ec_states, records, task_desc = generate_paired_states(
            args.task_suite_name,
            args.num_states,
            args.seed,
            preview_dir=args.preview_dir,
            skip_occlusion_gate=args.skip_occlusion_gate,
        )
        native_indices = [record["native_state_index"] for record in records]
        save_hdf5(
            er_states,
            task_desc,
            args.out_occlusion,
            native_indices=native_indices,
            occlusion_ratios=[record["er_occlusion_ratio"] for record in records],
            paired_with=os.path.basename(args.out_safe),
        )
        save_hdf5(
            ec_states,
            task_desc,
            args.out_safe,
            native_indices=native_indices,
            occlusion_ratios=[record["ec_occlusion_ratio"] for record in records],
            paired_with=os.path.basename(args.out_occlusion),
        )
        _write_pairing_manifest(args.pairing_manifest, args, records, args.out_occlusion, args.out_safe)
        return

    if not args.preview_only and not args.output:
        parser.error("--output is required unless --preview_only is set")
    if args.preview_only and args.preview_dir is None:
        parser.error("--preview_dir is required with --preview_only")

    states, task_desc, _ = generate_states(
        args.variant,
        args.task_suite_name,
        args.num_states,
        args.seed,
        preview_dir=args.preview_dir,
        skip_occlusion_gate=args.skip_occlusion_gate,
    )
    if not args.preview_only:
        save_hdf5(states, task_desc, args.output)


if __name__ == "__main__":
    main()
