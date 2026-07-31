"""Paired native-only pipeline for L1-A4 two-landmark attribution.

The selected native prompt is ``pick up the black bowl between the plate and
the ramekin and place it on the plate``. EB is the exact native serialized
state. In ER and EC, the native target bowl, plate, and ramekin undergo the
same planar rigid transform while preserving the unique between relation.
Only ER puts the second native black bowl at the paired EB target pose; EC
returns that bowl to its native initialization region.

No BDDL, prompt, asset, camera, or task-goal modification is performed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import DepthDisambiguationOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.validate_l1a4_spatial_native_preflight import (
    FORMAL_WAIT_STEPS,
    INTERVENTION_ID,
    MAX_RECEPTACLE_TILT_DEG,
    PHYSICAL_GATE_VERDICT,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    VERDICT as PREFLIGHT_VERDICT,
    resolve_native_bddl,
    validate_native_task,
    verify_state_file,
    write_preflight,
)


TARGET = "akita_black_bowl_1_main"
LURE = "akita_black_bowl_2_main"
PLATE = "plate_1_main"
RAMEKIN = "glazed_rim_porcelain_ramekin_1_main"
COOKIES = "cookies_1_main"
CABINET = "wooden_cabinet_1_main"
STOVE = "flat_stove_1_main"

BOWLS = (TARGET, LURE)
LANDMARKS = (PLATE, RAMEKIN)
MOVABLE_BODIES = BOWLS + LANDMARKS
VISUAL_REFERENTS = MOVABLE_BODIES
TRACKED_BODIES = MOVABLE_BODIES + (COOKIES, CABINET, STOVE)
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

MIN_BETWEEN_PROJECTION = 0.25
MAX_BETWEEN_PROJECTION = 0.75
MAX_BETWEEN_PERPENDICULAR_M = 0.050
# These coarse center-distance margins are secondary to the exact no-contact,
# post-wait tilt, velocity, support, and policy-mask gates below. The native
# Akita bowl has an approximately 0.094 m rim diameter, so 0.110 m still
# provides geometric separation without rejecting stable states solely due to
# native EB pose jitter.
MIN_LURE_RELATION_DISTANCE_M = 0.100
MIN_BOWL_DISTANCE_M = 0.110
MAX_LAYOUT_POSITION_ERROR_M = 0.020
MAX_TRANSIENT_WAIT_TILT_DEG = 10.0
MAX_POST_WAIT_LINEAR_SPEED_M_S = 1e-4
MAX_POST_WAIT_ANGULAR_SPEED_RAD_S = 1e-3
MAX_CONFIRM_DRIFT_M = 1e-4
FORMAL_CONFIRM_STEPS = 5
MIN_VISIBLE_PIXELS = 60
MIN_MASK_CENTROID_SEPARATION_PX = 12.0
POLICY_RESOLUTION = 256
SETTLE_STEPS = 80
STABILITY_CONFIRM_STEPS = 40
PAIR_TOLERANCE = 1e-10

# Centers of the selected native BDDL initialization regions.
NATIVE_TARGET_XY = np.array([-0.05, 0.20])
NATIVE_PLATE_XY = np.array([0.06, 0.20])
NATIVE_RAMEKIN_XY = np.array([-0.20, 0.20])
NATIVE_LURE_XY = np.array([-0.18, 0.32])

# Ordered, preregistered planar rigid transforms. The retired v4 translation
# placed the target bowl close enough to the native cookies object that the
# two contacted during the evaluator's ten-step no-op wait and the bowl tipped
# in place. These transforms move the unchanged native relation up and rotate
# it away from both native lure locations while keeping every Ec referent
# within 0.170 m of its native BDDL region center.
#
# Each tuple is (translation_x_m, translation_y_m, yaw_deg). Generation accepts
# the first candidate that passes the exact formal-reset/post-wait physical and
# policy-view gates for that native robot state.
RELATION_TRANSFORM_CANDIDATES = (
    (0.1275, 0.0400, -22.0),
    (0.1250, 0.0450, -22.0),
    (0.1225, 0.0475, -22.0),
    (0.1200, 0.0500, -22.0),
    (0.1175, 0.0525, -22.0),
    (0.1150, 0.0550, -22.0),
    (0.1180, 0.0470, -24.0),
    (0.1200, 0.0475, -23.0),
    (0.1175, 0.0500, -23.0),
    (0.1150, 0.0525, -23.0),
    (0.1125, 0.0550, -23.0),
    (0.1175, 0.0550, -21.0),
)
EC_LURE_XY = NATIVE_LURE_XY.copy()

# EC is a benign capability control, so every task-relevant movable body must
# remain close to the center of its native initialization region after
# settling. The largest preregistered rigid-transform displacement is below
# 0.170 m, leaving only a small allowance while rejecting retired layouts.
MAX_EC_NATIVE_CENTER_DISPLACEMENT_M = 0.17
NATIVE_REGION_CENTERS = {
    TARGET: NATIVE_TARGET_XY,
    LURE: NATIVE_LURE_XY,
    PLATE: NATIVE_PLATE_XY,
    RAMEKIN: NATIVE_RAMEKIN_XY,
}


def _relation_positions(
    transform: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    dx, dy, yaw_deg = transform
    yaw = np.deg2rad(float(yaw_deg))
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=float,
    )
    target = NATIVE_TARGET_XY + np.array([dx, dy], dtype=float)
    return {
        TARGET: target,
        PLATE: target + rotation @ (NATIVE_PLATE_XY - NATIVE_TARGET_XY),
        RAMEKIN: (
            target + rotation @ (NATIVE_RAMEKIN_XY - NATIVE_TARGET_XY)
        ),
    }


def _ensure_libero_importable() -> None:
    try:
        from libero.libero import benchmark  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    root = resolve_native_bddl().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _task_and_suite():
    _ensure_libero_importable()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID)).resolve(strict=True)
    validate_native_task(resolve_native_bddl(), bddl, task.language)
    if task.language != TASK_PROMPT or Path(task.bddl_file).name != TASK_FILE:
        raise RuntimeError(
            "LIBERO task map mismatch for L1-A4 spatial: "
            f"id={TASK_ID}, prompt={task.language!r}, bddl={task.bddl_file!r}"
        )
    return suite, task, bddl


def _load_native_init_states(task):
    import torch
    from libero.libero import get_libero_path

    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    ).resolve(strict=True)
    if path.name != task.init_states_file or path.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native LIBERO init-state source: {path}")
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _env(bddl: Path, *, render: bool = True):
    _ensure_libero_importable()
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=POLICY_RESOLUTION,
        camera_widths=POLICY_RESOLUTION,
        hard_reset=False,
    )


def _body_pos(env, body: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _body_tilt_deg(env, body: str) -> float:
    body_id = env.sim.model.body_name2id(body)
    quat = np.asarray(env.sim.data.body_xquat[body_id], dtype=float)
    _, x, y, _ = quat
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def _free_joint_addresses(sim, body: str) -> tuple[int, int]:
    body_id = sim.model.body_name2id(body)
    for joint_id in range(sim.model.njnt):
        if (
            int(sim.model.jnt_bodyid[joint_id]) == body_id
            and int(sim.model.jnt_type[joint_id]) == 0
        ):
            return (
                int(sim.model.jnt_qposadr[joint_id]),
                int(sim.model.jnt_dofadr[joint_id]),
            )
    raise RuntimeError(f"No native free joint for {body}")


def _set_xy(sim, body: str, xy: np.ndarray) -> None:
    qadr, dadr = _free_joint_addresses(sim, body)
    sim.data.qpos[qadr : qadr + 2] = np.asarray(xy, dtype=float)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    sim.forward()


def _capture_free_joint(
    sim, body: str
) -> tuple[np.ndarray, np.ndarray]:
    qadr, dadr = _free_joint_addresses(sim, body)
    return (
        np.asarray(sim.data.qpos[qadr : qadr + 7], dtype=float).copy(),
        np.asarray(sim.data.qvel[dadr : dadr + 6], dtype=float).copy(),
    )


def _body_twist(env, body: str) -> tuple[float, float]:
    _, dadr = _free_joint_addresses(env.sim, body)
    velocity = np.asarray(
        env.sim.data.qvel[dadr : dadr + 6], dtype=float
    )
    return (
        float(np.linalg.norm(velocity[:3])),
        float(np.linalg.norm(velocity[3:])),
    )


def _transplant(
    env,
    base_state,
    poses: Mapping[str, tuple[np.ndarray, np.ndarray]],
):
    env.set_init_state(base_state)
    for body, (qpos, _qvel) in poses.items():
        qadr, dadr = _free_joint_addresses(env.sim, body)
        env.sim.data.qpos[qadr : qadr + 7] = qpos
        env.sim.data.qvel[dadr : dadr + 6] = 0.0
    env.sim.forward()
    return env.sim.get_state().flatten()


def _settled_variant(
    env, base_state, positions: Mapping[str, np.ndarray]
):
    env.set_init_state(base_state)
    for body, xy in positions.items():
        _set_xy(env.sim, body, xy)
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    first = {body: _body_pos(env, body) for body in MOVABLE_BODIES}
    for _ in range(STABILITY_CONFIRM_STEPS):
        env.sim.step()
    drift = {
        body: float(np.linalg.norm(_body_pos(env, body) - first[body]))
        for body in MOVABLE_BODIES
    }
    poses = {
        body: _capture_free_joint(env.sim, body) for body in MOVABLE_BODIES
    }
    return _transplant(env, base_state, poses), drift


def _geom_ids_for_body(env, body: str) -> set[int]:
    body_id = env.sim.model.body_name2id(body)
    body_ids = {int(body_id)}
    changed = True
    while changed:
        changed = False
        for candidate in range(env.sim.model.nbody):
            if (
                int(env.sim.model.body_parentid[candidate]) in body_ids
                and candidate not in body_ids
            ):
                body_ids.add(candidate)
                changed = True
    return {
        geom_id
        for geom_id in range(env.sim.model.ngeom)
        if int(env.sim.model.geom_bodyid[geom_id]) in body_ids
    }


def _negative_contact_between(env, first: str, second: str) -> bool:
    return _negative_contact_between_geom_sets(
        env,
        _geom_ids_for_body(env, first),
        _geom_ids_for_body(env, second),
    )


def _negative_contact_between_geom_sets(
    env, first_geoms: set[int], second_geoms: set[int]
) -> bool:
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if float(getattr(contact, "dist", -1.0)) >= 0.0:
            continue
        if (
            contact.geom1 in first_geoms
            and contact.geom2 in second_geoms
        ) or (
            contact.geom2 in first_geoms
            and contact.geom1 in second_geoms
        ):
            return True
    return False


def _contact_between_geom_sets(
    env, first_geoms: set[int], second_geoms: set[int]
) -> bool:
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if (
            contact.geom1 in first_geoms
            and contact.geom2 in second_geoms
        ) or (
            contact.geom2 in first_geoms
            and contact.geom1 in second_geoms
        ):
            return True
    return False


def _robot_geom_ids(env) -> set[int]:
    geom_ids = set()
    for geom_id in range(env.sim.model.ngeom):
        body_id = int(env.sim.model.geom_bodyid[geom_id])
        body_name = env.sim.model.body_id2name(body_id) or ""
        lowered = body_name.lower()
        if (
            body_name.startswith(("robot0_", "gripper0_"))
            or "finger" in lowered
            or "hand" in lowered
            or "eef" in lowered
        ):
            geom_ids.add(geom_id)
    return geom_ids


def _forbidden_contact_pairs(env) -> list[str]:
    contact_bodies = MOVABLE_BODIES + (COOKIES, CABINET, STOVE)
    pairs = []
    for first_index, first in enumerate(contact_bodies):
        for second in contact_bodies[first_index + 1 :]:
            if _negative_contact_between(env, first, second):
                pairs.append(f"{first}/{second}")
    robot_geoms = _robot_geom_ids(env)
    for body in MOVABLE_BODIES:
        if _negative_contact_between_geom_sets(
            env, robot_geoms, _geom_ids_for_body(env, body)
        ):
            pairs.append(f"robot/{body}")
    return pairs


def _physical_snapshot(env) -> dict[str, dict[str, object]]:
    return {
        body: {
            "position": _body_pos(env, body),
            "tilt_deg": _body_tilt_deg(env, body),
            "linear_speed_m_s": _body_twist(env, body)[0],
            "angular_speed_rad_s": _body_twist(env, body)[1],
        }
        for body in MOVABLE_BODIES
    }


def _serializable_snapshot(
    snapshot: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        body: {
            name: (
                np.asarray(value, dtype=float).round(9).tolist()
                if name == "position"
                else float(value)
            )
            for name, value in values.items()
        }
        for body, values in snapshot.items()
    }


def _check_receptacle_tilts(
    condition: str,
    phase: str,
    tilts: Mapping[str, float],
) -> None:
    excessive = {
        body: float(tilt)
        for body, tilt in tilts.items()
        if float(tilt) > MAX_RECEPTACLE_TILT_DEG
    }
    if excessive:
        raise RuntimeError(
            f"{condition}: {phase} receptacle tilt exceeds "
            f"{MAX_RECEPTACLE_TILT_DEG:.1f}deg: {excessive}"
        )


def _formal_policy_state_gate(
    env, state, condition: str
) -> dict[str, object]:
    """Replay the evaluator reset/wait and gate its first policy frame."""
    env.reset()
    env.set_init_state(state)
    env.sim.forward()
    pre_wait = _physical_snapshot(env)
    max_tilt = {
        body: float(values["tilt_deg"])
        for body, values in pre_wait.items()
    }

    wait_drift_origin = {
        body: np.asarray(values["position"], dtype=float).copy()
        for body, values in pre_wait.items()
    }
    for step in range(1, FORMAL_WAIT_STEPS + 1):
        env.step(NOOP)
        forbidden = _forbidden_contact_pairs(env)
        if forbidden:
            raise RuntimeError(
                f"{condition}: forbidden contact during formal wait "
                f"step {step}: {forbidden}"
            )
        step_tilts = {
            body: _body_tilt_deg(env, body) for body in MOVABLE_BODIES
        }
        for body, tilt in step_tilts.items():
            max_tilt[body] = max(max_tilt[body], tilt)
        excessive_transient = {
            body: tilt
            for body, tilt in step_tilts.items()
            if tilt > MAX_TRANSIENT_WAIT_TILT_DEG
        }
        if excessive_transient:
            raise RuntimeError(
                f"{condition}: formal-wait step {step} transient tilt "
                f"exceeds {MAX_TRANSIENT_WAIT_TILT_DEG:.1f}deg: "
                f"{excessive_transient}"
            )

    first_policy = _physical_snapshot(env)
    _check_receptacle_tilts(
        condition,
        "first-policy-frame",
        {
            body: float(values["tilt_deg"])
            for body, values in first_policy.items()
        },
    )
    excessive_linear = {
        body: float(values["linear_speed_m_s"])
        for body, values in first_policy.items()
        if float(values["linear_speed_m_s"])
        > MAX_POST_WAIT_LINEAR_SPEED_M_S
    }
    excessive_angular = {
        body: float(values["angular_speed_rad_s"])
        for body, values in first_policy.items()
        if float(values["angular_speed_rad_s"])
        > MAX_POST_WAIT_ANGULAR_SPEED_RAD_S
    }
    if excessive_linear or excessive_angular:
        raise RuntimeError(
            f"{condition}: first-policy-frame velocity is unstable: "
            f"linear={excessive_linear}, angular={excessive_angular}"
        )

    table_geoms = _geom_ids_for_body(env, "table")
    unsupported = [
        body
        for body in MOVABLE_BODIES
        if not _contact_between_geom_sets(
            env, _geom_ids_for_body(env, body), table_geoms
        )
    ]
    if unsupported:
        raise RuntimeError(
            f"{condition}: first-policy-frame objects lack table support: "
            f"{unsupported}"
        )

    policy_stats = _mask_stats(env, "agentview")
    for body, body_stats in policy_stats.items():
        if int(body_stats["pixels"]) < MIN_VISIBLE_PIXELS:
            raise RuntimeError(
                f"{condition}: first-policy-frame {body} only "
                f"{body_stats['pixels']} policy-view pixels"
            )
    centroids = [
        np.asarray(policy_stats[body]["centroid"], dtype=float)
        for body in VISUAL_REFERENTS
    ]
    centroid_separation = min(
        float(np.linalg.norm(centroids[first] - centroids[second]))
        for first in range(len(centroids))
        for second in range(first + 1, len(centroids))
    )
    if centroid_separation < MIN_MASK_CENTROID_SEPARATION_PX:
        raise RuntimeError(
            f"{condition}: first-policy-frame referent mask centroid "
            f"separation={centroid_separation:.1f}px"
        )

    first_policy_positions = {
        body: np.asarray(values["position"], dtype=float).copy()
        for body, values in first_policy.items()
    }
    for step in range(1, FORMAL_CONFIRM_STEPS + 1):
        env.step(NOOP)
        forbidden = _forbidden_contact_pairs(env)
        if forbidden:
            raise RuntimeError(
                f"{condition}: forbidden contact during post-wait "
                f"confirmation step {step}: {forbidden}"
            )
        step_tilts = {
            body: _body_tilt_deg(env, body) for body in MOVABLE_BODIES
        }
        for body, tilt in step_tilts.items():
            max_tilt[body] = max(max_tilt[body], tilt)
        _check_receptacle_tilts(
            condition, f"post-wait confirmation step {step}", step_tilts
        )

    confirmed = _physical_snapshot(env)
    confirm_drift = {
        body: float(
            np.linalg.norm(
                np.asarray(values["position"], dtype=float)
                - first_policy_positions[body]
            )
        )
        for body, values in confirmed.items()
    }
    excessive_drift = {
        body: drift
        for body, drift in confirm_drift.items()
        if drift > MAX_CONFIRM_DRIFT_M
    }
    if excessive_drift:
        raise RuntimeError(
            f"{condition}: post-wait confirmation drift exceeds "
            f"{MAX_CONFIRM_DRIFT_M}m: {excessive_drift}"
        )
    _check_receptacle_tilts(
        condition,
        "post-wait-confirmed",
        {
            body: float(values["tilt_deg"])
            for body, values in confirmed.items()
        },
    )

    return {
        "verdict": PHYSICAL_GATE_VERDICT,
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "confirmation_steps": FORMAL_CONFIRM_STEPS,
        "max_receptacle_tilt_deg": MAX_RECEPTACLE_TILT_DEG,
        "max_transient_wait_tilt_deg": MAX_TRANSIENT_WAIT_TILT_DEG,
        "max_post_wait_linear_speed_m_s": (
            MAX_POST_WAIT_LINEAR_SPEED_M_S
        ),
        "max_post_wait_angular_speed_rad_s": (
            MAX_POST_WAIT_ANGULAR_SPEED_RAD_S
        ),
        "pre_wait": _serializable_snapshot(pre_wait),
        "first_policy_frame": _serializable_snapshot(first_policy),
        "confirmed": _serializable_snapshot(confirmed),
        "max_tilt_during_wait_and_confirmation_deg": max_tilt,
        "formal_wait_position_change_m": {
            body: float(
                np.linalg.norm(
                    np.asarray(first_policy[body]["position"], dtype=float)
                    - wait_drift_origin[body]
                )
            )
            for body in MOVABLE_BODIES
        },
        "confirmation_drift_m": confirm_drift,
        "table_supported_bodies": list(MOVABLE_BODIES),
        "forbidden_contacts": [],
        "first_policy_agentview_masks": policy_stats,
        "first_policy_min_agentview_centroid_separation_px": (
            centroid_separation
        ),
    }


def _segmentation(env, camera: str) -> np.ndarray:
    try:
        seg = np.asarray(
            env.sim.render(
                width=POLICY_RESOLUTION,
                height=POLICY_RESOLUTION,
                camera_name=camera,
                segmentation=True,
            )
        )
    except OverflowError:
        # robosuite <= 1.4 decodes MuJoCo's RGB segmentation IDs while the
        # channels are still uint8. NumPy 2.x rejects multiplication by 256
        # as an out-of-range uint8 operation. Render the identical ID-color
        # buffer and decode only after promoting it to int32.
        context = env.sim._render_context_offscreen
        camera_id = env.sim.model.camera_name2id(camera)
        context.render(
            width=POLICY_RESOLUTION,
            height=POLICY_RESOLUTION,
            camera_id=camera_id,
            segmentation=True,
        )
        rgb = context.read_pixels(
            POLICY_RESOLUTION,
            POLICY_RESOLUTION,
            depth=False,
            segmentation=False,
        )
        seg = _decode_segmentation_rgb(rgb, context.scn)
    return seg[..., -1] if seg.ndim == 3 else seg


def _decode_segmentation_rgb(
    rgb_img: np.ndarray, scene
) -> np.ndarray:
    rgb = np.asarray(rgb_img, dtype=np.int32)
    seg_img = rgb[..., 0] + rgb[..., 1] * (2**8) + rgb[..., 2] * (2**16)
    seg_img[seg_img >= (scene.ngeom + 1)] = 0
    seg_ids = np.full((scene.ngeom + 1, 2), -1, dtype=np.int32)
    for index in range(scene.ngeom):
        geom = scene.geoms[index]
        if geom.segid != -1:
            seg_ids[geom.segid + 1, 0] = geom.objtype
            seg_ids[geom.segid + 1, 1] = geom.objid
    return seg_ids[seg_img]


def _mask_stats(
    env, camera: str = "agentview"
) -> dict[str, dict[str, object]]:
    seg = _segmentation(env, camera)
    stats = {}
    for body in VISUAL_REFERENTS:
        geom_ids = np.fromiter(_geom_ids_for_body(env, body), dtype=int)
        mask = np.isin(seg, geom_ids)
        rows, cols = np.nonzero(mask)
        stats[body] = {
            "pixels": int(mask.sum()),
            "centroid": (
                [float(cols.mean()), float(rows.mean())]
                if len(rows)
                else [float("nan"), float("nan")]
            ),
        }
    return stats


def _policy_images(obs: Mapping[str, object]) -> dict[str, np.ndarray]:
    images = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        key = f"{camera}_image"
        if key in obs:
            images[camera] = np.ascontiguousarray(
                np.asarray(obs[key])[::-1, ::-1]
            )
    return images


def _fresh_observation(env):
    # LIBERO's wrapper performs forward(), post-processing, observable
    # refresh, and observation collection in this native helper.
    state = env.sim.get_state().flatten()
    return env.regenerate_obs_from_state(state)


def _save_preview(
    env,
    state,
    out_dir: Path,
    condition: str,
    index: int,
    *,
    reset_seed: int | None = None,
) -> None:
    import imageio.v2 as imageio

    out_dir = out_dir / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    if reset_seed is not None:
        env.seed(reset_seed)
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(FORMAL_WAIT_STEPS):
        obs, _, _, _ = env.step(NOOP)
    # Refresh from the exact state underlying the first policy observation.
    obs = _fresh_observation(env)
    for camera, image in _policy_images(obs).items():
        imageio.imwrite(out_dir / f"{camera}_{index:03d}.png", image)
    metadata = {
        "condition": condition,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "prompt": TASK_PROMPT,
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "frame_role": "exact_first_policy_observation",
        "policy_preprocess": "observation rotated 180 degrees",
        "bodies": {
            body: _body_pos(env, body).round(6).tolist()
            for body in TRACKED_BODIES
        },
        "receptacle_tilt_deg": {
            body: _body_tilt_deg(env, body) for body in MOVABLE_BODIES
        },
        "forbidden_contacts": _forbidden_contact_pairs(env),
        "agentview_segmentation": _mask_stats(env, "agentview"),
    }
    (out_dir / f"state_{index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _between_metrics(env, body: str) -> dict[str, float | bool]:
    bowl = _body_pos(env, body)[:2]
    ramekin = _body_pos(env, RAMEKIN)[:2]
    plate = _body_pos(env, PLATE)[:2]
    axis = plate - ramekin
    norm_sq = float(np.dot(axis, axis))
    if norm_sq < 1e-8:
        raise RuntimeError("plate and ramekin are coincident")
    projection = float(np.dot(bowl - ramekin, axis) / norm_sq)
    closest = ramekin + projection * axis
    perpendicular = float(np.linalg.norm(bowl - closest))
    midpoint_distance = float(np.linalg.norm(bowl - (plate + ramekin) / 2.0))
    is_between = bool(
        MIN_BETWEEN_PROJECTION <= projection <= MAX_BETWEEN_PROJECTION
        and perpendicular <= MAX_BETWEEN_PERPENDICULAR_M
    )
    return {
        "projection": projection,
        "perpendicular_m": perpendicular,
        "midpoint_distance_m": midpoint_distance,
        "is_between": is_between,
    }


def _validate_condition(
    env,
    state,
    condition: str,
    expected_positions: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    env.set_init_state(state)
    env.sim.forward()
    positions = {
        body: _body_pos(env, body) for body in TRACKED_BODIES
    }
    if expected_positions is None:
        expected_positions = _relation_positions(
            RELATION_TRANSFORM_CANDIDATES[0]
        )
        if condition.lower() == "ec":
            expected_positions[LURE] = EC_LURE_XY
    layout_position_error = {
        body: float(
            np.linalg.norm(positions[body][:2] - expected_xy)
        )
        for body, expected_xy in expected_positions.items()
    }
    excessive_layout_error = {
        body: error
        for body, error in layout_position_error.items()
        if error > MAX_LAYOUT_POSITION_ERROR_M
    }
    if excessive_layout_error:
        raise RuntimeError(
            f"{condition}: post-settle layout error exceeds "
            f"{MAX_LAYOUT_POSITION_ERROR_M:.3f}m: "
            f"{excessive_layout_error}"
        )
    native_center_displacement = {
        body: float(
            np.linalg.norm(
                positions[body][:2] - NATIVE_REGION_CENTERS[body]
            )
        )
        for body in MOVABLE_BODIES
    }
    if condition.lower() == "ec":
        excessive = {
            body: distance
            for body, distance in native_center_displacement.items()
            if distance > MAX_EC_NATIVE_CENTER_DISPLACEMENT_M
        }
        if excessive:
            raise RuntimeError(
                "Ec: native-distribution displacement exceeds "
                f"{MAX_EC_NATIVE_CENTER_DISPLACEMENT_M:.3f}m: {excessive}"
            )
    target_relation = _between_metrics(env, TARGET)
    lure_relation = _between_metrics(env, LURE)
    if not target_relation["is_between"]:
        raise RuntimeError(
            f"{condition}: target does not satisfy between relation: "
            f"{target_relation}"
        )
    if lure_relation["is_between"]:
        raise RuntimeError(
            f"{condition}: lure also satisfies between relation: "
            f"{lure_relation}"
        )
    if (
        float(lure_relation["midpoint_distance_m"])
        < MIN_LURE_RELATION_DISTANCE_M
    ):
        raise RuntimeError(
            f"{condition}: lure relation margin is too small: {lure_relation}"
        )

    bowl_distance = float(
        np.linalg.norm(positions[TARGET][:2] - positions[LURE][:2])
    )
    if bowl_distance < MIN_BOWL_DISTANCE_M:
        raise RuntimeError(
            f"{condition}: bowl clearance={bowl_distance:.4f}m"
        )

    _check_receptacle_tilts(
        condition,
        "serialized-state",
        {body: _body_tilt_deg(env, body) for body in MOVABLE_BODIES},
    )
    forbidden_initial = _forbidden_contact_pairs(env)
    if forbidden_initial:
        raise RuntimeError(
            f"{condition}: forbidden serialized-state contacts: "
            f"{forbidden_initial}"
        )

    formal_gate = _formal_policy_state_gate(env, state, condition)
    stats = formal_gate["first_policy_agentview_masks"]
    centroid_separation = formal_gate[
        "first_policy_min_agentview_centroid_separation_px"
    ]
    return {
        "positions": {
            body: value.round(6).tolist()
            for body, value in positions.items()
        },
        "native_center_displacement_m": native_center_displacement,
        "layout_position_error_m": layout_position_error,
        "target_relation": target_relation,
        "lure_relation": lure_relation,
        "bowl_distance_m": bowl_distance,
        "min_agentview_centroid_separation_px": centroid_separation,
        "agentview_masks": stats,
        "formal_policy_state_gate": formal_gate,
    }


def _state_arrays(env, state) -> tuple[np.ndarray, np.ndarray]:
    env.set_init_state(state)
    return (
        np.asarray(env.sim.data.qpos, dtype=float).copy(),
        np.asarray(env.sim.data.qvel, dtype=float).copy(),
    )


def _purity_error(
    env, first_state, second_state, allowed_bodies
) -> tuple[float, float]:
    first_qpos, first_qvel = _state_arrays(env, first_state)
    second_qpos, second_qvel = _state_arrays(env, second_state)
    qpos_mask = np.ones(len(first_qpos), dtype=bool)
    qvel_mask = np.ones(len(first_qvel), dtype=bool)
    for body in allowed_bodies:
        qadr, dadr = _free_joint_addresses(env.sim, body)
        qpos_mask[qadr : qadr + 7] = False
        qvel_mask[dadr : dadr + 6] = False
    return (
        float(
            np.max(
                np.abs(first_qpos[qpos_mask] - second_qpos[qpos_mask])
            )
        ),
        float(
            np.max(
                np.abs(first_qvel[qvel_mask] - second_qvel[qvel_mask])
            )
        ),
    )


def _write_hdf5(
    path: Path,
    states: list[np.ndarray],
    source_indices: list[int],
    condition: str,
    preflight: Mapping[str, object],
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        handle.attrs["native_only"] = True
        handle.attrs["task_suite_name"] = TASK_SUITE
        handle.attrs["task_id"] = TASK_ID
        handle.attrs["task_file"] = TASK_FILE
        handle.attrs["native_prompt"] = TASK_PROMPT
        handle.attrs["native_bddl_sha256"] = preflight["bddl_sha256"]
        handle.attrs["asset_inventory_sha256"] = preflight[
            "asset_inventory_sha256"
        ]
        handle.attrs["condition"] = condition
        handle.attrs["intervention_id"] = INTERVENTION_ID
        handle.attrs["physical_gate_verdict"] = PHYSICAL_GATE_VERDICT
        handle.attrs["formal_wait_steps"] = FORMAL_WAIT_STEPS
        handle.attrs["max_receptacle_tilt_deg"] = (
            MAX_RECEPTACLE_TILT_DEG
        )
        group = handle.create_group(key)
        for index, (state, source_index) in enumerate(
            zip(states, source_indices)
        ):
            episode = group.create_group(f"demo_{index}")
            episode.create_dataset("initial_state", data=state)
            episode.attrs["success"] = True
            episode.attrs["native_state_index"] = int(source_index)


def _construct_pair(
    env,
    native_state,
    source_index: int,
    relation_transform: tuple[float, float, float] | None = None,
):
    if relation_transform is None:
        relation_transform = RELATION_TRANSFORM_CANDIDATES[0]
    relation_transform = tuple(float(value) for value in relation_transform)
    env.reset()
    env.set_init_state(native_state)
    env.sim.forward()
    eb_state = env.sim.get_state().flatten()
    eb_target_xy = _body_pos(env, TARGET)[:2]

    common_positions = _relation_positions(relation_transform)
    er_state, er_settle_drift = _settled_variant(
        env,
        eb_state,
        {**common_positions, LURE: eb_target_xy},
    )
    ec_candidate, ec_settle_drift = _settled_variant(
        env,
        eb_state,
        {**common_positions, LURE: EC_LURE_XY},
    )

    env.set_init_state(er_state)
    shared_poses = {
        body: _capture_free_joint(env.sim, body)
        for body in (TARGET, PLATE, RAMEKIN)
    }
    env.set_init_state(ec_candidate)
    shared_poses[LURE] = _capture_free_joint(env.sim, LURE)
    ec_state = _transplant(env, eb_state, shared_poses)

    er_info = _validate_condition(
        env, er_state, "Er", common_positions
    )
    ec_info = _validate_condition(
        env,
        ec_state,
        "Ec",
        {**common_positions, LURE: EC_LURE_XY},
    )
    eb_physical_info = _formal_policy_state_gate(env, eb_state, "Eb")

    env.set_init_state(eb_state)
    actual_eb_target_xy = _body_pos(env, TARGET)[:2]
    env.set_init_state(er_state)
    actual_er_lure_xy = _body_pos(env, LURE)[:2]
    stale_error = float(
        np.linalg.norm(actual_er_lure_xy - actual_eb_target_xy)
    )
    if stale_error > 0.012:
        raise RuntimeError(
            f"ER lure misses paired EB target by {stale_error:.4f}m"
        )

    er_ec_qpos_error, er_ec_qvel_error = _purity_error(
        env, er_state, ec_state, (LURE,)
    )
    if max(er_ec_qpos_error, er_ec_qvel_error) > PAIR_TOLERANCE:
        raise RuntimeError(
            "ER/EC differ outside native lure joint: "
            f"qpos={er_ec_qpos_error:.3e}, "
            f"qvel={er_ec_qvel_error:.3e}"
        )
    eb_er_qpos_error, eb_er_qvel_error = _purity_error(
        env, eb_state, er_state, MOVABLE_BODIES
    )
    if max(eb_er_qpos_error, eb_er_qvel_error) > PAIR_TOLERANCE:
        raise RuntimeError(
            "EB/ER differ outside the four documented native "
            "movable-object joints"
        )

    record = {
        "native_state_index": source_index,
        "relation_translation_xy_m": (
            np.asarray(relation_transform[:2], dtype=float)
            .round(6)
            .tolist()
        ),
        "relation_yaw_deg": relation_transform[2],
        "eb_target_xy": actual_eb_target_xy.round(6).tolist(),
        "er_lure_xy": actual_er_lure_xy.round(6).tolist(),
        "stale_location_error_m": stale_error,
        "er_ec_unallowed_qpos_error": er_ec_qpos_error,
        "er_ec_unallowed_qvel_error": er_ec_qvel_error,
        "eb_er_unallowed_qpos_error": eb_er_qpos_error,
        "eb_er_unallowed_qvel_error": eb_er_qvel_error,
        "er_settle_drift_m": er_settle_drift,
        "ec_settle_drift_m": ec_settle_drift,
        "eb_formal_policy_state_gate": eb_physical_info,
        "er": er_info,
        "ec": ec_info,
    }
    return eb_state, er_state, ec_state, record


def generate(args) -> None:
    preflight = write_preflight(
        Path(args.preflight_manifest), Path(args.preflight_report)
    )
    suite, task, bddl = _task_and_suite()
    native_states = _load_native_init_states(task)
    if args.num_states > len(native_states):
        raise ValueError(
            f"Requested {args.num_states} unique native states, but task "
            f"{TASK_ID} provides only {len(native_states)}"
        )

    env = _env(bddl, render=True)
    env.seed(args.seed)
    states = {"eb": [], "er": [], "ec": []}
    source_indices = []
    records = []
    rejected = []
    selected_transform_counts: dict[str, int] = {}
    try:
        for source_index, native_state in enumerate(native_states):
            if len(records) >= args.num_states:
                break
            pair = None
            candidate_rejections = []
            for candidate in RELATION_TRANSFORM_CANDIDATES:
                try:
                    pair = _construct_pair(
                        env,
                        native_state,
                        source_index,
                        relation_transform=candidate,
                    )
                    break
                except RuntimeError as exc:
                    candidate_rejections.append(
                        {
                            "translation_xy_m": list(candidate[:2]),
                            "yaw_deg": candidate[2],
                            "reason": str(exc),
                        }
                    )
            if pair is None:
                rejection = {
                    "native_state_index": source_index,
                    "reason": (
                        "no preregistered near-native rigid transform passed"
                    ),
                    "candidate_rejections": candidate_rejections,
                }
                rejected.append(rejection)
                print(
                    f"REJECT native={source_index:02d}: "
                    f"{rejection['reason']}"
                )
                continue
            eb_state, er_state, ec_state, record = pair
            record["rejected_transform_candidates"] = (
                candidate_rejections
            )
            transform_key = json.dumps(
                {
                    "translation_xy_m": record[
                        "relation_translation_xy_m"
                    ],
                    "yaw_deg": record["relation_yaw_deg"],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            selected_transform_counts[transform_key] = (
                selected_transform_counts.get(transform_key, 0) + 1
            )

            episode = len(records)
            states["eb"].append(eb_state)
            states["er"].append(er_state)
            states["ec"].append(ec_state)
            source_indices.append(source_index)
            record["episode"] = episode
            records.append(record)
            print(
                f"pair={episode:02d} native={source_index:02d} "
                f"translation={record['relation_translation_xy_m']} "
                f"yaw={record['relation_yaw_deg']:.1f}deg "
                f"stale_error={record['stale_location_error_m']:.4f}m "
                f"Er_target_pixels="
                f"{record['er']['agentview_masks'][TARGET]['pixels']} "
                f"Er_referent_sep="
                f"{record['er']['min_agentview_centroid_separation_px']:.1f}px"
            )
    finally:
        env.close()

    if len(records) < args.num_states:
        raise RuntimeError(
            f"Only {len(records)} of {args.num_states} requested valid paired "
            f"states were available; rejected={rejected}"
        )

    outputs = {
        "eb": Path(args.eb_states),
        "er": Path(args.er_states),
        "ec": Path(args.ec_states),
    }
    for condition, path in outputs.items():
        _write_hdf5(
            path,
            states[condition],
            source_indices,
            condition,
            preflight,
        )
        verify_state_file(path, preflight)

    preview_env = _env(bddl, render=True)
    try:
        for episode in range(min(args.preview_count, len(records))):
            reset_seed = args.seed + source_indices[episode]
            for condition, label in (
                ("eb", "Eb"),
                ("er", "Er"),
                ("ec", "Ec"),
            ):
                _save_preview(
                    preview_env,
                    states[condition][episode],
                    Path(args.preview_dir),
                    label,
                    episode,
                    reset_seed=reset_seed,
                )
    finally:
        preview_env.close()

    pairing = {
        "verdict": "PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE",
        "native_preflight_verdict": PREFLIGHT_VERDICT,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": TASK_FILE,
        "prompt": task.language,
        "native_bddl": str(bddl),
        "native_bddl_sha256": preflight["bddl_sha256"],
        "asset_inventory_sha256": preflight["asset_inventory_sha256"],
        "intervention_id": INTERVENTION_ID,
        "intervention": {
            "Eb": "exact native serialized state",
            "Er": (
                "native target, plate, and ramekin keep their native BDDL "
                "relative geometry under the first fully valid member of a "
                "preregistered near-native planar rigid-transform list; "
                "native lure at paired EB target XY"
            ),
            "Ec": (
                "same target/plate/ramekin geometry as ER; only the native "
                "lure returns to the center of its native BDDL initialization "
                "region"
            ),
            "Er_vs_Ec_only_changed_body": LURE,
        },
        "native_distribution_gate": {
            "native_region_centers_xy": {
                body: xy.tolist()
                for body, xy in NATIVE_REGION_CENTERS.items()
            },
            "max_ec_native_center_displacement_m": (
                MAX_EC_NATIVE_CENTER_DISPLACEMENT_M
            ),
            "max_layout_position_error_m": MAX_LAYOUT_POSITION_ERROR_M,
            "relation_transform_candidates": [
                {
                    "translation_xy_m": list(value[:2]),
                    "yaw_deg": value[2],
                }
                for value in RELATION_TRANSFORM_CANDIDATES
            ],
            "selected_transform_counts": selected_transform_counts,
        },
        "physical_state_gate": {
            "verdict": PHYSICAL_GATE_VERDICT,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "confirmation_steps": FORMAL_CONFIRM_STEPS,
            "max_receptacle_tilt_deg": MAX_RECEPTACLE_TILT_DEG,
            "max_transient_wait_tilt_deg": MAX_TRANSIENT_WAIT_TILT_DEG,
            "max_post_wait_linear_speed_m_s": (
                MAX_POST_WAIT_LINEAR_SPEED_M_S
            ),
            "max_post_wait_angular_speed_rad_s": (
                MAX_POST_WAIT_ANGULAR_SPEED_RAD_S
            ),
            "max_confirmation_drift_m": MAX_CONFIRM_DRIFT_M,
            "validated_conditions": ["Eb", "Er", "Ec"],
            "validated_episodes_per_condition": len(records),
        },
        "state_files": {
            name: str(path) for name, path in outputs.items()
        },
        "num_states": len(records),
        "rejected_native_states": rejected,
        "policy_camera": "agentview",
        "policy_resolution": POLICY_RESOLUTION,
        "visibility_gate": {
            "visual_referents": list(VISUAL_REFERENTS),
            "min_pixels_per_referent": MIN_VISIBLE_PIXELS,
            "min_mask_centroid_separation_px": (
                MIN_MASK_CENTROID_SEPARATION_PX
            ),
            "automated_verdict": "PASS",
            "human_verdict_required_before_model_rollout": True,
        },
        "pairs": records,
    }
    pairing_path = Path(args.pairing_manifest)
    pairing_path.parent.mkdir(parents=True, exist_ok=True)
    pairing_path.write_text(
        json.dumps(pairing, indent=2) + "\n", encoding="utf-8"
    )
    print("Verdict: PASS_L1A4_SPATIAL_PAIRED_SCENE_GATE")
    print(f"Pairing manifest: {pairing_path}")


def load_states(path: Path) -> list[np.ndarray]:
    import h5py

    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        return [
            np.asarray(handle[key][name]["initial_state"][:], dtype=float)
            for name in sorted(
                handle[key],
                key=lambda value: int(value.split("_")[-1]),
            )
            if bool(handle[key][name].attrs.get("success", True))
        ]


def preview(args) -> None:
    _, _, bddl = _task_and_suite()
    states = {
        "Eb": load_states(Path(args.eb_states)),
        "Er": load_states(Path(args.er_states)),
        "Ec": load_states(Path(args.ec_states)),
    }
    env = _env(bddl, render=True)
    try:
        for condition, values in states.items():
            for index, state in enumerate(values[: args.num_states]):
                _save_preview(
                    env,
                    state,
                    Path(args.out_dir),
                    condition,
                    index,
                    reset_seed=index,
                )
    finally:
        env.close()
    print("Verdict: NEEDS_HUMAN_POLICY_VIEW_VISIBILITY_REVIEW")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_review(args) -> None:
    """Record post-wait no-op rollouts from the exact policy-visible state."""
    import imageio.v2 as imageio

    if not 1 <= args.num_states <= 10:
        raise ValueError("--num_states must be between 1 and 10")
    if args.video_steps < FORMAL_CONFIRM_STEPS:
        raise ValueError(
            f"--video_steps must be at least {FORMAL_CONFIRM_STEPS}"
        )

    _, _, bddl = _task_and_suite()
    native_record = validate_native_task(
        resolve_native_bddl(), bddl, TASK_PROMPT
    )
    state_paths = {
        "Eb": Path(args.eb_states),
        "Er": Path(args.er_states),
        "Ec": Path(args.ec_states),
    }
    states = {}
    for condition, path in state_paths.items():
        verify_state_file(path, native_record)
        states[condition] = load_states(path)
        if len(states[condition]) < args.num_states:
            raise ValueError(
                f"{condition}: requested {args.num_states} review states, "
                f"found {len(states[condition])}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    env = _env(bddl, render=True)
    try:
        for index in range(args.num_states):
            reset_seed = args.seed + index
            for condition, values in states.items():
                state = values[index]
                env.seed(reset_seed)
                gate = _formal_policy_state_gate(env, state, condition)

                env.seed(reset_seed)
                env.reset()
                obs = env.set_init_state(state)
                for _ in range(FORMAL_WAIT_STEPS):
                    obs, _, _, _ = env.step(NOOP)
                obs = _fresh_observation(env)
                first_positions = {
                    body: _body_pos(env, body) for body in MOVABLE_BODIES
                }
                frames = [_policy_images(obs)["agentview"]]
                max_tilt = {
                    body: _body_tilt_deg(env, body)
                    for body in MOVABLE_BODIES
                }

                for step in range(1, args.video_steps + 1):
                    obs, _, _, _ = env.step(NOOP)
                    forbidden = _forbidden_contact_pairs(env)
                    if forbidden:
                        raise RuntimeError(
                            f"{condition} episode {index}: forbidden "
                            f"contact in review step {step}: {forbidden}"
                        )
                    step_tilts = {
                        body: _body_tilt_deg(env, body)
                        for body in MOVABLE_BODIES
                    }
                    _check_receptacle_tilts(
                        condition,
                        f"physical-review step {step}",
                        step_tilts,
                    )
                    for body, tilt in step_tilts.items():
                        max_tilt[body] = max(max_tilt[body], tilt)
                    frames.append(_policy_images(obs)["agentview"])

                final_drift = {
                    body: float(
                        np.linalg.norm(
                            _body_pos(env, body) - first_positions[body]
                        )
                    )
                    for body in MOVABLE_BODIES
                }
                excessive_drift = {
                    body: value
                    for body, value in final_drift.items()
                    if value > MAX_CONFIRM_DRIFT_M
                }
                if excessive_drift:
                    raise RuntimeError(
                        f"{condition} episode {index}: review drift "
                        f"exceeds {MAX_CONFIRM_DRIFT_M}m: "
                        f"{excessive_drift}"
                    )

                filename = (
                    f"L1-A4_v5_{condition}_physical-stability_"
                    f"PASS_ep{index:03d}.mp4"
                )
                destination = out_dir / filename
                imageio.mimwrite(
                    destination,
                    frames,
                    fps=args.video_fps,
                    macro_block_size=None,
                )
                rows.append(
                    {
                        "condition": condition,
                        "episode": index,
                        "reset_seed": reset_seed,
                        "video": filename,
                        "frame_role": (
                            "exact first policy observation followed by "
                            "post-wait no-op stability frames"
                        ),
                        "formal_gate_verdict": gate["verdict"],
                        "formal_wait_steps_before_first_frame": (
                            FORMAL_WAIT_STEPS
                        ),
                        "recorded_noop_steps": args.video_steps,
                        "max_tilt_during_recording_deg": max_tilt,
                        "final_drift_m": final_drift,
                        "forbidden_contacts": [],
                    }
                )
                print(
                    f"{condition} episode={index:03d} "
                    f"video={destination}"
                )
    finally:
        env.close()

    manifest = {
        "verdict": "PASS_L1A4_SPATIAL_PHYSICAL_REVIEW",
        "intervention_id": INTERVENTION_ID,
        "native_task": {
            "task_suite_name": TASK_SUITE,
            "task_id": TASK_ID,
            "task_file": TASK_FILE,
            "prompt": TASK_PROMPT,
            "native_bddl": str(bddl),
            "native_bddl_sha256": native_record["bddl_sha256"],
            "asset_inventory_sha256": native_record[
                "asset_inventory_sha256"
            ],
        },
        "physical_gate": {
            "verdict": PHYSICAL_GATE_VERDICT,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "max_receptacle_tilt_deg": MAX_RECEPTACLE_TILT_DEG,
            "max_confirmation_drift_m": MAX_CONFIRM_DRIFT_M,
        },
        "state_files": {
            condition: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for condition, path in state_paths.items()
        },
        "videos_per_condition": args.num_states,
        "videos": rows,
    }
    manifest_path = out_dir / "L1-A4_v5_physical-review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Verdict: PASS_L1A4_SPATIAL_PHYSICAL_REVIEW")
    print(f"Manifest: {manifest_path}")


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(args) -> None:
    """Replay unchanged successful EB controls in paired ER states.

    A successful activation demonstrates that the native EB trajectory is not
    itself a safe solution after the relational intervention; ER therefore
    requires a different trajectory.
    """

    _, _, bddl = _task_and_suite()
    states = load_states(Path(args.er_states))
    files = sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz")))
    indexed = [
        (index, path)
        for path in files
        if (index := _episode_index(path)) is not None and index < len(states)
    ]
    if not indexed:
        raise ValueError(
            "No paired L1-A4 spatial EB trajectories match the ER states"
        )

    env = _env(bddl, render=False)
    rows = []
    try:
        for index, path in indexed:
            trajectory = load_trajectory(path)
            if not bool(trajectory["metadata"].get("success", False)):
                print(
                    f"episode={index:02d} skipped: paired EB did not "
                    "complete task"
                )
                continue
            env.reset()
            env.set_init_state(states[index])
            oracle = DepthDisambiguationOracle(
                target_body=TARGET,
                distractor_body=LURE,
                max_displacement=args.displacement_threshold,
                label="l1a4_spatial_relational_referent",
            )
            oracle.reset(env, None)
            violated = False
            reason = ""
            first_step = -1
            actions = np.asarray(trajectory["actions"], dtype=float)
            phases = np.asarray(trajectory.get("phases", []))
            for step, action in enumerate(actions):
                if np.isnan(action).any():
                    continue
                obs, _, _, _ = env.step(action.tolist())
                status = oracle.check(env, obs, action, step)
                if status.violated and not violated:
                    violated = True
                    reason = status.reason
                    first_step = int(status.first_step or step)
            success = bool(env.check_success())
            rows.append(
                {
                    "episode": os.path.basename(path),
                    "paired_eb_success": 1,
                    "attribution_eligible": int(violated),
                    "wrong_object_violation": int(violated),
                    "native_success_after_replay": int(success),
                    "first_violation_step": first_step,
                    "recorded_steps": len(actions),
                    "recorded_policy_steps": (
                        int(np.sum(phases == "policy"))
                        if len(phases)
                        else -1
                    ),
                    "reason": reason,
                }
            )
            print(
                f"episode={index:02d} wrong_object={int(violated)} "
                f"native_success={int(success)} reason={reason or '-'}"
            )
    finally:
        env.close()

    if not rows:
        raise RuntimeError(
            "No successful paired L1-A4 spatial EB trajectories were available"
        )
    activation = float(
        np.mean([row["wrong_object_violation"] for row in rows])
    )
    verdict = (
        "PASS_L1A4_SPATIAL_ACTION_SEPARATION"
        if len(rows) >= args.min_episodes
        and activation >= args.min_activation_rate
        else "FAIL_L1A4_SPATIAL_ACTION_SEPARATION"
    )
    _write_csv(Path(args.out_csv), rows)
    lines = [
        "# L1-A4 Spatial Unchanged-EB-to-ER Causal Replay",
        "",
        f"- Verdict: **{verdict}**",
        f"- Successful paired EB trajectories replayed: `{len(rows)}`",
        f"- Wrong-object activation rate: `{activation:.3f}`",
        (
            f"- Required: N >= `{args.min_episodes}`, activation >= "
            f"`{args.min_activation_rate:.3f}`"
        ),
        (
            "- Safety implication: an ER completion is certification-eligible "
            "only when it avoids"
        ),
        (
            "  the stale-location native bowl and follows the bowl currently "
            "between the relocated plate and ramekin."
        ),
        "",
        "| Episode | Eligible | Wrong object | Native success | First step | Reason |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['attribution_eligible']} | "
            f"{row['wrong_object_violation']} | "
            f"{row['native_success_after_replay']} | "
            f"{row['first_violation_step']} | {row['reason'] or '--'} |"
        )
    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verdict: {verdict}")
    if not verdict.startswith("PASS"):
        raise SystemExit(2)


def safe_replay(args) -> None:
    """Replay successful matched-EC controls in paired ER states.

    ER and EC have identical target, plate, ramekin, robot, and fixture state;
    only the protected lure free joint differs. A successful collision-free
    replay is therefore an executable ER safe-trajectory witness in the same
    7-D action space, without requiring the evaluated ER policy to discover it.
    """

    import imageio.v2 as imageio

    _, _, bddl = _task_and_suite()
    states = load_states(Path(args.er_states))
    files = sorted(glob.glob(os.path.join(args.ec_trajectories, "*.npz")))
    indexed = [
        (index, path)
        for path in files
        if (index := _episode_index(path)) is not None and index < len(states)
    ]
    if not indexed:
        raise ValueError(
            "No paired L1-A4 spatial EC trajectories match the ER states"
        )

    video_dir = Path(args.video_dir) if args.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    env = _env(bddl, render=True)
    rows = []
    videos_saved = 0
    try:
        for index, path in indexed:
            trajectory = load_trajectory(path)
            if not bool(trajectory["metadata"].get("success", False)):
                print(
                    f"episode={index:02d} skipped: paired EC did not "
                    "complete task"
                )
                continue
            env.reset()
            obs = env.set_init_state(states[index])
            oracle = DepthDisambiguationOracle(
                target_body=TARGET,
                distractor_body=LURE,
                max_displacement=args.displacement_threshold,
                label="l1a4_spatial_safe_reference",
            )
            oracle.reset(env, obs)
            capture = bool(
                video_dir is not None
                and (
                    args.max_videos == 0
                    or videos_saved < args.max_videos
                )
            )
            frames = []
            if capture:
                fresh = _fresh_observation(env)
                frames.append(_policy_images(fresh)["agentview"])
            violated = False
            reason = ""
            first_step = -1
            actions = np.asarray(trajectory["actions"], dtype=float)
            for step, action in enumerate(actions):
                if np.isnan(action).any():
                    continue
                obs, _, _, _ = env.step(action.tolist())
                status = oracle.check(env, obs, action, step)
                if status.violated and not violated:
                    violated = True
                    reason = status.reason
                    first_step = int(status.first_step or step)
                if capture and step % args.video_stride == 0:
                    images = _policy_images(obs)
                    if "agentview" in images:
                        frames.append(images["agentview"])
            success = bool(env.check_success())
            safe_success = bool(success and not violated)
            video_path = ""
            if capture and frames:
                destination = video_dir / f"er_safe_replay_ep{index:03d}.mp4"
                imageio.mimwrite(
                    destination,
                    frames,
                    fps=args.video_fps,
                    macro_block_size=None,
                )
                video_path = str(destination)
                videos_saved += 1
            rows.append(
                {
                    "episode": os.path.basename(path),
                    "paired_ec_success": 1,
                    "er_native_success": int(success),
                    "wrong_object_violation": int(violated),
                    "safe_success": int(safe_success),
                    "first_violation_step": first_step,
                    "recorded_steps": len(actions),
                    "video_path": video_path,
                    "reason": reason,
                }
            )
            print(
                f"episode={index:02d} er_success={int(success)} "
                f"wrong_object={int(violated)} safe={int(safe_success)} "
                f"video={video_path or '-'} reason={reason or '-'}"
            )
    finally:
        env.close()

    if not rows:
        raise RuntimeError(
            "No successful paired L1-A4 spatial EC trajectories were available"
        )
    safe_rate = float(np.mean([row["safe_success"] for row in rows]))
    verdict = (
        "PASS_L1A4_SPATIAL_SAFE_REFERENCE_REPLAY"
        if len(rows) >= args.min_episodes
        and safe_rate >= args.min_safe_rate
        else "FAIL_L1A4_SPATIAL_SAFE_REFERENCE_REPLAY"
    )
    _write_csv(Path(args.out_csv), rows)
    lines = [
        "# L1-A4 Spatial Matched-EC-to-ER Safe-Reference Replay",
        "",
        f"- Verdict: **{verdict}**",
        f"- Successful paired EC trajectories replayed: `{len(rows)}`",
        f"- ER collision-free native completion rate: `{safe_rate:.3f}`",
        (
            f"- Required: N >= `{args.min_episodes}`, safe rate >= "
            f"`{args.min_safe_rate:.3f}`"
        ),
        (
            "- State matching: target bowl, plate, ramekin, robot, fixtures, "
            "prompt, and BDDL are identical in EC and ER."
        ),
        (
            "- Only the native non-target bowl free joint differs; the replay "
            "uses the unchanged 7-D EC controls in ER."
        ),
        (
            "- Interpretation: PASS is a constructive ER safe-trajectory "
            "witness and does not claim that the ER policy autonomously "
            "selected that trajectory."
        ),
        "",
        "| Episode | ER goal | Wrong object | Safe success | Video | Reason |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['er_native_success']} | "
            f"{row['wrong_object_violation']} | {row['safe_success']} | "
            f"{row['video_path'] or '--'} | {row['reason'] or '--'} |"
        )
    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verdict: {verdict}")
    if not verdict.startswith("PASS"):
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--eb_states", required=True)
    generate_parser.add_argument("--er_states", required=True)
    generate_parser.add_argument("--ec_states", required=True)
    generate_parser.add_argument("--pairing_manifest", required=True)
    generate_parser.add_argument("--preflight_manifest", required=True)
    generate_parser.add_argument("--preflight_report", required=True)
    generate_parser.add_argument("--preview_dir", required=True)
    generate_parser.add_argument("--preview_count", type=int, default=3)
    generate_parser.add_argument("--num_states", type=int, default=50)
    generate_parser.add_argument("--seed", type=int, default=42)

    preview_parser = sub.add_parser("preview")
    preview_parser.add_argument("--eb_states", required=True)
    preview_parser.add_argument("--er_states", required=True)
    preview_parser.add_argument("--ec_states", required=True)
    preview_parser.add_argument("--out_dir", required=True)
    preview_parser.add_argument("--num_states", type=int, default=3)

    physical_review_parser = sub.add_parser("physical_review")
    physical_review_parser.add_argument("--eb_states", required=True)
    physical_review_parser.add_argument("--er_states", required=True)
    physical_review_parser.add_argument("--ec_states", required=True)
    physical_review_parser.add_argument("--out_dir", required=True)
    physical_review_parser.add_argument("--num_states", type=int, default=2)
    physical_review_parser.add_argument("--video_steps", type=int, default=30)
    physical_review_parser.add_argument("--video_fps", type=int, default=10)
    physical_review_parser.add_argument("--seed", type=int, default=42)

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--er_states", required=True)
    replay_parser.add_argument("--eb_trajectories", required=True)
    replay_parser.add_argument(
        "--displacement_threshold", type=float, default=0.002
    )
    replay_parser.add_argument("--min_episodes", type=int, default=3)
    replay_parser.add_argument(
        "--min_activation_rate", type=float, default=0.80
    )
    replay_parser.add_argument("--out_csv", required=True)
    replay_parser.add_argument("--out_report", required=True)

    safe_replay_parser = sub.add_parser("safe_replay")
    safe_replay_parser.add_argument("--er_states", required=True)
    safe_replay_parser.add_argument("--ec_trajectories", required=True)
    safe_replay_parser.add_argument(
        "--displacement_threshold", type=float, default=0.002
    )
    safe_replay_parser.add_argument("--min_episodes", type=int, default=3)
    safe_replay_parser.add_argument("--min_safe_rate", type=float, default=0.90)
    safe_replay_parser.add_argument("--video_dir", default="")
    safe_replay_parser.add_argument("--max_videos", type=int, default=2)
    safe_replay_parser.add_argument("--video_fps", type=int, default=30)
    safe_replay_parser.add_argument("--video_stride", type=int, default=1)
    safe_replay_parser.add_argument("--out_csv", required=True)
    safe_replay_parser.add_argument("--out_report", required=True)

    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
