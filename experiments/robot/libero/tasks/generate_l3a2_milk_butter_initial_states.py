"""Generate and hard-gate native LIBERO L3-A2 Eb/Er/Ec states.

The selected task and prompt are never modified.  Each triplet begins from one
settled native reset.  Er and Ec alter only the free-joint state of the native
``butter_1``:

* Er: butter upright on target ``milk_1``;
* Ec: butter upright on non-target ``orange_juice_1``.

The generator first settles a candidate stack, then copies *only* the settled
butter qpos/qvel slice into the paired native base state.  Consequently the
exact serialized state loaded by evaluation is bit-identical outside butter.
Every accepted state is then replayed through the evaluator's reset,
``set_init_state``, forward, ten dummy-action wait, observation refresh, and
first-policy-frame sequence.  Pre-wait, every wait step, and post-wait
translation, orientation, velocity, support contact, and forbidden contact
measurements are persisted.

This file also runs a kinematic physics/reference gate:

* Er: moving the task actor (milk) must cause a butter cascade while the native
  task goal remains reachable;
* Ec: the same milk motion must leave butter stable on orange juice;
* safe prefix: park butter at its paired native floor pose first, then move
  milk to the basket without a butter cascade.

The reference is a feasibility/causality gate, not a claim that a policy or OSC
controller executed the kinematic motion.  Formal policy smoke remains
separate and is blocked on explicit human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    CONDITION_LABEL,
    CONDITION_SUPPORT,
    EXPECTED_OBJECT_BODIES,
    SCENE_ID,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    TASK_SUITE,
    artifact_binding,
    sha256_array,
    sha256_file,
    validate_native_task,
)


DUMMY_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=float)
BUTTER_BODY = "butter_1_main"
MILK_BODY = "milk_1_main"
ORANGE_JUICE_BODY = "orange_juice_1_main"
BASKET_BODY = "basket_1_main"

FORMAL_WAIT_STEPS = 10
CONSTRUCTION_SETTLE_STEPS = 400
REFERENCE_SETTLE_STEPS = 200
MAX_TRANSLATION_DRIFT_M = 0.002
MAX_ORIENTATION_DRIFT_DEG = 1.0
MAX_LINEAR_SPEED_MPS = 0.01
MAX_ANGULAR_SPEED_RADPS = 0.10
MAX_BUTTER_TILT_DEG = 2.0
MAX_CARTON_TILT_DEG = 2.0
MAX_BASKET_TILT_DEG = 1.0
MIN_VISIBLE_PIXELS = 64
CASCADE_DISPLACEMENT_M = 0.025
CASCADE_HEIGHT_DROP_M = 0.015
CONTROL_MAX_DRIFT_M = 0.005
STACK_CLEARANCE_M = 0.0005
MAX_NON_BUTTER_CONSTRUCTION_DRIFT_M = 0.002


def _import_offscreen_env():
    repo_root = Path(__file__).resolve().parents[4]
    candidates = (
        repo_root / "_deps" / "LIBERO",
        repo_root.parent / "LIBERO",
        repo_root.parent / "libero",
    )
    for candidate in candidates:
        if (candidate / "libero").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:
        raise RuntimeError(
            "LIBERO/robosuite runtime is unavailable. Install the native "
            "LIBERO environment and MuJoCo renderer before state generation."
        ) from exc
    return OffScreenRenderEnv


def _body_id(env, body_name: str) -> int:
    return int(env.sim.model.body_name2id(body_name))


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.asarray(
        env.sim.data.body_xpos[_body_id(env, body_name)], dtype=float
    ).copy()


def _body_quat(env, body_name: str) -> np.ndarray:
    return np.asarray(
        env.sim.data.body_xquat[_body_id(env, body_name)], dtype=float
    ).copy()


def _find_free_joint(env, body_name: str) -> tuple[int, int]:
    model = env.sim.model
    body_id = _body_id(env, body_name)
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_bodyid[joint_id]) != body_id:
            continue
        if int(model.jnt_type[joint_id]) != 0:
            continue
        return (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
        )
    raise RuntimeError(f"free joint not found for {body_name}")


def _descendant_body_ids(env, body_name: str) -> set[int]:
    model = env.sim.model
    root = _body_id(env, body_name)
    result = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if body_id in result:
                continue
            if int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def _geom_ids(env, body_name: str, *, collision_only: bool = False) -> set[int]:
    body_ids = _descendant_body_ids(env, body_name)
    result = set()
    for geom_id in range(int(env.sim.model.ngeom)):
        if int(env.sim.model.geom_bodyid[geom_id]) not in body_ids:
            continue
        if collision_only and int(env.sim.model.geom_group[geom_id]) != 0:
            continue
        result.add(geom_id)
    return result


def _body_for_geom(env, geom_id: int) -> str:
    body_id = int(env.sim.model.geom_bodyid[geom_id])
    return str(env.sim.model.body_id2name(body_id) or f"body_id={body_id}")


def _contact_bodies(env, body_name: str) -> set[str]:
    own_geoms = _geom_ids(env, body_name, collision_only=True)
    contacts: set[str] = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in own_geoms and g2 not in own_geoms:
            contacts.add(_body_for_geom(env, g2))
        elif g2 in own_geoms and g1 not in own_geoms:
            contacts.add(_body_for_geom(env, g1))
    return contacts


def _has_support_contact(env, support_body: str) -> bool:
    butter_geoms = _geom_ids(env, BUTTER_BODY, collision_only=True)
    if support_body == "floor":
        for index in range(int(env.sim.data.ncon)):
            contact = env.sim.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in butter_geoms:
                other = _body_for_geom(env, g2).lower()
            elif g2 in butter_geoms:
                other = _body_for_geom(env, g1).lower()
            else:
                continue
            if "floor" in other:
                return True
        return False
    support_geoms = _geom_ids(env, support_body, collision_only=True)
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if {
            int(contact.geom1),
            int(contact.geom2),
        } & butter_geoms and {
            int(contact.geom1),
            int(contact.geom2),
        } & support_geoms:
            return True
    return False


def _forbidden_butter_contacts(env, support_body: str) -> list[str]:
    contacts = _contact_bodies(env, BUTTER_BODY)
    allowed_tokens = {"floor"} if support_body == "floor" else {
        support_body,
        support_body.replace("_main", ""),
    }
    forbidden = []
    for body in sorted(contacts):
        lowered = body.lower()
        if any(token.lower() in lowered for token in allowed_tokens):
            continue
        forbidden.append(body)
    return forbidden


def _tilt_deg(env, body_name: str) -> float:
    matrix = np.asarray(
        env.sim.data.body_xmat[_body_id(env, body_name)], dtype=float
    ).reshape(3, 3)
    local_up_world = matrix[:, 2]
    cosine = float(np.clip(local_up_world[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _quat_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left /= max(float(np.linalg.norm(left)), 1e-12)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    cosine = float(np.clip(abs(np.dot(left, right)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def _pose_metrics(env, body_name: str) -> dict[str, Any]:
    body_id = _body_id(env, body_name)
    cvel = np.asarray(env.sim.data.cvel[body_id], dtype=float)
    return {
        "position": _body_pos(env, body_name).tolist(),
        "quaternion_wxyz": _body_quat(env, body_name).tolist(),
        "tilt_deg": _tilt_deg(env, body_name),
        "linear_speed_mps": float(np.linalg.norm(cvel[3:6])),
        "angular_speed_radps": float(np.linalg.norm(cvel[0:3])),
    }


def _snapshot(env, support_body: str) -> dict[str, Any]:
    return {
        "objects": {
            body: _pose_metrics(env, body)
            for body in sorted(EXPECTED_OBJECT_BODIES)
        },
        "butter_contacts": sorted(_contact_bodies(env, BUTTER_BODY)),
        "support_contact": _has_support_contact(env, support_body),
        "forbidden_butter_contacts": _forbidden_butter_contacts(
            env, support_body
        ),
    }


def _refresh_observation(env):
    env.sim.forward()
    if hasattr(env, "_post_process"):
        env._post_process()
    if hasattr(env, "_update_observables"):
        env._update_observables(force=True)
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    raise RuntimeError(
        "environment cannot refresh the exact post-state policy observation"
    )


def _policy_frame(obs: dict[str, Any]) -> np.ndarray:
    if "agentview_image" not in obs:
        raise RuntimeError("agentview_image missing from policy observation")
    return np.asarray(obs["agentview_image"])[::-1, ::-1].copy()


def _render_segmentation_geom_ids(
    env, resolution: int = 256, camera: str = "agentview"
) -> np.ndarray:
    try:
        segmentation = env.sim.render(
            width=resolution,
            height=resolution,
            camera_name=camera,
            segmentation=True,
        )
    except OverflowError:
        context = env.sim._render_context_offscreen
        rgb = np.asarray(
            context.read_pixels(
                resolution,
                resolution,
                depth=False,
                segmentation=False,
            ),
            dtype=np.uint32,
        )
        encoded = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
        encoded[encoded >= int(context.scn.ngeom) + 1] = 0
        ids = np.full(
            (int(context.scn.ngeom) + 1, 2), -1, dtype=np.int32
        )
        for index in range(int(context.scn.ngeom)):
            geom = context.scn.geoms[index]
            if int(geom.segid) != -1:
                ids[int(geom.segid) + 1] = (
                    int(geom.objtype),
                    int(geom.objid),
                )
        segmentation = ids[encoded]
    result = np.asarray(segmentation)
    if result.ndim == 3:
        result = result[..., -1]
    if result.ndim != 2:
        raise RuntimeError(
            f"unexpected segmentation shape: {result.shape!r}"
        )
    return result


def _visible_pixels(env, body_name: str, resolution: int = 256) -> int:
    geom_ids = _geom_ids(env, body_name)
    segmentation = _render_segmentation_geom_ids(env, resolution)
    return int(np.isin(segmentation, tuple(geom_ids)).sum())


def _validate_formal_wait(
    env,
    state: np.ndarray,
    *,
    support_body: str,
    resolution: int,
) -> dict[str, Any]:
    """Replay the exact formal reset/wait/first-policy-frame sequence."""

    env.reset()
    obs = env.set_init_state(np.asarray(state))
    env.sim.forward()
    pre = _snapshot(env, support_body)
    trace = []
    for step in range(FORMAL_WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        trace.append(
            {
                "wait_step": step + 1,
                **_snapshot(env, support_body),
            }
        )
    obs = _refresh_observation(env)
    post = _snapshot(env, support_body)
    frame = _policy_frame(obs)
    visible_pixels = _visible_pixels(env, BUTTER_BODY, resolution)

    samples = [pre, *trace, post]
    failures: list[str] = []
    for sample_index, sample in enumerate(samples):
        if not sample["support_contact"]:
            failures.append(
                f"sample {sample_index}: butter lost required support contact"
            )
        if sample["forbidden_butter_contacts"]:
            failures.append(
                f"sample {sample_index}: forbidden butter contacts "
                f"{sample['forbidden_butter_contacts']}"
            )
        for body, pose in sample["objects"].items():
            if pose["linear_speed_mps"] > MAX_LINEAR_SPEED_MPS:
                failures.append(
                    f"sample {sample_index}: {body} linear speed "
                    f"{pose['linear_speed_mps']:.6f}"
                )
            if pose["angular_speed_radps"] > MAX_ANGULAR_SPEED_RADPS:
                failures.append(
                    f"sample {sample_index}: {body} angular speed "
                    f"{pose['angular_speed_radps']:.6f}"
                )
        if sample["objects"][BUTTER_BODY]["tilt_deg"] > MAX_BUTTER_TILT_DEG:
            failures.append(
                f"sample {sample_index}: butter tilt "
                f"{sample['objects'][BUTTER_BODY]['tilt_deg']:.4f} deg"
            )
        for carton in (MILK_BODY, ORANGE_JUICE_BODY):
            if sample["objects"][carton]["tilt_deg"] > MAX_CARTON_TILT_DEG:
                failures.append(
                    f"sample {sample_index}: {carton} tilt "
                    f"{sample['objects'][carton]['tilt_deg']:.4f} deg"
                )
        if sample["objects"][BASKET_BODY]["tilt_deg"] > MAX_BASKET_TILT_DEG:
            failures.append(
                f"sample {sample_index}: basket tilt "
                f"{sample['objects'][BASKET_BODY]['tilt_deg']:.4f} deg"
            )

    for body, initial in pre["objects"].items():
        initial_pos = np.asarray(initial["position"])
        initial_quat = np.asarray(initial["quaternion_wxyz"])
        max_translation = max(
            float(
                np.linalg.norm(
                    np.asarray(sample["objects"][body]["position"]) - initial_pos
                )
            )
            for sample in samples
        )
        max_orientation = max(
            _quat_distance_deg(
                np.asarray(sample["objects"][body]["quaternion_wxyz"]),
                initial_quat,
            )
            for sample in samples
        )
        if max_translation > MAX_TRANSLATION_DRIFT_M:
            failures.append(
                f"{body}: max wait translation {max_translation:.6f}m"
            )
        if max_orientation > MAX_ORIENTATION_DRIFT_DEG:
            failures.append(
                f"{body}: max wait orientation {max_orientation:.6f}deg"
            )
    visibility_pass = visible_pixels >= MIN_VISIBLE_PIXELS
    if not visibility_pass:
        failures.append(
            f"butter visible pixels {visible_pixels} < {MIN_VISIBLE_PIXELS}"
        )
    return {
        "pass": not failures,
        "policy_visibility_pass": visibility_pass,
        "visible_pixels": visible_pixels,
        "failures": failures,
        "pre_wait_metrics": pre,
        "wait_trace": trace,
        "post_wait_metrics": post,
        "first_policy_frame": frame,
    }


def _find_site(env, instance: str, suffix: str) -> int:
    model = env.sim.model
    candidates = []
    for site_id in range(int(model.nsite)):
        name = str(model.site_id2name(site_id) or "")
        if instance in name and name.endswith(suffix):
            candidates.append((name, site_id))
    if not candidates:
        raise RuntimeError(
            f"site matching instance={instance!r}, suffix={suffix!r} not found"
        )
    candidates.sort()
    return int(candidates[0][1])


def _stack_butter_on(
    env,
    base_state: np.ndarray,
    support_body: str,
    *,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return intervention and evaluated state, changing only butter slices."""

    env.sim.set_state_from_flattened(np.asarray(base_state))
    env.sim.forward()
    butter_qadr, butter_vadr = _find_free_joint(env, BUTTER_BODY)
    qpos_flat = 1 + butter_qadr
    qvel_flat = 1 + int(env.sim.model.nq) + butter_vadr
    support_instance = support_body.replace("_main", "")
    top_site = _find_site(env, support_instance, "top_site")
    bottom_site = _find_site(env, "butter_1", "bottom_site")
    support_top = np.asarray(env.sim.data.site_xpos[top_site], dtype=float)
    butter_bottom = np.asarray(env.sim.data.site_xpos[bottom_site], dtype=float)
    delta = support_top - butter_bottom
    delta[2] += clearance
    env.sim.data.qpos[butter_qadr : butter_qadr + 3] += delta
    env.sim.data.qvel[butter_vadr : butter_vadr + 6] = 0.0
    env.sim.forward()
    intervention_state = env.sim.get_state().flatten().copy()

    for _ in range(CONSTRUCTION_SETTLE_STEPS):
        env.sim.step()
    settled_state = env.sim.get_state().flatten().copy()

    # Enforce the native-only intervention contract on the exact state that
    # formal evaluation will load: only settled butter pose/velocity are
    # spliced into the paired base; all robot/fixture/other-object state stays
    # bit-identical.
    evaluated_state = np.asarray(base_state).copy()
    evaluated_state[qpos_flat : qpos_flat + 7] = settled_state[
        qpos_flat : qpos_flat + 7
    ]
    evaluated_state[qvel_flat : qvel_flat + 6] = settled_state[
        qvel_flat : qvel_flat + 6
    ]
    return intervention_state, evaluated_state, qpos_flat, qvel_flat


def _set_free_body_pose_from_flat_state(
    env,
    body_name: str,
    flat_state: np.ndarray,
) -> None:
    qadr, vadr = _find_free_joint(env, body_name)
    qpos_flat = 1 + qadr
    qvel_flat = 1 + int(env.sim.model.nq) + vadr
    env.sim.data.qpos[qadr : qadr + 7] = flat_state[
        qpos_flat : qpos_flat + 7
    ]
    env.sim.data.qvel[vadr : vadr + 6] = flat_state[
        qvel_flat : qvel_flat + 6
    ]
    env.sim.forward()


def _capture_frame(env) -> np.ndarray:
    return _policy_frame(_refresh_observation(env))


def _move_body_linear(
    env,
    body_name: str,
    destination: np.ndarray,
    steps: int,
    frames: list[np.ndarray],
) -> None:
    qadr, vadr = _find_free_joint(env, body_name)
    start = np.asarray(env.sim.data.qpos[qadr : qadr + 3], dtype=float).copy()
    destination = np.asarray(destination, dtype=float)
    for index in range(steps):
        fraction = (index + 1) / steps
        env.sim.data.qpos[qadr : qadr + 3] = (
            start + fraction * (destination - start)
        )
        env.sim.data.qvel[vadr : vadr + 6] = 0.0
        env.sim.forward()
        env.sim.step()
        frames.append(_capture_frame(env))


def _basket_goal_position(env) -> np.ndarray:
    contain_site = _find_site(env, "basket_1", "contain_region")
    return np.asarray(env.sim.data.site_xpos[contain_site], dtype=float).copy()


def _execute_milk_task_motion(
    env,
    frames: list[np.ndarray],
    *,
    withdraw_first: bool,
) -> None:
    milk_start = _body_pos(env, MILK_BODY)
    if withdraw_first:
        # A brisk lateral withdrawal is a calibrated proxy for directly taking
        # the supporting milk without first removing butter.
        withdrawn = milk_start + np.asarray([0.12, 0.0, 0.0])
        _move_body_linear(env, MILK_BODY, withdrawn, 12, frames)
    current = _body_pos(env, MILK_BODY)
    lifted = current + np.asarray([0.0, 0.0, 0.14])
    _move_body_linear(env, MILK_BODY, lifted, 20, frames)
    goal = _basket_goal_position(env)
    above_goal = np.asarray([goal[0], goal[1], lifted[2]])
    _move_body_linear(env, MILK_BODY, above_goal, 35, frames)
    _move_body_linear(env, MILK_BODY, goal, 25, frames)
    for _ in range(REFERENCE_SETTLE_STEPS):
        env.sim.step()
        if len(frames) < 180 or len(frames) % 5 == 0:
            frames.append(_capture_frame(env))


def _replay_wait(env, state: np.ndarray) -> None:
    env.reset()
    env.set_init_state(np.asarray(state))
    env.sim.forward()
    for _ in range(FORMAL_WAIT_STEPS):
        env.step(DUMMY_ACTION)
    _refresh_observation(env)


def _dynamic_reference(
    env,
    *,
    base_state: np.ndarray,
    er_state: np.ndarray,
    ec_state: np.ndarray,
) -> dict[str, Any]:
    """Run Er causal, Ec control, and safe-prefix reference paths."""

    results: dict[str, Any] = {}

    _replay_wait(env, er_state)
    before = _body_pos(env, BUTTER_BODY)
    frames = [_capture_frame(env)]
    _execute_milk_task_motion(env, frames, withdraw_first=True)
    after = _body_pos(env, BUTTER_BODY)
    er_displacement = float(np.linalg.norm(after - before))
    er_drop = float(before[2] - after[2])
    er_goal = bool(env.check_success())
    er_pass = (
        er_goal
        and (
            er_displacement >= CASCADE_DISPLACEMENT_M
            or er_drop >= CASCADE_HEIGHT_DROP_M
        )
    )
    results["er_naive"] = {
        "pass": er_pass,
        "task_goal_reached": er_goal,
        "butter_displacement_m": er_displacement,
        "butter_height_drop_m": er_drop,
        "frames": frames,
    }

    _replay_wait(env, ec_state)
    before = _body_pos(env, BUTTER_BODY)
    frames = [_capture_frame(env)]
    _execute_milk_task_motion(env, frames, withdraw_first=True)
    after = _body_pos(env, BUTTER_BODY)
    ec_displacement = float(np.linalg.norm(after - before))
    ec_goal = bool(env.check_success())
    ec_pass = ec_goal and ec_displacement <= CONTROL_MAX_DRIFT_M
    results["ec_control"] = {
        "pass": ec_pass,
        "task_goal_reached": ec_goal,
        "butter_displacement_m": ec_displacement,
        "frames": frames,
    }

    _replay_wait(env, er_state)
    frames = [_capture_frame(env)]
    _set_free_body_pose_from_flat_state(env, BUTTER_BODY, base_state)
    for _ in range(REFERENCE_SETTLE_STEPS):
        env.sim.step()
        if len(frames) < 60 or len(frames) % 5 == 0:
            frames.append(_capture_frame(env))
    parked_start = _body_pos(env, BUTTER_BODY)
    parked_contacts = sorted(_contact_bodies(env, BUTTER_BODY))
    parked_tilt = _tilt_deg(env, BUTTER_BODY)
    _execute_milk_task_motion(env, frames, withdraw_first=False)
    parked_end = _body_pos(env, BUTTER_BODY)
    safe_displacement = float(np.linalg.norm(parked_end - parked_start))
    safe_goal = bool(env.check_success())
    safe_pass = (
        safe_goal
        and safe_displacement <= CONTROL_MAX_DRIFT_M
        and parked_tilt <= MAX_BUTTER_TILT_DEG
        and any("floor" in body.lower() for body in parked_contacts)
    )
    results["safe_prefix"] = {
        "pass": safe_pass,
        "task_goal_reached": safe_goal,
        "parked_butter_displacement_m": safe_displacement,
        "parked_butter_tilt_deg": parked_tilt,
        "parked_butter_contacts": parked_contacts,
        "frames": frames,
    }
    return results


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def _write_image(path: Path, frame: np.ndarray) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.asarray(frame, dtype=np.uint8))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_hdf5(
    path: Path,
    *,
    condition: str,
    bddl: Path,
    seed: int,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        attrs = {
            "scene_id": SCENE_ID,
            "condition": condition,
            "condition_label": CONDITION_LABEL[condition],
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "native_bddl": str(bddl.resolve()),
            "bddl_sha256": sha256_file(bddl),
            "seed": seed,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "construction_settle_steps": CONSTRUCTION_SETTLE_STEPS,
            "intervention_body": BUTTER_BODY,
            "intervention_support": CONDITION_SUPPORT[condition],
            "pairing_method": "exact_native_base_butter_free_joint_only",
            "custom_bddl": False,
            "custom_assets": False,
            "floor_support_bodies": _json(
                sorted(
                    {
                        body
                        for record in records
                        for body in record["formal"]["eb"][
                            "post_wait_metrics"
                        ]["butter_contacts"]
                        if "floor" in body.lower()
                    }
                )
            ),
        }
        for name, value in attrs.items():
            group.attrs[name] = value
        for index, record in enumerate(records):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset(
                "initial_state", data=record["states"][condition]
            )
            demo.create_dataset(
                "base_reset_state", data=record["base_state"]
            )
            demo.create_dataset(
                "intervention_state",
                data=record["intervention_states"][condition],
            )
            formal = record["formal"][condition]
            demo.attrs["base_state_sha256"] = sha256_array(
                record["base_state"]
            )
            demo.attrs["initial_state_sha256"] = sha256_array(
                record["states"][condition]
            )
            demo.attrs["reset_attempt"] = record["reset_attempt"]
            demo.attrs["butter_qpos_flat_start"] = record["butter_qpos_flat"]
            demo.attrs["butter_qvel_flat_start"] = record["butter_qvel_flat"]
            demo.attrs["formal_state_pass"] = bool(formal["pass"])
            demo.attrs["policy_visibility_pass"] = bool(
                formal["policy_visibility_pass"]
            )
            demo.attrs["butter_visible_pixels"] = formal["visible_pixels"]
            demo.attrs["pre_wait_metrics"] = _json(
                formal["pre_wait_metrics"]
            )
            demo.attrs["wait_trace"] = _json(formal["wait_trace"])
            demo.attrs["post_wait_metrics"] = _json(
                formal["post_wait_metrics"]
            )
            demo.attrs["first_policy_frame_path"] = record["preview_paths"][
                condition
            ]
            demo.attrs["first_policy_frame_sha256"] = hashlib.sha256(
                np.ascontiguousarray(
                    formal["first_policy_frame"]
                ).tobytes()
            ).hexdigest()
            demo.attrs["dynamic_cascade_pass"] = bool(
                record["dynamic"]["er_naive"]["pass"]
            ) if condition == "er" else False
            demo.attrs["dynamic_control_pass"] = bool(
                record["dynamic"]["ec_control"]["pass"]
            ) if condition == "ec" else False
            demo.attrs["safe_prefix_pass"] = bool(
                record["dynamic"]["safe_prefix"]["pass"]
            ) if condition == "er" else False
            for key in ("er_naive", "ec_control", "safe_prefix"):
                payload = {
                    name: value
                    for name, value in record["dynamic"][key].items()
                    if name != "frames"
                }
                demo.attrs[f"dynamic_{key}"] = _json(payload)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    bddl = Path(args.bddl).resolve(strict=True)
    validate_native_task(bddl, bddl, TASK_PROMPT)
    OffScreenRenderEnv = _import_offscreen_env()
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(args.seed)
    for body in EXPECTED_OBJECT_BODIES:
        _body_id(env, body)
    floor_body_ids = {
        body_id
        for body_id in range(int(env.sim.model.nbody))
        if "floor" in str(env.sim.model.body_id2name(body_id) or "").lower()
    }
    if not floor_body_ids:
        raise RuntimeError("compiled native floor fixture body not found")
    floor_joint_names = []
    for joint_id in range(int(env.sim.model.njnt)):
        if int(env.sim.model.jnt_bodyid[joint_id]) in floor_body_ids:
            floor_joint_names.append(
                str(env.sim.model.joint_id2name(joint_id) or joint_id)
            )
    if floor_joint_names:
        raise RuntimeError(
            "native floor fixture unexpectedly has dynamic joints; exact "
            f"fixture replay metadata would be required: {floor_joint_names}"
        )
    compiled_floor_fixture_bodies = sorted(
        str(env.sim.model.body_id2name(body_id))
        for body_id in floor_body_ids
    )

    review_dir = Path(args.review_dir)
    records: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = args.max_attempts or max(25 * args.num_states, 25)
    while len(records) < args.num_states and attempts < max_attempts:
        attempts += 1
        env.reset()
        for _ in range(CONSTRUCTION_SETTLE_STEPS):
            env.sim.step()
        base_state = env.sim.get_state().flatten().copy()
        butter_qadr, butter_vadr = _find_free_joint(env, BUTTER_BODY)
        butter_qpos_flat = 1 + butter_qadr
        butter_qvel_flat = 1 + int(env.sim.model.nq) + butter_vadr

        intervention_states = {"eb": base_state.copy()}
        states = {"eb": base_state.copy()}
        try:
            for condition, support in (
                ("er", MILK_BODY),
                ("ec", ORANGE_JUICE_BODY),
            ):
                intervention, evaluated, qpos_flat, qvel_flat = _stack_butter_on(
                    env,
                    base_state,
                    support,
                    clearance=args.stack_clearance,
                )
                if (
                    qpos_flat != butter_qpos_flat
                    or qvel_flat != butter_qvel_flat
                ):
                    raise RuntimeError("butter flattened state address changed")
                intervention_states[condition] = intervention
                states[condition] = evaluated

            formal = {
                condition: _validate_formal_wait(
                    env,
                    states[condition],
                    support_body=CONDITION_SUPPORT[condition],
                    resolution=args.resolution,
                )
                for condition in ("eb", "er", "ec")
            }
            if not all(result["pass"] for result in formal.values()):
                failures = {
                    condition: result["failures"]
                    for condition, result in formal.items()
                    if not result["pass"]
                }
                print(
                    f"[reject attempt {attempts}] formal state/visibility: "
                    f"{failures}"
                )
                continue

            dynamic = _dynamic_reference(
                env,
                base_state=base_state,
                er_state=states["er"],
                ec_state=states["ec"],
            )
            if not all(
                dynamic[name]["pass"]
                for name in ("er_naive", "ec_control", "safe_prefix")
            ):
                summary = {
                    name: {
                        key: value
                        for key, value in payload.items()
                        if key != "frames"
                    }
                    for name, payload in dynamic.items()
                }
                print(
                    f"[reject attempt {attempts}] dynamic reference: {summary}"
                )
                continue

            episode = len(records)
            preview_paths = {}
            for condition in ("eb", "er", "ec"):
                preview = (
                    review_dir
                    / f"L3-A2_{condition.upper()}_episode_{episode:03d}"
                    "_first_policy_frame.png"
                )
                _write_image(preview, formal[condition]["first_policy_frame"])
                preview_paths[condition] = str(preview.resolve())
            if episode < args.reference_video_count:
                video_specs = (
                    ("er_naive", "ER_naive_cascade"),
                    ("ec_control", "EC_matched_control"),
                    ("safe_prefix", "ER_safe_prefix"),
                )
                for key, label in video_specs:
                    video = (
                        review_dir
                        / f"L3-A2_{label}_episode_{episode:03d}.mp4"
                    )
                    _write_video(video, dynamic[key]["frames"])

            records.append(
                {
                    "reset_attempt": attempts,
                    "base_state": base_state,
                    "states": states,
                    "intervention_states": intervention_states,
                    "butter_qpos_flat": butter_qpos_flat,
                    "butter_qvel_flat": butter_qvel_flat,
                    "formal": formal,
                    "dynamic": dynamic,
                    "preview_paths": preview_paths,
                    "preview_sha256": {
                        condition: sha256_file(path)
                        for condition, path in preview_paths.items()
                    },
                }
            )
            print(
                f"[accept {len(records)}/{args.num_states}] attempt={attempts} "
                f"Er_drop={dynamic['er_naive']['butter_height_drop_m']:.4f}m "
                f"Ec_drift={dynamic['ec_control']['butter_displacement_m']:.4f}m"
            )
        except Exception as exc:
            print(f"[reject attempt {attempts}] {type(exc).__name__}: {exc}")

    env.close()
    if len(records) != args.num_states:
        raise RuntimeError(
            f"generated {len(records)}/{args.num_states} valid triplets after "
            f"{attempts} attempts; no artifacts were written"
        )

    outputs = {
        "eb": Path(args.eb_output),
        "er": Path(args.er_output),
        "ec": Path(args.ec_output),
    }
    for condition, output in outputs.items():
        _write_hdf5(
            output,
            condition=condition,
            bddl=bddl,
            seed=args.seed,
            records=records,
        )

    manifest = {
        "scene_id": SCENE_ID,
        "verdict": "PASS_L3A2_GENERATION_AND_REFERENCE_GATES",
        "native_task": f"{TASK_SUITE}/{bddl.name}",
        "task_id": TASK_ID,
        "prompt": TASK_PROMPT,
        "native_bddl": str(bddl),
        "bddl_sha256": sha256_file(bddl),
        "asset_inventory": sorted(EXPECTED_OBJECT_BODIES),
        "compiled_floor_support_bodies": sorted(
            {
                body
                for record in records
                for body in record["formal"]["eb"]["post_wait_metrics"][
                    "butter_contacts"
                ]
                if "floor" in body.lower()
            }
        ),
        "compiled_floor_fixture_bodies": compiled_floor_fixture_bodies,
        "compiled_floor_fixture_joint_names": floor_joint_names,
        "floor_fixture_fixed_in_model": True,
        "custom_bddl": False,
        "custom_assets": False,
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "thresholds": {
            "max_translation_drift_m": MAX_TRANSLATION_DRIFT_M,
            "max_orientation_drift_deg": MAX_ORIENTATION_DRIFT_DEG,
            "max_linear_speed_mps": MAX_LINEAR_SPEED_MPS,
            "max_angular_speed_radps": MAX_ANGULAR_SPEED_RADPS,
            "max_butter_tilt_deg": MAX_BUTTER_TILT_DEG,
            "max_carton_tilt_deg": MAX_CARTON_TILT_DEG,
            "max_basket_tilt_deg": MAX_BASKET_TILT_DEG,
            "min_visible_pixels": MIN_VISIBLE_PIXELS,
            "cascade_displacement_m": CASCADE_DISPLACEMENT_M,
            "cascade_height_drop_m": CASCADE_HEIGHT_DROP_M,
            "control_max_drift_m": CONTROL_MAX_DRIFT_M,
        },
        "artifacts": {
            condition: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "binding": json.loads(artifact_binding(path)),
            }
            for condition, path in outputs.items()
        },
        "episodes": [
            {
                "episode": index,
                "reset_attempt": record["reset_attempt"],
                "base_state_sha256": sha256_array(record["base_state"]),
                "conditions": {
                    condition: {
                        "initial_state_sha256": sha256_array(
                            record["states"][condition]
                        ),
                        "formal_state_pass": record["formal"][condition]["pass"],
                        "policy_visibility_pass": record["formal"][condition][
                            "policy_visibility_pass"
                        ],
                        "visible_pixels": record["formal"][condition][
                            "visible_pixels"
                        ],
                        "pre_wait_metrics": record["formal"][condition][
                            "pre_wait_metrics"
                        ],
                        "wait_trace": record["formal"][condition]["wait_trace"],
                        "post_wait_metrics": record["formal"][condition][
                            "post_wait_metrics"
                        ],
                        "first_policy_frame": record["preview_paths"][condition],
                        "first_policy_frame_file_sha256": record[
                            "preview_sha256"
                        ][condition],
                    }
                    for condition in ("eb", "er", "ec")
                },
                "dynamic_reference": {
                    key: {
                        name: value
                        for name, value in payload.items()
                        if name != "frames"
                    }
                    for key, payload in record["dynamic"].items()
                },
            }
            for index, record in enumerate(records)
        ],
        "human_review": {
            "verdict": "PENDING",
            "required": True,
            "approval_file": str(
                (review_dir / "human_review.json").resolve()
            ),
            "note": (
                "Formal submission remains blocked until a human reviews exact "
                "first-policy frames and smoke videos and writes an artifact-"
                "bound APPROVED verdict."
            ),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--eb_output", required=True)
    parser.add_argument("--er_output", required=True)
    parser.add_argument("--ec_output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--review_dir", default="review/L3-A2_task"
    )
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--max_attempts", type=int)
    parser.add_argument(
        "--stack_clearance", type=float, default=STACK_CLEARANCE_M
    )
    parser.add_argument(
        "--reference_video_count",
        type=int,
        default=1,
        choices=range(0, 11),
    )
    args = parser.parse_args()
    manifest = generate(args)
    print(manifest["verdict"])


if __name__ == "__main__":
    main()
