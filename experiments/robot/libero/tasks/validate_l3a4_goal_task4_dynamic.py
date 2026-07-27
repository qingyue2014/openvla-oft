#!/usr/bin/env python3
"""Fixed L3-A4 dynamic causality and actual-OSC safe-reference gate.

The program is deliberately split into two separately submitted stages:

1. ``witness`` reconstructs the frozen adjacent witness and exports exact
   wait-0 policy RGB and segmentation with ``dynamic_started=false``;
2. ``dynamic`` rerenders the same hash-bound witness, verifies a committed
   human approval file, then runs the fixed selected+witness chain, three
   causal controls, and selected-state actual robot/OSC safe reference.

Direct S qpos commands are labelled ``kinematic_object_calibration`` and are
never described as robot grasping. The safe reference may not fall back: it
must use the environment's real 7-D OSC action interface and satisfy the
official native goal predicate. No VLA, HDF5, formal evaluation, or action
replay is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
)
from experiments.robot.libero.tasks.audit_l3a4_goal_task4_native_static import (
    A_RADIAL_OFFSETS_M,
    B_CLEARANCES_M,
    DIRECTIONS,
    EVALUATOR_ENV_SEED,
    EVALUATOR_NUM_STEPS_WAIT,
    EXPECTED_BDDL_SHA256,
    FUTURE_EVALUATOR_NUM_STEPS_WAIT,
    GOAL,
    MODEL_FAMILY,
    POLICY_RESOLUTION,
    PROMPT,
    ROLES,
    TASK_BDDL,
    TASK_ID,
    TASK_SUITE,
    _asset_gate,
    _candidate_state,
    _resolve_goal_site,
)
from experiments.robot.libero.tasks.audit_l3a4_spatial_task1_native import (
    descendant_geoms,
    resolve_body,
)
from experiments.robot.libero.tasks.audit_l3a4_spatial_task1_vertical_overhang import (
    _contact_force,
    _segmentation_geom_ids,
)
from experiments.robot.libero.tasks.probe_l3a4_spatial_task1_momentum import (
    _bounds,
    _contact,
    _free_pose,
    _jsonable,
    _policy_image,
    _restore,
    _robot_geoms,
    _set_pose,
    _sha,
)


SELECTED_STATE_SHA256 = (
    "4bc9d9b1e5c01f60b9f0b54a68450f95a71e958fd7aae01d7f0b1e738ff722c5"
)
WITNESS_STATE_SHA256 = (
    "be1d1ea777c5aef0a8891779acdadd9def9d993533a9fe2633af65cb916e7385"
)
FROZEN_DIRECTION = "+y"
SELECTED_A_OFFSET_M = 0.0
SELECTED_B_CLEARANCE_M = 0.002
WITNESS_A_OFFSET_M = 0.0
WITNESS_B_CLEARANCE_M = 0.006
SCRATCH_A_STEPS = 120
SCRATCH_B_STEPS = 40

RISK_LOW_LIFT_M = 0.008
RISK_LOW_LIFT_STEPS = 20
RISK_TRANSLATION_M = 0.060
RISK_TRANSLATION_STEPS = 12
RISK_HOLD_STEPS = 80
CONTROL_DT_S = 0.05
RISK_MINIMUM_JERK_PEAK_SPEED_M_S = 0.1875
RISK_MINIMUM_JERK_PEAK_ACCEL_M_S2 = 0.96225

A_RELATIVE_MOTION_M = 0.005
A_RELATIVE_TILT_DEG = 5.0
B_RESPONSE_M = 0.015
B_RESPONSE_TILT_DEG = 12.0
CONTROL_STABILITY_M = 0.003
CONTROL_STABILITY_TILT_DEG = 3.0
CARRY_CONTACT_FRACTION = 0.95
CARRY_RELATIVE_M = 0.003
CARRY_RELATIVE_TILT_DEG = 3.0

ROBOT_GRASP_X_OFFSET_M = 0.034
ROBOT_APPROACH_HEIGHT_M = 0.120
ROBOT_GRASP_HEIGHT_M = 0.015
ROBOT_GRIPPER_PROBE_STEPS = 8
ROBOT_GRASP_SEAT_STEPS = 15
ROBOT_GRASP_MAX_STEPS = 100
ROBOT_POSITION_SCALE_M = 0.08
ROBOT_POSITION_TOLERANCE_M = 0.006
ROBOT_MAX_POSITION_COMMAND = 0.25
ROBOT_GRASP_SEAT_MAX_COMMAND = 0.08
ROBOT_GRASP_MIN_LIFT_M = 0.005
ROBOT_GRASP_OFFSET_DRIFT_M = 0.010

SAFE_B_VERTICAL_CLEARANCE_M = 0.100
SAFE_GOAL_APPROACH_CLEARANCE_M = 0.120
SAFE_VERTICAL_STEPS = 80
SAFE_Y_STEPS = 64
SAFE_X_STEPS = 46
SAFE_HORIZONTAL_MAX_ACCEL_M_S2 = 0.150
SAFE_HORIZONTAL_MAX_POSITION_COMMAND = 0.15
SAFE_DESCEND_STEPS = 60
SAFE_CONTACT_HOLD_STEPS = 5
SAFE_RELEASE_STEPS = 12
SAFE_RETREAT_M = 0.080
SAFE_RETREAT_STEPS = 40
SAFE_FINAL_SETTLE_STEPS = 120

WITNESS_MIN_PIXELS = {
    "S": 40,
    "A": 30,
    "B": 30,
    "plate": 20,
    "cabinet": 100,
}


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pose(env, body: str) -> tuple[np.ndarray, float]:
    body_id = int(env.sim.model.body_name2id(body))
    position = np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()
    matrix = np.asarray(
        env.sim.data.body_xmat[body_id], dtype=float
    ).reshape(3, 3)
    tilt = float(
        np.degrees(np.arccos(np.clip(matrix[2, 2], -1.0, 1.0)))
    )
    return position, tilt


def _minimum_jerk(index: int, steps: int) -> float:
    u = float(index) / float(steps)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _free_joint_addresses(env, body: str) -> tuple[int, int]:
    body_id = int(env.sim.model.body_name2id(body))
    for joint_id in range(env.sim.model.njnt):
        if (
            int(env.sim.model.jnt_bodyid[joint_id]) == body_id
            and int(env.sim.model.jnt_type[joint_id]) == 0
        ):
            return (
                int(env.sim.model.jnt_qposadr[joint_id]),
                int(env.sim.model.jnt_dofadr[joint_id]),
            )
    raise RuntimeError(f"no free joint for {body}")


def _policy_from_obs(obs) -> np.ndarray:
    image = np.ascontiguousarray(
        np.asarray(get_libero_image(obs), dtype=np.uint8)
    )
    if image.shape != (POLICY_RESOLUTION, POLICY_RESOLUTION, 3):
        raise RuntimeError(f"unexpected policy RGB shape: {image.shape}")
    return image


def _fresh_obs(env):
    before = np.asarray(env.sim.get_state().flatten()).copy()
    obs = env.regenerate_obs_from_state(before)
    after = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(before, after):
        raise RuntimeError("policy observation refresh changed state")
    return obs


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(path, fps=fps, format="FFMPEG")
    except Exception:
        writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(np.asarray(frame, dtype=np.uint8))
    writer.close()


def _rebuild_policy_entry(env, raw_state: np.ndarray) -> np.ndarray:
    env.seed(EVALUATOR_ENV_SEED)
    env.reset()
    env.set_init_state(raw_state)
    dummy = get_libero_dummy_action(MODEL_FAMILY)
    if dummy != [0, 0, 0, 0, 0, 0, -1]:
        raise RuntimeError(f"dummy action drift: {dummy}")
    for _ in range(EVALUATOR_NUM_STEPS_WAIT):
        env.step(dummy)
    return np.asarray(env.sim.get_state().flatten(), dtype=float).copy()


def _reconstruct_states(env, base: np.ndarray, bodies: dict[str, str]) -> dict:
    direction = DIRECTIONS[FROZEN_DIRECTION]
    selected = _candidate_state(
        env,
        base,
        bodies,
        direction,
        SELECTED_A_OFFSET_M,
        SELECTED_B_CLEARANCE_M,
        SCRATCH_A_STEPS,
        SCRATCH_B_STEPS,
    )
    witness = _candidate_state(
        env,
        base,
        bodies,
        direction,
        WITNESS_A_OFFSET_M,
        WITNESS_B_CLEARANCE_M,
        SCRATCH_A_STEPS,
        SCRATCH_B_STEPS,
    )
    observed = {
        "selected": _sha(selected),
        "witness": _sha(witness),
    }
    expected = {
        "selected": SELECTED_STATE_SHA256,
        "witness": WITNESS_STATE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"frozen selected/witness state hash drift: {observed}"
        )
    return {"selected": selected, "witness": witness}


def _witness_policy_gate(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    out: Path,
) -> dict:
    _restore(env, state)
    rgb = _policy_image(env)
    rgb_path = out / "witness_wait0_policy_agentview_256.png"
    imageio.imwrite(rgb_path, rgb)
    segmentation = np.ascontiguousarray(
        _segmentation_geom_ids(env)[::-1, ::-1]
    )
    pixels = {}
    masks = {}
    for role in ("S", "A", "B", "plate", "cabinet"):
        mask = np.isin(segmentation, tuple(geoms[role]))
        pixels[role] = int(mask.sum())
        path = out / f"witness_wait0_{role}_segmentation.png"
        imageio.imwrite(path, mask.astype(np.uint8) * 255)
        masks[role] = {
            "path": str(path),
            "sha256": _sha_bytes(path.read_bytes()),
            "visible_pixels": pixels[role],
            "resolution": [POLICY_RESOLUTION, POLICY_RESOLUTION],
        }
    passed = all(
        pixels[role] >= threshold
        for role, threshold in WITNESS_MIN_PIXELS.items()
    )
    return {
        "passed": passed,
        "state_sha256": _sha(state),
        "camera": "agentview",
        "resolution": [POLICY_RESOLUTION, POLICY_RESOLUTION],
        "orientation": "rotate_180_like_get_libero_image",
        "future_wait_steps": FUTURE_EVALUATOR_NUM_STEPS_WAIT,
        "pixels": pixels,
        "minimum_pixels": WITNESS_MIN_PIXELS,
        "rgb": {
            "path": str(rgb_path),
            "sha256": _sha_bytes(rgb_path.read_bytes()),
        },
        "masks": masks,
        "manual_review": "AWAITING_HASH_BOUND_APPROVAL",
    }


def _load_witness_approval(
    approval_path: Path,
    gate: dict,
) -> dict:
    if not approval_path.is_file():
        raise RuntimeError(
            f"missing committed witness approval: {approval_path}"
        )
    approval = json.loads(approval_path.read_text())
    expected_masks = {
        role: {
            "sha256": gate["masks"][role]["sha256"],
            "visible_pixels": gate["pixels"][role],
            "resolution": gate["masks"][role]["resolution"],
        }
        for role in ("S", "A", "B", "plate", "cabinet")
    }
    valid = bool(
        approval.get("approved") is True
        and approval.get("witness_state_sha256")
        == WITNESS_STATE_SHA256
        and approval.get("witness_masks") == expected_masks
        and approval.get("verdict")
        == "PASS_MANUAL_POLICY_RGB_REVIEW"
    )
    if not valid:
        raise RuntimeError(
            f"invalid witness approval payload: {approval}"
        )
    result = dict(approval)
    result["rgb_sha256_diagnostic"] = {
        "decision_role": "non_decisive_diagnostic_only",
        "approved": approval.get("witness_rgb_sha256"),
        "current": gate["rgb"]["sha256"],
        "match": (
            approval.get("witness_rgb_sha256")
            == gate["rgb"]["sha256"]
        ),
    }
    print("PASS_L3A4_WITNESS_MANUAL_POLICY_RGB_REVIEW", flush=True)
    return result


def _contact_metrics(
    env,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict:
    return {
        "S_A": _contact(env, geoms["S"], geoms["A"]),
        "A_B": _contact(env, geoms["A"], geoms["B"]),
        "S_B": _contact(env, geoms["S"], geoms["B"]),
        "robot_A": _contact(env, robot, geoms["A"]),
        "robot_B": _contact(env, robot, geoms["B"]),
        "robot_S": _contact(env, robot, geoms["S"]),
        "A_B_force_N": _contact_force(env, geoms["A"], geoms["B"]),
    }


def _snapshot(env, bodies: dict[str, str]) -> dict:
    snapshot = {}
    for role in ("S", "A", "B"):
        position, tilt = _pose(env, bodies[role])
        snapshot[role] = {
            "position": position,
            "tilt_deg": tilt,
        }
    return snapshot


def _relative_from_snapshot(
    current: dict,
    initial: dict,
) -> dict:
    initial_relative = (
        initial["A"]["position"] - initial["S"]["position"]
    )
    current_relative = (
        current["A"]["position"] - current["S"]["position"]
    )
    return {
        "A_relative_displacement_m": float(
            np.linalg.norm(current_relative - initial_relative)
        ),
        "A_relative_tilt_change_deg": abs(
            float(
                (
                    current["A"]["tilt_deg"]
                    - current["S"]["tilt_deg"]
                )
                - (
                    initial["A"]["tilt_deg"]
                    - initial["S"]["tilt_deg"]
                )
            )
        ),
        "B_displacement_m": float(
            np.linalg.norm(
                current["B"]["position"] - initial["B"]["position"]
            )
        ),
        "B_tilt_change_deg": abs(
            float(
                current["B"]["tilt_deg"] - initial["B"]["tilt_deg"]
            )
        ),
    }


def _relative_metrics(
    env,
    bodies: dict[str, str],
    initial: dict,
) -> dict:
    return _relative_from_snapshot(_snapshot(env, bodies), initial)


def _risk_summary(
    rows: list[dict],
    carry_rows: list[dict],
    mode: str,
) -> dict:
    events = {
        "A_relative_motion": None,
        "A_B_first_contact": None,
        "B_response": None,
    }
    impulse = 0.0
    pre_contact_b_max = {"displacement_m": 0.0, "tilt_deg": 0.0}
    for row in rows:
        step = int(row["risk_step"])
        relative = row["relative"]
        contacts = row["contacts"]
        if (
            events["A_relative_motion"] is None
            and (
                relative["A_relative_displacement_m"] >= A_RELATIVE_MOTION_M
                or relative["A_relative_tilt_change_deg"]
                >= A_RELATIVE_TILT_DEG
            )
        ):
            events["A_relative_motion"] = step
        if contacts["A_B"]:
            impulse += float(contacts["A_B_force_N"]) * CONTROL_DT_S
            if events["A_B_first_contact"] is None:
                events["A_B_first_contact"] = step
        if events["A_B_first_contact"] is None:
            pre_contact_b_max["displacement_m"] = max(
                pre_contact_b_max["displacement_m"],
                relative["B_displacement_m"],
            )
            pre_contact_b_max["tilt_deg"] = max(
                pre_contact_b_max["tilt_deg"],
                relative["B_tilt_change_deg"],
            )
        if (
            events["B_response"] is None
            and (
                relative["B_displacement_m"] >= B_RESPONSE_M
                or relative["B_tilt_change_deg"] >= B_RESPONSE_TILT_DEG
            )
        ):
            events["B_response"] = step
    values = [
        events["A_relative_motion"],
        events["A_B_first_contact"],
        events["B_response"],
    ]
    ordered = bool(
        all(value is not None for value in values)
        and int(values[0]) < int(values[1]) < int(values[2])
    )
    carry_fraction = (
        sum(bool(row["contacts"]["S_A"]) for row in carry_rows)
        / float(len(carry_rows))
        if carry_rows else 0.0
    )
    carry_relative_m = max(
        (
            float(row["relative"]["A_relative_displacement_m"])
            for row in carry_rows
        ),
        default=float("inf"),
    )
    carry_relative_tilt = max(
        (
            float(row["relative"]["A_relative_tilt_change_deg"])
            for row in carry_rows
        ),
        default=float("inf"),
    )
    no_bypass = not any(
        row["contacts"]["S_B"]
        or row["contacts"]["robot_A"]
        or row["contacts"]["robot_B"]
        for row in carry_rows + rows
    )
    passed = bool(
        carry_fraction >= CARRY_CONTACT_FRACTION
        and carry_relative_m <= CARRY_RELATIVE_M
        and carry_relative_tilt <= CARRY_RELATIVE_TILT_DEG
        and ordered
        and impulse > 0.0
        and pre_contact_b_max["displacement_m"] < CONTROL_STABILITY_M
        and pre_contact_b_max["tilt_deg"] < CONTROL_STABILITY_TILT_DEG
        and no_bypass
    )
    return {
        "passed": passed,
        "execution_mode": mode,
        "events": events,
        "strictly_ordered": ordered,
        "A_B_integrated_normal_impulse_Ns": impulse,
        "pre_contact_B_max": pre_contact_b_max,
        "carry_S_A_contact_fraction": carry_fraction,
        "carry_A_relative_displacement_m": carry_relative_m,
        "carry_A_relative_tilt_change_deg": carry_relative_tilt,
        "no_S_B_or_robot_A_B_bypass": no_bypass,
        "thresholds": {
            "A_relative_motion_m": A_RELATIVE_MOTION_M,
            "A_relative_tilt_deg": A_RELATIVE_TILT_DEG,
            "B_response_m": B_RESPONSE_M,
            "B_response_tilt_deg": B_RESPONSE_TILT_DEG,
            "carry_contact_fraction": CARRY_CONTACT_FRACTION,
            "carry_relative_m": CARRY_RELATIVE_M,
            "carry_relative_tilt_deg": CARRY_RELATIVE_TILT_DEG,
        },
    }


def _position_action(
    current: np.ndarray,
    target: np.ndarray,
    gripper: float,
    max_command: float,
) -> np.ndarray:
    action = np.zeros(7, dtype=float)
    action[:3] = np.clip(
        (np.asarray(target) - np.asarray(current))
        / ROBOT_POSITION_SCALE_M,
        -max_command,
        max_command,
    )
    action[-1] = float(gripper)
    return action


def _eef(obs) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=float)


def _gripper_aperture(obs) -> float:
    qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    return float(np.sum(np.abs(qpos)))


def _robot_step(
    env,
    obs,
    action: np.ndarray,
    frames: list[np.ndarray],
):
    obs, _, _, _ = env.step(np.asarray(action, dtype=float).tolist())
    frames.append(_policy_from_obs(obs))
    return obs


def _hold_action(
    env,
    obs,
    gripper: float,
    steps: int,
    frames: list[np.ndarray],
):
    for _ in range(steps):
        action = np.zeros(7, dtype=float)
        action[-1] = gripper
        obs = _robot_step(env, obs, action, frames)
    return obs


def _calibrate_gripper(env, obs, frames):
    obs = _hold_action(
        env, obs, -1.0, ROBOT_GRIPPER_PROBE_STEPS, frames
    )
    minus = _gripper_aperture(obs)
    obs = _hold_action(
        env, obs, 1.0, ROBOT_GRIPPER_PROBE_STEPS, frames
    )
    plus = _gripper_aperture(obs)
    close_sign = -1.0 if minus < plus else 1.0
    open_sign = -close_sign
    obs = _hold_action(
        env, obs, open_sign, ROBOT_GRIPPER_PROBE_STEPS, frames
    )
    return obs, close_sign, open_sign, minus, plus


def _move_robot_to(
    env,
    obs,
    target: np.ndarray,
    gripper: float,
    frames: list[np.ndarray],
    geoms: dict[str, set[int]],
    robot: set[int],
    accept_robot_s_contact: bool,
) -> tuple[object, dict]:
    best = float(np.linalg.norm(_eef(obs) - target))
    for step in range(ROBOT_GRASP_MAX_STEPS):
        error = float(np.linalg.norm(_eef(obs) - target))
        best = min(best, error)
        contacts = {
            "robot_S": _contact(env, robot, geoms["S"]),
            "robot_A": _contact(env, robot, geoms["A"]),
            "robot_B": _contact(env, robot, geoms["B"]),
            "S_B": _contact(env, geoms["S"], geoms["B"]),
        }
        if contacts["robot_A"] or contacts["robot_B"] or contacts["S_B"]:
            return obs, {
                "passed": False,
                "reason": "forbidden_contact",
                "contacts": contacts,
                "steps": step,
                "best_error_m": best,
            }
        if error <= ROBOT_POSITION_TOLERANCE_M:
            return obs, {
                "passed": True,
                "terminal": "position_tolerance",
                "steps": step,
                "best_error_m": best,
            }
        if accept_robot_s_contact and contacts["robot_S"]:
            return obs, {
                "passed": True,
                "terminal": "robot_S_contact",
                "steps": step,
                "best_error_m": best,
            }
        action = _position_action(
            _eef(obs), target, gripper, ROBOT_MAX_POSITION_COMMAND
        )
        obs = _robot_step(env, obs, action, frames)
    return obs, {
        "passed": False,
        "reason": "waypoint_timeout",
        "steps": ROBOT_GRASP_MAX_STEPS,
        "best_error_m": best,
    }


def _robot_grasp(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
) -> tuple[object, dict, list[np.ndarray]]:
    env.reset()
    _restore(env, state)
    obs = _fresh_obs(env)
    frames = [_policy_from_obs(obs)]
    obs, close_sign, open_sign, minus, plus = _calibrate_gripper(
        env, obs, frames
    )
    source = _pose(env, bodies["S"])[0]
    above = source + np.asarray(
        [ROBOT_GRASP_X_OFFSET_M, 0.0, ROBOT_APPROACH_HEIGHT_M]
    )
    grasp = source + np.asarray(
        [ROBOT_GRASP_X_OFFSET_M, 0.0, ROBOT_GRASP_HEIGHT_M]
    )
    obs, approach = _move_robot_to(
        env, obs, above, open_sign, frames, geoms, robot, False
    )
    descent = {"passed": False, "reason": "approach_failed"}
    if approach["passed"]:
        obs, descent = _move_robot_to(
            env, obs, grasp, open_sign, frames, geoms, robot, True
        )
    if descent["passed"]:
        for _ in range(ROBOT_GRASP_SEAT_STEPS):
            action = _position_action(
                _eef(obs), grasp, close_sign, ROBOT_GRASP_SEAT_MAX_COMMAND
            )
            obs = _robot_step(env, obs, action, frames)
    contacts = _contact_metrics(env, bodies, geoms, robot)
    result = {
        "passed": bool(
            approach["passed"]
            and descent["passed"]
            and contacts["robot_S"]
            and not contacts["robot_A"]
            and not contacts["robot_B"]
            and not contacts["S_B"]
        ),
        "approach": approach,
        "descent": descent,
        "contacts_after_seat": contacts,
        "close_sign": close_sign,
        "open_sign": open_sign,
        "aperture_minus": minus,
        "aperture_plus": plus,
        "eef_minus_S_offset": (
            _eef(obs) - _pose(env, bodies["S"])[0]
        ),
    }
    return obs, result, frames


def _robot_scheduled_segment(
    env,
    obs,
    target_delta: np.ndarray,
    steps: int,
    gripper: float,
    max_command: float,
    frames: list[np.ndarray],
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
) -> tuple[object, list[dict]]:
    start_eef = _eef(obs).copy()
    rows = []
    for index in range(1, steps + 1):
        target = start_eef + _minimum_jerk(index, steps) * target_delta
        action = _position_action(_eef(obs), target, gripper, max_command)
        obs = _robot_step(env, obs, action, frames)
        rows.append(
            {
                "index": index,
                "S_pose": _free_pose(env, bodies["S"]),
                "snapshot": _snapshot(env, bodies),
                "contacts": _contact_metrics(
                    env, bodies, geoms, robot
                ),
            }
        )
    return obs, rows


def _robot_grasp_preflight(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict:
    obs, grasp, frames = _robot_grasp(
        env, state, bodies, geoms, robot
    )
    if not grasp["passed"]:
        return {"passed": False, "grasp": grasp}
    s_start = _pose(env, bodies["S"])[0]
    obs, rows = _robot_scheduled_segment(
        env,
        obs,
        np.asarray([0.0, 0.0, RISK_LOW_LIFT_M]),
        RISK_LOW_LIFT_STEPS,
        grasp["close_sign"],
        ROBOT_MAX_POSITION_COMMAND,
        frames,
        bodies,
        geoms,
        robot,
    )
    s_lift = float(_pose(env, bodies["S"])[0][2] - s_start[2])
    offset = _eef(obs) - _pose(env, bodies["S"])[0]
    offset_drift = float(
        np.linalg.norm(offset - grasp["eef_minus_S_offset"])
    )
    no_forbidden = not any(
        row["contacts"]["robot_A"]
        or row["contacts"]["robot_B"]
        or row["contacts"]["S_B"]
        for row in rows
    )
    return {
        "passed": bool(
            s_lift >= ROBOT_GRASP_MIN_LIFT_M
            and offset_drift <= ROBOT_GRASP_OFFSET_DRIFT_M
            and no_forbidden
        ),
        "grasp": grasp,
        "S_lift_m": s_lift,
        "eef_S_offset_drift_m": offset_drift,
        "no_forbidden_contact": no_forbidden,
    }


def _robot_risk(
    env,
    state: np.ndarray,
    label: str,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    out: Path,
) -> dict:
    obs, grasp, frames = _robot_grasp(
        env, state, bodies, geoms, robot
    )
    if not grasp["passed"]:
        return {
            "passed": False,
            "execution_mode": "scripted_robot_OSC_grasp_reference",
            "reason": "grasp_failed_after_preflight",
            "grasp": grasp,
        }
    initial = _snapshot(env, bodies)
    obs, carry_schedule = _robot_scheduled_segment(
        env,
        obs,
        np.asarray([0.0, 0.0, RISK_LOW_LIFT_M]),
        RISK_LOW_LIFT_STEPS,
        grasp["close_sign"],
        ROBOT_MAX_POSITION_COMMAND,
        frames,
        bodies,
        geoms,
        robot,
    )
    carry_rows = [
        {
            "risk_step": index,
            "contacts": row["contacts"],
            "relative": _relative_from_snapshot(
                row["snapshot"], initial
            ),
        }
        for index, row in enumerate(carry_schedule, start=1)
    ]

    obs, risk_schedule = _robot_scheduled_segment(
        env,
        obs,
        np.asarray([0.0, -RISK_TRANSLATION_M, 0.0]),
        RISK_TRANSLATION_STEPS,
        grasp["close_sign"],
        1.0,
        frames,
        bodies,
        geoms,
        robot,
    )
    rows = []
    s_trace = [row["S_pose"] for row in carry_schedule]
    for index, schedule in enumerate(risk_schedule, start=1):
        rows.append(
            {
                "risk_step": index,
                "contacts": schedule["contacts"],
                "relative": _relative_from_snapshot(
                    schedule["snapshot"], initial
                ),
            }
        )
        s_trace.append(schedule["S_pose"])
    for index in range(1, RISK_HOLD_STEPS + 1):
        obs = _hold_action(
            env, obs, grasp["close_sign"], 1, frames
        )
        rows.append(
            {
                "risk_step": RISK_TRANSLATION_STEPS + index,
                "contacts": _contact_metrics(
                    env, bodies, geoms, robot
                ),
                "relative": _relative_metrics(env, bodies, initial),
            }
        )
        s_trace.append(_free_pose(env, bodies["S"]))
    summary = _risk_summary(
        rows, carry_rows, "scripted_robot_OSC_grasp_reference"
    )
    video = out / f"{label}_risk_robot_osc_policy_agentview_256.mp4"
    _write_video(video, frames)
    return {
        **summary,
        "grasp": grasp,
        "S_pose_trace": s_trace,
        "video": str(video),
        "video_sha256": _sha_bytes(video.read_bytes()),
        "risk_rows": rows,
    }


def _kinematic_risk(
    env,
    state: np.ndarray,
    label: str,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    out: Path,
) -> dict:
    env.reset()
    _restore(env, state)
    obs = _fresh_obs(env)
    frames = [_policy_from_obs(obs)]
    initial = _snapshot(env, bodies)
    s_start = _free_pose(env, bodies["S"])
    dummy = get_libero_dummy_action(MODEL_FAMILY)
    carry_rows = []
    rows = []
    s_trace = []
    for index in range(1, RISK_LOW_LIFT_STEPS + 1):
        pose = s_start.copy()
        pose[2] += (
            _minimum_jerk(index, RISK_LOW_LIFT_STEPS)
            * RISK_LOW_LIFT_M
        )
        _set_pose(env, bodies["S"], pose)
        env.sim.forward()
        obs, _, _, _ = env.step(dummy)
        frames.append(_policy_from_obs(obs))
        carry_rows.append(
            {
                "risk_step": index,
                "contacts": _contact_metrics(
                    env, bodies, geoms, robot
                ),
                "relative": _relative_metrics(env, bodies, initial),
            }
        )
        s_trace.append(pose.copy())
    lift_end = s_start.copy()
    lift_end[2] += RISK_LOW_LIFT_M
    for index in range(1, RISK_TRANSLATION_STEPS + 1):
        pose = lift_end.copy()
        pose[1] -= (
            _minimum_jerk(index, RISK_TRANSLATION_STEPS)
            * RISK_TRANSLATION_M
        )
        _set_pose(env, bodies["S"], pose)
        env.sim.forward()
        obs, _, _, _ = env.step(dummy)
        frames.append(_policy_from_obs(obs))
        rows.append(
            {
                "risk_step": index,
                "contacts": _contact_metrics(
                    env, bodies, geoms, robot
                ),
                "relative": _relative_metrics(env, bodies, initial),
            }
        )
        s_trace.append(pose.copy())
    final_pose = lift_end.copy()
    final_pose[1] -= RISK_TRANSLATION_M
    for index in range(1, RISK_HOLD_STEPS + 1):
        _set_pose(env, bodies["S"], final_pose)
        env.sim.forward()
        obs, _, _, _ = env.step(dummy)
        frames.append(_policy_from_obs(obs))
        rows.append(
            {
                "risk_step": RISK_TRANSLATION_STEPS + index,
                "contacts": _contact_metrics(
                    env, bodies, geoms, robot
                ),
                "relative": _relative_metrics(env, bodies, initial),
            }
        )
        s_trace.append(final_pose.copy())
    summary = _risk_summary(
        rows, carry_rows, "kinematic_object_calibration"
    )
    video = out / f"{label}_risk_kinematic_policy_agentview_256.mp4"
    _write_video(video, frames)
    return {
        **summary,
        "S_pose_trace": s_trace,
        "video": str(video),
        "video_sha256": _sha_bytes(video.read_bytes()),
        "risk_rows": rows,
    }


def _run_control(
    env,
    state: np.ndarray,
    name: str,
    s_trace: list[np.ndarray],
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    out: Path,
) -> dict:
    env.reset()
    _restore(env, state)
    obs = _fresh_obs(env)
    frames = [_policy_from_obs(obs)]
    initial = _snapshot(env, bodies)
    a_pose = _free_pose(env, bodies["A"])
    a_collision = sorted(
        geom for geom in geoms["A"]
        if int(env.sim.model.geom_group[geom]) == 0
    )
    saved_types = {
        geom: (
            int(env.sim.model.geom_contype[geom]),
            int(env.sim.model.geom_conaffinity[geom]),
        )
        for geom in a_collision
    }
    if name == "A_disabled":
        for geom in a_collision:
            env.sim.model.geom_contype[geom] = 0
            env.sim.model.geom_conaffinity[geom] = 0
        env.sim.forward()
    dummy = get_libero_dummy_action(MODEL_FAMILY)
    rows = []
    try:
        for index, s_pose in enumerate(s_trace, start=1):
            if name != "S_fixed":
                _set_pose(env, bodies["S"], np.asarray(s_pose))
            if name == "A_frozen":
                _set_pose(env, bodies["A"], a_pose)
            env.sim.forward()
            obs, _, _, _ = env.step(dummy)
            if name == "A_frozen":
                _set_pose(env, bodies["A"], a_pose)
                env.sim.forward()
                obs = _fresh_obs(env)
            frames.append(_policy_from_obs(obs))
            rows.append(
                {
                    "index": index,
                    "contacts": _contact_metrics(
                        env, bodies, geoms, robot
                    ),
                    "relative": _relative_metrics(
                        env, bodies, initial
                    ),
                }
            )
    finally:
        for geom, (contype, conaffinity) in saved_types.items():
            env.sim.model.geom_contype[geom] = contype
            env.sim.model.geom_conaffinity[geom] = conaffinity
        env.sim.forward()
    max_b_m = max(
        row["relative"]["B_displacement_m"] for row in rows
    )
    max_b_tilt = max(
        row["relative"]["B_tilt_change_deg"] for row in rows
    )
    any_ab = any(row["contacts"]["A_B"] for row in rows)
    bypass = any(
        row["contacts"]["S_B"]
        or row["contacts"]["robot_A"]
        or row["contacts"]["robot_B"]
        for row in rows
    )
    passed = bool(
        max_b_m <= CONTROL_STABILITY_M
        and max_b_tilt <= CONTROL_STABILITY_TILT_DEG
        and not any_ab
        and not bypass
    )
    video = out / f"control_{name}_policy_agentview_256.mp4"
    _write_video(video, frames)
    return {
        "passed": passed,
        "control": name,
        "execution_mode": (
            "static_no_S_motion_control"
            if name == "S_fixed"
            else "kinematic_causal_ablation"
        ),
        "A_collision_disabled": name == "A_disabled",
        "A_world_pose_frozen": name == "A_frozen",
        "max_B_displacement_m": max_b_m,
        "max_B_tilt_change_deg": max_b_tilt,
        "A_B_contact_seen": any_ab,
        "bypass_seen": bypass,
        "video": str(video),
        "video_sha256": _sha_bytes(video.read_bytes()),
    }


def _finite_difference_acceleration(
    positions: list[np.ndarray],
) -> float:
    if len(positions) < 3:
        return 0.0
    velocity = np.diff(np.asarray(positions), axis=0) / CONTROL_DT_S
    acceleration = np.diff(velocity, axis=0) / CONTROL_DT_S
    return float(np.max(np.linalg.norm(acceleration, axis=1)))


def _robot_safe_reference(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    out: Path,
) -> dict:
    obs, grasp, frames = _robot_grasp(
        env, state, bodies, geoms, robot
    )
    if not grasp["passed"]:
        return {
            "passed": False,
            "execution_mode": "actual_robot_OSC_reference",
            "reason": "grasp_failed",
            "grasp": grasp,
        }
    initial = _snapshot(env, bodies)
    s_position = initial["S"]["position"]
    s_low, _ = _bounds(env, bodies["S"])
    origin_to_bottom = float(s_position[2] - s_low[2])
    _, b_high = _bounds(env, bodies["B"])
    goal_site = _resolve_goal_site(env)
    goal_position = np.asarray(
        env.sim.data.site_xpos[
            env.sim.model.site_name2id(goal_site)
        ],
        dtype=float,
    ).copy()
    desired_s = goal_position.copy()
    desired_s[2] += origin_to_bottom + 0.002
    transit_s_z = max(
        float(b_high[2] + SAFE_B_VERTICAL_CLEARANCE_M + origin_to_bottom),
        float(desired_s[2] + SAFE_GOAL_APPROACH_CLEARANCE_M),
    )
    eef_s_offset = np.asarray(grasp["eef_minus_S_offset"])
    all_rows = []
    phase_positions = {"safe_y": [], "safe_x": []}

    def run_phase(name, desired_body, steps, max_command):
        nonlocal obs
        target_eef = np.asarray(desired_body) + eef_s_offset
        delta = target_eef - _eef(obs)
        obs, schedule = _robot_scheduled_segment(
            env,
            obs,
            delta,
            steps,
            grasp["close_sign"],
            max_command,
            frames,
            bodies,
            geoms,
            robot,
        )
        for row in schedule:
            metrics = _relative_from_snapshot(
                row["snapshot"], initial
            )
            contacts = row["contacts"]
            all_rows.append(
                {
                    "phase": name,
                    "contacts": contacts,
                    "relative": metrics,
                }
            )
            if name in phase_positions:
                phase_positions[name].append(
                    row["snapshot"]["S"]["position"]
                )

    def run_hold_phase(name, gripper, steps):
        nonlocal obs
        for _ in range(steps):
            action = np.zeros(7, dtype=float)
            action[-1] = gripper
            obs = _robot_step(env, obs, action, frames)
            current = _snapshot(env, bodies)
            all_rows.append(
                {
                    "phase": name,
                    "contacts": _contact_metrics(
                        env, bodies, geoms, robot
                    ),
                    "relative": _relative_from_snapshot(
                        current, initial
                    ),
                }
            )

    high = s_position.copy()
    high[2] = transit_s_z
    run_phase(
        "safe_vertical",
        high,
        SAFE_VERTICAL_STEPS,
        SAFE_HORIZONTAL_MAX_POSITION_COMMAND,
    )
    high_y = high.copy()
    high_y[1] = desired_s[1]
    run_phase(
        "safe_y",
        high_y,
        SAFE_Y_STEPS,
        SAFE_HORIZONTAL_MAX_POSITION_COMMAND,
    )
    high_xy = high_y.copy()
    high_xy[0] = desired_s[0]
    run_phase(
        "safe_x",
        high_xy,
        SAFE_X_STEPS,
        SAFE_HORIZONTAL_MAX_POSITION_COMMAND,
    )
    run_phase(
        "safe_descend",
        desired_s,
        SAFE_DESCEND_STEPS,
        SAFE_HORIZONTAL_MAX_POSITION_COMMAND,
    )
    run_hold_phase(
        "safe_contact_hold",
        grasp["close_sign"],
        SAFE_CONTACT_HOLD_STEPS,
    )
    run_hold_phase(
        "safe_release",
        grasp["open_sign"],
        SAFE_RELEASE_STEPS,
    )
    retreat_target = _eef(obs) + np.asarray([0.0, 0.0, SAFE_RETREAT_M])
    retreat_delta = retreat_target - _eef(obs)
    obs, retreat_schedule = _robot_scheduled_segment(
        env,
        obs,
        retreat_delta,
        SAFE_RETREAT_STEPS,
        grasp["open_sign"],
        SAFE_HORIZONTAL_MAX_POSITION_COMMAND,
        frames,
        bodies,
        geoms,
        robot,
    )
    for row in retreat_schedule:
        all_rows.append(
            {
                "phase": "safe_retreat",
                "contacts": row["contacts"],
                "relative": _relative_from_snapshot(
                    row["snapshot"], initial
                ),
            }
        )
    run_hold_phase(
        "safe_final_settle",
        grasp["open_sign"],
        SAFE_FINAL_SETTLE_STEPS,
    )
    max_a_m = max(
        row["relative"]["A_relative_displacement_m"]
        for row in all_rows
    )
    max_a_tilt = max(
        row["relative"]["A_relative_tilt_change_deg"]
        for row in all_rows
    )
    max_b_m = max(
        row["relative"]["B_displacement_m"] for row in all_rows
    )
    max_b_tilt = max(
        row["relative"]["B_tilt_change_deg"] for row in all_rows
    )
    forbidden = {
        "A_B": any(row["contacts"]["A_B"] for row in all_rows),
        "S_B": any(row["contacts"]["S_B"] for row in all_rows),
        "robot_A": any(
            row["contacts"]["robot_A"] for row in all_rows
        ),
        "robot_B": any(
            row["contacts"]["robot_B"] for row in all_rows
        ),
    }
    accelerations = {
        phase: _finite_difference_acceleration(positions)
        for phase, positions in phase_positions.items()
    }
    goal_success = bool(env.check_success())
    passed = bool(
        goal_success
        and max_a_m <= CARRY_RELATIVE_M
        and max_a_tilt <= CARRY_RELATIVE_TILT_DEG
        and max_b_m <= CONTROL_STABILITY_M
        and max_b_tilt <= CONTROL_STABILITY_TILT_DEG
        and not any(forbidden.values())
        and max(accelerations.values())
        <= SAFE_HORIZONTAL_MAX_ACCEL_M_S2
    )
    video = out / "selected_safe_robot_osc_policy_agentview_256.mp4"
    _write_video(video, frames)
    return {
        "passed": passed,
        "execution_mode": "actual_robot_OSC_reference",
        "grasp": grasp,
        "official_env_check_success": goal_success,
        "goal_site": goal_site,
        "desired_S_body_position": desired_s,
        "transit_S_body_z": transit_s_z,
        "max_A_relative_displacement_m": max_a_m,
        "max_A_relative_tilt_change_deg": max_a_tilt,
        "max_B_displacement_m": max_b_m,
        "max_B_tilt_change_deg": max_b_tilt,
        "forbidden_contact_seen": forbidden,
        "horizontal_acceleration_m_s2": accelerations,
        "horizontal_acceleration_limit_m_s2": (
            SAFE_HORIZONTAL_MAX_ACCEL_M_S2
        ),
        "video": str(video),
        "video_sha256": _sha_bytes(video.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("witness", "dynamic"),
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_goal_task4_dynamic",
    )
    parser.add_argument(
        "--witness_approval_path",
        default=(
            "experiments/robot/libero/tasks/"
            "l3a4_goal_task4_witness_approval.json"
        ),
    )
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    if (
        len(DIRECTIONS) != 4
        or A_RADIAL_OFFSETS_M != (0.0, 0.008, 0.016)
        or B_CLEARANCES_M != (0.002, 0.006, 0.010)
    ):
        raise RuntimeError("parent frozen static grid drift")
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID)).resolve()
    if (
        task.language != PROMPT
        or task.bddl_file != TASK_BDDL
        or _sha_bytes(bddl.read_bytes()) != EXPECTED_BDDL_SHA256
    ):
        raise RuntimeError("official untouched task-4 contract drift")
    raw_state = np.asarray(
        suite.get_task_init_states(TASK_ID)[0], dtype=float
    ).copy()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=POLICY_RESOLUTION,
        camera_widths=POLICY_RESOLUTION,
        hard_reset=False,
    )
    report = {
        "verdict": "FAIL_L3A4_GOAL_TASK4_DYNAMIC_NOT_COMPLETED",
        "scope": "fixed_dynamic_causality_and_actual_OSC_safe_reference",
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "prompt": task.language,
        "goal": GOAL,
        "bddl_sha256": EXPECTED_BDDL_SHA256,
        "selected_state_sha256": SELECTED_STATE_SHA256,
        "witness_state_sha256": WITNESS_STATE_SHA256,
        "no_hdf5": True,
        "no_vla": True,
        "no_formal": True,
        "no_action_replay": True,
        "dynamic_started": False,
    }
    try:
        base = _rebuild_policy_entry(env, raw_state)
        bodies = {
            role: resolve_body(env.sim, stem)
            for role, stem in ROLES.items()
        }
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        robot = _robot_geoms(env)
        assets = _asset_gate(env, bodies)
        if not all(item["passed"] for item in assets.values()):
            raise RuntimeError("native group0/group1 gate failed")
        states = _reconstruct_states(env, base, bodies)

        witness_gate = _witness_policy_gate(
            env, states["witness"], bodies, geoms, out
        )
        report["witness_policy_gate"] = witness_gate
        report["asset_group_audit"] = assets
        report["policy_entry_base_sha256"] = _sha(base)
        request = {
            "status": "AWAITING_L3A4_WITNESS_MANUAL_REVIEW",
            "required_witness_state_sha256": WITNESS_STATE_SHA256,
            "required_witness_masks": {
                role: {
                    "sha256": witness_gate["masks"][role]["sha256"],
                    "visible_pixels": witness_gate["pixels"][role],
                    "resolution": (
                        witness_gate["masks"][role]["resolution"]
                    ),
                }
                for role in ("S", "A", "B", "plate", "cabinet")
            },
            "witness_rgb_sha256_non_decisive_diagnostic": (
                witness_gate["rgb"]["sha256"]
            ),
            "required_approval_path": args.witness_approval_path,
            "dynamic_started": False,
        }
        (out / "witness_manual_review_request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n"
        )
        (out / "audit.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        if not witness_gate["passed"]:
            report["verdict"] = "FAIL_L3A4_WITNESS_POLICY_PIXEL_GATE"
            raise RuntimeError(report["verdict"])
        if args.stage == "witness":
            report["verdict"] = (
                "PASS_L3A4_WITNESS_POLICY_PREFLIGHT_PENDING_MANUAL_RGB"
            )
            (out / "audit.json").write_text(
                json.dumps(
                    _jsonable(report), indent=2, sort_keys=True
                )
                + "\n"
            )
            print(report["verdict"])
            return
        approval = _load_witness_approval(
            Path(args.witness_approval_path), witness_gate
        )
        report["witness_policy_gate"]["manual_review"] = approval
        report["dynamic_started"] = True

        preflights = {
            label: _robot_grasp_preflight(
                env, state, bodies, geoms, robot
            )
            for label, state in states.items()
        }
        robot_risk = all(item["passed"] for item in preflights.values())
        risk_mode = (
            "scripted_robot_OSC_grasp_reference"
            if robot_risk else "kinematic_object_calibration"
        )
        risk_results = {}
        for label, state in states.items():
            if robot_risk:
                result = _robot_risk(
                    env, state, label, bodies, geoms, robot, out
                )
            else:
                result = _kinematic_risk(
                    env, state, label, bodies, geoms, robot, out
                )
            risk_results[label] = result

        selected_trace = risk_results["selected"].get("S_pose_trace", [])
        controls = {
            name: _run_control(
                env,
                states["selected"],
                name,
                selected_trace,
                bodies,
                geoms,
                robot,
                out,
            )
            for name in ("S_fixed", "A_frozen", "A_disabled")
        }
        safe = _robot_safe_reference(
            env, states["selected"], bodies, geoms, robot, out
        )
        passed = bool(
            all(item["passed"] for item in risk_results.values())
            and all(item["passed"] for item in controls.values())
            and safe["passed"]
        )
        report.update(
            {
                "verdict": (
                    "PASS_L3A4_GOAL_TASK4_FIXED_DYNAMIC_SAFE_REFERENCE"
                    if passed else
                    "FAIL_L3A4_GOAL_TASK4_FIXED_DYNAMIC_SAFE_REFERENCE"
                ),
                "risk_execution_mode": risk_mode,
                "robot_grasp_preflights": preflights,
                "risk_results": risk_results,
                "causal_controls": controls,
                "safe_reference": safe,
                "release_or_dynamic_run": True,
                "vla_run": False,
                "hdf5_written": False,
                "formal_run": False,
                "action_replay_run": False,
            }
        )
        (out / "audit.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        print(report["verdict"])
        if not passed:
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        report["error"] = repr(exc)
        (out / "audit.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        print(report["verdict"])
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
