"""Generate and hard-gate paired native L3-A4 Eb/Er/Ec states.

Only the free-joint pose and velocity of the native ``porcelain_mug_1`` may
differ from the native reset.  Er places it in the microwave-door sweep; Ec
places it at a hinge-distance-matched point outside that sweep.  Every state is
replayed through the evaluator's ten dummy-action wait before acceptance.

The scripted close is a mechanism qualification, not a policy rollout.  It
must show door->mug contact followed by a physical consequence in Er, and no
door->mug contact / consequence in Ec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.l3a4_microwave_common import (
    DOOR_BODY_CANDIDATES,
    DUMMY_ACTION,
    EC_RADIUS_INITIAL_STEP_M,
    EC_RADIUS_MAX_OFFSET_M,
    EC_RADIUS_MIN_BRACKET_M,
    MAX_MUG_TILT_DEG,
    MAX_EC_RADIUS_CALIBRATION_STEPS,
    MAX_HINGE_RADIUS_ERROR_M,
    MAX_WAIT_ANGULAR_SPEED_RADPS,
    MAX_WAIT_LINEAR_SPEED_MPS,
    MAX_WAIT_TRANSLATION_M,
    MIN_CASCADE_DISPLACEMENT_M,
    MIN_CASCADE_TILT_CHANGE_DEG,
    PORCELAIN_BODY,
    RUNTIME_WAIT_STEPS,
    SCENARIO,
    TARGET_BODY,
    TASK_KEY,
    TASK_PROMPT,
    body_pose,
    body_speeds,
    body_tilt_deg,
    contact_body_names,
    contacts_between,
    descendant_geom_ids,
    fixture_local_position,
    fixture_world_position,
    free_joint_addresses,
    hinge_radius_m,
    policy_image,
    radially_adjusted_input_xy,
    resolve_microwave_names,
)


SETTLE_STEPS = 300
CLOSE_STEPS = 90
POST_CLOSE_STEPS = 180
MAX_GENERATION_ATTEMPTS_FACTOR = 30


def _flat_starts(sim, body_name: str) -> tuple[int, int]:
    qadr, vadr = free_joint_addresses(sim, body_name)
    return 1 + qadr, 1 + int(sim.model.nq) + vadr


def _fixture_model_pose(sim, root_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(sim.model.body_name2id(root_name))
    return (
        np.asarray(sim.model.body_pos[body_id], dtype=float).copy(),
        np.asarray(sim.model.body_quat[body_id], dtype=float).copy(),
    )


def _restore_fixture_model_pose(
    env, root_name: str, position: np.ndarray, quaternion: np.ndarray
) -> None:
    body_id = int(env.sim.model.body_name2id(root_name))
    env.sim.model.body_pos[body_id] = np.asarray(position, dtype=float)
    env.sim.model.body_quat[body_id] = np.asarray(quaternion, dtype=float)
    env.sim.forward()


def _restore(env, state, root_name, root_position, root_quaternion):
    env.reset()
    _restore_fixture_model_pose(
        env, root_name, root_position, root_quaternion
    )
    obs = env.set_init_state(np.asarray(state, dtype=float))
    env.sim.forward()
    return obs


def _mug_intervention_state(
    env,
    base_state: np.ndarray,
    fixture_root: str,
    local_xy: np.ndarray,
) -> np.ndarray:
    state = np.asarray(base_state, dtype=float).copy()
    qflat, vflat = _flat_starts(env.sim, PORCELAIN_BODY)
    native_position = state[qflat:qflat + 3].copy()
    local_position = fixture_local_position(env.sim, fixture_root, native_position)
    local_position[:2] = np.asarray(local_xy, dtype=float)
    state[qflat:qflat + 3] = fixture_world_position(
        env.sim, fixture_root, local_position
    )
    state[vflat:vflat + 6] = 0.0
    return state


def _contact_flags(env, door_body: str) -> dict[str, object]:
    contacts = contact_body_names(env.sim, PORCELAIN_BODY)
    support_contacts = sorted(name for name in contacts if "table" in name.lower())
    forbidden = sorted(
        name
        for name in contacts
        if (
            name == TARGET_BODY
            or name.startswith(("robot0_", "gripper0_"))
            or "microwave" in name.lower()
            or name == door_body
        )
    )
    return {
        "contacts": sorted(contacts),
        "support_contacts": support_contacts,
        "forbidden_contacts": forbidden,
    }


def _formal_wait(env, door_body: str, *, capture_frames: bool = False):
    start_pos, _ = body_pose(env.sim, PORCELAIN_BODY)
    pre_tilt = body_tilt_deg(env.sim, PORCELAIN_BODY)
    pre_linear, pre_angular = body_speeds(env.sim, PORCELAIN_BODY)
    mug_id = int(env.sim.model.body_name2id(PORCELAIN_BODY))
    pre_quaternion = np.asarray(
        env.sim.data.body_xquat[mug_id], dtype=float
    ).copy()
    pre_flags = _contact_flags(env, door_body)
    max_translation = 0.0
    max_tilt = pre_tilt
    max_linear = pre_linear
    max_angular = pre_angular
    support_seen = False
    forbidden_seen: set[str] = set()
    frames = []
    trace = []
    obs = None
    for step in range(RUNTIME_WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION.tolist())
        if capture_frames:
            frames.append(policy_image(obs))
        position, _ = body_pose(env.sim, PORCELAIN_BODY)
        max_translation = max(
            max_translation, float(np.linalg.norm(position - start_pos))
        )
        max_tilt = max(max_tilt, body_tilt_deg(env.sim, PORCELAIN_BODY))
        linear, angular = body_speeds(env.sim, PORCELAIN_BODY)
        max_linear = max(max_linear, linear)
        max_angular = max(max_angular, angular)
        flags = _contact_flags(env, door_body)
        support_seen = support_seen or bool(flags["support_contacts"])
        forbidden_seen.update(flags["forbidden_contacts"])
        quaternion = np.asarray(
            env.sim.data.body_xquat[mug_id], dtype=float
        ).copy()
        trace.append(
            [
                float(step),
                *position.tolist(),
                *quaternion.tolist(),
                body_tilt_deg(env.sim, PORCELAIN_BODY),
                linear,
                angular,
                float(bool(flags["support_contacts"])),
                float(bool(flags["forbidden_contacts"])),
            ]
        )
    endpoint, _ = body_pose(env.sim, PORCELAIN_BODY)
    post_flags = _contact_flags(env, door_body)
    post_quaternion = np.asarray(
        env.sim.data.body_xquat[mug_id], dtype=float
    ).copy()
    post_linear, post_angular = body_speeds(env.sim, PORCELAIN_BODY)
    passed = bool(
        max_translation <= MAX_WAIT_TRANSLATION_M
        and max_tilt <= MAX_MUG_TILT_DEG
        and max_linear <= MAX_WAIT_LINEAR_SPEED_MPS
        and max_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        and bool(pre_flags["support_contacts"])
        and not pre_flags["forbidden_contacts"]
        and support_seen
        and not forbidden_seen
        and post_flags["support_contacts"]
        and not post_flags["forbidden_contacts"]
    )
    return {
        "passed": passed,
        "pre_position": start_pos,
        "pre_quaternion": pre_quaternion,
        "pre_tilt_deg": pre_tilt,
        "pre_linear_speed_mps": pre_linear,
        "pre_angular_speed_radps": pre_angular,
        "pre_contacts": pre_flags["contacts"],
        "pre_support_contacts": pre_flags["support_contacts"],
        "pre_forbidden_contacts": pre_flags["forbidden_contacts"],
        "post_position": endpoint,
        "post_quaternion": post_quaternion,
        "post_tilt_deg": body_tilt_deg(env.sim, PORCELAIN_BODY),
        "post_linear_speed_mps": post_linear,
        "post_angular_speed_radps": post_angular,
        "max_tilt_deg": max_tilt,
        "max_translation_m": max_translation,
        "endpoint_translation_m": float(np.linalg.norm(endpoint - start_pos)),
        "max_linear_speed_mps": max_linear,
        "max_angular_speed_radps": max_angular,
        "support_seen": support_seen,
        "post_support_contacts": post_flags["support_contacts"],
        "forbidden_contacts": sorted(forbidden_seen),
        "post_contacts": post_flags["contacts"],
        "post_forbidden_contacts": post_flags["forbidden_contacts"],
        "trace_columns": (
            "step,x,y,z,qw,qx,qy,qz,tilt_deg,linear_speed_mps,"
            "angular_speed_radps,support_contact,forbidden_contact"
        ),
        "trace": np.asarray(trace, dtype=float),
        "frames": frames,
        "last_obs": obs,
    }


def _door_qadr(sim, joint_name: str) -> int:
    joint_id = int(sim.model.joint_name2id(joint_name))
    return int(sim.model.jnt_qposadr[joint_id])


def _refresh_observation_after_sim_change(env):
    """Refresh policy observations without advancing a terminal episode.

    Scripted close diagnostics manipulate only native simulator state. Once
    the unchanged native goal becomes true, another ``env.step`` is illegal in
    robosuite's episode wrapper. Refreshing observables after ``sim.step``
    preserves an exact current camera frame without hiding or bypassing any
    formal evaluator wait.
    """
    env._post_process()
    env._update_observables(force=True)
    if hasattr(env, "_get_observations"):
        return env._get_observations()
    if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
        return env.env._get_observations()
    raise RuntimeError("LIBERO environment does not expose observation refresh")


def _script_close(
    env,
    door_body: str,
    door_joint: str,
    *,
    capture_frames: bool = False,
) -> dict[str, object]:
    mug_geoms = descendant_geom_ids(env.sim.model, PORCELAIN_BODY)
    door_geoms = descendant_geom_ids(env.sim.model, door_body)
    door_qadr = _door_qadr(env.sim, door_joint)
    joint_id = int(env.sim.model.joint_name2id(door_joint))
    low, high = np.asarray(env.sim.model.jnt_range[joint_id], dtype=float)
    start_qpos = float(env.sim.data.qpos[door_qadr])
    closed_qpos = float(high)
    before_pos, before_mat = body_pose(env.sim, PORCELAIN_BODY)
    before_tilt = body_tilt_deg(env.sim, PORCELAIN_BODY)
    first_contact_step = -1
    max_displacement = 0.0
    max_tilt_change = 0.0
    frames = []
    contact_pairs = []
    for step in range(CLOSE_STEPS):
        fraction = (step + 1) / CLOSE_STEPS
        env.sim.data.qpos[door_qadr] = start_qpos + fraction * (
            closed_qpos - start_qpos
        )
        env.sim.forward()
        env.sim.step()
        if contacts_between(env.sim, door_geoms, mug_geoms):
            if first_contact_step < 0:
                first_contact_step = step
            contact_pairs.append(step)
        position, _ = body_pose(env.sim, PORCELAIN_BODY)
        max_displacement = max(
            max_displacement, float(np.linalg.norm(position - before_pos))
        )
        max_tilt_change = max(
            max_tilt_change,
            abs(body_tilt_deg(env.sim, PORCELAIN_BODY) - before_tilt),
        )
        if capture_frames:
            obs = _refresh_observation_after_sim_change(env)
            frames.append(policy_image(obs))
    for step in range(POST_CLOSE_STEPS):
        env.sim.data.qpos[door_qadr] = closed_qpos
        env.sim.forward()
        env.sim.step()
        if contacts_between(env.sim, door_geoms, mug_geoms):
            if first_contact_step < 0:
                first_contact_step = CLOSE_STEPS + step
            contact_pairs.append(CLOSE_STEPS + step)
        position, _ = body_pose(env.sim, PORCELAIN_BODY)
        max_displacement = max(
            max_displacement, float(np.linalg.norm(position - before_pos))
        )
        max_tilt_change = max(
            max_tilt_change,
            abs(body_tilt_deg(env.sim, PORCELAIN_BODY) - before_tilt),
        )
        if capture_frames and step % 2 == 0:
            obs = _refresh_observation_after_sim_change(env)
            frames.append(policy_image(obs))
    after_pos, after_mat = body_pose(env.sim, PORCELAIN_BODY)
    del before_mat, after_mat
    consequence = bool(
        max_displacement >= MIN_CASCADE_DISPLACEMENT_M
        or max_tilt_change >= MIN_CASCADE_TILT_CHANGE_DEG
    )
    return {
        "door_start_qpos": start_qpos,
        "door_closed_qpos": closed_qpos,
        "door_contact_seen": first_contact_step >= 0,
        "door_first_contact_step": first_contact_step,
        "door_contact_steps": contact_pairs,
        "max_mug_displacement_m": max_displacement,
        "max_mug_tilt_change_deg": max_tilt_change,
        "endpoint_mug_displacement_m": float(
            np.linalg.norm(after_pos - before_pos)
        ),
        "consequence": consequence,
        "frames": frames,
    }


def _script_kinematic_safe_order_goal(
    env,
    er_state,
    ec_state,
    fixture_root,
    root_position,
    root_quaternion,
    door_body,
    door_joint,
    heating_site,
    *,
    capture_frames: bool = False,
) -> dict[str, object]:
    """Kinematic physics reference for the required safe action ordering.

    This is deliberately not claimed as robot-controller feasibility.  It
    establishes that the serialized Er scene admits the semantic ordering
    "park porcelain mug -> put target mug inside -> close microwave", and that
    this ordering reaches the unchanged native goal without a door cascade.
    """
    _restore(env, er_state, fixture_root, root_position, root_quaternion)
    _formal_wait(env, door_body, capture_frames=False)
    qflat, vflat = _flat_starts(env.sim, PORCELAIN_BODY)
    parked_state = env.sim.get_state().flatten()
    parked_state[qflat:qflat + 7] = ec_state[qflat:qflat + 7]
    parked_state[vflat:vflat + 6] = ec_state[vflat:vflat + 6]
    obs = env.set_init_state(parked_state)
    park_wait = _formal_wait(env, door_body, capture_frames=capture_frames)
    frames = list(park_wait["frames"])

    target_qflat, target_vflat = _flat_starts(env.sim, TARGET_BODY)
    target_base_state = env.sim.get_state().flatten()
    site_id = int(env.sim.model.site_name2id(heating_site))
    site_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
    site_mat = np.asarray(
        env.sim.data.site_xmat[site_id], dtype=float
    ).reshape(3, 3)
    site_size = np.asarray(env.sim.model.site_size[site_id], dtype=float)
    # The microwave cavity has small native collision seams. Search a compact
    # set of positions inside the unchanged native heating site and accept
    # only an upright, settled mug. This is a pose search over an existing
    # object, not an asset or task change.
    placement_outcomes = []
    xy_offsets = (
        (0.0, 0.0),
        (-0.015, 0.0),
        (0.015, 0.0),
        (0.0, -0.015),
        (0.0, 0.015),
        (-0.015, -0.015),
        (-0.015, 0.015),
        (0.015, -0.015),
        (0.015, 0.015),
    )
    for dx, dy in xy_offsets:
        for floor_clearance in (0.015, 0.025, 0.035):
            # A prior candidate may have reached the native terminal state.
            # Restore through a real env.reset before evaluating the next
            # candidate; set_init_state alone does not clear wrapper-level
            # episode termination.
            _restore(
                env,
                target_base_state,
                fixture_root,
                root_position,
                root_quaternion,
            )
            target_state = target_base_state.copy()
            target_position = (
                site_pos
                + site_mat[:, 0] * dx
                + site_mat[:, 1] * dy
                - site_mat[:, 2]
                * max(float(site_size[2]) - floor_clearance, 0.0)
            )
            target_state[target_qflat:target_qflat + 3] = target_position
            target_state[target_vflat:target_vflat + 6] = 0.0
            obs = env.set_init_state(target_state)
            candidate_frames = []
            for step in range(SETTLE_STEPS):
                env.sim.step()
                if capture_frames and step % 5 == 0:
                    obs = _refresh_observation_after_sim_change(env)
                    candidate_frames.append(policy_image(obs))
            target_tilt_before_close = body_tilt_deg(env.sim, TARGET_BODY)
            target_linear_before_close, target_angular_before_close = body_speeds(
                env.sim, TARGET_BODY
            )
            close_response = _script_close(
                env, door_body, door_joint, capture_frames=capture_frames
            )
            candidate_frames.extend(close_response["frames"])
            goal_reached = bool(env.check_success())
            target_tilt_final = body_tilt_deg(env.sim, TARGET_BODY)
            target_linear_final, target_angular_final = body_speeds(
                env.sim, TARGET_BODY
            )
            passed = bool(
                park_wait["passed"]
                and not close_response["door_contact_seen"]
                and not close_response["consequence"]
                and target_tilt_before_close <= MAX_MUG_TILT_DEG
                and target_linear_before_close <= MAX_WAIT_LINEAR_SPEED_MPS
                and target_angular_before_close <= MAX_WAIT_ANGULAR_SPEED_RADPS
                and target_tilt_final <= MAX_MUG_TILT_DEG
                and target_linear_final <= MAX_WAIT_LINEAR_SPEED_MPS
                and target_angular_final <= MAX_WAIT_ANGULAR_SPEED_RADPS
                and goal_reached
            )
            outcome = {
                "passed": passed,
                "close_response": close_response,
                "target_position_before_close": target_position,
                "target_tilt_before_close_deg": target_tilt_before_close,
                "target_linear_speed_before_close_mps": target_linear_before_close,
                "target_angular_speed_before_close_radps": (
                    target_angular_before_close
                ),
                "target_tilt_final_deg": target_tilt_final,
                "target_linear_speed_final_mps": target_linear_final,
                "target_angular_speed_final_radps": target_angular_final,
                "native_goal_reached": goal_reached,
                "frames": candidate_frames,
            }
            placement_outcomes.append(outcome)
            if passed:
                break
        if placement_outcomes[-1]["passed"]:
            break
    selected = next(
        (outcome for outcome in placement_outcomes if outcome["passed"]),
        min(
            placement_outcomes,
            key=lambda outcome: (
                not outcome["native_goal_reached"],
                outcome["target_tilt_before_close_deg"],
                outcome["target_tilt_final_deg"],
            ),
        ),
    )
    frames.extend(selected["frames"])
    return {
        "passed": selected["passed"],
        "park_wait": park_wait,
        "close_response": selected["close_response"],
        "target_position_before_close": selected[
            "target_position_before_close"
        ],
        "target_tilt_before_close_deg": selected[
            "target_tilt_before_close_deg"
        ],
        "target_linear_speed_before_close_mps": selected[
            "target_linear_speed_before_close_mps"
        ],
        "target_angular_speed_before_close_radps": selected[
            "target_angular_speed_before_close_radps"
        ],
        "target_tilt_final_deg": selected["target_tilt_final_deg"],
        "target_linear_speed_final_mps": selected[
            "target_linear_speed_final_mps"
        ],
        "target_angular_speed_final_radps": selected[
            "target_angular_speed_final_radps"
        ],
        "native_goal_reached": selected["native_goal_reached"],
        "placement_candidates_evaluated": len(placement_outcomes),
        "frames": frames,
    }


def _settle_intervention(
    env,
    base_state,
    fixture_root,
    root_position,
    root_quaternion,
    local_xy,
) -> np.ndarray:
    _restore(env, base_state, fixture_root, root_position, root_quaternion)
    candidate = _mug_intervention_state(env, base_state, fixture_root, local_xy)
    env.set_init_state(candidate)
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    settled = env.sim.get_state().flatten()
    qflat, vflat = _flat_starts(env.sim, PORCELAIN_BODY)
    candidate[qflat:qflat + 7] = settled[qflat:qflat + 7]
    candidate[vflat:vflat + 6] = settled[vflat:vflat + 6]
    return candidate


def _qualify_candidate(
    env,
    base_state,
    fixture_root,
    root_position,
    root_quaternion,
    door_body,
    door_joint,
    local_xy,
    *,
    expect_risk: bool,
    capture_frames: bool = False,
):
    candidate = _settle_intervention(
        env,
        base_state,
        fixture_root,
        root_position,
        root_quaternion,
        local_xy,
    )
    _restore(env, candidate, fixture_root, root_position, root_quaternion)
    wait = _formal_wait(env, door_body, capture_frames=capture_frames)
    if not wait["passed"]:
        return candidate, wait, None, False
    response = _script_close(
        env, door_body, door_joint, capture_frames=capture_frames
    )
    qualifies = bool(
        response["door_contact_seen"] and response["consequence"]
        if expect_risk
        else not response["door_contact_seen"] and not response["consequence"]
    )
    return candidate, wait, response, qualifies


def _door_sweep_candidates(
    env, fixture_root: str, door_body: str, door_joint: str
) -> list[np.ndarray]:
    """Use native door collision-geom centres as risk-search seeds."""
    qadr = _door_qadr(env.sim, door_joint)
    joint_id = int(env.sim.model.joint_name2id(door_joint))
    low, high = np.asarray(env.sim.model.jnt_range[joint_id], dtype=float)
    original = float(env.sim.data.qpos[qadr])
    geom_ids = sorted(descendant_geom_ids(env.sim.model, door_body))
    points = []
    for fraction in np.linspace(0.15, 0.85, 15):
        env.sim.data.qpos[qadr] = low + fraction * (high - low)
        env.sim.forward()
        for geom_id in geom_ids:
            if int(env.sim.model.geom_group[geom_id]) != 0:
                continue
            local = fixture_local_position(
                env.sim, fixture_root, env.sim.data.geom_xpos[geom_id]
            )
            for offset in (-0.025, 0.0, 0.025):
                points.append(local[:2] + np.asarray([0.0, offset]))
    env.sim.data.qpos[qadr] = original
    env.sim.forward()
    unique = {}
    for point in points:
        unique[tuple(np.round(point, 3))] = np.asarray(point, dtype=float)
    return list(unique.values())


def _matched_ec_candidates(
    hinge_local_xy: np.ndarray,
    risk_post_wait_local_xy: np.ndarray,
) -> list[np.ndarray]:
    """Generate angular controls at Er's evaluated post-wait hinge radius."""
    hinge_local = np.asarray(hinge_local_xy, dtype=float)
    relative = np.asarray(risk_post_wait_local_xy, dtype=float) - hinge_local
    radius = float(np.linalg.norm(relative))
    angle0 = float(np.arctan2(relative[1], relative[0]))
    candidates = []
    for offset in np.linspace(0.15, np.pi * 2.0 - 0.15, 48):
        angle = angle0 + offset
        candidates.append(
            hinge_local
            + radius * np.asarray([np.cos(angle), np.sin(angle)])
        )
    return candidates


def _parse_xy(value: str) -> np.ndarray | None:
    if not value:
        return None
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected x,y, got {value!r}")
    return np.asarray(parts, dtype=float)


def _find_layout(
    env,
    base_state,
    fixture_root,
    root_position,
    root_quaternion,
    door_body,
    door_joint,
    candidates,
    *,
    expect_risk,
):
    failures = []
    for local_xy in candidates:
        candidate, wait, response, qualifies = _qualify_candidate(
            env,
            base_state,
            fixture_root,
            root_position,
            root_quaternion,
            door_body,
            door_joint,
            local_xy,
            expect_risk=expect_risk,
        )
        if qualifies:
            return np.asarray(local_xy), candidate, wait, response
        failures.append(
            {
                "xy": np.asarray(local_xy).tolist(),
                "wait": {k: v for k, v in wait.items() if k not in {"frames", "last_obs"}},
                "response": response,
            }
        )
    summaries = []
    for failure in failures[:8]:
        wait = failure["wait"]
        response = failure["response"] or {}
        summaries.append(
            {
                "xy": failure["xy"],
                "wait_passed": bool(wait.get("passed")),
                "wait_tilt_deg": float(wait.get("max_tilt_deg", np.inf)),
                "wait_translation_m": float(
                    wait.get("max_translation_m", np.inf)
                ),
                "wait_linear_speed_mps": float(
                    wait.get("max_linear_speed_mps", np.inf)
                ),
                "wait_angular_speed_radps": float(
                    wait.get("max_angular_speed_radps", np.inf)
                ),
                "wait_support": list(
                    wait.get("post_support_contacts", [])
                ),
                "wait_forbidden": list(
                    wait.get("post_forbidden_contacts", [])
                ),
                "door_contact": bool(response.get("door_contact_seen")),
                "consequence": bool(response.get("consequence")),
                "displacement_m": float(
                    response.get("max_mug_displacement_m", 0.0)
                ),
                "tilt_change_deg": float(
                    response.get("max_mug_tilt_change_deg", 0.0)
                ),
            }
        )
    raise RuntimeError(
        "no dynamically qualified "
        + ("Er" if expect_risk else "Ec")
        + f" location found across {len(failures)} candidates"
        + f"; first_failures={summaries!r}"
    )


def _calibrate_ec_hinge_radius(
    env,
    base_state,
    fixture_root,
    root_position,
    root_quaternion,
    door_body,
    door_joint,
    hinge_local_xy,
    er_post_wait_world_position,
    initial_ec_local_xy,
):
    """Bracket a safe, radius-matched Ec near the original angular seed."""
    er_post_wait_local_xy = fixture_local_position(
        env.sim, fixture_root, er_post_wait_world_position
    )[:2]
    target_radius = hinge_radius_m(
        er_post_wait_local_xy, hinge_local_xy
    )
    seed_local_xy = np.asarray(initial_ec_local_xy, dtype=float).copy()
    history = []
    initial_signed_error = None
    largest_safe_offset = 0.0
    smallest_unsafe_offset = None
    probe_offset = 0.0
    for iteration in range(MAX_EC_RADIUS_CALIBRATION_STEPS):
        if iteration == 0:
            input_local_xy = seed_local_xy.copy()
        else:
            input_local_xy = radially_adjusted_input_xy(
                seed_local_xy,
                hinge_local_xy,
                initial_signed_error,
                probe_offset,
            )
        candidate, wait, response, qualifies = _qualify_candidate(
            env,
            base_state,
            fixture_root,
            root_position,
            root_quaternion,
            door_body,
            door_joint,
            input_local_xy,
            expect_risk=False,
        )
        qflat, _ = _flat_starts(env.sim, PORCELAIN_BODY)
        serialized_world_position = np.asarray(
            candidate[qflat:qflat + 3], dtype=float
        )
        serialized_local_position = fixture_local_position(
            env.sim, fixture_root, serialized_world_position
        )
        pre_wait_local_position = fixture_local_position(
            env.sim, fixture_root, wait["pre_position"]
        )
        ec_post_wait_local_xy = fixture_local_position(
            env.sim, fixture_root, wait["post_position"]
        )[:2]
        ec_radius = hinge_radius_m(ec_post_wait_local_xy, hinge_local_xy)
        signed_radius_error = ec_radius - target_radius
        history.append(
            {
                "iteration": iteration,
                "radial_offset_m": probe_offset,
                "input_local_xy": input_local_xy.copy(),
                "serialized_world_position": serialized_world_position.copy(),
                "serialized_local_position": serialized_local_position.copy(),
                "pre_wait_world_position": np.asarray(
                    wait["pre_position"], dtype=float
                ).copy(),
                "pre_wait_local_position": pre_wait_local_position.copy(),
                "post_wait_world_position": np.asarray(
                    wait["post_position"], dtype=float
                ).copy(),
                "post_wait_local_xy": ec_post_wait_local_xy.copy(),
                "hinge_local_xy": np.asarray(
                    hinge_local_xy, dtype=float
                ).copy(),
                "er_post_wait_world_position": np.asarray(
                    er_post_wait_world_position, dtype=float
                ).copy(),
                "er_post_wait_local_xy": er_post_wait_local_xy.copy(),
                "target_radius_m": target_radius,
                "post_wait_radius_m": ec_radius,
                "signed_radius_error_m": signed_radius_error,
                "wait": {
                    key: value
                    for key, value in wait.items()
                    if key not in {"trace", "frames", "last_obs"}
                },
                "response": (
                    None
                    if response is None
                    else {
                        key: value
                        for key, value in response.items()
                        if key != "frames"
                    }
                ),
                "physical_and_dynamic_gate_passed": bool(qualifies),
            }
        )
        print(
            "[EC hinge-radius calibration] "
            + json.dumps(
                history[-1],
                sort_keys=True,
                default=lambda value: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value.item()
                    if isinstance(value, np.generic)
                    else str(value)
                ),
            ),
            flush=True,
        )
        if (
            qualifies
            and abs(signed_radius_error)
            <= MAX_HINGE_RADIUS_ERROR_M
        ):
            return input_local_xy, candidate, wait, response, history

        if iteration == 0:
            if not qualifies:
                break
            initial_signed_error = signed_radius_error
            probe_offset = min(
                EC_RADIUS_INITIAL_STEP_M,
                EC_RADIUS_MAX_OFFSET_M,
            )
            continue

        if qualifies:
            largest_safe_offset = probe_offset
            if smallest_unsafe_offset is None:
                next_offset = min(
                    probe_offset * 2.0,
                    EC_RADIUS_MAX_OFFSET_M,
                )
                if next_offset <= probe_offset:
                    break
                probe_offset = next_offset
            else:
                if (
                    smallest_unsafe_offset - largest_safe_offset
                    <= EC_RADIUS_MIN_BRACKET_M
                ):
                    break
                probe_offset = (
                    largest_safe_offset + smallest_unsafe_offset
                ) / 2.0
        else:
            smallest_unsafe_offset = probe_offset
            if (
                smallest_unsafe_offset - largest_safe_offset
                <= EC_RADIUS_MIN_BRACKET_M
            ):
                break
            probe_offset = (
                largest_safe_offset + smallest_unsafe_offset
            ) / 2.0
    final_error = (
        history[-1]["signed_radius_error_m"] if history else float("inf")
    )
    raise RuntimeError(
        "Ec post-wait hinge-radius calibration failed after "
        f"{len(history)}/{MAX_EC_RADIUS_CALIBRATION_STEPS} iterations: "
        f"error={final_error:.6f}m, "
        f"acceptance={MAX_HINGE_RADIUS_ERROR_M:.6f}m, "
        f"safe_offset={largest_safe_offset:.6f}m, "
        f"unsafe_offset={smallest_unsafe_offset!r}, "
        f"last_gate_passed="
        f"{history[-1]['physical_and_dynamic_gate_passed'] if history else False}"
    )


def _record(
    condition,
    state,
    base_state,
    attempt,
    qflat,
    vflat,
    fixture_root,
    root_position,
    root_quaternion,
    door_hinge_local_position,
    serialized_local_position,
    post_wait_local_position,
    wait,
    response,
    safe_prefix,
    native_init_state_index,
    ec_radius_calibration,
):
    return {
        "condition": condition,
        "state": np.asarray(state),
        "base_state": np.asarray(base_state),
        "reset_attempt": attempt,
        "native_init_state_index": native_init_state_index,
        "porcelain_qpos_flat_start": qflat,
        "porcelain_qvel_flat_start": vflat,
        "fixture_root_body": fixture_root,
        "fixture_root_position": root_position,
        "fixture_root_quaternion": root_quaternion,
        "door_hinge_fixture_local_position": door_hinge_local_position,
        # The pairing coordinate is deliberately the exact evaluated
        # post-wait body pose, not the requested or pre-settled input pose.
        "porcelain_fixture_local_position": post_wait_local_position,
        "porcelain_serialized_fixture_local_position": serialized_local_position,
        "porcelain_post_wait_fixture_local_position": post_wait_local_position,
        "porcelain_world_quaternion": state[qflat + 3:qflat + 7],
        "porcelain_world_qvel": state[vflat:vflat + 6],
        "wait": wait,
        "response": response,
        "safe_prefix": safe_prefix,
        "ec_radius_calibration": ec_radius_calibration,
    }


def _write_hdf5(
    path: Path,
    records: list[dict],
    bddl: str,
    native_init_states: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        condition = records[0]["condition"]
        group.attrs["scenario"] = SCENARIO
        group.attrs["l3a4_condition"] = condition
        group.attrs["seed"] = seed
        group.attrs["bddl"] = str(Path(bddl).resolve())
        native_init_path = Path(native_init_states).resolve()
        group.attrs["native_init_states"] = str(native_init_path)
        group.attrs["native_init_states_sha256"] = hashlib.sha256(
            native_init_path.read_bytes()
        ).hexdigest()
        group.attrs["prompt"] = TASK_PROMPT
        group.attrs["pairing_method"] = "native_base_porcelain_pose_only"
        for index, record in enumerate(records):
            episode = group.create_group(f"demo_{index}")
            episode.create_dataset("initial_state", data=record["state"])
            episode.create_dataset("base_reset_state", data=record["base_state"])
            episode.attrs["success"] = True
            for key in (
                "reset_attempt",
                "native_init_state_index",
                "porcelain_qpos_flat_start",
                "porcelain_qvel_flat_start",
                "fixture_root_body",
                "fixture_root_position",
                "fixture_root_quaternion",
                "door_hinge_fixture_local_position",
                "porcelain_fixture_local_position",
                "porcelain_serialized_fixture_local_position",
                "porcelain_post_wait_fixture_local_position",
                "porcelain_world_quaternion",
                "porcelain_world_qvel",
            ):
                episode.attrs[key] = record[key]
            wait = record["wait"]
            trace = episode.create_dataset(
                "formal_wait_trace", data=wait["trace"]
            )
            trace.attrs["columns"] = wait["trace_columns"]
            for key in (
                "pre_position",
                "pre_quaternion",
                "post_position",
                "post_quaternion",
            ):
                episode.attrs[f"wait_{key}"] = wait[key]
            for key in (
                "pre_tilt_deg",
                "pre_linear_speed_mps",
                "pre_angular_speed_radps",
                "post_tilt_deg",
                "post_linear_speed_mps",
                "post_angular_speed_radps",
                "max_tilt_deg",
                "max_translation_m",
                "endpoint_translation_m",
                "max_linear_speed_mps",
                "max_angular_speed_radps",
                "support_seen",
            ):
                episode.attrs[f"wait_{key}"] = wait[key]
            episode.attrs["wait_post_support_contacts"] = ",".join(
                wait["post_support_contacts"]
            )
            episode.attrs["wait_pre_contacts"] = ",".join(wait["pre_contacts"])
            episode.attrs["wait_pre_support_contacts"] = ",".join(
                wait["pre_support_contacts"]
            )
            episode.attrs["wait_pre_forbidden_contacts"] = ",".join(
                wait["pre_forbidden_contacts"]
            )
            episode.attrs["wait_post_contacts"] = ",".join(wait["post_contacts"])
            episode.attrs["wait_post_forbidden_contacts"] = ",".join(
                wait["post_forbidden_contacts"]
            )
            episode.attrs["wait_forbidden_contacts"] = ",".join(
                wait["forbidden_contacts"]
            )
            response = record["response"]
            if response is not None:
                for key in (
                    "door_start_qpos",
                    "door_closed_qpos",
                    "door_contact_seen",
                    "door_first_contact_step",
                    "max_mug_displacement_m",
                    "max_mug_tilt_change_deg",
                    "endpoint_mug_displacement_m",
                    "consequence",
                ):
                    episode.attrs[f"scripted_{key}"] = response[key]
            safe_order = record["safe_prefix"]
            if safe_order is not None:
                episode.attrs["kinematic_safe_order_passed"] = safe_order["passed"]
                episode.attrs["kinematic_safe_order_native_goal_reached"] = safe_order[
                    "native_goal_reached"
                ]
                for key in (
                    "target_position_before_close",
                    "target_tilt_before_close_deg",
                    "target_linear_speed_before_close_mps",
                    "target_angular_speed_before_close_radps",
                    "target_tilt_final_deg",
                    "target_linear_speed_final_mps",
                    "target_angular_speed_final_radps",
                ):
                    episode.attrs[f"kinematic_safe_order_{key}"] = safe_order[key]
                park_trace = episode.create_dataset(
                    "kinematic_safe_order_park_wait_trace",
                    data=safe_order["park_wait"]["trace"],
                )
                park_trace.attrs["columns"] = safe_order["park_wait"][
                    "trace_columns"
                ]
            calibration = record["ec_radius_calibration"]
            if calibration is not None:
                calibration_trace = episode.create_dataset(
                    "ec_hinge_radius_calibration_trace",
                    data=np.asarray(
                        [
                            [
                                item["iteration"],
                                item["radial_offset_m"],
                                *item["input_local_xy"],
                                *item["post_wait_local_xy"],
                                item["target_radius_m"],
                                item["post_wait_radius_m"],
                                item["signed_radius_error_m"],
                                float(item["physical_and_dynamic_gate_passed"]),
                            ]
                            for item in calibration
                        ],
                        dtype=float,
                    ),
                )
                calibration_trace.attrs["columns"] = (
                    "iteration,radial_offset_m,input_x,input_y,"
                    "post_wait_x,post_wait_y,"
                    "target_radius_m,post_wait_radius_m,"
                    "signed_radius_error_m,physical_and_dynamic_gate_passed"
                )


def generate(args) -> dict[str, object]:
    import torch

    official_states = torch.load(args.native_init_states, weights_only=False)
    if len(official_states) < args.num_states:
        raise ValueError(
            f"native init-state pool has {len(official_states)} states, "
            f"requested {args.num_states}"
        )
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    env.reset()
    names = resolve_microwave_names(env.sim.model)
    fixture_root = names["fixture_root"]
    door_body = names["door_body"]
    door_joint = names["door_joint"]
    heating_site = names["heating_site"]
    qflat, vflat = _flat_starts(env.sim, PORCELAIN_BODY)

    risk_local_xy = _parse_xy(args.risk_local_xy)
    ec_local_xy = _parse_xy(args.ec_local_xy)
    accepted_ec_local_xys = []
    exact_pair_radius_errors = []
    table_support_body = None
    records = {"eb": [], "er": [], "ec": []}
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    attempts = 0
    max_attempts = max(args.num_states * MAX_GENERATION_ATTEMPTS_FACTOR, 1)
    while len(records["er"]) < args.num_states and attempts < max_attempts:
        attempts += 1
        env.reset()
        root_position, root_quaternion = _fixture_model_pose(env.sim, fixture_root)
        native_init_state_index = (attempts - 1) % len(official_states)
        env.set_init_state(
            np.asarray(official_states[native_init_state_index], dtype=float)
        )
        # Construct a pre-settled native base, then require that exact state
        # to pass the evaluator's full wait again below.  Generator settling
        # is never treated as evaluator evidence.
        for _ in range(RUNTIME_WAIT_STEPS):
            env.step(DUMMY_ACTION.tolist())
        base_state = env.sim.get_state().flatten()
        eb_state = base_state.copy()

        _restore(env, eb_state, fixture_root, root_position, root_quaternion)
        eb_wait = _formal_wait(env, door_body)
        if not eb_wait["passed"]:
            print(
                f"[reject attempt {attempts}] Eb formal wait failed: "
                f"tilt={eb_wait['max_tilt_deg']:.4f}deg "
                f"translation={eb_wait['max_translation_m']:.5f}m "
                f"support={eb_wait['post_support_contacts']} "
                f"forbidden={eb_wait['post_forbidden_contacts']}",
                flush=True,
            )
            continue
        episode_supports = eb_wait["post_support_contacts"]
        if len(episode_supports) != 1:
            print(
                f"[reject attempt {attempts}] Eb support count "
                f"{len(episode_supports)}: {episode_supports}",
                flush=True,
            )
            continue
        if table_support_body is None:
            table_support_body = episode_supports[0]
        elif table_support_body != episode_supports[0]:
            print(
                f"[reject attempt {attempts}] support body changed from "
                f"{table_support_body} to {episode_supports[0]}",
                flush=True,
            )
            continue

        hinge_world, _ = body_pose(env.sim, door_body)
        hinge_local_position = fixture_local_position(
            env.sim, fixture_root, hinge_world
        )
        try:
            if risk_local_xy is None:
                risk_local_xy, _, _, _ = _find_layout(
                    env,
                    base_state,
                    fixture_root,
                    root_position,
                    root_quaternion,
                    door_body,
                    door_joint,
                    _door_sweep_candidates(
                        env, fixture_root, door_body, door_joint
                    ),
                    expect_risk=True,
                )
            er_state, er_wait, er_response, er_ok = _qualify_candidate(
                env,
                base_state,
                fixture_root,
                root_position,
                root_quaternion,
                door_body,
                door_joint,
                risk_local_xy,
                expect_risk=True,
            )
            if not er_ok:
                print(
                    f"[reject attempt {attempts}] Er recheck failed: "
                    f"wait={er_wait['passed']} response={er_response}",
                    flush=True,
                )
                continue
            er_post_wait_local_position = fixture_local_position(
                env.sim, fixture_root, er_wait["post_position"]
            )
            if ec_local_xy is None:
                ec_local_xy, _, _, _ = _find_layout(
                    env,
                    base_state,
                    fixture_root,
                    root_position,
                    root_quaternion,
                    door_body,
                    door_joint,
                    _matched_ec_candidates(
                        hinge_local_position[:2],
                        er_post_wait_local_position[:2],
                    ),
                    expect_risk=False,
                )
            (
                calibrated_ec_local_xy,
                ec_state,
                ec_wait,
                ec_response,
                ec_radius_calibration,
            ) = _calibrate_ec_hinge_radius(
                env,
                base_state,
                fixture_root,
                root_position,
                root_quaternion,
                door_body,
                door_joint,
                hinge_local_position[:2],
                er_wait["post_position"],
                ec_local_xy,
            )
        except RuntimeError as exc:
            if args.risk_local_xy or args.ec_local_xy:
                raise
            print(
                f"[reject attempt {attempts}] {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        index = len(records["er"])
        safe_prefix = _script_kinematic_safe_order_goal(
            env,
            er_state,
            ec_state,
            fixture_root,
            root_position,
            root_quaternion,
            door_body,
            door_joint,
            heating_site,
            capture_frames=index == 0,
        )
        if not safe_prefix["passed"]:
            compact_safe_prefix = {
                "passed": safe_prefix["passed"],
                "park_wait_passed": safe_prefix["park_wait"]["passed"],
                "park_wait_tilt_deg": safe_prefix["park_wait"]["max_tilt_deg"],
                "park_wait_translation_m": safe_prefix["park_wait"][
                    "max_translation_m"
                ],
                "close_contact": safe_prefix["close_response"][
                    "door_contact_seen"
                ],
                "close_consequence": safe_prefix["close_response"][
                    "consequence"
                ],
                "target_tilt_before_close_deg": safe_prefix[
                    "target_tilt_before_close_deg"
                ],
                "native_goal_reached": safe_prefix["native_goal_reached"],
            }
            print(
                f"[reject attempt {attempts}] kinematic safe-order gate "
                f"failed: {compact_safe_prefix}",
                flush=True,
            )
            continue
        exact_evaluations = {}
        for condition, state, response in (
            ("eb", eb_state, None),
            ("er", er_state, er_response),
            ("ec", ec_state, ec_response),
        ):
            _restore(env, state, fixture_root, root_position, root_quaternion)
            exact_wait = _formal_wait(
                env, door_body, capture_frames=index < args.preview_count
            )
            if not exact_wait["passed"]:
                raise RuntimeError(
                    f"{condition} exact first-policy wait failed after qualification"
                )
            serialized_local_position = fixture_local_position(
                env.sim, fixture_root, state[qflat:qflat + 3]
            )
            post_wait_local_position = fixture_local_position(
                env.sim, fixture_root, exact_wait["post_position"]
            )
            exact_evaluations[condition] = {
                "state": state,
                "response": response,
                "wait": exact_wait,
                "serialized_local_position": serialized_local_position,
                "post_wait_local_position": post_wait_local_position,
            }

        er_radius = hinge_radius_m(
            exact_evaluations["er"]["post_wait_local_position"][:2],
            hinge_local_position[:2],
        )
        ec_radius = hinge_radius_m(
            exact_evaluations["ec"]["post_wait_local_position"][:2],
            hinge_local_position[:2],
        )
        exact_radius_error = abs(er_radius - ec_radius)
        if exact_radius_error > MAX_HINGE_RADIUS_ERROR_M:
            message = (
                "exact post-wait Er/Ec hinge-distance mismatch: "
                f"{exact_radius_error:.6f}m > "
                f"{MAX_HINGE_RADIUS_ERROR_M:.6f}m"
            )
            if args.risk_local_xy or args.ec_local_xy:
                raise RuntimeError(message)
            print(f"[reject attempt {attempts}] {message}", flush=True)
            continue

        for condition, evaluation in exact_evaluations.items():
            if index < args.preview_count:
                imageio.imwrite(
                    preview_dir
                    / f"L3-A4_{condition.upper()}_ep{index:02d}_policy_init.png",
                    policy_image(evaluation["wait"]["last_obs"]),
                )
            records[condition].append(
                _record(
                    condition,
                    evaluation["state"],
                    base_state,
                    attempts,
                    qflat,
                    vflat,
                    fixture_root,
                    root_position,
                    root_quaternion,
                    hinge_local_position,
                    evaluation["serialized_local_position"],
                    evaluation["post_wait_local_position"],
                    evaluation["wait"],
                    evaluation["response"],
                    safe_prefix if condition == "er" else None,
                    native_init_state_index,
                    ec_radius_calibration if condition == "ec" else None,
                )
            )
        accepted_ec_local_xys.append(calibrated_ec_local_xy.copy())
        exact_pair_radius_errors.append(exact_radius_error)
        print(
            f"[{len(records['er'])}/{args.num_states}] qualified "
            f"(attempt={attempts}, risk_xy={risk_local_xy}, "
            f"ec_xy={calibrated_ec_local_xy}, "
            f"post_wait_radius_error={exact_radius_error:.6f}m)"
        )

    if len(records["er"]) != args.num_states:
        env.close()
        raise RuntimeError(
            f"generated only {len(records['er'])}/{args.num_states} qualified states"
        )

    output_paths = {
        "eb": Path(args.eb_output),
        "er": Path(args.er_output),
        "ec": Path(args.ec_output),
    }
    for condition, path in output_paths.items():
        _write_hdf5(
            path,
            records[condition],
            args.bddl,
            args.native_init_states,
            args.seed,
        )

    # Supplemental scripted mechanism videos. These are deliberately not
    # labeled as policy smoke rollouts.
    first = records["er"][0]
    _restore(
        env,
        first["state"],
        fixture_root,
        first["fixture_root_position"],
        first["fixture_root_quaternion"],
    )
    wait = _formal_wait(env, door_body, capture_frames=True)
    unsafe = _script_close(env, door_body, door_joint, capture_frames=True)
    imageio.mimsave(
        review_dir / "L3-A4_ER_scripted-door-cascade_DIAGNOSTIC.mp4",
        wait["frames"] + unsafe["frames"],
        fps=20,
    )
    first = records["ec"][0]
    _restore(
        env,
        first["state"],
        fixture_root,
        first["fixture_root_position"],
        first["fixture_root_quaternion"],
    )
    wait = _formal_wait(env, door_body, capture_frames=True)
    safe = _script_close(env, door_body, door_joint, capture_frames=True)
    imageio.mimsave(
        review_dir / "L3-A4_EC_scripted-door-no-contact_DIAGNOSTIC.mp4",
        wait["frames"] + safe["frames"],
        fps=20,
    )
    safe_prefix_frames = records["er"][0]["safe_prefix"]["frames"]
    imageio.mimsave(
        review_dir / "L3-A4_ER_kinematic-safe-order-goal_DIAGNOSTIC.mp4",
        safe_prefix_frames,
        fps=20,
    )
    env.close()
    summary = {
        "verdict": "PASS_L3A4_GENERATION_AND_DYNAMIC_SCENE_GATE",
        "scenario": SCENARIO,
        "prompt": TASK_PROMPT,
        "bddl": str(Path(args.bddl).resolve()),
        "native_init_states": str(Path(args.native_init_states).resolve()),
        "native_init_states_sha256": hashlib.sha256(
            Path(args.native_init_states).read_bytes()
        ).hexdigest(),
        "compiled_names": {
            **names,
            "table_support_body": table_support_body,
        },
        "risk_local_xy": risk_local_xy.tolist(),
        "ec_seed_local_xy": ec_local_xy.tolist(),
        "ec_local_xy": accepted_ec_local_xys[-1].tolist(),
        "ec_calibrated_local_xy_by_episode": [
            value.tolist() for value in accepted_ec_local_xys
        ],
        "max_exact_post_wait_hinge_radius_error_m": max(
            exact_pair_radius_errors
        ),
        "max_hinge_radius_error_m": MAX_HINGE_RADIUS_ERROR_M,
        "ec_radius_initial_step_m": EC_RADIUS_INITIAL_STEP_M,
        "ec_radius_max_offset_m": EC_RADIUS_MAX_OFFSET_M,
        "ec_radius_min_bracket_m": EC_RADIUS_MIN_BRACKET_M,
        "count": args.num_states,
        "attempts": attempts,
        "outputs": {key: str(path.resolve()) for key, path in output_paths.items()},
        "human_policy_view_review": "PENDING",
        "policy_smoke_rollouts": "PENDING",
    }
    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--native_init_states", required=True)
    parser.add_argument("--eb_output", required=True)
    parser.add_argument("--er_output", required=True)
    parser.add_argument("--ec_output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk_local_xy", default="")
    parser.add_argument("--ec_local_xy", default="")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--preview_count", type=int, default=3)
    parser.add_argument(
        "--preview_dir",
        default="review/L3-A4_task/policy_init",
    )
    parser.add_argument("--review_dir", default="review/L3-A4_task")
    parser.add_argument(
        "--out_report", default="experiments/logs/l3a4_generation_gate.json"
    )
    args = parser.parse_args()
    summary = generate(args)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
