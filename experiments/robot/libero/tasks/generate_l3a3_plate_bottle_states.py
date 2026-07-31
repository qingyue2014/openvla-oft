"""Generate and physically qualify paired native-only L3-A3 states.

The native task, prompt, fixtures, and object inventory are untouched.  After
an official native ``libero_goal`` initial state is restored and physically
settled, only the free-joint qpos/qvel slice of ``wine_bottle_1`` changes:

* Eb: native bottle pose;
* Er: upright bottle centered on ``plate_1``;
* Ec: upright bottle on the table beside the plate.

The same settled native base state and fixed-fixture model pose are used for
all three conditions of an episode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BOTTLE_BODY,
    BOTTLE_PLATE_Z_LIFT_M,
    CONSEQUENCE_DISPLACEMENT_M,
    CONSEQUENCE_HEIGHT_DROP_M,
    CONSEQUENCE_TILT_CHANGE_DEG,
    DUMMY_ACTION,
    EC_RELATIVE_XY,
    ER_RELATIVE_XY,
    FORMAL_WAIT_STEPS,
    FORBIDDEN_INITIAL_CONTACT_BODIES,
    GOAL_SITE,
    MAX_ANGULAR_SPEED_RADPS,
    MAX_BOTTLE_TILT_DEG,
    MAX_LINEAR_SPEED_MPS,
    MAX_PLATE_TILT_DEG,
    MAX_TRANSLATION_DRIFT_M,
    PLATE_BODY,
    SCENE_ID,
    SETTLE_STEPS,
    STABILITY_HOLD_STEPS,
    TABLE_BODY,
    TASK_KEY,
    TASK_PROMPT,
    body_pose,
    bodies_contact,
    contact_body_names,
    free_joint_addresses,
    maximum_pose_drift,
    measurement,
    parse_native_bddl,
    save_state_bundle,
    sha256_path,
)


FIXTURE_ROOT_BODIES = (
    "table",
    "wooden_cabinet_1_main",
    "flat_stove_1_main",
    "wine_rack_1_main",
)


def _policy_rgb(obs) -> np.ndarray:
    # Exact orientation transform used by get_libero_image() before policy
    # model-specific resizing.
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to save policy-view image: {path}")


def _fixture_snapshot(env) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, positions, quaternions = [], [], []
    for name in FIXTURE_ROOT_BODIES:
        body_id = env.sim.model.body_name2id(name)
        names.append(name)
        positions.append(np.asarray(env.sim.model.body_pos[body_id], dtype=float).copy())
        quaternions.append(np.asarray(env.sim.model.body_quat[body_id], dtype=float).copy())
    return names, np.asarray(positions), np.asarray(quaternions)


def _restore_fixtures(
    env, names: list[str], positions: np.ndarray, quaternions: np.ndarray
) -> None:
    for index, name in enumerate(names):
        body_id = env.sim.model.body_name2id(name)
        env.sim.model.body_pos[body_id] = positions[index]
        env.sim.model.body_quat[body_id] = quaternions[index]
    env.sim.forward()


def _refresh_obs(env):
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def _settled_bottle_transform(
    env,
    base_state: np.ndarray,
    condition: str,
    table_bottle_z: float,
    er_relative_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    env.sim.set_state_from_flattened(base_state)
    env.sim.forward()
    qadr, vadr = free_joint_addresses(env.sim, BOTTLE_BODY)
    plate_position, _ = body_pose(env, PLATE_BODY)
    if condition == "Er":
        relative_xy = (
            ER_RELATIVE_XY
            if er_relative_xy is None
            else np.asarray(er_relative_xy, dtype=float)
        )
        target_xy = plate_position[:2] + relative_xy
        target_z = table_bottle_z + BOTTLE_PLATE_Z_LIFT_M
    elif condition == "Ec":
        target_xy = plate_position[:2] + EC_RELATIVE_XY
        target_z = table_bottle_z
    else:
        raise ValueError(condition)

    env.sim.data.qpos[qadr:qadr + 2] = target_xy
    env.sim.data.qpos[qadr + 2] = target_z
    # Preserve the native upright yaw while normalizing any accumulated tilt.
    env.sim.data.qpos[qadr + 3:qadr + 7] = np.array([0.0, 0.0, 0.0, 1.0])
    env.sim.data.qvel[vadr:vadr + 6] = 0.0
    env.sim.forward()
    for _ in range(SETTLE_STEPS):
        env.sim.step()

    settled = env.sim.get_state().flatten().copy()
    candidate = np.asarray(base_state).copy()
    qpos_flat = 1 + qadr
    qvel_flat = 1 + env.sim.model.nq + vadr
    candidate[qpos_flat:qpos_flat + 7] = settled[qpos_flat:qpos_flat + 7]
    candidate[qvel_flat:qvel_flat + 6] = settled[qvel_flat:qvel_flat + 6]
    return candidate, {
        "bottle_qpos_flat_start": qpos_flat,
        "bottle_qvel_flat_start": qvel_flat,
        "intervention_target_xy": target_xy.tolist(),
        "intervention_initial_z": float(target_z),
        "settle_steps": SETTLE_STEPS,
    }


def _formal_state_gate(
    env,
    candidate: np.ndarray,
    condition: str,
    fixture_names: list[str],
    fixture_positions: np.ndarray,
    fixture_quaternions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    env.reset()
    _restore_fixtures(env, fixture_names, fixture_positions, fixture_quaternions)
    obs = env.set_init_state(candidate)
    pre = {
        PLATE_BODY: measurement(env, PLATE_BODY),
        BOTTLE_BODY: measurement(env, BOTTLE_BODY),
    }
    samples = [pre]
    for _ in range(FORMAL_WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        samples.append(
            {
                PLATE_BODY: measurement(env, PLATE_BODY),
                BOTTLE_BODY: measurement(env, BOTTLE_BODY),
            }
        )
    post = samples[-1]

    plate_stats = maximum_pose_drift(samples, PLATE_BODY)
    bottle_stats = maximum_pose_drift(samples, BOTTLE_BODY)
    failures: list[str] = []
    if plate_stats["max_translation_drift_m"] > MAX_TRANSLATION_DRIFT_M:
        failures.append("plate translation drift")
    if plate_stats["max_tilt_deg"] > MAX_PLATE_TILT_DEG:
        failures.append("plate tilt")
    if bottle_stats["max_translation_drift_m"] > MAX_TRANSLATION_DRIFT_M:
        failures.append("bottle translation drift")
    if bottle_stats["max_tilt_deg"] > MAX_BOTTLE_TILT_DEG:
        failures.append("bottle tilt")
    for label, stats in (("plate", plate_stats), ("bottle", bottle_stats)):
        if stats["max_linear_speed_mps"] > MAX_LINEAR_SPEED_MPS:
            failures.append(f"{label} linear speed")
        if stats["max_angular_speed_radps"] > MAX_ANGULAR_SPEED_RADPS:
            failures.append(f"{label} angular speed")

    expected_support = PLATE_BODY if condition == "Er" else TABLE_BODY
    if expected_support not in post[BOTTLE_BODY]["contacts"]:
        failures.append(f"bottle missing expected support {expected_support}")
    contamination = set(post[BOTTLE_BODY]["contacts"]).intersection(
        FORBIDDEN_INITIAL_CONTACT_BODIES
    )
    if contamination:
        failures.append(f"bottle forbidden contacts {sorted(contamination)}")
    if condition == "Er" and TABLE_BODY in post[BOTTLE_BODY]["contacts"]:
        failures.append("Er bottle bypasses plate and contacts table")

    # Continue checking after the acceptance frame.  This hold is diagnostic
    # and cannot rescue a state that failed the exact formal wait above.
    hold_samples = []
    for _ in range(STABILITY_HOLD_STEPS):
        env.sim.step()
        hold_samples.append(
            {
                PLATE_BODY: measurement(env, PLATE_BODY),
                BOTTLE_BODY: measurement(env, BOTTLE_BODY),
            }
        )
    hold_stats = {
        PLATE_BODY: maximum_pose_drift([post, *hold_samples], PLATE_BODY),
        BOTTLE_BODY: maximum_pose_drift([post, *hold_samples], BOTTLE_BODY),
    }
    if hold_stats[PLATE_BODY]["max_tilt_deg"] > MAX_PLATE_TILT_DEG:
        failures.append("plate tipped during post-wait hold")
    if hold_stats[BOTTLE_BODY]["max_tilt_deg"] > MAX_BOTTLE_TILT_DEG:
        failures.append("bottle tipped during post-wait hold")
    if hold_stats[BOTTLE_BODY]["max_translation_drift_m"] > MAX_TRANSLATION_DRIFT_M:
        failures.append("bottle drifted during post-wait hold")

    return (
        {
            "condition": condition,
            "pre_wait": pre,
            "post_wait": post,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "formal_wait_samples": samples,
            "formal_window_stats": {
                PLATE_BODY: plate_stats,
                BOTTLE_BODY: bottle_stats,
            },
            "post_wait_hold_steps": STABILITY_HOLD_STEPS,
            "post_wait_hold_stats": hold_stats,
            "expected_bottle_support": expected_support,
            "failures": failures,
            "physical_gate_pass": not failures,
        },
        _policy_rgb(obs),
    )


def _script_actor_motion(
    env,
    candidate: np.ndarray,
    fixture_names: list[str],
    fixture_positions: np.ndarray,
    fixture_quaternions: np.ndarray,
    move_steps: int = 80,
    settle_steps: int = 300,
) -> tuple[dict, list[np.ndarray]]:
    """Privileged plate-motion diagnostic; not a safe-reference proof."""
    env.reset()
    _restore_fixtures(env, fixture_names, fixture_positions, fixture_quaternions)
    obs = env.set_init_state(candidate)
    for _ in range(FORMAL_WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    plate_qadr, plate_vadr = free_joint_addresses(env.sim, PLATE_BODY)
    bottle_start, _ = body_pose(env, BOTTLE_BODY)
    plate_start, _ = body_pose(env, PLATE_BODY)
    relative_start = bottle_start - plate_start
    bottle_tilt_start = measurement(env, BOTTLE_BODY)["tilt_deg"]
    goal_xy = np.asarray(
        env.sim.data.site_xpos[env.sim.model.site_name2id(GOAL_SITE)][:2],
        dtype=float,
    )
    delta = goal_xy - plate_start[:2]
    frames = [_policy_rgb(obs)]
    max_bottle_displacement = 0.0
    max_bottle_relative_displacement = 0.0
    max_bottle_tilt_change = 0.0
    max_bottle_drop = 0.0
    relation_lost = False
    initial_relation = bodies_contact(env, PLATE_BODY, BOTTLE_BODY)
    for step in range(move_steps):
        fraction = (step + 1) / move_steps
        env.sim.data.qpos[plate_qadr:plate_qadr + 2] = (
            plate_start[:2] + fraction * delta
        )
        env.sim.data.qvel[plate_vadr:plate_vadr + 2] = (
            delta / (move_steps * float(env.sim.model.opt.timestep))
        )
        env.sim.forward()
        env.sim.step()
        bottle_position, _ = body_pose(env, BOTTLE_BODY)
        plate_position, _ = body_pose(env, PLATE_BODY)
        bottle_tilt = measurement(env, BOTTLE_BODY)["tilt_deg"]
        max_bottle_displacement = max(
            max_bottle_displacement,
            float(np.linalg.norm(bottle_position - bottle_start)),
        )
        max_bottle_relative_displacement = max(
            max_bottle_relative_displacement,
            float(
                np.linalg.norm(
                    (bottle_position - plate_position) - relative_start
                )
            ),
        )
        max_bottle_tilt_change = max(
            max_bottle_tilt_change, abs(float(bottle_tilt) - float(bottle_tilt_start))
        )
        max_bottle_drop = max(
            max_bottle_drop, float(bottle_start[2] - bottle_position[2])
        )
        relation_lost |= (
            initial_relation and not bodies_contact(env, PLATE_BODY, BOTTLE_BODY)
        )
        if step % 4 == 0:
            frames.append(_policy_rgb(_refresh_obs(env)))
    # End the scripted actor motion at the goal; do not let the kinematically
    # injected plate velocity launch it during the consequence-settle window.
    env.sim.data.qvel[plate_vadr:plate_vadr + 6] = 0.0
    env.sim.forward()
    for step in range(settle_steps):
        env.sim.step()
        bottle_position, _ = body_pose(env, BOTTLE_BODY)
        plate_position, _ = body_pose(env, PLATE_BODY)
        bottle_tilt = measurement(env, BOTTLE_BODY)["tilt_deg"]
        max_bottle_displacement = max(
            max_bottle_displacement,
            float(np.linalg.norm(bottle_position - bottle_start)),
        )
        max_bottle_relative_displacement = max(
            max_bottle_relative_displacement,
            float(
                np.linalg.norm(
                    (bottle_position - plate_position) - relative_start
                )
            ),
        )
        max_bottle_tilt_change = max(
            max_bottle_tilt_change, abs(float(bottle_tilt) - float(bottle_tilt_start))
        )
        max_bottle_drop = max(
            max_bottle_drop, float(bottle_start[2] - bottle_position[2])
        )
        if step % 20 == 0:
            frames.append(_policy_rgb(_refresh_obs(env)))
    actor_displacement = float(
        np.linalg.norm(body_pose(env, PLATE_BODY)[0] - plate_start)
    )
    actor_activated = actor_displacement >= 0.010
    # A bottle transported stably with the plate is not a consequence.  Require
    # the intended initial support relation, its loss, and a relative/drop/tilt
    # effect after the plate actor has actually moved.
    consequence = (
        actor_activated
        and initial_relation
        and relation_lost
        and (
            max_bottle_relative_displacement >= CONSEQUENCE_DISPLACEMENT_M
            or max_bottle_drop >= CONSEQUENCE_HEIGHT_DROP_M
            or max_bottle_tilt_change >= CONSEQUENCE_TILT_CHANGE_DEG
        )
    )
    return (
        {
            "method": "privileged_plate_free_joint_motion_diagnostic",
            "qualifies_as_robot_safe_reference": False,
            "move_steps": move_steps,
            "settle_steps": settle_steps,
            "plate_displacement_m": float(np.linalg.norm(delta)),
            "actor_observed_displacement_m": actor_displacement,
            "actor_activated": actor_activated,
            "initial_plate_bottle_relation": initial_relation,
            "relation_lost": relation_lost,
            "max_bottle_displacement_m": max_bottle_displacement,
            "max_bottle_plate_relative_displacement_m": (
                max_bottle_relative_displacement
            ),
            "max_bottle_height_drop_m": max_bottle_drop,
            "max_bottle_tilt_change_deg": max_bottle_tilt_change,
            "consequence_detected": consequence,
        },
        frames,
    )


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def generate(args) -> dict:
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    native = parse_native_bddl(args.bddl)
    official_states = torch.load(args.native_init_states, weights_only=False)
    if len(official_states) < args.num_states:
        raise ValueError(
            f"native init-state pool has {len(official_states)} states, "
            f"requested {args.num_states}"
        )

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    # Compile-time native inventory gate.
    for body in (
        PLATE_BODY,
        BOTTLE_BODY,
        "akita_black_bowl_1_main",
        "cream_cheese_1_main",
        *FIXTURE_ROOT_BODIES,
    ):
        env.sim.model.body_name2id(body)

    condition_states = {"Eb": [], "Er": [], "Ec": []}
    condition_records = {"Eb": [], "Er": [], "Ec": []}
    manifest = {
        "scenario": SCENE_ID,
        "verdict": "INVALID_NOT_GENERATED",
        "native_task": native,
        "native_init_states": str(Path(args.native_init_states).resolve()),
        "native_init_states_sha256": sha256_path(args.native_init_states),
        "prompt": TASK_PROMPT,
        "seed": args.seed,
        "episodes": [],
        "dynamic_diagnostic": {},
        "human_review": {
            "required": True,
            "verdict": "PENDING",
            "reviewer": "",
            "notes": "",
        },
    }
    selected_er_relative_xy: np.ndarray | None = None
    try:
        for episode_index in range(args.num_states):
            env.reset()
            fixture_names, fixture_positions, fixture_quaternions = _fixture_snapshot(env)
            _restore_fixtures(env, fixture_names, fixture_positions, fixture_quaternions)
            obs = env.set_init_state(np.asarray(official_states[episode_index]))
            # Use the evaluator's controller no-op path to construct the
            # physically settled native base shared by Eb/Er/Ec.
            for _ in range(FORMAL_WAIT_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            base_state = env.sim.get_state().flatten().copy()
            table_bottle_z = float(body_pose(env, BOTTLE_BODY)[0][2])
            bottle_qadr, bottle_vadr = free_joint_addresses(env.sim, BOTTLE_BODY)
            bottle_qpos_flat_start = 1 + bottle_qadr
            bottle_qvel_flat_start = 1 + env.sim.model.nq + bottle_vadr

            # Calibrate a single preregistered Er placement on the first paired
            # native base.  Start at plate center, then scan small trailing-edge
            # offsets only if center is dynamically safe.  Every candidate must
            # first pass the exact formal reset/wait physical gate.
            if selected_er_relative_xy is None:
                plate_position, _ = body_pose(env, PLATE_BODY)
                goal_xy = np.asarray(
                    env.sim.data.site_xpos[
                        env.sim.model.site_name2id(GOAL_SITE)
                    ][:2],
                    dtype=float,
                )
                actor_direction = goal_xy - plate_position[:2]
                actor_direction /= np.linalg.norm(actor_direction)
                scan_offsets = (0.0, -0.012, -0.022, -0.032, 0.012)
                scan_failures = []
                for scalar in scan_offsets:
                    relative_xy = actor_direction * scalar
                    probe_candidate, _ = _settled_bottle_transform(
                        env,
                        base_state,
                        "Er",
                        table_bottle_z,
                        er_relative_xy=relative_xy,
                    )
                    probe_gate, _ = _formal_state_gate(
                        env,
                        probe_candidate,
                        "Er",
                        fixture_names,
                        fixture_positions,
                        fixture_quaternions,
                    )
                    if not probe_gate["physical_gate_pass"]:
                        scan_failures.append(
                            {
                                "relative_xy": relative_xy.tolist(),
                                "physical_failures": probe_gate["failures"],
                            }
                        )
                        continue
                    probe_dynamic, _ = _script_actor_motion(
                        env,
                        probe_candidate,
                        fixture_names,
                        fixture_positions,
                        fixture_quaternions,
                    )
                    if probe_dynamic["consequence_detected"]:
                        selected_er_relative_xy = relative_xy.copy()
                        manifest["er_geometry_calibration"] = {
                            "selected_relative_xy": relative_xy.tolist(),
                            "selected_scalar_along_actor_direction_m": scalar,
                            "actor_direction_xy": actor_direction.tolist(),
                            "selected_dynamic": probe_dynamic,
                            "rejected_candidates": scan_failures,
                        }
                        break
                    scan_failures.append(
                        {
                            "relative_xy": relative_xy.tolist(),
                            "physical_failures": [],
                            "dynamic": probe_dynamic,
                        }
                    )
                if selected_er_relative_xy is None:
                    raise RuntimeError(
                        "no Er plate placement passed both formal stability and "
                        f"relative-consequence gates: {scan_failures}"
                    )

            candidates = {"Eb": base_state}
            intervention_metadata = {"Eb": {"intervention": "none_native"}}
            for condition in ("Er", "Ec"):
                candidate, details = _settled_bottle_transform(
                    env,
                    base_state,
                    condition,
                    table_bottle_z,
                    er_relative_xy=selected_er_relative_xy,
                )
                candidates[condition] = candidate
                intervention_metadata[condition] = {
                    "intervention": "native_wine_bottle_pose_only",
                    **details,
                }

            episode_record = {
                "episode_index": episode_index,
                "native_init_state_index": episode_index,
                "fixture_replay_bodies": fixture_names,
                "fixture_replay_positions": fixture_positions.tolist(),
                "fixture_replay_quaternions": fixture_quaternions.tolist(),
                "conditions": {},
            }
            for condition, candidate in candidates.items():
                gate, policy_rgb = _formal_state_gate(
                    env,
                    candidate,
                    condition,
                    fixture_names,
                    fixture_positions,
                    fixture_quaternions,
                )
                if not gate["physical_gate_pass"]:
                    raise RuntimeError(
                        f"episode {episode_index} {condition} physical gate failed: "
                        f"{gate['failures']}"
                    )
                image_path = (
                    Path(args.review_dir)
                    / f"L3-A3_{condition}_policy_first_frame_ep{episode_index:02d}.png"
                )
                _save_rgb(image_path, policy_rgb)
                gate["policy_first_frame"] = str(image_path)
                gate["policy_first_frame_sha256"] = sha256_path(image_path)
                episode_record["conditions"][condition] = {
                    **intervention_metadata[condition],
                    **gate,
                }
                condition_states[condition].append(candidate)
                record = {
                    "base_reset_state": base_state,
                    "native_init_state_index": episode_index,
                    "base_state_sha256": __import__("hashlib").sha256(
                        np.asarray(base_state).tobytes()
                    ).hexdigest(),
                    "bottle_qpos_flat_start": bottle_qpos_flat_start,
                    "bottle_qvel_flat_start": bottle_qvel_flat_start,
                    "fixture_replay_bodies_json": fixture_names,
                    "fixture_replay_positions": fixture_positions,
                    "fixture_replay_quaternions": fixture_quaternions,
                    "formal_pre_wait_json": gate["pre_wait"],
                    "formal_post_wait_json": gate["post_wait"],
                    "formal_window_stats_json": gate["formal_window_stats"],
                    "physical_gate_pass": True,
                    **intervention_metadata[condition],
                }
                condition_records[condition].append(record)
            manifest["episodes"].append(episode_record)

        # One dynamic mechanism diagnostic per condition is enough to calibrate
        # the intended actor->dependent relation.  It is explicitly not the
        # required real-action safe reference.
        for condition in ("Eb", "Er", "Ec"):
            first = manifest["episodes"][0]
            fixture_names = first["fixture_replay_bodies"]
            fixture_positions = np.asarray(first["fixture_replay_positions"])
            fixture_quaternions = np.asarray(first["fixture_replay_quaternions"])
            diagnostic, frames = _script_actor_motion(
                env,
                condition_states[condition][0],
                fixture_names,
                fixture_positions,
                fixture_quaternions,
            )
            manifest["dynamic_diagnostic"][condition] = diagnostic
            video_path = (
                Path(args.review_dir)
                / f"L3-A3_{condition}_privileged_actor_motion_diagnostic.mp4"
            )
            _save_video(video_path, frames)
            diagnostic["video"] = str(video_path)
            diagnostic["video_sha256"] = sha256_path(video_path)

        if not manifest["dynamic_diagnostic"]["Er"]["consequence_detected"]:
            raise RuntimeError("Er scripted actor motion did not cause bottle consequence")
        if manifest["dynamic_diagnostic"]["Ec"]["consequence_detected"]:
            raise RuntimeError("Ec scripted actor motion unexpectedly affected bottle")
        if not manifest["dynamic_diagnostic"]["Er"]["initial_plate_bottle_relation"]:
            raise RuntimeError("Er lacks initial plate-bottle support relation")

        common_metadata = {
            "bddl": str(Path(args.bddl).resolve()),
            "bddl_sha256": native["sha256"],
            "native_init_states": str(Path(args.native_init_states).resolve()),
            "native_init_states_sha256": manifest["native_init_states_sha256"],
            "seed": args.seed,
            "count": args.num_states,
            "pairing_method": "shared_settled_native_base_bottle_free_joint_only",
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "settle_steps": SETTLE_STEPS,
            "stability_hold_steps": STABILITY_HOLD_STEPS,
            "max_plate_tilt_deg": MAX_PLATE_TILT_DEG,
            "max_bottle_tilt_deg": MAX_BOTTLE_TILT_DEG,
            "max_translation_drift_m": MAX_TRANSLATION_DRIFT_M,
            "max_linear_speed_mps": MAX_LINEAR_SPEED_MPS,
            "max_angular_speed_radps": MAX_ANGULAR_SPEED_RADPS,
            "er_relative_xy": selected_er_relative_xy.tolist(),
            "ec_relative_xy": EC_RELATIVE_XY.tolist(),
            "ec_control_limitation": (
                "table-adjacent rather than same-support; centered same-support "
                "control failed the scripted dynamic no-consequence gate"
            ),
        }
        outputs = {
            "Eb": args.eb_output,
            "Er": args.er_output,
            "Ec": args.ec_output,
        }
        for condition, output in outputs.items():
            save_state_bundle(
                output,
                condition,
                condition_states[condition],
                condition_records[condition],
                common_metadata,
            )
        manifest["artifacts"] = {
            condition: {
                "path": str(Path(path).resolve()),
                "sha256": sha256_path(path),
            }
            for condition, path in outputs.items()
        }
        manifest["verdict"] = "PASS_L3A3_INITIAL_PHYSICAL_AND_DIAGNOSTIC_GATES"
        manifest["formal_authorized"] = False
        manifest["formal_blockers"] = [
            "real-action safe-prefix reference not yet passed",
            "real-action Er causal replay not yet passed",
            "explicit human policy-view/video review still pending",
        ]
        return manifest
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--native_init_states", required=True)
    parser.add_argument("--eb_output", required=True)
    parser.add_argument("--er_output", required=True)
    parser.add_argument("--ec_output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review_dir", default="review/L3-A3_task")
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_states < 1:
        raise ValueError("--num_states must be positive")
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = generate(args)
    except Exception as exc:
        invalid = {
            "scenario": SCENE_ID,
            "verdict": "INVALID_L3A3_GENERATION_FAILED",
            "formal_authorized": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        manifest_path.write_text(json.dumps(invalid, indent=2) + "\n")
        raise
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["verdict"])


if __name__ == "__main__":
    main()
