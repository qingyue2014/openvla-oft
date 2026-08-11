#!/usr/bin/env python3
"""Screen model-informed L1-C5 ER offsets without learned ER rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.robot.libero.tasks.l1c_occupied_common import (
    body_in_anchor_region,
    body_pos,
    body_speeds,
    body_tilt_deg,
    get_spec,
    load_states,
    native_success,
    place_at_anchor,
    resolve_bddl,
    settle,
)
from experiments.robot.libero.tasks.l1c_occupied_pipeline import (
    _body_contact,
    _body_pose_relative_to_anchor,
    _env,
    _finite,
    _model_policy_camera,
    _paired_state_diff_audit,
    _restore_native_with_anchor_relative_occupant,
    _robot_contact,
    _rotation_matrix_separation_deg,
    _visible_pixels_in_policy_crop,
)

NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
EXPECTED_AUTHORIZATION_SHA256 = (
    "15d636be1ed64437be5920cc730a753d665a3b2187c6136f1b65bf7f9486f165"
)
EXPECTED_CANDIDATE_SET_SHA256 = (
    "79778ddd9532fab8adece094a4c862dcb30de7f67a3e83600126f24e4dfc7094"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _policy_image(env, obs, model_family: str) -> np.ndarray:
    image = obs.get("agentview_image") if obs is not None else None
    if image is None:
        image = env.sim.render(256, 256, camera_name="agentview")
    return _model_policy_camera(np.asarray(image), model_family)


def _baseline_anchor_timeline(env, state, anchor_body: str, steps: int):
    env.reset()
    env.set_init_state(state)
    positions = [body_pos(env, anchor_body)]
    for _ in range(steps):
        env.step(NOOP)
        positions.append(body_pos(env, anchor_body))
    return positions


def _candidate_state(
    env,
    native_state,
    spec,
    offset,
    baseline_anchor,
    args,
    image_prefix: Path | None,
):
    env.reset()
    env.set_init_state(native_state)
    place_at_anchor(env, spec, spec.occupant_body, offset)
    settle(env, spec.settle_steps)
    relative_start, rotation_start = _body_pose_relative_to_anchor(
        env, spec.occupant_body, spec.anchor_body
    )
    settle_max_drift = 0.0
    settle_max_rotation = 0.0
    settle_max_linear_speed = 0.0
    settle_max_angular_speed = 0.0
    settle_support = True
    settle_inside = True
    settle_finite = _finite(env)
    for _ in range(args.stability_steps):
        env.step(NOOP)
        settle_finite &= _finite(env)
        relative, rotation = _body_pose_relative_to_anchor(
            env, spec.occupant_body, spec.anchor_body
        )
        linear_speed, angular_speed = body_speeds(env, spec.occupant_body)
        settle_max_drift = max(
            settle_max_drift, float(np.linalg.norm(relative - relative_start))
        )
        settle_max_rotation = max(
            settle_max_rotation,
            _rotation_matrix_separation_deg(rotation, rotation_start),
        )
        settle_max_linear_speed = max(settle_max_linear_speed, linear_speed)
        settle_max_angular_speed = max(settle_max_angular_speed, angular_speed)
        settle_support &= _body_contact(
            env, spec.occupant_body, spec.anchor_body
        )
        settle_inside &= body_in_anchor_region(
            env, spec, spec.occupant_body
        )

    _restore_native_with_anchor_relative_occupant(
        env, native_state, spec.occupant_body, spec.anchor_body
    )
    variant_state = env.sim.get_state().flatten()
    diff = _paired_state_diff_audit(
        env, native_state, variant_state, spec.occupant_body
    )

    env.reset()
    obs = env.set_init_state(variant_state)
    relative_t0, rotation_t0 = _body_pose_relative_to_anchor(
        env, spec.occupant_body, spec.anchor_body
    )
    pixels_t0 = _visible_pixels_in_policy_crop(
        env, spec.occupant_body, model_family=args.policy_model_family
    )
    if image_prefix is not None:
        Image.fromarray(_policy_image(env, obs, args.policy_model_family)).save(
            image_prefix.with_name(image_prefix.name + "_policy_t0.png")
        )

    wait_max_drift = 0.0
    wait_max_rotation = 0.0
    wait_max_linear_speed = 0.0
    wait_max_angular_speed = 0.0
    wait_max_anchor_excess = 0.0
    full_window_inside = body_in_anchor_region(env, spec, spec.occupant_body)
    full_window_support = _body_contact(
        env, spec.occupant_body, spec.anchor_body
    )
    forbidden_target_contact = _body_contact(
        env, spec.occupant_body, spec.target_body
    )
    forbidden_robot_contact = _robot_contact(env, spec.occupant_body)
    wait_finite = _finite(env)
    for step in range(1, args.policy_wait_steps + 1):
        obs, _, _, _ = env.step(NOOP)
        wait_finite &= _finite(env)
        relative, rotation = _body_pose_relative_to_anchor(
            env, spec.occupant_body, spec.anchor_body
        )
        linear_speed, angular_speed = body_speeds(env, spec.occupant_body)
        wait_max_drift = max(
            wait_max_drift, float(np.linalg.norm(relative - relative_t0))
        )
        wait_max_rotation = max(
            wait_max_rotation,
            _rotation_matrix_separation_deg(rotation, rotation_t0),
        )
        wait_max_linear_speed = max(wait_max_linear_speed, linear_speed)
        wait_max_angular_speed = max(wait_max_angular_speed, angular_speed)
        wait_max_anchor_excess = max(
            wait_max_anchor_excess,
            float(
                np.linalg.norm(
                    body_pos(env, spec.anchor_body) - baseline_anchor[step]
                )
            ),
        )
        full_window_inside &= body_in_anchor_region(
            env, spec, spec.occupant_body
        )
        full_window_support &= _body_contact(
            env, spec.occupant_body, spec.anchor_body
        )
        forbidden_target_contact |= _body_contact(
            env, spec.occupant_body, spec.target_body
        )
        forbidden_robot_contact |= _robot_contact(env, spec.occupant_body)

    pixels_policy_start = _visible_pixels_in_policy_crop(
        env, spec.occupant_body, model_family=args.policy_model_family
    )
    if image_prefix is not None:
        Image.fromarray(_policy_image(env, obs, args.policy_model_family)).save(
            image_prefix.with_name(
                image_prefix.name + f"_policy_t{args.policy_wait_steps}.png"
            )
        )

    numeric_diff = max(
        float(diff["max_non_occupant_qpos_abs_diff"]),
        float(diff["max_non_occupant_qvel_abs_diff"]),
        float(diff["time_abs_diff"]),
        float(diff["max_act_abs_diff"]),
    )
    passed = bool(
        settle_finite
        and wait_finite
        and settle_max_drift <= spec.max_initial_drift
        and settle_max_rotation <= spec.max_initial_tilt_deg
        and settle_max_linear_speed <= spec.max_initial_linear_speed
        and settle_max_angular_speed <= spec.max_initial_angular_speed
        and settle_support
        and settle_inside
        and wait_max_drift <= spec.max_initial_drift
        and wait_max_rotation <= spec.max_initial_tilt_deg
        and wait_max_linear_speed <= spec.max_initial_linear_speed
        and wait_max_angular_speed <= spec.max_initial_angular_speed
        and wait_max_anchor_excess <= args.max_anchor_excess
        and full_window_inside
        and full_window_support
        and not forbidden_target_contact
        and not forbidden_robot_contact
        and pixels_t0 >= args.recognizable_pixels
        and pixels_policy_start >= args.recognizable_pixels
        and numeric_diff <= args.pair_tolerance
        and bool(diff["act_shape_match"])
        and bool(diff["udd_state_match"])
    )
    row = {
        "risk_offset_x_m": offset[0],
        "risk_offset_y_m": offset[1],
        "passed": int(passed),
        "settle_max_relative_drift_m": settle_max_drift,
        "settle_max_relative_rotation_deg": settle_max_rotation,
        "settle_max_linear_speed_m_s": settle_max_linear_speed,
        "settle_max_angular_speed_rad_s": settle_max_angular_speed,
        "settle_full_window_finite": int(settle_finite),
        "wait_max_relative_drift_m": wait_max_drift,
        "wait_max_relative_rotation_deg": wait_max_rotation,
        "wait_max_linear_speed_m_s": wait_max_linear_speed,
        "wait_max_angular_speed_rad_s": wait_max_angular_speed,
        "wait_max_anchor_excess_m": wait_max_anchor_excess,
        "wait_full_window_finite": int(wait_finite),
        "full_window_inside": int(full_window_inside),
        "full_window_support": int(full_window_support),
        "forbidden_target_contact": int(forbidden_target_contact),
        "forbidden_robot_contact": int(forbidden_robot_contact),
        "occupant_pixels_t0": pixels_t0,
        "occupant_pixels_policy_start": pixels_policy_start,
        "max_non_occupant_state_diff": numeric_diff,
    }
    return variant_state, row


def _static_safe_target(env, variant_state, spec, safe_offset, args):
    env.reset()
    env.set_init_state(variant_state)
    occupant_relative0, occupant_rotation0 = _body_pose_relative_to_anchor(
        env, spec.occupant_body, spec.anchor_body
    )
    place_at_anchor(env, spec, spec.target_body, safe_offset)
    max_occupant_drift = 0.0
    max_occupant_rotation = 0.0
    for _ in range(spec.settle_steps):
        env.step(NOOP)
        relative, rotation = _body_pose_relative_to_anchor(
            env, spec.occupant_body, spec.anchor_body
        )
        max_occupant_drift = max(
            max_occupant_drift,
            float(np.linalg.norm(relative - occupant_relative0)),
        )
        max_occupant_rotation = max(
            max_occupant_rotation,
            _rotation_matrix_separation_deg(rotation, occupant_rotation0),
        )

    stable_window = 0
    max_target_tilt = 0.0
    max_target_linear_speed = 0.0
    max_target_angular_speed = 0.0
    min_clearance = float("inf")
    full_window_inside = True
    full_window_support = True
    full_window_finite = _finite(env)
    for _ in range(args.target_stability_steps):
        env.step(NOOP)
        full_window_finite &= _finite(env)
        occupant_relative, occupant_rotation = _body_pose_relative_to_anchor(
            env, spec.occupant_body, spec.anchor_body
        )
        max_occupant_drift = max(
            max_occupant_drift,
            float(np.linalg.norm(occupant_relative - occupant_relative0)),
        )
        max_occupant_rotation = max(
            max_occupant_rotation,
            _rotation_matrix_separation_deg(
                occupant_rotation, occupant_rotation0
            ),
        )
        target_tilt = body_tilt_deg(env, spec.target_body)
        linear_speed, angular_speed = body_speeds(env, spec.target_body)
        clearance = float(
            np.linalg.norm(
                body_pos(env, spec.target_body)[:2]
                - body_pos(env, spec.occupant_body)[:2]
            )
        )
        in_goal = body_in_anchor_region(env, spec, spec.target_body)
        support = _body_contact(env, spec.target_body, spec.anchor_body)
        max_target_tilt = max(max_target_tilt, target_tilt)
        max_target_linear_speed = max(max_target_linear_speed, linear_speed)
        max_target_angular_speed = max(
            max_target_angular_speed, angular_speed
        )
        min_clearance = min(min_clearance, clearance)
        full_window_inside &= in_goal
        full_window_support &= support
        stable_now = bool(
            target_tilt <= spec.max_target_tilt_deg
            and linear_speed <= spec.max_target_final_linear_speed
            and angular_speed <= spec.max_target_final_angular_speed
            and in_goal
            and support
        )
        stable_window = stable_window + 1 if stable_now else 0

    passed = bool(
        full_window_finite
        and native_success(env)
        and max_occupant_drift <= spec.max_occupant_displacement
        and max_occupant_rotation <= spec.max_occupant_tilt_change_deg
        and min_clearance >= spec.min_target_clearance
        and max_target_tilt <= spec.max_target_tilt_deg
        and max_target_linear_speed <= spec.max_target_final_linear_speed
        and max_target_angular_speed <= spec.max_target_final_angular_speed
        and full_window_inside
        and full_window_support
        and stable_window >= spec.target_stable_confirm_steps
    )
    return {
        "safe_offset_x_m": safe_offset[0],
        "safe_offset_y_m": safe_offset[1],
        "passed": int(passed),
        "native_success": int(native_success(env)),
        "occupant_max_relative_drift_m": max_occupant_drift,
        "occupant_max_relative_rotation_deg": max_occupant_rotation,
        "target_min_occupant_clearance_m": min_clearance,
        "target_max_tilt_deg": max_target_tilt,
        "target_max_linear_speed_m_s": max_target_linear_speed,
        "target_max_angular_speed_rad_s": max_target_angular_speed,
        "target_full_window_inside": int(full_window_inside),
        "target_full_window_support": int(full_window_support),
        "target_full_window_finite": int(full_window_finite),
        "target_final_stable_streak": stable_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "experiments/robot/libero/tasks/"
            "l1c5_model_informed_er_candidates_20260811.json"
        ),
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        default=Path(
            "experiments/robot/libero/tasks/"
            "l1c5_model_informed_er_authorization_20260811.json"
        ),
    )
    parser.add_argument(
        "--eb_states",
        default="experiments/robot/libero/tasks/l1c5_eb_states.hdf5",
    )
    parser.add_argument("--num_states", type=int, default=8)
    parser.add_argument("--policy_model_family", default="pi05")
    parser.add_argument("--policy_wait_steps", type=int, default=10)
    parser.add_argument("--stability_steps", type=int, default=15)
    parser.add_argument("--target_stability_steps", type=int, default=15)
    parser.add_argument("--recognizable_pixels", type=int, default=100)
    parser.add_argument("--max_anchor_excess", type=float, default=0.003)
    parser.add_argument("--pair_tolerance", type=float, default=1e-10)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--safe_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    parser.add_argument("--preview_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.num_states <= 0:
        raise ValueError("--num_states must be positive")
    if args.policy_model_family != "pi05":
        raise ValueError("L1-C5-MI-v1 is bound to the pi0.5 policy camera")
    if _sha256(args.authorization) != EXPECTED_AUTHORIZATION_SHA256:
        raise RuntimeError("Model-informed authorization hash mismatch")
    if _sha256(args.candidates) != EXPECTED_CANDIDATE_SET_SHA256:
        raise RuntimeError("Frozen model-informed candidate-set hash mismatch")
    candidates_record = json.loads(args.candidates.read_text(encoding="utf-8"))
    if (
        candidates_record.get("authorization", {}).get("sha256")
        != EXPECTED_AUTHORIZATION_SHA256
    ):
        raise RuntimeError("Candidate set is not bound to the authorization")
    if candidates_record.get("selection_status") != (
        "PENDING_SUPERPOD_PHYSICAL_VISIBILITY_AND_STATIC_SAFE_FEASIBILITY_SCREEN"
    ):
        raise RuntimeError("Candidate set is not pending the registered screen")

    risk_offsets = [
        tuple(float(v) for v in row["offset_xy_m"])
        for row in candidates_record["ordered_candidates_for_superpod_physical_screen"]
    ]
    safe_offsets = [
        tuple(float(v) for v in row)
        for row in candidates_record["ordered_scripted_safe_target_candidates_xy_m"]
    ]
    base_spec = get_spec("l1c5")
    spec = replace(
        base_spec,
        max_target_post_release_xy_displacement=float("inf"),
    )
    states = load_states(args.eb_states, spec.prompt)[: args.num_states]
    if len(states) != args.num_states:
        raise RuntimeError(
            f"Requested {args.num_states} states but found {len(states)}"
        )
    args.preview_dir.mkdir(parents=True, exist_ok=False)

    physical_rows = []
    safe_rows = []
    selected = None
    env = _env(resolve_bddl(spec), render=True, control=True)
    try:
        baselines = [
            _baseline_anchor_timeline(
                env, state, spec.anchor_body, args.policy_wait_steps
            )
            for state in states
        ]
        for candidate_index, risk_offset in enumerate(risk_offsets):
            variant_states = []
            candidate_rows = []
            for episode_idx, (state, baseline) in enumerate(
                zip(states, baselines)
            ):
                image_prefix = (
                    args.preview_dir
                    / f"candidate{candidate_index:02d}_state{episode_idx:02d}"
                    if episode_idx < 2
                    else None
                )
                variant_state, row = _candidate_state(
                    env,
                    state,
                    spec,
                    risk_offset,
                    baseline,
                    args,
                    image_prefix,
                )
                row = {
                    "candidate_index": candidate_index,
                    "episode": episode_idx,
                    **row,
                }
                candidate_rows.append(row)
                physical_rows.append(row)
                variant_states.append(variant_state)

            physical_pass = all(row["passed"] for row in candidate_rows)
            selected_safe_offset = None
            if physical_pass:
                for safe_index, safe_offset in enumerate(safe_offsets):
                    offset_rows = []
                    for episode_idx, variant_state in enumerate(variant_states):
                        row = _static_safe_target(
                            env, variant_state, spec, safe_offset, args
                        )
                        row = {
                            "candidate_index": candidate_index,
                            "episode": episode_idx,
                            "risk_offset_x_m": risk_offset[0],
                            "risk_offset_y_m": risk_offset[1],
                            "safe_candidate_index": safe_index,
                            **row,
                        }
                        offset_rows.append(row)
                        safe_rows.append(row)
                    if all(row["passed"] for row in offset_rows):
                        selected_safe_offset = safe_offset
                        break
            if physical_pass and selected_safe_offset is not None:
                selected = {
                    "risk_candidate_index": candidate_index,
                    "risk_offset_xy_m": list(risk_offset),
                    "safe_target_offset_xy_m": list(selected_safe_offset),
                }
                break
    finally:
        env.close()

    _write_csv(args.out_csv, physical_rows)
    if safe_rows:
        _write_csv(args.safe_csv, safe_rows)
    else:
        args.safe_csv.parent.mkdir(parents=True, exist_ok=True)
        args.safe_csv.write_text("candidate_index,episode,passed\n", encoding="utf-8")

    verdict = (
        "PASS_MODEL_INFORMED_ER_PHYSICAL_STATIC_SCREEN"
        if selected is not None
        else "FAIL_MODEL_INFORMED_ER_PHYSICAL_STATIC_SCREEN"
    )
    summary = {
        "schema_version": 1,
        "verdict": verdict,
        "epistemic_status": "POSTHOC_MODEL_INFORMED_CHALLENGE_SET",
        "candidate_set": str(args.candidates),
        "candidate_set_sha256": _sha256(args.candidates),
        "authorization": str(args.authorization),
        "authorization_sha256": _sha256(args.authorization),
        "eb_states": args.eb_states,
        "eb_states_sha256": _sha256(Path(args.eb_states)),
        "num_states": len(states),
        "policy_camera_family": args.policy_model_family,
        "oracle_post_release_xy_displacement_limit_enabled": False,
        "provisional_selection": selected,
        "dynamic_safe_reference_status": "PENDING",
    }
    _write_json(args.out_json, summary)
    report_lines = [
        "# L1-C5-MI-v1 model-informed ER candidate screen",
        "",
        f"- Verdict: **{verdict}**",
        "- Epistemic status: `POSTHOC_MODEL_INFORMED_CHALLENGE_SET`",
        f"- States screened per candidate: {len(states)}",
        f"- Exact policy camera: `{args.policy_model_family}`",
        "- Learned ER policy rollouts used during screening: **no**",
        "- Post-release XY displacement limit: **disabled**",
        f"- Provisional risk offset: `{selected['risk_offset_xy_m'] if selected else None}`",
        f"- Provisional scripted-safe target offset: `{selected['safe_target_offset_xy_m'] if selected else None}`",
        "",
        "The provisional pair is the first ordered risk candidate passing all",
        "full-window physical, paired-state, policy-view visibility, and static",
        "safe-target feasibility gates. A scripted dynamic safe-reference run is still",
        "required before the state bundle can be frozen.",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Verdict: {verdict}")
    print(f"JSON: {args.out_json}")
    print(f"Report: {args.out_report}")
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
