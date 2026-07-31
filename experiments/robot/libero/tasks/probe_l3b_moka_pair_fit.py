"""Search for stable two-pot final states on the native LIBERO stove.

This is a diagnostic geometry probe, not an Eb/Er/Ec generator.  It modifies
only the serialized poses and velocities of the two moka pots already present
in the native ``libero_10`` task, pre-settles each proposal, then reloads that
exact state through the evaluator-style ten-step wait.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks.l3b_order_candidates import (
    CANDIDATES,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    SUITE,
    _refresh_observation,
    _trusted_native_states,
    _write_policy_image,
    body_measurement,
    native_bddl_path,
    static_preflight,
)


POT_BODIES = ("moka_pot_1_main", "moka_pot_2_main")
STOVE_BODY = "flat_stove_1_main"
COOK_SITE = "flat_stove_1_cook_region"
MAX_TILT_DEG = 1.0
MAX_FORMAL_DRIFT_M = 0.005
MAX_FINAL_LINEAR_SPEED_MPS = 0.01
MAX_FINAL_ANGULAR_SPEED_RADPS = 0.05


def yaw_quaternion_wxyz(yaw_deg: float) -> np.ndarray:
    half = np.radians(yaw_deg) / 2.0
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _free_joint_addresses(env, body_name: str) -> tuple[int, int]:
    model = env.sim.model
    body_id = int(model.body_name2id(body_name))
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_bodyid[joint_id]) != body_id:
            continue
        if int(model.jnt_type[joint_id]) != 0:
            continue
        return (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
        )
    raise ValueError(f"no free joint for {body_name}")


def _set_free_body_pose(
    env,
    body_name: str,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> None:
    qpos_address, velocity_address = _free_joint_addresses(env, body_name)
    env.sim.data.qpos[qpos_address : qpos_address + 3] = position
    env.sim.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion_wxyz
    env.sim.data.qvel[velocity_address : velocity_address + 6] = 0.0


def _pot_contact_with(measurement: dict, prefix: str) -> bool:
    return any(str(name).startswith(prefix) for name in measurement["contacts"])


def _sample_record(samples: list[dict[str, dict]]) -> dict:
    first, final = samples[0], samples[-1]
    stability = {}
    for body_name in POT_BODIES:
        origin = np.asarray(first[body_name]["position"], dtype=float)
        stability[body_name] = {
            "pre_wait": first[body_name],
            "first_policy": final[body_name],
            "max_translation_drift_m": max(
                float(
                    np.linalg.norm(
                        np.asarray(sample[body_name]["position"], dtype=float)
                        - origin
                    )
                )
                for sample in samples
            ),
            "max_tilt_deg": max(
                float(sample[body_name]["tilt_deg"]) for sample in samples
            ),
        }
    return stability


def _acceptance(stability: dict, task_success: bool) -> tuple[bool, list[str]]:
    reasons = []
    if not task_success:
        reasons.append("native_goal_false")
    for body_name in POT_BODIES:
        record = stability[body_name]
        final = record["first_policy"]
        if record["max_translation_drift_m"] > MAX_FORMAL_DRIFT_M:
            reasons.append(f"{body_name}:formal_drift")
        if record["max_tilt_deg"] > MAX_TILT_DEG:
            reasons.append(f"{body_name}:tilt")
        if (
            final["linear_speed_mps"] is None
            or final["linear_speed_mps"] > MAX_FINAL_LINEAR_SPEED_MPS
        ):
            reasons.append(f"{body_name}:linear_speed")
        if (
            final["angular_speed_radps"] is None
            or final["angular_speed_radps"] > MAX_FINAL_ANGULAR_SPEED_RADPS
        ):
            reasons.append(f"{body_name}:angular_speed")
        if not _pot_contact_with(final, "flat_stove_1_"):
            reasons.append(f"{body_name}:no_stove_support")
        other_prefix = (
            "moka_pot_2_" if body_name == "moka_pot_1_main" else "moka_pot_1_"
        )
        if _pot_contact_with(final, other_prefix):
            reasons.append(f"{body_name}:pot_contact")
    return not reasons, sorted(set(reasons))


def _candidate_layouts(
    center: np.ndarray,
    gripper_xy: np.ndarray,
) -> list[dict]:
    toward_robot = gripper_xy - center[:2]
    toward_robot /= max(float(np.linalg.norm(toward_robot)), 1e-9)
    across_robot = np.asarray([-toward_robot[1], toward_robot[0]], dtype=float)
    axes = (("near_far", toward_robot), ("lateral", across_robot))
    yaw_pairs = (
        (0.0, 180.0),
        (90.0, 270.0),
        (180.0, 0.0),
        (270.0, 90.0),
        (0.0, 0.0),
        (90.0, 90.0),
    )
    records = []
    for axis_name, axis in axes:
        for separation_m in (0.08, 0.09, 0.10, 0.11, 0.12):
            first_xy = center[:2] - axis * separation_m / 2.0
            second_xy = center[:2] + axis * separation_m / 2.0
            for first_yaw_deg, second_yaw_deg in yaw_pairs:
                records.append(
                    {
                        "axis": axis_name,
                        "separation_m": separation_m,
                        "positions_xy": {
                            POT_BODIES[0]: first_xy.tolist(),
                            POT_BODIES[1]: second_xy.tolist(),
                        },
                        "yaw_deg": {
                            POT_BODIES[0]: first_yaw_deg,
                            POT_BODIES[1]: second_yaw_deg,
                        },
                    }
                )
    return records


def run_probe(
    *,
    output_dir: Path,
    settle_steps: int,
    max_accepted: int,
    render_gpu_device_id: int,
) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    candidate = CANDIDATES["moka"]
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(candidate.task_id)
    if task.bddl_file != candidate.task_file or task.language != candidate.prompt:
        raise ValueError("runtime native moka task does not match the locked task")
    runtime_bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if runtime_bddl.resolve() != native_bddl_path(candidate).resolve():
        raise ValueError("runtime BDDL is not the locked native BDDL")
    native_states = _trusted_native_states(suite, task)

    env = OffScreenRenderEnv(
        bddl_file_name=str(runtime_bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(0)
    trials = []
    accepted = []
    try:
        env.reset()
        env.set_init_state(native_states[0])
        _refresh_observation(env)
        site_id = int(env.sim.model.site_name2id(COOK_SITE))
        site_position = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
        site_size = np.asarray(env.sim.model.site_size[site_id], dtype=float)
        gripper_id = int(env.sim.model.body_name2id("gripper0_eef"))
        gripper_xy = np.asarray(
            env.sim.data.body_xpos[gripper_id][:2], dtype=float
        )
        layouts = _candidate_layouts(site_position, gripper_xy)

        for trial_index, layout in enumerate(layouts):
            env.reset()
            env.set_init_state(native_states[0])
            spawn_z = float(site_position[2] + 0.13)
            for body_name in POT_BODIES:
                xy = np.asarray(layout["positions_xy"][body_name], dtype=float)
                position = np.asarray([xy[0], xy[1], spawn_z], dtype=float)
                quaternion = yaw_quaternion_wxyz(
                    layout["yaw_deg"][body_name]
                )
                _set_free_body_pose(env, body_name, position, quaternion)
            env.sim.forward()
            for _ in range(settle_steps):
                env.step(DUMMY_ACTION)

            settled_state = np.asarray(env.sim.get_state().flatten(), dtype=float)
            env.reset()
            env.set_init_state(settled_state)
            observation = _refresh_observation(env)
            samples = [
                {
                    body_name: body_measurement(env, body_name)
                    for body_name in POT_BODIES
                }
            ]
            for _ in range(FORMAL_WAIT_STEPS):
                observation, _, _, _ = env.step(DUMMY_ACTION)
                samples.append(
                    {
                        body_name: body_measurement(env, body_name)
                        for body_name in POT_BODIES
                    }
                )
            stability = _sample_record(samples)
            task_success = bool(env.check_success())
            passed, rejection_reasons = _acceptance(stability, task_success)
            record = {
                "trial_index": trial_index,
                **layout,
                "native_goal_at_first_policy": task_success,
                "stability": stability,
                "passed": passed,
                "rejection_reasons": rejection_reasons,
            }
            trials.append(record)
            if passed:
                image_path = (
                    output_dir
                    / "accepted"
                    / f"moka_pair_fit_{len(accepted):02d}_first_policy.png"
                )
                _write_policy_image(image_path, observation)
                record["policy_image"] = str(image_path.resolve())
                accepted.append(record)
                if len(accepted) >= max_accepted:
                    break
    finally:
        env.close()

    return {
        **static_preflight(candidate),
        "diagnostic_only": True,
        "formal_authorized": False,
        "intervention": {
            "changed_bodies": list(POT_BODIES),
            "changed_fields": ["free_joint_qpos", "free_joint_qvel"],
            "asset_inventory_changed": False,
            "prompt_changed": False,
            "bddl_changed": False,
        },
        "cook_region": {
            "site": COOK_SITE,
            "position": site_position.tolist(),
            "half_size_m": site_size.tolist(),
            "full_xy_size_m": (2.0 * site_size[:2]).tolist(),
        },
        "thresholds": {
            "max_tilt_deg": MAX_TILT_DEG,
            "max_formal_drift_m": MAX_FORMAL_DRIFT_M,
            "max_final_linear_speed_mps": MAX_FINAL_LINEAR_SPEED_MPS,
            "max_final_angular_speed_radps": MAX_FINAL_ANGULAR_SPEED_RADPS,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "construction_settle_steps": settle_steps,
        },
        "accepted_count": len(accepted),
        "tested_count": len(trials),
        "accepted": accepted,
        "trials": trials,
        "status": (
            "PASS_STABLE_NATIVE_MOKA_PAIR_FOUND"
            if accepted
            else "REJECT_NO_STABLE_NATIVE_MOKA_PAIR_FOUND"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe stable paired moka-pot fits on the native stove"
    )
    parser.add_argument(
        "--output-dir",
        default="review/L3-B_order_candidates_task/moka_pair_fit",
    )
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--max-accepted", type=int, default=5)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    if args.settle_steps < 1 or args.max_accepted < 1:
        raise ValueError("settle steps and max accepted must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_probe(
        output_dir=output_dir,
        settle_steps=args.settle_steps,
        max_accepted=args.max_accepted,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    report_path = output_dir / "moka_pair_fit.json"
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{result['status']} accepted={result['accepted_count']} "
        f"tested={result['tested_count']} report={report_path}"
    )


if __name__ == "__main__":
    main()
