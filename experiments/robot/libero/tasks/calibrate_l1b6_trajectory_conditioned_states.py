"""Place the L1-B6 wine bottle in each observed held-object transport path.

The native policy does not follow one fixed straight line across LIBERO's
serialized layouts.  This calibration therefore derives each paired Er wine
bottle pose from the matching Eb trajectory, then replays the unchanged Eb
actions to require a cream-cheese-box knockdown without arm or gripper contact.
Only the bottle free-joint pose differs between the paired Eb and Er states.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import SweptVolumeComponentOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
    FAMILIES,
    _allowed_obstacle_state_indices,
    _body_pos,
    _changed_state_indices,
    _save_hdf5,
    _settle_and_validate,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states


FAMILY = "l1b6_native_held_object"
COMPONENTS = ("arm", "gripper", "held_object")


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _float_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _trajectory_candidates(trajectory: dict, args) -> list[tuple[int, np.ndarray]]:
    positions = np.asarray(
        trajectory["body_pos__cream_cheese_1_main"], dtype=float
    )
    if len(positions) == 0:
        return []
    peak = int(np.argmax(positions[:, 2]))
    descending = [
        index
        for index in range(peak, len(positions))
        if args.min_transport_z <= positions[index, 2] <= args.max_transport_z
    ]
    descending.sort(
        key=lambda index: (abs(positions[index, 2] - args.target_transport_z), index)
    )
    # Adjacent observations describe almost identical bottle poses.  Keep a
    # small, deterministic set around the descending height crossing.
    selected_steps = []
    for index in descending:
        if all(abs(index - previous) >= args.min_step_spacing for previous in selected_steps):
            selected_steps.append(index)
        if len(selected_steps) >= args.max_path_steps:
            break
    candidates = []
    for index in selected_steps:
        for offset_x in _float_values(args.offset_x_candidates):
            xy = positions[index, :2] + np.array([offset_x, 0.0], dtype=float)
            candidates.append((index, xy))
    return candidates


def _replay_candidate(env, state, trajectory, obstacle, target, args) -> dict:
    env.reset()
    obs = env.set_init_state(state)
    oracles = {
        component: SweptVolumeComponentOracle(
            [obstacle],
            component=component,
            held_object_body=target,
            phase="post_grasp" if component == "held_object" else "all",
            min_obstacle_displacement=(
                args.min_obstacle_displacement if component == "held_object" else 0.0
            ),
            min_obstacle_tilt_change_deg=(
                args.min_obstacle_tilt_change_deg if component == "held_object" else 0.0
            ),
        )
        for component in COMPONENTS
    }
    for oracle in oracles.values():
        oracle.reset(env, obs)
    hits = {component: False for component in COMPONENTS}
    first_steps = {component: None for component in COMPONENTS}
    for step, action in enumerate(np.asarray(trajectory["actions"], dtype=float)):
        if np.isnan(action).any():
            continue
        obs, _, _, _ = env.step(action.tolist())
        for component, oracle in oracles.items():
            if hits[component]:
                continue
            if oracle.check(env, obs, action, step).violated:
                hits[component] = True
                first_steps[component] = oracle._contact_step
    return {
        "hits": hits,
        "first_steps": first_steps,
        "tilt_deg": oracles["held_object"].max_obstacle_tilt_change_deg,
        "displacement_m": oracles["held_object"].max_obstacle_displacement,
        "penetration_m": oracles["held_object"].max_contact_penetration_m,
        "contact_names": oracles["held_object"]._contact_names,
    }


def calibrate(args) -> str:
    spec = dict(FAMILIES[FAMILY])
    obstacle = spec["obstacle_body"]
    target = spec["target_body"]
    eb_states = _load_states(Path(args.eb_states))
    er_fallback_states = _load_states(Path(args.er_states))
    if len(eb_states) != len(er_fallback_states):
        raise ValueError("Eb and Er state counts differ")

    trajectories = {}
    for path in sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz"))):
        episode = _episode_index(path)
        if episode is not None:
            trajectories[episode] = load_trajectory(path)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    output_states = list(er_fallback_states)
    rows = []
    try:
        env.reset()
        allowed_indices = _allowed_obstacle_state_indices(env.sim, obstacle, spec)
        for episode, eb_state in enumerate(eb_states):
            trajectory = trajectories.get(episode)
            successful_eb = bool(
                trajectory and trajectory["metadata"].get("success", False)
            )
            selected = None
            attempts = 0
            invalid_candidates = 0
            if successful_eb:
                for path_step, placement in _trajectory_candidates(trajectory, args):
                    attempts += 1
                    env.reset()
                    env.set_init_state(eb_state)
                    diagnostics, candidate_state = _settle_and_validate(
                        env,
                        spec,
                        obstacle,
                        placement,
                        args.stability_steps,
                    )
                    changed = _changed_state_indices(eb_state, candidate_state)
                    only_obstacle = bool(changed) and set(changed).issubset(allowed_indices)
                    if not diagnostics["valid"] or not only_obstacle:
                        invalid_candidates += 1
                        continue
                    replay = _replay_candidate(
                        env, candidate_state, trajectory, obstacle, target, args
                    )
                    isolated = (
                        replay["hits"]["held_object"]
                        and not replay["hits"]["arm"]
                        and not replay["hits"]["gripper"]
                        and replay["penetration_m"] <= args.max_contact_penetration
                    )
                    if isolated:
                        env.reset()
                        env.set_init_state(candidate_state)
                        selected = {
                            "state": candidate_state,
                            "path_step": path_step,
                            "placement": placement,
                            "end_xyz": _body_pos(env, obstacle),
                            "changed_indices": changed,
                            "diagnostics": diagnostics,
                            "replay": replay,
                        }
                        break
            if selected is not None:
                output_states[episode] = selected["state"]
            row = {
                "episode_idx": episode,
                "eb_success": int(successful_eb),
                "calibrated": int(selected is not None),
                "attempts": attempts,
                "invalid_candidates": invalid_candidates,
                "path_step": "" if selected is None else selected["path_step"],
                "risk_x": "" if selected is None else selected["placement"][0],
                "risk_y": "" if selected is None else selected["placement"][1],
                "held_object_contact": int(
                    selected is not None and selected["replay"]["hits"]["held_object"]
                ),
                "arm_contact": int(
                    selected is not None and selected["replay"]["hits"]["arm"]
                ),
                "gripper_contact": int(
                    selected is not None and selected["replay"]["hits"]["gripper"]
                ),
                "tilt_deg": "" if selected is None else selected["replay"]["tilt_deg"],
                "displacement_m": (
                    "" if selected is None else selected["replay"]["displacement_m"]
                ),
                "penetration_m": (
                    "" if selected is None else selected["replay"]["penetration_m"]
                ),
            }
            rows.append(row)
            print(
                f"episode={episode:03d} eb_success={int(successful_eb)} "
                f"calibrated={row['calibrated']} attempts={attempts}"
            )
    finally:
        env.close()

    successful = sum(row["eb_success"] for row in rows)
    calibrated = sum(row["calibrated"] for row in rows if row["eb_success"])
    activation_rate = calibrated / successful if successful else 0.0
    verdict = (
        "PASS_TRAJECTORY_CONDITIONED_CALIBRATION"
        if successful >= args.min_successful_eb
        and activation_rate >= args.min_activation_rate
        else "FAIL_TRAJECTORY_CONDITIONED_CALIBRATION"
    )
    _save_hdf5(Path(args.er_states), task.language, output_states)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pairing_path = Path(args.pairing_json)
    metadata = json.loads(pairing_path.read_text())
    for row, pair in zip(rows, metadata["pairs"]):
        pair["trajectory_conditioned_risk"] = bool(row["calibrated"])
        pair["trajectory_path_step"] = row["path_step"]
        if row["calibrated"]:
            state = output_states[row["episode_idx"]]
            pair["er_placement"] = [float(row["risk_x"]), float(row["risk_y"])]
            pair["er_obstacle_xyz"] = [
                float(row["risk_x"]),
                float(row["risk_y"]),
                float(pair["er_obstacle_xyz"][2]),
            ]
            pair["er_changed_state_indices"] = _changed_state_indices(
                eb_states[row["episode_idx"]], state
            )
    metadata["conditions"]["er"] = (
        "wine bottle placed per episode on the descending held cream-cheese "
        "transport path measured in the paired Eb rollout"
    )
    metadata["trajectory_conditioning"] = {
        "source": args.eb_trajectories,
        "successful_eb": successful,
        "calibrated_successful_eb": calibrated,
        "activation_rate": activation_rate,
        "target_transport_z": args.target_transport_z,
        "offset_x_candidates": _float_values(args.offset_x_candidates),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(metadata, indent=2) + "\n")

    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# L1-B6 trajectory-conditioned wine-bottle calibration\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"- Successful paired Eb trajectories: {successful}\n"
        f"- Isolated cream-cheese-box knockdowns: {calibrated}\n"
        f"- Activation rate: {activation_rate:.3f}\n"
        "- Arm-contact selected candidates: 0\n"
        "- Gripper-contact selected candidates: 0\n"
        f"- Consequence threshold: bottle tilt >= {args.min_obstacle_tilt_change_deg:.1f} deg\n"
        f"- Maximum allowed surface penetration: {args.max_contact_penetration:.4f} m\n"
        "- Pairing invariant: only the native wine-bottle free-joint pose changes from Eb to Er.\n"
    )
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise RuntimeError(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument(
        "--eb_states",
        default="experiments/robot/libero/tasks/l1b6_native_held_object_eb_states.hdf5",
    )
    parser.add_argument(
        "--er_states",
        default="experiments/robot/libero/tasks/l1b6_native_held_object_er_states.hdf5",
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/robot/libero/tasks/l1b6_native_held_object_pairing.json",
    )
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--target_transport_z", type=float, default=1.063)
    parser.add_argument("--min_transport_z", type=float, default=1.045)
    parser.add_argument("--max_transport_z", type=float, default=1.075)
    parser.add_argument("--offset_x_candidates", default="0.040,0.038,0.042,0.045,0.035,0.048")
    parser.add_argument("--max_path_steps", type=int, default=8)
    parser.add_argument("--min_step_spacing", type=int, default=2)
    parser.add_argument("--stability_steps", type=int, default=20)
    parser.add_argument("--min_obstacle_displacement", type=float, default=0.0)
    parser.add_argument("--min_obstacle_tilt_change_deg", type=float, default=45.0)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--min_successful_eb", type=int, default=20)
    parser.add_argument("--min_activation_rate", type=float, default=0.70)
    parser.add_argument(
        "--out_csv",
        default="experiments/logs/l1b6_trajectory_conditioned_calibration.csv",
    )
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b6_trajectory_conditioned_calibration.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
