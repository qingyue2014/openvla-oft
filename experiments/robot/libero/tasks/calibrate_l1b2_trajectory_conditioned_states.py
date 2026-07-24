"""Place the L1-B2 wine bottle in each observed held-object transport path.

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
import shutil
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


FAMILY = "l1b2_native_held_object"
COMPONENTS = ("arm", "gripper", "held_object")


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _float_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _rotate_xy(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    return rotation @ vector


def _trajectory_candidates(trajectory: dict, args) -> list[tuple[int, np.ndarray]]:
    positions = np.asarray(
        trajectory["body_pos__cream_cheese_1_main"], dtype=float
    )
    eef_positions = np.asarray(trajectory["eef_pos"], dtype=float)
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
    step_data = []
    for index in selected_steps:
        # The grasp can rotate by more than 90 degrees across native layouts.
        # Use the measured gripper-to-box-center direction as the primary
        # protrusion axis, then search small angular and radial variations.
        outward = positions[index, :2] - eef_positions[index, :2]
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm < 1e-5:
            outward = np.array([1.0, 0.0], dtype=float)
        else:
            outward = outward / outward_norm
        step_data.append((index, outward))

    candidates = []
    # Cover the whole descending transport segment before densely expanding a
    # single instant.  This makes a fixed search budget a true swept-path
    # search rather than a local pose search.
    for distance in _float_values(args.radial_distance_candidates):
        for angle_deg in _float_values(args.angular_offset_deg_candidates):
            for index, outward in step_data:
                direction = _rotate_xy(outward, angle_deg)
                candidates.append(
                    (index, positions[index, :2] + distance * direction)
                )

    for index, _ in step_data:
        # Retain the original world-axis grid as a conservative fallback for
        # nearly centered grasps whose sub-centimetre EEF offset is noisy.
        for offset_y in _float_values(args.offset_y_candidates):
            for offset_x in _float_values(args.offset_x_candidates):
                xy = positions[index, :2] + np.array(
                    [offset_x, offset_y], dtype=float
                )
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


def _rewrite_selected_trajectories(
    trajectory_dir: Path,
    trajectories: dict[int, dict],
    selected_indices: list[int],
    task_id: int,
) -> Path:
    """Archive the qualification pool and expose a reindexed formal subset."""
    pool_dir = trajectory_dir.with_name(trajectory_dir.name + "_pool")
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    trajectory_dir.rename(pool_dir)
    trajectory_dir.mkdir(parents=True)
    index_rows = []
    for episode_idx, pool_episode_idx in enumerate(selected_indices):
        trajectory = trajectories[pool_episode_idx]
        metadata = dict(trajectory["metadata"])
        metadata["episode_idx"] = episode_idx
        metadata["qualification_pool_episode_idx"] = pool_episode_idx
        arrays = {
            key: value for key, value in trajectory.items() if key != "metadata"
        }
        path = trajectory_dir / f"task{task_id}_ep{episode_idx:03d}.npz"
        np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)
        index_rows.append(metadata)
    (trajectory_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    return pool_dir


def calibrate(args) -> str:
    spec = dict(FAMILIES[FAMILY])
    obstacle = spec["obstacle_body"]
    target = spec["target_body"]
    eb_states = _load_states(Path(args.eb_states))
    er_fallback_states = _load_states(Path(args.er_states))
    ec_states = _load_states(Path(args.ec_states))
    if len({len(eb_states), len(er_fallback_states), len(ec_states)}) != 1:
        raise ValueError("Eb, Er, and Ec state counts differ")

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
    selected_indices = []
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
            held_hits = 0
            arm_hits = 0
            gripper_hits = 0
            if successful_eb:
                candidates = _trajectory_candidates(trajectory, args)
                if args.max_candidates_per_episode > 0:
                    candidates = candidates[: args.max_candidates_per_episode]
                for path_step, placement in candidates:
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
                    held_hits += int(replay["hits"]["held_object"])
                    arm_hits += int(replay["hits"]["arm"])
                    gripper_hits += int(replay["hits"]["gripper"])
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
                "candidate_held_hits": held_hits,
                "candidate_arm_hits": arm_hits,
                "candidate_gripper_hits": gripper_hits,
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
            if selected is not None:
                selected_indices.append(episode)
            print(
                f"episode={episode:03d} eb_success={int(successful_eb)} "
                f"calibrated={row['calibrated']} attempts={attempts}"
            )
            if args.select_count > 0 and len(selected_indices) >= args.select_count:
                break
    finally:
        env.close()

    pool_successful = sum(row["eb_success"] for row in rows)
    pool_calibrated = sum(row["calibrated"] for row in rows if row["eb_success"])
    pool_yield = pool_calibrated / pool_successful if pool_successful else 0.0
    if args.select_count > 0:
        selected_ok = len(selected_indices) == args.select_count
        successful = len(selected_indices)
        calibrated = len(selected_indices)
        activation_rate = 1.0 if selected_indices else 0.0
    else:
        selected_ok = True
        successful = pool_successful
        calibrated = pool_calibrated
        activation_rate = pool_yield
    verdict = (
        "PASS_TRAJECTORY_CONDITIONED_CALIBRATION"
        if selected_ok
        and successful >= args.min_successful_eb
        and activation_rate >= args.min_activation_rate
        else "FAIL_TRAJECTORY_CONDITIONED_CALIBRATION"
    )
    if args.select_count > 0 and selected_ok:
        _save_hdf5(
            Path(args.eb_states), task.language,
            [eb_states[index] for index in selected_indices],
        )
        _save_hdf5(
            Path(args.er_states), task.language,
            [output_states[index] for index in selected_indices],
        )
        _save_hdf5(
            Path(args.ec_states), task.language,
            [ec_states[index] for index in selected_indices],
        )
        pool_trajectory_dir = _rewrite_selected_trajectories(
            Path(args.eb_trajectories), trajectories, selected_indices, args.task_id
        )
    else:
        _save_hdf5(Path(args.er_states), task.language, output_states)
        pool_trajectory_dir = None
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
    if args.select_count > 0 and selected_ok:
        selected_pairs = []
        for episode_idx, pool_episode_idx in enumerate(selected_indices):
            pair = dict(metadata["pairs"][pool_episode_idx])
            pair["qualification_pool_episode_idx"] = pool_episode_idx
            pair["episode_idx"] = episode_idx
            selected_pairs.append(pair)
        metadata["pairs"] = selected_pairs
        metadata["num_states"] = len(selected_pairs)
        metadata["unique_source_state_indices"] = len(
            {pair["source_state_index"] for pair in selected_pairs}
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
        "qualification_pool_processed": len(rows),
        "qualification_pool_successful_eb": pool_successful,
        "qualification_pool_calibrated": pool_calibrated,
        "qualification_pool_yield": pool_yield,
        "selected_pool_episode_indices": selected_indices,
        "selected_count": args.select_count,
        "pool_trajectory_dir": (
            None if pool_trajectory_dir is None else str(pool_trajectory_dir)
        ),
        "target_transport_z": args.target_transport_z,
        "offset_x_candidates": _float_values(args.offset_x_candidates),
        "offset_y_candidates": _float_values(args.offset_y_candidates),
        "radial_distance_candidates": _float_values(
            args.radial_distance_candidates
        ),
        "angular_offset_deg_candidates": _float_values(
            args.angular_offset_deg_candidates
        ),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(metadata, indent=2) + "\n")

    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# L1-B2 trajectory-conditioned wine-bottle calibration\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"- Successful paired Eb trajectories: {successful}\n"
        f"- Isolated cream-cheese-box knockdowns: {calibrated}\n"
        f"- Activation rate: {activation_rate:.3f}\n"
        f"- Qualification pool processed: {len(rows)}\n"
        f"- Qualification pool calibrated yield: {pool_calibrated}/{pool_successful} ({pool_yield:.3f})\n"
        f"- Selected formal states: {len(selected_indices) if args.select_count > 0 else 'not applied'}\n"
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
        default="experiments/robot/libero/tasks/l1b2_native_held_object_eb_states.hdf5",
    )
    parser.add_argument(
        "--er_states",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_er_states.hdf5",
    )
    parser.add_argument(
        "--ec_states",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_ec_states.hdf5",
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
    )
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--target_transport_z", type=float, default=1.063)
    parser.add_argument("--min_transport_z", type=float, default=1.000)
    parser.add_argument("--max_transport_z", type=float, default=1.090)
    parser.add_argument(
        "--radial_distance_candidates",
        default="0.020,0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060,0.065,0.070",
    )
    parser.add_argument(
        "--angular_offset_deg_candidates",
        default="0,20,-20,40,-40,60,-60,90,-90,120,-120,180",
    )
    parser.add_argument(
        "--offset_x_candidates",
        default="0.040,0.038,0.042,0.045,0.035,0.048,0.055,0.060,0.065,0.070",
    )
    parser.add_argument(
        "--offset_y_candidates",
        default="0.000,0.020,0.040,0.050,-0.020,-0.040",
    )
    parser.add_argument("--max_path_steps", type=int, default=8)
    parser.add_argument(
        "--max_candidates_per_episode",
        type=int,
        default=400,
        help="Bound per-trajectory physics search; 0 evaluates every candidate",
    )
    parser.add_argument("--min_step_spacing", type=int, default=2)
    parser.add_argument("--stability_steps", type=int, default=20)
    parser.add_argument("--min_obstacle_displacement", type=float, default=0.0)
    parser.add_argument("--min_obstacle_tilt_change_deg", type=float, default=45.0)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--min_successful_eb", type=int, default=20)
    parser.add_argument("--min_activation_rate", type=float, default=0.70)
    parser.add_argument(
        "--select_count",
        type=int,
        default=0,
        help="Select and reindex this many qualified states from a larger pool",
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/logs/l1b2_trajectory_conditioned_calibration.csv",
    )
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b2_trajectory_conditioned_calibration.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
