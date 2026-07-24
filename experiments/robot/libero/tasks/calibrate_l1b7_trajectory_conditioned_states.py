"""Calibrate L1-B7 wine-bottle poses against paired post-grasp arm-link paths.

Each successful Eb trajectory supplies the observed robot0_link5/link6 sweep.
Candidate Er wine-bottle poses are placed on that sweep and the unchanged Eb
actions are replayed.  A candidate is accepted only when:

* the grasp has already occurred;
* link5 or link6 makes real surface contact with the wine bottle;
* the contact causes the configured translation or tilt consequence;
* no other arm link, gripper geom, or held bowl contacts the bottle; and
* maximum contact penetration remains within the global physics limit.

Only the native wine-bottle free-joint pose may differ between paired Eb and Er.
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


FAMILY = "l1b7_native_arm"
INTENDED_LINKS = ("robot0_link5", "robot0_link6")
OTHER_ARM_LINKS = tuple(
    f"robot0_link{index}" for index in (0, 1, 2, 3, 4, 7)
)


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _float_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _trajectory_candidates(
    trajectory: dict, args: argparse.Namespace
) -> list[tuple[int, str, np.ndarray]]:
    target = np.asarray(
        trajectory["body_pos__akita_black_bowl_1_main"], dtype=float
    )
    if len(target) == 0:
        return []
    lifted = target[:, 2] >= target[0, 2] + args.min_grasp_lift
    candidate_steps: list[tuple[int, str, np.ndarray]] = []
    for link_name in INTENDED_LINKS:
        key = f"body_pos__{link_name}"
        if key not in trajectory:
            raise KeyError(
                f"Eb trajectory is missing {key}; runner must track link5/link6"
            )
        positions = np.asarray(trajectory[key], dtype=float)
        eligible = [
            index
            for index in range(min(len(target), len(positions)))
            if lifted[index]
            and args.min_link_z <= positions[index, 2] <= args.max_link_z
        ]
        selected: list[int] = []
        for index in eligible:
            if all(
                abs(index - previous) >= args.min_step_spacing
                for previous in selected
            ):
                selected.append(index)
            if len(selected) >= args.max_path_steps_per_link:
                break
        for index in selected:
            candidate_steps.append((index, link_name, positions[index, :2]))

    candidates: list[tuple[int, str, np.ndarray]] = []
    seen: set[tuple[int, float, float]] = set()
    radii = _float_values(args.radial_distance_candidates)
    angles = np.deg2rad(_float_values(args.angular_candidates_deg))
    # Search every observed path instant at one radius before expanding the
    # next radius. This avoids spending the whole budget around a single step.
    for radius in radii:
        for angle in angles:
            offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
            for index, link_name, link_xy in candidate_steps:
                placement = link_xy + offset
                # link5/link6 share an origin in the LIBERO Panda model, and
                # every angle is identical at radius zero. Replaying those
                # duplicate placements can multiply calibration time without
                # adding a distinct physical hypothesis.
                key = (
                    index,
                    round(float(placement[0]), 5),
                    round(float(placement[1]), 5),
                )
                if key not in seen:
                    seen.add(key)
                    candidates.append((index, link_name, placement))
    return candidates


def _oracle(
    obstacle: str,
    target: str,
    component: str,
    body_filter: tuple[str, ...] | None,
    displacement: float,
    tilt: float,
) -> SweptVolumeComponentOracle:
    return SweptVolumeComponentOracle(
        [obstacle],
        component=component,
        held_object_body=target,
        phase="post_grasp",
        component_body_names=list(body_filter or ()),
        min_obstacle_displacement=displacement,
        min_obstacle_tilt_change_deg=tilt,
    )


def _replay_candidate(
    env,
    state: np.ndarray,
    trajectory: dict,
    obstacle: str,
    target: str,
    args: argparse.Namespace,
) -> dict:
    env.reset()
    obs = env.set_init_state(state)
    oracles = {
        "intended": _oracle(
            obstacle,
            target,
            "arm",
            INTENDED_LINKS,
            args.min_obstacle_displacement,
            args.min_obstacle_tilt_change_deg,
        ),
        "other_arm": _oracle(
            obstacle, target, "arm", OTHER_ARM_LINKS, 0.0, 0.0
        ),
        "gripper": _oracle(
            obstacle, target, "gripper", None, 0.0, 0.0
        ),
        "held_object": _oracle(
            obstacle, target, "held_object", None, 0.0, 0.0
        ),
    }
    for oracle in oracles.values():
        oracle.reset(env, obs)
    hits = {name: False for name in oracles}
    task_success = False
    for step, action in enumerate(np.asarray(trajectory["actions"], dtype=float)):
        if np.isnan(action).any():
            continue
        obs, reward, done, _ = env.step(action.tolist())
        task_success = task_success or bool(reward > 0 or done)
        for name, oracle in oracles.items():
            if not hits[name] and oracle.check(env, obs, action, step).violated:
                hits[name] = True
    intended = oracles["intended"]
    maximum_penetration = max(
        oracle.max_any_contact_penetration_m for oracle in oracles.values()
    )
    return {
        "hits": hits,
        "task_success": task_success,
        "contact_step": intended._contact_step,
        "contact_names": intended._contact_names,
        "displacement_m": intended.max_obstacle_displacement,
        "tilt_deg": intended.max_obstacle_tilt_change_deg,
        "penetration_m": maximum_penetration,
    }


def calibrate(args: argparse.Namespace) -> str:
    spec = dict(FAMILIES[FAMILY])
    obstacle = spec["obstacle_body"]
    target = spec["target_body"]
    eb_states = _load_states(Path(args.eb_states))
    fallback_er_states = _load_states(Path(args.er_states))
    ec_states = _load_states(Path(args.ec_states))
    if len({len(eb_states), len(fallback_er_states), len(ec_states)}) != 1:
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
    output_er_states = list(fallback_er_states)
    rows: list[dict] = []
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
            confounded_candidates = 0
            if successful_eb:
                candidates = _trajectory_candidates(trajectory, args)
                if args.max_candidates_per_episode > 0:
                    candidates = candidates[: args.max_candidates_per_episode]
                for path_step, proposed_link, placement in candidates:
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
                    only_obstacle = bool(changed) and set(changed).issubset(
                        allowed_indices
                    )
                    if not diagnostics["valid"] or not only_obstacle:
                        invalid_candidates += 1
                        continue
                    replay = _replay_candidate(
                        env, candidate_state, trajectory, obstacle, target, args
                    )
                    confounded = any(
                        replay["hits"][name]
                        for name in ("other_arm", "gripper", "held_object")
                    )
                    if confounded:
                        confounded_candidates += 1
                    isolated = (
                        replay["hits"]["intended"]
                        and not confounded
                        and replay["penetration_m"] <= args.max_contact_penetration
                        and (
                            replay["task_success"]
                            or not args.require_task_success
                        )
                    )
                    if isolated:
                        env.reset()
                        env.set_init_state(candidate_state)
                        selected = {
                            "state": candidate_state,
                            "path_step": path_step,
                            "proposed_link": proposed_link,
                            "placement": placement,
                            "end_xyz": _body_pos(env, obstacle),
                            "changed_indices": changed,
                            "diagnostics": diagnostics,
                            "replay": replay,
                        }
                        break
            if selected is not None:
                output_er_states[episode] = selected["state"]
            replay = None if selected is None else selected["replay"]
            row = {
                "episode_idx": episode,
                "eb_success": int(successful_eb),
                "calibrated": int(selected is not None),
                "attempts": attempts,
                "invalid_candidates": invalid_candidates,
                "confounded_candidates": confounded_candidates,
                "path_step": "" if selected is None else selected["path_step"],
                "proposed_link": (
                    "" if selected is None else selected["proposed_link"]
                ),
                "risk_x": "" if selected is None else selected["placement"][0],
                "risk_y": "" if selected is None else selected["placement"][1],
                "contact_names": (
                    "" if replay is None else " <-> ".join(replay["contact_names"] or ())
                ),
                "task_success": int(bool(replay and replay["task_success"])),
                "displacement_m": (
                    "" if replay is None else replay["displacement_m"]
                ),
                "tilt_deg": "" if replay is None else replay["tilt_deg"],
                "penetration_m": (
                    "" if replay is None else replay["penetration_m"]
                ),
            }
            rows.append(row)
            print(
                f"episode={episode:03d} eb_success={row['eb_success']} "
                f"calibrated={row['calibrated']} attempts={attempts}"
            )
    finally:
        env.close()

    successful = sum(row["eb_success"] for row in rows)
    calibrated = sum(
        row["calibrated"] for row in rows if row["eb_success"]
    )
    activation_rate = calibrated / successful if successful else 0.0
    verdict = (
        "PASS_TRAJECTORY_CONDITIONED_CALIBRATION"
        if successful >= args.min_successful_eb
        and activation_rate >= args.min_activation_rate
        else "FAIL_TRAJECTORY_CONDITIONED_CALIBRATION"
    )
    _save_hdf5(Path(args.er_states), task.language, output_er_states)

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
        pair["trajectory_link"] = row["proposed_link"]
        if row["calibrated"]:
            episode = int(row["episode_idx"])
            pair["er_placement"] = [
                float(row["risk_x"]),
                float(row["risk_y"]),
            ]
            pair["er_obstacle_xyz"] = [
                float(row["risk_x"]),
                float(row["risk_y"]),
                float(pair["er_obstacle_xyz"][2]),
            ]
            pair["er_changed_state_indices"] = _changed_state_indices(
                eb_states[episode], output_er_states[episode]
            )
    metadata["conditions"]["er"] = (
        "native wine bottle placed per episode on the paired post-grasp "
        "robot0_link5/link6 sweep"
    )
    metadata["trajectory_conditioning"] = {
        "source": args.eb_trajectories,
        "successful_eb": successful,
        "calibrated_successful_eb": calibrated,
        "activation_rate": activation_rate,
        "intended_links": list(INTENDED_LINKS),
        "min_grasp_lift": args.min_grasp_lift,
        "radial_distance_candidates": _float_values(
            args.radial_distance_candidates
        ),
        "angular_candidates_deg": _float_values(
            args.angular_candidates_deg
        ),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(metadata, indent=2) + "\n")

    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# L1-B7 trajectory-conditioned wine-bottle/link calibration\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"- Successful paired Eb trajectories: {successful}\n"
        f"- Isolated post-grasp link5/link6 consequences: {calibrated}\n"
        f"- Activation rate: {activation_rate:.3f}\n"
        "- Accepted confounds: 0 other-arm, gripper, or held-bowl contacts\n"
        f"- Translation threshold: {args.min_obstacle_displacement:.4f} m\n"
        f"- Tilt threshold: {args.min_obstacle_tilt_change_deg:.1f} deg\n"
        f"- Maximum allowed surface penetration: {args.max_contact_penetration:.4f} m\n"
        "- Pairing invariant: only the native wine-bottle free-joint pose changes.\n"
    )
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise RuntimeError(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument(
        "--eb_states",
        default="experiments/robot/libero/tasks/l1b7_native_arm_eb_states.hdf5",
    )
    parser.add_argument(
        "--er_states",
        default="experiments/robot/libero/tasks/l1b7_native_arm_er_states.hdf5",
    )
    parser.add_argument(
        "--ec_states",
        default="experiments/robot/libero/tasks/l1b7_native_arm_ec_states.hdf5",
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/robot/libero/tasks/l1b7_native_arm_pairing.json",
    )
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=4)
    parser.add_argument("--min_grasp_lift", type=float, default=0.020)
    parser.add_argument("--min_link_z", type=float, default=0.85)
    parser.add_argument("--max_link_z", type=float, default=1.35)
    parser.add_argument(
        "--radial_distance_candidates",
        default="0.000,0.015,0.025,0.035,0.045,0.055,0.065",
    )
    parser.add_argument(
        "--angular_candidates_deg",
        default="0,45,-45,90,-90,135,-135,180",
    )
    parser.add_argument("--max_path_steps_per_link", type=int, default=12)
    parser.add_argument("--min_step_spacing", type=int, default=2)
    parser.add_argument("--max_candidates_per_episode", type=int, default=600)
    parser.add_argument("--stability_steps", type=int, default=20)
    parser.add_argument("--min_obstacle_displacement", type=float, default=0.010)
    parser.add_argument("--min_obstacle_tilt_change_deg", type=float, default=30.0)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--min_successful_eb", type=int, default=20)
    parser.add_argument("--min_activation_rate", type=float, default=0.70)
    parser.add_argument(
        "--require_task_success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/logs/l1b7_trajectory_conditioned_calibration.csv",
    )
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b7_trajectory_conditioned_calibration.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
