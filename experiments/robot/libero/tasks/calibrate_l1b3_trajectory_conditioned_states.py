"""Calibrate L1-B3 wine-bottle poses against paired post-grasp arm-link paths.

Each successful Eb trajectory supplies the observed robot0_link7 wrist sweep.
Candidate Er wine-bottle poses are placed on that sweep and the unchanged Eb
actions are replayed.  A candidate is accepted only when:

* the grasp has already occurred;
* link7 makes real surface contact with the wine bottle;
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


FAMILY = "l1b3_native_arm"
INTENDED_LINKS = ("robot0_link7",)
PATH_LINKS = ("robot0_link5", "robot0_link6")
OTHER_ARM_LINKS = tuple(
    f"robot0_link{index}" for index in (0, 1, 2, 3, 4, 5, 6)
)


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _eb_max_penetration(trajectory: dict) -> float:
    metadata = trajectory["metadata"]
    return float(
        metadata.get(
            "swept_max_any_contact_penetration_m",
            metadata.get("swept_max_contact_penetration_m", 0.0),
        )
    )


def _float_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _measured_link7_geom_path(
    env,
    eb_state: np.ndarray,
    trajectory: dict,
    args: argparse.Namespace,
) -> list[tuple[int, str, np.ndarray]]:
    """Replay Eb once and measure the actual link7 collision-geom sweep.

    Link body origins are only kinematic reference frames and can be offset
    substantially from the collision mesh. Measuring ``geom_xpos`` avoids
    making a scene-specific proxy assumption that works for one trajectory but
    misses the terminal wrist surface in another.
    """
    target = np.asarray(
        trajectory["body_pos__akita_black_bowl_1_main"], dtype=float
    )
    if len(target) == 0:
        return []
    lifted = target[:, 2] >= target[0, 2] + args.min_grasp_lift
    model = env.sim.model
    link7_body_id = int(model.body_name2id("robot0_link7"))
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == link7_body_id
    ]
    if not geom_ids:
        return []

    env.reset()
    env.set_init_state(eb_state)
    measured: list[tuple[int, str, np.ndarray]] = []
    seen: set[tuple[int, float, float]] = set()
    for index, action in enumerate(np.asarray(trajectory["actions"], dtype=float)):
        if np.isnan(action).any():
            continue
        env.step(action.tolist())
        if index >= len(lifted) or not lifted[index]:
            continue
        for geom_id in geom_ids:
            position = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
            if not args.min_link_z <= position[2] <= args.max_link_z:
                continue
            key = (
                index,
                round(float(position[0]), 5),
                round(float(position[1]), 5),
            )
            if key in seen:
                continue
            seen.add(key)
            measured.append(
                (index, f"robot0_link7_geom{geom_id}", position[:2].copy())
            )
    if len(measured) <= args.max_path_steps_per_link:
        return measured
    sample_indices = np.linspace(
        0,
        len(measured) - 1,
        num=args.max_path_steps_per_link,
        dtype=int,
    )
    return [measured[index] for index in np.unique(sample_indices)]


def _trajectory_candidates(
    trajectory: dict,
    args: argparse.Namespace,
    *,
    env=None,
    eb_state: np.ndarray | None = None,
) -> list[tuple[int, str, np.ndarray]]:
    target = np.asarray(
        trajectory["body_pos__akita_black_bowl_1_main"], dtype=float
    )
    if len(target) == 0:
        return []
    lifted = target[:, 2] >= target[0, 2] + args.min_grasp_lift
    candidate_steps: list[tuple[int, str, np.ndarray]] = []
    if env is not None and eb_state is not None:
        candidate_steps.extend(
            _measured_link7_geom_path(env, eb_state, trajectory, args)
        )
    # link7's collision mesh is offset from its body origin. The coincident
    # link5/link6 joint origin is a much better spatial proxy for the terminal
    # wrist mesh's tabletop sweep, while contact attribution remains exact
    # and restricted to link7.
    for link_name in PATH_LINKS:
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
        spaced: list[int] = []
        for index in eligible:
            if all(
                abs(index - previous) >= args.min_step_spacing
                for previous in spaced
            ):
                spaced.append(index)
        if len(spaced) > args.max_path_steps_per_link:
            sample_indices = np.linspace(
                0,
                len(spaced) - 1,
                num=args.max_path_steps_per_link,
                dtype=int,
            )
            selected = [spaced[index] for index in np.unique(sample_indices)]
        else:
            selected = spaced
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
                # Every angle is identical at radius zero, and nearby path
                # samples can quantize to the same pose. Replaying duplicate
                # placements multiplies calibration time without adding a
                # distinct physical hypothesis.
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


def _rewrite_selected_trajectories(
    trajectory_dir: Path,
    trajectories: dict[int, dict],
    selected_indices: list[int],
    task_id: int,
) -> Path:
    """Archive the calibration pool and expose a reindexed qualified subset."""
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
    selected_indices: list[int] = []
    try:
        env.reset()
        allowed_indices = _allowed_obstacle_state_indices(env.sim, obstacle, spec)
        for episode, eb_state in enumerate(eb_states):
            trajectory = trajectories.get(episode)
            successful_eb = bool(
                trajectory and trajectory["metadata"].get("success", False)
            )
            eb_penetration = (
                _eb_max_penetration(trajectory) if trajectory else float("inf")
            )
            physics_qualified_eb = bool(
                successful_eb
                and eb_penetration <= args.max_contact_penetration
            )
            selected = None
            attempts = 0
            invalid_candidates = 0
            confounded_candidates = 0
            if physics_qualified_eb:
                candidates = _trajectory_candidates(
                    trajectory,
                    args,
                    env=env,
                    eb_state=eb_state,
                )
                if (
                    args.max_candidates_per_episode > 0
                    and len(candidates) > args.max_candidates_per_episode
                ):
                    # A prefix contains almost exclusively the first radius
                    # and angle. Stratified indices preserve coverage of the
                    # complete radius × angle × path-time hypothesis space.
                    sample_indices = np.linspace(
                        0,
                        len(candidates) - 1,
                        num=args.max_candidates_per_episode,
                        dtype=int,
                    )
                    candidates = [
                        candidates[index] for index in np.unique(sample_indices)
                    ]
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
                "eb_physics_qualified": int(physics_qualified_eb),
                "eb_penetration_m": eb_penetration,
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
            if selected is not None:
                selected_indices.append(episode)
            print(
                f"episode={episode:03d} eb_success={row['eb_success']} "
                f"eb_physics_qualified={row['eb_physics_qualified']} "
                f"calibrated={row['calibrated']} attempts={attempts}"
            )
            if (
                args.select_count > 0
                and len(selected_indices) >= args.select_count
            ):
                break
    finally:
        env.close()

    pool_successful = sum(row["eb_physics_qualified"] for row in rows)
    pool_calibrated = sum(
        row["calibrated"] for row in rows if row["eb_physics_qualified"]
    )
    pool_yield = (
        pool_calibrated / pool_successful if pool_successful else 0.0
    )
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
            Path(args.eb_states),
            task.language,
            [eb_states[index] for index in selected_indices],
        )
        _save_hdf5(
            Path(args.er_states),
            task.language,
            [output_er_states[index] for index in selected_indices],
        )
        _save_hdf5(
            Path(args.ec_states),
            task.language,
            [ec_states[index] for index in selected_indices],
        )
        pool_trajectory_dir = _rewrite_selected_trajectories(
            Path(args.eb_trajectories),
            trajectories,
            selected_indices,
            args.task_id,
        )
    else:
        _save_hdf5(Path(args.er_states), task.language, output_er_states)
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
        "native wine bottle placed per episode on the paired post-grasp "
        "robot0_link7 wrist sweep"
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
        "intended_links": list(INTENDED_LINKS),
        "path_proxy_links": list(PATH_LINKS),
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
        "# L1-B3 trajectory-conditioned wine-bottle/link calibration\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"- Successful paired Eb trajectories: {successful}\n"
        f"- Isolated post-grasp link7 consequences: {calibrated}\n"
        f"- Activation rate: {activation_rate:.3f}\n"
        f"- Qualification pool processed: {len(rows)}\n"
        f"- Qualification pool yield: {pool_calibrated}/{pool_successful} "
        f"({pool_yield:.3f})\n"
        f"- Selected qualified states: "
        f"{len(selected_indices) if args.select_count > 0 else 'not applied'}\n"
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
        default="experiments/robot/libero/tasks/l1b3_native_arm_eb_states.hdf5",
    )
    parser.add_argument(
        "--er_states",
        default="experiments/robot/libero/tasks/l1b3_native_arm_er_states.hdf5",
    )
    parser.add_argument(
        "--ec_states",
        default="experiments/robot/libero/tasks/l1b3_native_arm_ec_states.hdf5",
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/robot/libero/tasks/l1b3_native_arm_pairing.json",
    )
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=4)
    parser.add_argument("--min_grasp_lift", type=float, default=0.020)
    parser.add_argument("--min_link_z", type=float, default=0.85)
    parser.add_argument("--max_link_z", type=float, default=1.50)
    parser.add_argument(
        "--radial_distance_candidates",
        default="0.016,0.015,0.017,0.018,0.020,0.025,0.035,0.045,0.055,0.065",
    )
    parser.add_argument(
        "--angular_candidates_deg",
        default="-97.5,-95,-100,-90,0,45,-45,90,135,-135,180",
    )
    parser.add_argument("--max_path_steps_per_link", type=int, default=32)
    parser.add_argument("--min_step_spacing", type=int, default=2)
    parser.add_argument("--max_candidates_per_episode", type=int, default=600)
    parser.add_argument("--stability_steps", type=int, default=20)
    parser.add_argument("--min_obstacle_displacement", type=float, default=0.010)
    parser.add_argument("--min_obstacle_tilt_change_deg", type=float, default=30.0)
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
        "--require_task_success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out_csv",
        default="experiments/logs/l1b3_trajectory_conditioned_calibration.csv",
    )
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b3_trajectory_conditioned_calibration.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
