"""Search native wine-bottle poses for an isolated L1-B4 arm sweep."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from experiments.robot.libero.physcog_oracles import SweptVolumeComponentOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    _find_free_joint_qadr,
)

TARGET_BODY = "akita_black_bowl_1_main"
OBSTACLE_BODY = "wine_bottle_1_main"
COMPONENTS = ("arm", "gripper", "held_object")


def _values(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def _load_states(path: Path) -> list[np.ndarray]:
    with h5py.File(path) as handle:
        group = handle[next(iter(handle.keys()))]
        return [
            np.asarray(group[name]["initial_state"])
            for name in sorted(group, key=lambda value: int(value.split("_")[-1]))
        ]


def _free_joint_vadr(sim, qadr: int) -> int:
    for joint_id in range(sim.model.njnt):
        if int(sim.model.jnt_qposadr[joint_id]) == qadr:
            return int(sim.model.jnt_dofadr[joint_id])
    raise ValueError(f"No velocity address for qpos address {qadr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_states", type=Path, required=True)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--task_id", type=int, default=4)
    parser.add_argument("--fractions", default="-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.1")
    parser.add_argument("--laterals", default="-0.18,-0.12,-0.06,0,0.06,0.12,0.18")
    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--obstacle_z", type=float, default=None)
    parser.add_argument("--min_displacement", type=float, default=0.004)
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    states = _load_states(args.eb_states)
    trajectories = {}
    for path in glob.glob(os.path.join(args.eb_trajectories, "task*_ep*.npz")):
        match = re.search(r"_ep(\d+)\.npz$", path)
        if match:
            trajectory = load_trajectory(path)
            if trajectory["metadata"].get("success"):
                trajectories[int(match.group(1))] = trajectory
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    rows = []
    try:
        for fraction in _values(args.fractions):
            for lateral in _values(args.laterals):
                hits = {component: 0 for component in COMPONENTS}
                hit_episodes = {component: [] for component in COMPONENTS}
                hit_reasons = {component: [] for component in COMPONENTS}
                valid = 0
                for episode_idx, trajectory in trajectories.items():
                    env.reset()
                    env.set_init_state(states[episode_idx])
                    phases = np.asarray(trajectory["phases"])
                    actions = np.asarray(trajectory["actions"], dtype=float)
                    for action in actions[phases != "policy"]:
                        if not np.isnan(action).any():
                            env.step(action.tolist())
                    target_start = np.asarray(
                        env.sim.data.body_xpos[env.sim.model.body_name2id(TARGET_BODY)][:2]
                    )
                    target_end = np.asarray(
                        trajectory[f"body_pos__{TARGET_BODY}"][-1, :2], dtype=float
                    )
                    delta = target_end - target_start
                    distance = float(np.linalg.norm(delta))
                    if distance < 0.05:
                        continue
                    direction = delta / distance
                    placement = (
                        target_start
                        + fraction * delta
                        + lateral * np.array([-direction[1], direction[0]])
                    )
                    qadr = _find_free_joint_qadr(env.sim, OBSTACLE_BODY)
                    vadr = _free_joint_vadr(env.sim, qadr)
                    env.sim.data.qpos[qadr:qadr + 2] = placement
                    if args.obstacle_z is not None:
                        env.sim.data.qpos[qadr + 2] = args.obstacle_z
                    env.sim.data.qvel[vadr:vadr + 6] = 0.0
                    env.sim.forward()
                    before = np.asarray(
                        env.sim.data.body_xpos[env.sim.model.body_name2id(OBSTACLE_BODY)]
                    ).copy()
                    for _ in range(args.settle_steps):
                        env.sim.step()
                    drift = float(
                        np.linalg.norm(
                            np.asarray(
                                env.sim.data.body_xpos[
                                    env.sim.model.body_name2id(OBSTACLE_BODY)
                                ]
                            )
                            - before
                        )
                    )
                    if drift > 0.02:
                        continue
                    valid += 1
                    oracles = {
                        component: SweptVolumeComponentOracle(
                            [OBSTACLE_BODY],
                            component=component,
                            held_object_body=TARGET_BODY,
                            phase="post_grasp" if component == "held_object" else "all",
                            min_obstacle_displacement=(
                                args.min_displacement if component == "arm" else 0.0
                            ),
                        )
                        for component in COMPONENTS
                    }
                    for oracle in oracles.values():
                        oracle.reset(env, {})
                    seen = {component: False for component in COMPONENTS}
                    for step, action in enumerate(actions[phases == "policy"]):
                        if np.isnan(action).any():
                            continue
                        obs, _, _, _ = env.step(action.tolist())
                        for component, oracle in oracles.items():
                            if not seen[component]:
                                seen[component] = oracle.check(
                                    env, obs, action, step
                                ).violated
                    for component in COMPONENTS:
                        hits[component] += int(seen[component])
                        if seen[component]:
                            hit_episodes[component].append(episode_idx)
                            names = oracles[component]._contact_names or ("unknown", "unknown")
                            hit_reasons[component].append(
                                f"ep{episode_idx}:{names[0]}<->{names[1]}"
                            )
                total = len(trajectories)
                row = {
                    "fraction": fraction,
                    "lateral_m": lateral,
                    "episodes": total,
                    "valid": valid,
                    **{
                        f"{component}_rate": hits[component] / total
                        for component in COMPONENTS
                    },
                    **{
                        f"{component}_episodes": ",".join(
                            str(index) for index in hit_episodes[component]
                        )
                        for component in COMPONENTS
                    },
                    **{
                        f"{component}_reasons": ";".join(hit_reasons[component])
                        for component in COMPONENTS
                    },
                }
                row["rank"] = (
                    int(valid == total) * 100
                    + 10 * row["arm_rate"]
                    - 10 * row["gripper_rate"]
                    - 10 * row["held_object_rate"]
                )
                rows.append(row)
                print(row)
    finally:
        env.close()
    rows.sort(key=lambda row: row["rank"], reverse=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
