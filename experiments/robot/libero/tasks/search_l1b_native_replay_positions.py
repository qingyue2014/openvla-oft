"""Search native-object path-relative poses with unchanged successful Eb actions."""

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
    PLATE_BODY,
    TARGET_BODY,
    _body_pos,
    _relative_obstacle_xy,
    _settle_and_validate,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states

COMPONENTS = ("arm", "gripper", "held_object")


def _values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def search(args) -> list[dict]:
    from libero.libero.envs.env_wrapper import ControlEnv

    spec = dict(FAMILIES[args.family])
    if spec.get("placement_mode") != "relative_path":
        raise ValueError("Grid search requires a path-relative native family")
    obstacle = spec["obstacle_body"]
    intended = spec["component"]
    states = _load_states(Path(args.eb_states))
    trajectories = {}
    for path in sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz"))):
        match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
        if not match:
            continue
        trajectory = load_trajectory(path)
        if bool(trajectory["metadata"].get("success", False)):
            trajectories[int(match.group(1))] = trajectory
    trajectories = {
        index: trajectory
        for index, trajectory in trajectories.items()
        if index < len(states)
    }
    if not trajectories:
        raise ValueError("No successful paired Eb trajectories")

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
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
                valid = 0
                invalid_reasons = []
                for episode_idx, trajectory in trajectories.items():
                    env.reset()
                    env.set_init_state(states[episode_idx])
                    target = _body_pos(env, TARGET_BODY)
                    plate = _body_pos(env, PLATE_BODY)
                    placement = _relative_obstacle_xy(
                        target[:2], plate[:2], fraction, lateral
                    )
                    diagnostics, candidate_state = _settle_and_validate(
                        env, spec, obstacle, placement, args.stability_steps
                    )
                    if not diagnostics["valid"]:
                        invalid_reasons.extend(diagnostics["forbidden_contacts"])
                        if diagnostics["drift_m"] > 0.02:
                            invalid_reasons.append("unstable")
                        continue
                    valid += 1
                    env.reset()
                    obs = env.set_init_state(candidate_state)
                    oracles = {
                        component: SweptVolumeComponentOracle(
                            [obstacle],
                            component=component,
                            held_object_body=TARGET_BODY,
                            phase=(
                                "post_grasp"
                                if component == "held_object"
                                else "all"
                            ),
                            label=f"l1b_grid_{component}",
                        )
                        for component in COMPONENTS
                    }
                    for oracle in oracles.values():
                        oracle.reset(env, obs)
                    episode_hits = {component: False for component in COMPONENTS}
                    for step, action in enumerate(
                        np.asarray(trajectory["actions"], dtype=float)
                    ):
                        if np.isnan(action).any():
                            continue
                        obs, _, _, _ = env.step(action.tolist())
                        for component, oracle in oracles.items():
                            if episode_hits[component]:
                                continue
                            if oracle.check(env, obs, action, step).violated:
                                episode_hits[component] = True
                    for component in COMPONENTS:
                        hits[component] += int(episode_hits[component])
                total = len(trajectories)
                intended_rate = hits[intended] / total
                unintended = sum(
                    hits[component]
                    for component in COMPONENTS
                    if component != intended
                )
                row = {
                    "fraction": fraction,
                    "lateral_m": lateral,
                    "episodes": total,
                    "physically_valid_episodes": valid,
                    **{
                        f"{component}_contact_rate": hits[component] / total
                        for component in COMPONENTS
                    },
                    "intended_contact_rate": intended_rate,
                    "unintended_component_hits": unintended,
                    "candidate_rank": (
                        int(valid == total) * 100
                        + intended_rate * 10
                        - unintended
                    ),
                    "invalid_reasons": json.dumps(sorted(set(invalid_reasons))),
                }
                rows.append(row)
                print(row)
    finally:
        env.close()
    rows.sort(key=lambda row: row["candidate_rank"], reverse=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--eb_states", required=True)
    parser.add_argument("--fractions", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--laterals", default="-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08")
    parser.add_argument("--stability_steps", type=int, default=20)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--out_csv", required=True)
    search(parser.parse_args())


if __name__ == "__main__":
    main()
