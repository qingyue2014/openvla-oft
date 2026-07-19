"""Replay successful native Eb actions unchanged in paired L1-B4/B5/B6 Er states.

The replay is a geometry/component calibration gate, not a model evaluation.
It restores the paired Er state, executes the exact action sequence recorded in
the same-index Eb rollout, and monitors arm, gripper, and held-object contacts
independently.  Formal calibration requires 70--95% intended-component
activation, at most 10% unintended-component activation, and at least 90%
component purity among observed component contacts.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    TARGET_BODY,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states


COMPONENTS = ("arm", "gripper", "held_object")


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def replay(args) -> str:
    from libero.libero.envs.env_wrapper import ControlEnv

    spec = FAMILIES[args.family]
    if not spec.get("native_assets_only") or spec.get("bddl_file") is not None:
        raise ValueError(f"{args.family} is not an all-native L1-B alternative")
    obstacle_body = spec["obstacle_body"]
    intended_component = spec["component"]

    files = sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No Eb .npz trajectories in {args.eb_trajectories}"
        )
    states = _load_states(Path(args.risk_states))
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
        for path in files:
            episode_idx = _episode_index(path)
            if episode_idx is None or episode_idx >= len(states):
                continue
            trajectory = load_trajectory(path)
            metadata = trajectory["metadata"]
            if args.successful_eb_only and not bool(metadata.get("success", False)):
                continue
            actions = np.asarray(trajectory["actions"], dtype=float)
            phases = np.asarray(trajectory["phases"])
            env.reset()
            obs = env.set_init_state(states[episode_idx])
            oracles = {
                component: SweptVolumeComponentOracle(
                    [obstacle_body],
                    component=component,
                    held_object_body=TARGET_BODY,
                    phase="post_grasp" if component == "held_object" else "all",
                    label=f"l1b_replay_{component}",
                )
                for component in COMPONENTS
            }
            for oracle in oracles.values():
                oracle.reset(env, obs)
            hits = {component: False for component in COMPONENTS}
            reasons = {component: "" for component in COMPONENTS}
            task_success = False
            for step, action in enumerate(actions):
                if np.isnan(action).any():
                    continue
                obs, reward, done, info = env.step(action.tolist())
                task_success = task_success or bool(reward > 0 or done)
                for component, oracle in oracles.items():
                    if hits[component]:
                        continue
                    status = oracle.check(env, obs, action, step)
                    if status.violated:
                        hits[component] = True
                        reasons[component] = status.reason
            unintended = any(
                hits[component]
                for component in COMPONENTS
                if component != intended_component
            )
            row = {
                "episode": os.path.basename(path),
                "episode_idx": episode_idx,
                "eb_success": int(bool(metadata.get("success", False))),
                "actions_replayed": len(actions),
                "policy_actions_replayed": int(np.sum(phases == "policy")),
                "er_task_success": int(task_success),
                **{f"{component}_contact": int(hits[component]) for component in COMPONENTS},
                "intended_contact": int(hits[intended_component]),
                "unintended_contact": int(unintended),
                **{f"{component}_reason": reasons[component] for component in COMPONENTS},
            }
            rows.append(row)
            print(
                f"episode={episode_idx:03d} intended={row['intended_contact']} "
                f"unintended={row['unintended_contact']} "
                + " ".join(
                    f"{component}={int(hits[component])}"
                    for component in COMPONENTS
                )
            )
    finally:
        env.close()

    if not rows:
        raise ValueError("No paired successful Eb trajectories matched the Er states")
    intended_rate = float(np.mean([row["intended_contact"] for row in rows]))
    unintended_rate = float(np.mean([row["unintended_contact"] for row in rows]))
    component_hits = sum(
        row[f"{component}_contact"]
        for row in rows
        for component in COMPONENTS
    )
    intended_hits = sum(row["intended_contact"] for row in rows)
    purity = intended_hits / component_hits if component_hits else 0.0
    enough = len(rows) >= args.min_episodes
    activation_ok = args.min_activation_rate <= intended_rate <= args.max_activation_rate
    isolation_ok = (
        unintended_rate <= args.max_unintended_rate
        and purity >= args.min_component_purity
    )
    passed = enough and activation_ok and isolation_ok
    verdict = "PASS_NATIVE_REPLAY_CALIBRATION" if passed else "FAIL_NATIVE_REPLAY_CALIBRATION"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.family} unchanged-Eb native replay",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Intended component: `{intended_component}`",
        f"- Protected native body: `{obstacle_body}`",
        f"- Eligible successful Eb episodes: `{len(rows)}`",
        f"- Required episodes: `>= {args.min_episodes}`",
        f"- Intended activation rate: `{intended_rate:.3f}`",
        f"- Required activation interval: `[{args.min_activation_rate:.3f}, {args.max_activation_rate:.3f}]`",
        f"- Any unintended-component contact rate: `{unintended_rate:.3f}`",
        f"- Maximum unintended rate: `{args.max_unintended_rate:.3f}`",
        f"- Intended-component purity among component hits: `{purity:.3f}`",
        f"- Required component purity: `>= {args.min_component_purity:.3f}`",
        "",
        "This gate replays unchanged successful Eb actions in paired Er states; it",
        "does not measure obstacle-aware policy adaptation in Er.",
    ]
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--risk_states", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--successful_eb_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_episodes", type=int, default=20)
    parser.add_argument("--min_activation_rate", type=float, default=0.70)
    parser.add_argument("--max_activation_rate", type=float, default=0.95)
    parser.add_argument("--max_unintended_rate", type=float, default=0.10)
    parser.add_argument("--min_component_purity", type=float, default=0.90)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--fail_on_invalid", action="store_true")
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
