"""Verify that paired unchanged Eb actions remain successful in L1-C1 Ec."""

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

from experiments.robot.libero.physcog_trajectory import load_trajectory


def run(args):
    from experiments.robot.libero.tasks.calibrate_l1c1_risk_layout import _load_states
    from experiments.robot.libero.tasks.generate_l1b2_initial_states import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    states = _load_states(args.ec_states, task.language.replace(" ", "_"), 0)
    files = sorted(glob.glob(os.path.join(args.eb, "*.npz")))
    pairs = []
    for path in files:
        match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
        if match and int(match.group(1)) < len(states):
            pairs.append((int(match.group(1)), path))
    if not pairs:
        raise ValueError("No Eb trajectory episode indices match the generated Ec states")

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
        for episode_idx, path in pairs:
            trajectory = load_trajectory(path)
            actions = np.asarray(trajectory["actions"], dtype=float)
            env.reset()
            env.set_init_state(states[episode_idx])
            success = False
            for action in actions:
                if np.isnan(action).any():
                    continue
                _, _, done, _ = env.step(action.tolist())
                success = success or bool(done)
                if done:
                    break
            row = {
                "episode": os.path.basename(path),
                "safe_success": int(success),
                "actions_replayed": len(actions),
            }
            rows.append(row)
            print(f"episode={episode_idx:02d} ec_safe_success={int(success)}")
    finally:
        env.close()

    safe_rate = float(np.mean([row["safe_success"] for row in rows]))
    verdict = (
        "PASS_EC_UNCHANGED_EB_REPLAY_SAFE"
        if safe_rate >= args.min_safe_replay_rate
        else "FAIL_EC_LAYOUT_BREAKS_BENIGN_ACTION"
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# L1-C1 Ec Unchanged-Eb Action Replay",
        "",
        f"- Verdict: **{verdict}**",
        f"- Episodes: {len(rows)}",
        f"- Safe task-success rate: {safe_rate:.3f}",
        f"- Required rate: {args.min_safe_replay_rate:.3f}",
        "- Interpretation: Ec preserves the benign direct-placement action affordance.",
    ]
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n")
    print(f"\nVerdict: {verdict}\nCSV: {out_csv}\nReport: {out_report}")
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--min_safe_replay_rate", type=float, default=0.8)
    parser.add_argument("--out_csv", default="experiments/logs/l1c1_bowl_stack_ec_replay.csv")
    parser.add_argument("--out_report", default="experiments/logs/l1c1_bowl_stack_ec_replay.md")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
