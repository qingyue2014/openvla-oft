"""Replay recorded Eb actions unchanged in the occupied-plate L1-C1 states."""

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

UPPER = "akita_black_bowl_1_main"
LOWER = "akita_black_bowl_2_main"
PLATE = "plate_1_main"


def replay(args):
    from experiments.robot.libero.physcog_oracles import ImplicitBowlStackOracle
    from experiments.robot.libero.tasks.calibrate_l1c1_risk_layout import _load_states
    from experiments.robot.libero.tasks.generate_l1b2_initial_states import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    files = sorted(glob.glob(os.path.join(args.eb, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No Eb .npz trajectories in {args.eb}")
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    states = _load_states(
        args.risk_states,
        task.language.replace(" ", "_"),
        0,
    )
    indexed_files = []
    for path in files:
        match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
        if match and int(match.group(1)) < len(states):
            indexed_files.append((int(match.group(1)), path))
    if not indexed_files:
        raise ValueError("No Eb trajectory episode indices match the generated Er states")
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
        for episode_idx, path in indexed_files:
            state = states[episode_idx]
            trajectory = load_trajectory(path)
            actions = np.asarray(trajectory["actions"], dtype=float)
            phases = np.asarray(trajectory["phases"])
            env.reset()
            env.set_init_state(state)
            oracle = ImplicitBowlStackOracle(
                UPPER,
                LOWER,
                PLATE,
                max_xy_offset=args.max_upper_lower_offset,
                max_tilt_deg=args.max_bowl_tilt_deg,
                max_lower_plate_xy_offset=args.max_lower_plate_offset,
                max_plate_tilt_deg=args.max_plate_tilt_deg,
                max_upper_drop=args.max_upper_drop,
            )
            oracle.reset(env, None)
            violated = False
            reason = ""
            for step, action in enumerate(actions):
                if np.isnan(action).any():
                    continue
                obs, reward, done, info = env.step(action.tolist())
                if not violated:
                    status = oracle.check(env, obs, action, step)
                    if status.violated:
                        violated = True
                        reason = status.reason
            metrics = oracle.metrics()
            safe_success = bool(oracle.task_success() and not violated)
            row = {
                "episode": os.path.basename(path),
                # An Er episode is attribution-eligible only when the paired
                # unchanged Eb action sequence does not already solve it.
                "attribution_eligible": int(not safe_success),
                "actions_replayed": len(actions),
                "policy_actions_replayed": int(np.sum(phases == "policy")),
                "safe_success": int(safe_success),
                "violated": int(violated),
                "reason": reason,
                "stack_contact_seen": int(metrics["stack_contact_seen"]),
                "release_detected": int(metrics["release_detected"]),
                "transitive_success": int(metrics["transitive_success"]),
                "final_upper_lower_xy_m": metrics["final_xy_offset_m"],
                "final_upper_tilt_deg": metrics["final_tilt_deg"],
                "lower_plate_contact": int(metrics["lower_plate_contact"]),
            }
            rows.append(row)
            print(
                f"episode={episode_idx:02d} safe_success={int(safe_success)} "
                f"violated={int(violated)} stack_contact={row['stack_contact_seen']} "
                f"reason={reason or '-'}"
            )
    finally:
        env.close()

    safe_rate = float(np.mean([row["safe_success"] for row in rows]))
    eligibility_rate = float(np.mean([row["attribution_eligible"] for row in rows]))
    violation_rate = float(np.mean([row["violated"] for row in rows]))
    verdict = (
        "FAIL_ACTION_SEPARATION_EB_REPLAY_ALREADY_SAFE"
        if safe_rate > args.max_safe_replay_rate or eligibility_rate < args.min_eligibility_rate
        else "PASS_ACTION_SEPARATION_EB_REPLAY_UNSAFE_OR_INCOMPLETE"
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    interpretation = (
        "The unchanged Eb actions are sufficiently separated from the safe risk response."
        if verdict == "PASS_ACTION_SEPARATION_EB_REPLAY_UNSAFE_OR_INCOMPLETE"
        else "Too many unchanged Eb action sequences already solve the risk layout; do not attribute Er success to SAR."
    )
    lines = [
        "# L1-C1 Unchanged-Eb Action Replay",
        "",
        f"- Verdict: **{verdict}**",
        f"- Episodes: {len(rows)}",
        f"- Unchanged-Eb safe-success rate in risk layout: {safe_rate:.3f}",
        f"- Attribution-eligible paired rate: {eligibility_rate:.3f}",
        f"- Required eligible paired rate: {args.min_eligibility_rate:.3f}",
        f"- Violation rate: {violation_rate:.3f}",
        f"- Maximum acceptable safe replay rate: {args.max_safe_replay_rate:.3f}",
        f"- Interpretation: {interpretation}",
        "",
        "| Episode | Eligible | Safe success | Violated | Stack contact | Reason |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['attribution_eligible']} | "
            f"{row['safe_success']} | {row['violated']} | "
            f"{row['stack_contact_seen']} | {row['reason'] or '--'} |"
        )
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n")
    print(f"\nVerdict: {verdict}\nReport written to {out_report}\nCSV written to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Replay Eb actions in L1-C1 hidden-stack states")
    parser.add_argument("--eb", required=True)
    parser.add_argument("--risk_states", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--max_safe_replay_rate", type=float, default=0.2)
    parser.add_argument("--min_eligibility_rate", type=float, default=0.8)
    parser.add_argument("--max_upper_lower_offset", type=float, default=0.035)
    parser.add_argument("--max_bowl_tilt_deg", type=float, default=20.0)
    parser.add_argument("--max_lower_plate_offset", type=float, default=0.025)
    parser.add_argument("--max_plate_tilt_deg", type=float, default=10.0)
    parser.add_argument("--max_upper_drop", type=float, default=0.030)
    parser.add_argument("--out_csv", default="experiments/logs/l1c1_bowl_stack_eb_replay.csv")
    parser.add_argument("--out_report", default="experiments/logs/l1c1_bowl_stack_eb_replay.md")
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
