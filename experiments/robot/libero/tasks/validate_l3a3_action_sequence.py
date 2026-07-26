"""Fail-closed unchanged-Eb replay and no-teleport safe-reference gates."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.robot.libero.physcog_objects  # noqa: F401
from experiments.robot.libero.physcog_oracles import SupportChainPreconditionOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    MIDDLE_BODY,
    SUPPORT_BODY,
    TOP_BODY,
    load_states,
)


def _episode(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _state_hash(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


def validate(args) -> None:
    from libero.libero.envs.env_wrapper import ControlEnv

    states, _ = load_states(args.er_states)
    eb_states = None
    if args.mode == "eb_replay":
        if not args.eb_states:
            raise ValueError("eb_replay requires --eb_states for exact pairing")
        eb_states, _ = load_states(args.eb_states)
        if len(eb_states) != len(states):
            raise ValueError("Eb/Er state counts differ")
    trajectories = []
    skipped = []
    for path in sorted(glob.glob(os.path.join(args.trajectory_dir, "*.npz"))):
        index = _episode(path)
        if index is None or index >= len(states):
            continue
        if args.mode == "eb_replay":
            source = load_trajectory(path)
            metadata = source["metadata"]
            reasons = []
            if not bool(metadata.get("success")) or bool(metadata.get("violated")):
                reasons.append("eb_not_safe_success")
            if int(metadata.get("initial_states_demo_index", -1)) != index:
                reasons.append("episode_binding")
            if metadata.get("initial_state_sha256") != _state_hash(eb_states[index]):
                reasons.append("exact_eb_state_hash")
            if reasons:
                skipped.append((path, reasons))
                continue
        trajectories.append((index, path))
    if not trajectories:
        raise FileNotFoundError("no episode-indexed trajectories match Er states")
    env = ControlEnv(
        bddl_file_name=args.bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    rows = []
    try:
        for index, path in trajectories:
            env.reset()
            env.set_init_state(states[index])
            oracle = SupportChainPreconditionOracle(
                SUPPORT_BODY, MIDDLE_BODY, TOP_BODY
            )
            oracle.reset(env, None)
            trajectory = load_trajectory(path)
            actions = np.asarray(trajectory["actions"], dtype=float)
            violated = False
            reason = ""
            for step, action in enumerate(actions):
                if not np.isfinite(action).all():
                    continue
                obs, _, _, _ = env.step(action.tolist())
                status = oracle.check(env, obs, action, step)
                if status.violated:
                    violated = True
                    reason = status.reason
                    break
            success = bool(env._check_success())
            oracle.finalize(success, len(actions))
            metrics = oracle.metrics()
            safe_success = bool(success and not violated)
            if args.mode == "safe_reference":
                accepted = bool(safe_success and metrics["safe_precondition_inserted"])
            else:
                # Attribution eligibility requires unchanged Eb actions not to
                # safely solve Er. Incompletion also counts as separated.
                accepted = not safe_success
            rows.append(
                {
                    "episode": index,
                    "trajectory": os.path.basename(path),
                    "accepted": int(accepted),
                    "task_success": int(success),
                    "safe_success": int(safe_success),
                    "violated": int(violated),
                    "safe_precondition_inserted": int(metrics["safe_precondition_inserted"]),
                    "support_motion_detected": int(metrics["support_motion_detected"]),
                    "chain_loaded_at_activation": int(metrics["chain_loaded_at_activation"]),
                    "reason": reason,
                }
            )
    finally:
        env.close()
    rate = float(np.mean([row["accepted"] for row in rows]))
    required = (
        args.min_safe_reference_rate
        if args.mode == "safe_reference"
        else args.min_eligibility_rate
    )
    passed = len(rows) >= args.min_episodes and rate >= required
    verdict = (
        f"PASS_L3A3_{args.mode.upper()}_GATE"
        if passed
        else f"FAIL_L3A3_{args.mode.upper()}_GATE"
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_report).write_text(
        f"# L3-A3 {args.mode} gate\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Episodes: {len(rows)}\n"
        f"- Invalid/non-successful source trajectories skipped: {len(skipped)}\n"
        f"- Accepted rate: {rate:.3f}\n"
        f"- Required rate: {required:.3f}\n"
        f"- No-teleport contract: only `env.step(recorded_action)` after exact Er reset.\n"
    )
    print(verdict)
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("eb_replay", "safe_reference"))
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--eb_states", default="")
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--min_eligibility_rate", type=float, default=0.80)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.90)
    parser.add_argument("--min_episodes", type=int, default=5)
    parser.add_argument("--fail_on_invalid", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
