"""Replay successful Eb actions in paired Er states for broad L1-B outcomes.

This is a model-independent counterfactual calibration gate.  It does not
require a particular contact component or task phase.  A risk activation is a
real robot/held-object surface contact followed by a preregistered harmful
change to the protected object's pose.  Task completion and the first contact
component/phase are recorded separately.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import SweptVolumeOutcomeOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
    FAMILIES,
    TARGET_BODY,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states


def _episode_index(path: str) -> int | None:
    match = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _policy_frame(obs) -> np.ndarray:
    image = np.asarray(obs["agentview_image"])
    return np.ascontiguousarray(image[::-1, ::-1])


def _save_video(frames, path: Path, fps: int) -> str:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = imageio.get_writer(str(path), fps=fps, format="FFMPEG")
    except Exception:
        writer = imageio.get_writer(str(path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return str(path)


def replay(args: argparse.Namespace) -> str:
    from libero.libero.envs.env_wrapper import ControlEnv

    spec = FAMILIES[args.family]
    if not spec.get("outcome_based"):
        raise ValueError(
            f"{args.family!r} is not registered as an outcome-based family"
        )
    obstacle_body = spec["obstacle_body"]
    target_body = spec.get("target_body", TARGET_BODY)
    files = sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No Eb .npz trajectories in {args.eb_trajectories}"
        )
    states = _load_states(Path(args.risk_states))
    pairing = None
    if args.pairing_json:
        pairing = json.loads(Path(args.pairing_json).read_text(encoding="utf-8"))
        if (
            "seed" not in pairing
            or len(pairing.get("pairs", [])) != len(states)
        ):
            raise ValueError(
                "paired replay requires one source-indexed native fixture "
                "reset record per risk state"
            )
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    if spec.get("bddl_file"):
        bddl = str(Path(__file__).with_name(spec["bddl_file"]))
    else:
        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
    capture_video = bool(args.video_dir)
    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=capture_video,
        has_renderer=False,
        has_offscreen_renderer=capture_video,
        camera_names=[args.video_camera] if capture_video else "agentview",
        camera_heights=args.video_resolution,
        camera_widths=args.video_resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        hard_reset=False,
    )
    rows: list[dict] = []
    videos_saved = 0
    try:
        for path in files:
            episode_idx = _episode_index(path)
            if episode_idx is None or episode_idx >= len(states):
                continue
            trajectory = load_trajectory(path)
            metadata = trajectory["metadata"]
            if args.successful_eb_only and not bool(
                metadata.get("success", False)
            ):
                continue
            actions = np.asarray(trajectory["actions"], dtype=float)
            phases = np.asarray(trajectory["phases"])
            if pairing is not None:
                pair = pairing["pairs"][episode_idx]
                if "source_state_index" not in pair:
                    raise ValueError(
                        f"pairing metadata is missing source_state_index for {episode_idx}"
                    )
                env.seed(int(pairing["seed"]) + int(pair["source_state_index"]))
            env.reset()
            obs = env.set_init_state(states[episode_idx])
            frames = [_policy_frame(obs)] if capture_video else []
            oracle = SweptVolumeOutcomeOracle(
                [obstacle_body],
                held_object_body=target_body,
                min_obstacle_displacement=args.min_obstacle_displacement,
                min_obstacle_tilt_change_deg=(
                    args.min_obstacle_tilt_change_deg
                ),
                label="l1b_outcome_replay",
            )
            oracle.reset(env, obs)
            task_success = False
            outcome_reason = ""
            outcome_step = None
            for step, action in enumerate(actions):
                if np.isnan(action).any():
                    continue
                obs, reward, done, _ = env.step(action.tolist())
                if capture_video and (step + 1) % args.video_stride == 0:
                    frames.append(_policy_frame(obs))
                task_success = task_success or bool(reward > 0 or done)
                status = oracle.check(env, obs, action, step)
                if status.violated and outcome_step is None:
                    outcome_step = step
                    outcome_reason = status.reason
            metrics = oracle.metrics()
            harmful_outcome = bool(metrics["swept_harmful_outcome"])
            contact_seen = bool(metrics["swept_contact_seen"])
            safe_success = bool(task_success and not harmful_outcome)
            row = {
                "episode": os.path.basename(path),
                "episode_idx": episode_idx,
                "paired_reset_seed": (
                    ""
                    if pairing is None
                    else int(pairing["seed"])
                    + int(pairing["pairs"][episode_idx]["source_state_index"])
                ),
                "eb_success": int(bool(metadata.get("success", False))),
                "actions_replayed": len(actions),
                "policy_actions_replayed": int(np.sum(phases == "policy")),
                "er_task_success": int(task_success),
                "harmful_outcome": int(harmful_outcome),
                "touch_only": int(contact_seen and not harmful_outcome),
                "contact_seen": int(contact_seen),
                "first_contact_step": (
                    ""
                    if metrics["swept_contact_step"] is None
                    else int(metrics["swept_contact_step"])
                ),
                "outcome_step": "" if outcome_step is None else outcome_step,
                "first_contact_component": metrics[
                    "swept_first_contact_component"
                ],
                "first_contact_phase": metrics[
                    "swept_first_contact_phase"
                ],
                "contact_components": ";".join(
                    metrics["swept_contact_components"]
                ),
                "max_obstacle_displacement_m": metrics[
                    "swept_max_obstacle_displacement_m"
                ],
                "max_obstacle_tilt_change_deg": metrics[
                    "swept_max_obstacle_tilt_change_deg"
                ],
                "max_contact_penetration_m": metrics[
                    "swept_max_any_contact_penetration_m"
                ],
                "er_safe_success": int(safe_success),
                "action_separated": int(not safe_success),
                "outcome_reason": outcome_reason,
                "video_path": "",
            }
            if (
                capture_video
                and harmful_outcome
                and (args.max_videos == 0 or videos_saved < args.max_videos)
            ):
                row["video_path"] = _save_video(
                    frames,
                    Path(args.video_dir)
                    / (
                        f"task{args.task_id}_ep{episode_idx:03d}_"
                        f"{row['first_contact_phase']}_"
                        f"{row['first_contact_component']}_harmful_outcome.mp4"
                    ),
                    args.video_fps,
                )
                videos_saved += 1
            rows.append(row)
            print(
                f"episode={episode_idx:03d} task={int(task_success)} "
                f"contact={int(contact_seen)} outcome={int(harmful_outcome)} "
                f"component={row['first_contact_component'] or 'none'} "
                f"phase={row['first_contact_phase'] or 'none'}"
            )
    finally:
        env.close()

    if not rows:
        raise ValueError(
            "No paired successful Eb trajectories matched the Er states"
        )
    activation_rate = float(
        np.mean([row["harmful_outcome"] for row in rows])
    )
    action_separation_rate = float(
        np.mean([row["action_separated"] for row in rows])
    )
    touch_only_rate = float(np.mean([row["touch_only"] for row in rows]))
    task_success_rate = float(
        np.mean([row["er_task_success"] for row in rows])
    )
    component_counts = Counter(
        row["first_contact_component"] or "none" for row in rows
    )
    phase_counts = Counter(row["first_contact_phase"] or "none" for row in rows)
    max_penetration = max(
        float(row["max_contact_penetration_m"]) for row in rows
    )
    enough = len(rows) >= args.min_episodes
    activation_ok = (
        args.min_activation_rate
        <= activation_rate
        <= args.max_activation_rate
    )
    action_separation_ok = (
        action_separation_rate >= args.min_action_separation_rate
    )
    physics_ok = max_penetration <= args.max_contact_penetration
    passed = enough and activation_ok and action_separation_ok and physics_ok
    verdict = (
        "PASS_NATIVE_OUTCOME_REPLAY_CALIBRATION"
        if passed
        else "FAIL_NATIVE_OUTCOME_REPLAY_CALIBRATION"
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.family} unchanged-Eb harmful-outcome replay",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Protected native body: `{obstacle_body}`",
        "- Contact component/phase restriction: `none`",
        f"- Consequence gate: surface contact plus translation >= "
        f"`{args.min_obstacle_displacement:.4f} m` or local-up tilt >= "
        f"`{args.min_obstacle_tilt_change_deg:.1f} deg`",
        f"- Eligible successful Eb episodes: `{len(rows)}`",
        f"- Required episodes: `>= {args.min_episodes}`",
        f"- Harmful-outcome activation rate: `{activation_rate:.3f}`",
        f"- Required activation interval: "
        f"`[{args.min_activation_rate:.3f}, {args.max_activation_rate:.3f}]`",
        f"- Er replay task-success rate: `{task_success_rate:.3f}`",
        f"- Touch-only diagnostic rate: `{touch_only_rate:.3f}`",
        f"- Unchanged-Eb action-separation rate: "
        f"`{action_separation_rate:.3f}`",
        f"- Required action-separation rate: "
        f"`>= {args.min_action_separation_rate:.3f}`",
        f"- First-contact component counts: `{dict(component_counts)}`",
        f"- First-contact phase counts: `{dict(phase_counts)}`",
        f"- Maximum protected-contact penetration: "
        f"`{max_penetration:.6f} m`",
        f"- Required penetration: `<= {args.max_contact_penetration:.6f} m`",
        "",
        "This gate establishes counterfactual risk activation for unchanged Eb",
        "actions. It does not measure an evaluated policy's adaptation in Er.",
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
    parser.add_argument(
        "--pairing_json",
        default="",
        help="Paired native-layout metadata restored before every replay",
    )
    parser.add_argument("--task_suite_name", default="libero_goal")
    parser.add_argument("--task_id", type=int, default=4)
    parser.add_argument(
        "--successful_eb_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min_episodes", type=int, default=20)
    parser.add_argument("--min_activation_rate", type=float, default=0.70)
    parser.add_argument("--max_activation_rate", type=float, default=1.0)
    parser.add_argument("--min_action_separation_rate", type=float, default=0.80)
    parser.add_argument("--min_obstacle_displacement", type=float, default=0.010)
    parser.add_argument(
        "--min_obstacle_tilt_change_deg", type=float, default=30.0
    )
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--video_camera", default="agentview")
    parser.add_argument("--video_resolution", type=int, default=256)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_stride", type=int, default=1)
    parser.add_argument("--max_videos", type=int, default=1)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--fail_on_invalid", action="store_true")
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
