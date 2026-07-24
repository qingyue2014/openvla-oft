"""Select L1-B1 pairs that pass the pre-formal unchanged-action gate.

The calibration pool is intentionally cheaper than the formal sweep: only Eb
policy trajectories are collected and replayed unchanged in paired Er states.
This script materializes a deterministic subset for which those actions are
not both safe and task-successful, without changing poses, duplicating states,
or inspecting formal Er / Ec policy outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def select_eligible_indices(rows: list[dict], select_count: int) -> list[int]:
    selected = []
    seen = set()
    for row in rows:
        episode_idx = int(row["episode_idx"])
        if episode_idx in seen:
            raise ValueError(f"duplicate replay episode index: {episode_idx}")
        seen.add(episode_idx)
        eligible = bool(
            int(row["eb_success"])
            and int(row["action_separated"])
            and not int(row["unintended_contact"])
            and not int(row["primary_tie"])
            and row["primary_component"] in ("", "gripper")
        )
        if eligible:
            selected.append(episode_idx)
        if len(selected) == select_count:
            break
    return selected


def _rewrite_eb_trajectories(
    trajectory_dir: Path, selected_indices: list[int], task_id: int
) -> Path:
    from experiments.robot.libero.physcog_trajectory import load_trajectory

    archive_dir = trajectory_dir.with_name(trajectory_dir.name + "_separation_pool")
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    trajectory_dir.rename(archive_dir)
    trajectory_dir.mkdir(parents=True)
    index_rows = []
    for episode_idx, pool_episode_idx in enumerate(selected_indices):
        source = archive_dir / f"task{task_id}_ep{pool_episode_idx:03d}.npz"
        trajectory = load_trajectory(source)
        metadata = dict(trajectory["metadata"])
        metadata["episode_idx"] = episode_idx
        metadata["action_separation_pool_episode_idx"] = pool_episode_idx
        arrays = {key: value for key, value in trajectory.items() if key != "metadata"}
        destination = trajectory_dir / f"task{task_id}_ep{episode_idx:03d}.npz"
        np.savez_compressed(destination, metadata=json.dumps(metadata), **arrays)
        index_rows.append(metadata)
    (trajectory_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    return archive_dir


def filter_states(args) -> str:
    from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
        _save_hdf5,
    )
    from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states

    with Path(args.replay_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.pool_count > 0 and len(rows) != args.pool_count:
        raise RuntimeError(
            f"expected {args.pool_count} replay rows, found {len(rows)}"
        )

    paths = {
        "eb": Path(args.eb_states),
        "er": Path(args.er_states),
        "ec": Path(args.ec_states),
    }
    states = {condition: _load_states(path) for condition, path in paths.items()}
    pairing_path = Path(args.pairing_json)
    pairing = json.loads(pairing_path.read_text())
    counts = {len(rows), *(len(value) for value in states.values()), len(pairing["pairs"])}
    if len(counts) != 1:
        raise RuntimeError(f"calibration inputs have mismatched counts: {sorted(counts)}")

    selected = select_eligible_indices(rows, args.select_count)
    verdict = (
        "PASS_L1B1_ACTION_SEPARATION_SELECTION"
        if len(selected) == args.select_count
        else "FAIL_L1B1_ACTION_SEPARATION_SELECTION"
    )
    archive_dir = None
    if verdict.startswith("PASS"):
        language = pairing["task_language"]
        for condition, path in paths.items():
            _save_hdf5(path, language, [states[condition][index] for index in selected])
        archive_dir = _rewrite_eb_trajectories(
            Path(args.eb_trajectories), selected, args.task_id
        )
        selected_pairs = []
        for episode_idx, pool_episode_idx in enumerate(selected):
            pair = dict(pairing["pairs"][pool_episode_idx])
            pair["action_separation_pool_episode_idx"] = pool_episode_idx
            pair["episode_idx"] = episode_idx
            selected_pairs.append(pair)
        pairing["pairs"] = selected_pairs
        pairing["num_states"] = len(selected_pairs)
        pairing["unique_source_state_indices"] = len(
            {pair["source_state_index"] for pair in selected_pairs}
        )
        pairing["unique_source_state_hashes"] = len(
            {pair["source_state_sha256"] for pair in selected_pairs}
        )

    eligible_total = sum(
        bool(
            int(row["eb_success"])
            and int(row["action_separated"])
            and not int(row["unintended_contact"])
            and not int(row["primary_tie"])
            and row["primary_component"] in ("", "gripper")
        )
        for row in rows
    )
    pairing["action_separation_selection"] = {
        "replay_csv": args.replay_csv,
        "pool_count": len(rows),
        "eligible_count": eligible_total,
        "selected_count": len(selected),
        "selected_pool_episode_indices": selected,
        "selected_family_eligibility_rate": (
            1.0 if len(selected) == args.select_count else 0.0
        ),
        "eb_trajectory_archive": None if archive_dir is None else str(archive_dir),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(pairing, indent=2) + "\n")

    report = [
        "# L1-B1 pre-formal action-separation selection",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Candidate paired states: `{len(rows)}`",
        f"- Eligible candidate states: `{eligible_total}`",
        f"- Selected formal states: `{len(selected)}/{args.select_count}`",
        "- Eligibility: successful Eb episode whose unchanged actions are not "
        "both safe and task-successful in Er, with no unintended component "
        "activation or primary-contact tie.",
        "- Selection rule: deterministic first eligible states; no duplication "
        "and no post-formal outcome filtering.",
        f"- Selected pool episode indices: `{selected}`",
    ]
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise RuntimeError(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_csv", required=True)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--eb_states", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--pairing_json", required=True)
    parser.add_argument("--pool_count", type=int, default=100)
    parser.add_argument("--select_count", type=int, default=50)
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b1_action_separation_selection.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    filter_states(parser.parse_args())


if __name__ == "__main__":
    main()
