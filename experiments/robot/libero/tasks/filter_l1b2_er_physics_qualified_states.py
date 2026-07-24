"""Select L1-B2 pairs whose observed Er rollout passes the global physics gate.

Trajectory-conditioned bottle placement first proves that the unchanged Eb
actions produce an isolated held-object knockdown.  This second, independent
qualification rejects pairs where the actual Er policy rollout produces any
contact deeper than the project-wide penetration limit.  It never relaxes the
limit, duplicates a state, or changes anything except selecting and reindexing
already paired Eb / Er / Ec states.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _depth(row: dict) -> float:
    return float(
        row.get(
            "swept_max_any_contact_penetration_m",
            row.get("swept_max_contact_penetration_m", 0.0),
        )
        or 0.0
    )


def select_qualified_indices(
    rows: list[dict], select_count: int, max_contact_penetration: float
) -> list[int]:
    """Return a deterministic, unique prefix of globally physics-valid rows."""
    qualified = []
    seen = set()
    for fallback_idx, row in enumerate(rows):
        episode_idx = int(row.get("episode_idx", fallback_idx))
        if episode_idx in seen:
            raise ValueError(f"duplicate Er episode index: {episode_idx}")
        seen.add(episode_idx)
        if _depth(row) <= max_contact_penetration:
            qualified.append(episode_idx)
        if len(qualified) == select_count:
            break
    return qualified


def _rewrite_eb_trajectories(
    trajectory_dir: Path, selected_indices: list[int], task_id: int
) -> Path:
    from experiments.robot.libero.physcog_trajectory import load_trajectory

    archive_dir = trajectory_dir.with_name(
        trajectory_dir.name + "_trajectory_qualified"
    )
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    trajectory_dir.rename(archive_dir)
    trajectory_dir.mkdir(parents=True)
    index_rows = []
    for episode_idx, source_episode_idx in enumerate(selected_indices):
        source = archive_dir / f"task{task_id}_ep{source_episode_idx:03d}.npz"
        trajectory = load_trajectory(source)
        metadata = dict(trajectory["metadata"])
        metadata["episode_idx"] = episode_idx
        metadata["er_physics_qualification_episode_idx"] = source_episode_idx
        arrays = {key: value for key, value in trajectory.items() if key != "metadata"}
        destination = trajectory_dir / f"task{task_id}_ep{episode_idx:03d}.npz"
        np.savez_compressed(
            destination, metadata=json.dumps(metadata), **arrays
        )
        index_rows.append(metadata)
    (trajectory_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    return archive_dir


def _rewrite_er_trajectories(
    trajectory_dir: Path, selected_indices: list[int], task_id: int
) -> Path:
    """Materialize the exact qualified Er rollouts as the formal trajectory set."""
    from experiments.robot.libero.physcog_trajectory import load_trajectory

    archive_dir = trajectory_dir.with_name(
        trajectory_dir.name + "_physics_qualification"
    )
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    trajectory_dir.rename(archive_dir)
    trajectory_dir.mkdir(parents=True)
    index_rows = []
    for episode_idx, qualification_episode_idx in enumerate(selected_indices):
        source = archive_dir / f"task{task_id}_ep{qualification_episode_idx:03d}.npz"
        trajectory = load_trajectory(source)
        metadata = dict(trajectory["metadata"])
        metadata["episode_idx"] = episode_idx
        metadata["er_physics_qualification_episode_idx"] = qualification_episode_idx
        arrays = {key: value for key, value in trajectory.items() if key != "metadata"}
        destination = trajectory_dir / f"task{task_id}_ep{episode_idx:03d}.npz"
        np.savez_compressed(destination, metadata=json.dumps(metadata), **arrays)
        index_rows.append(metadata)
    (trajectory_dir / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows)
    )
    return archive_dir


def filter_states(args) -> str:
    # Keep the pure selection helper importable in lightweight local test
    # environments that do not have LIBERO / torch installed.
    from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
        _save_hdf5,
    )
    from experiments.robot.libero.tasks.validate_l1b_swept_states import _load_states

    er_trajectory_dir = Path(args.er_trajectories)
    rows = [
        json.loads(line)
        for line in (er_trajectory_dir / "index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = args.qualification_count
    if expected > 0 and len(rows) != expected:
        raise RuntimeError(
            f"expected {expected} Er qualification trajectories, found {len(rows)}"
        )

    eb_states = _load_states(Path(args.eb_states))
    er_states = _load_states(Path(args.er_states))
    ec_states = _load_states(Path(args.ec_states))
    pairing_path = Path(args.pairing_json)
    pairing = json.loads(pairing_path.read_text())
    counts = {len(rows), len(eb_states), len(er_states), len(ec_states), len(pairing["pairs"])}
    if len(counts) != 1:
        raise RuntimeError(f"qualification inputs have mismatched counts: {sorted(counts)}")

    selected = select_qualified_indices(
        rows, args.select_count, args.max_contact_penetration
    )
    if len(selected) != args.select_count:
        verdict = "FAIL_ER_POLICY_PHYSICS_QUALIFICATION"
        archive_dir = None
        er_archive_dir = None
    else:
        verdict = "PASS_ER_POLICY_PHYSICS_QUALIFICATION"
        language = pairing.get("task_language", "put the cream cheese in the bowl")
        _save_hdf5(Path(args.eb_states), language, [eb_states[index] for index in selected])
        _save_hdf5(Path(args.er_states), language, [er_states[index] for index in selected])
        _save_hdf5(Path(args.ec_states), language, [ec_states[index] for index in selected])
        archive_dir = _rewrite_eb_trajectories(
            Path(args.eb_trajectories), selected, args.task_id
        )
        er_archive_dir = _rewrite_er_trajectories(
            er_trajectory_dir, selected, args.task_id
        )

        selected_pairs = []
        for episode_idx, qualification_episode_idx in enumerate(selected):
            pair = dict(pairing["pairs"][qualification_episode_idx])
            pair["er_physics_qualification_episode_idx"] = qualification_episode_idx
            pair["episode_idx"] = episode_idx
            selected_pairs.append(pair)
        pairing["pairs"] = selected_pairs
        pairing["num_states"] = len(selected_pairs)
        pairing["unique_source_state_indices"] = len(
            {pair["source_state_index"] for pair in selected_pairs}
        )

    qualified_total = sum(
        _depth(row) <= args.max_contact_penetration for row in rows
    )
    rejected = [
        {
            "episode_idx": int(row.get("episode_idx", index)),
            "max_any_contact_penetration_m": _depth(row),
        }
        for index, row in enumerate(rows)
        if _depth(row) > args.max_contact_penetration
    ]
    pairing["er_policy_physics_qualification"] = {
        "source": str(er_trajectory_dir),
        "evaluated_count": len(rows),
        "globally_physics_valid_count": qualified_total,
        "selected_count": len(selected),
        "selected_qualification_episode_indices": selected,
        "max_contact_penetration_m": args.max_contact_penetration,
        "rejected": rejected,
        "eb_trajectory_archive": None if archive_dir is None else str(archive_dir),
        "er_trajectory_archive": None if er_archive_dir is None else str(er_archive_dir),
        "formal_er_trajectories": None if er_archive_dir is None else str(er_trajectory_dir),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(pairing, indent=2) + "\n")

    report_lines = [
        "# L1-B2 Er policy physics qualification",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Evaluated unique paired states: `{len(rows)}`",
        f"- Globally physics-valid states: `{qualified_total}`",
        f"- Selected formal states: `{len(selected)}/{args.select_count}`",
        f"- Maximum allowed contact penetration: `{args.max_contact_penetration:.6f} m`",
        "- Selection rule: deterministic first valid states; no duplication and no threshold relaxation.",
        f"- Selected qualification episode indices: `{selected}`",
        f"- Rejected states: `{len(rejected)}`",
        *(
            f"  - `ep{row['episode_idx']:03d}: {row['max_any_contact_penetration_m']:.6f} m`"
            for row in rejected
        ),
    ]
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))

    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise RuntimeError(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--er_trajectories", required=True)
    parser.add_argument(
        "--eb_states",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_eb_states.hdf5",
    )
    parser.add_argument(
        "--er_states",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_er_states.hdf5",
    )
    parser.add_argument(
        "--ec_states",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_ec_states.hdf5",
    )
    parser.add_argument(
        "--pairing_json",
        default="experiments/robot/libero/tasks/l1b2_native_held_object_pairing.json",
    )
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--qualification_count", type=int, default=100)
    parser.add_argument("--select_count", type=int, default=50)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b2_er_physics_qualification.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    filter_states(parser.parse_args())


if __name__ == "__main__":
    main()
