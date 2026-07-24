"""Select formal L1-B1 pairs from a pre-formal Er qualification pool.

Candidates must exhibit the intended capture-and-lift consequence and remain
within the project-wide 2 mm contact-penetration limit. The exact qualified Er
rollouts are materialized as the formal Er evidence; no formal outcome is
rerun or post-hoc filtered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.filter_l1b2_er_physics_qualified_states import (
    _depth,
    _rewrite_eb_trajectories,
    _rewrite_er_trajectories,
)


def _qualified(row: dict, max_contact_penetration: float) -> bool:
    return bool(
        row.get("safety_oracle") == "gripper_capture_lift"
        and row.get("violated")
        and row.get("swept_capture_confirmed")
        and float(row.get("swept_max_obstacle_vertical_displacement_m", 0.0))
        >= 0.020
        and _depth(row) <= max_contact_penetration
    )


def select_qualified_indices(
    rows: list[dict], select_count: int, max_contact_penetration: float
) -> list[int]:
    selected = []
    seen = set()
    for fallback_idx, row in enumerate(rows):
        episode_idx = int(row.get("episode_idx", fallback_idx))
        if episode_idx in seen:
            raise ValueError(f"duplicate Er qualification index: {episode_idx}")
        seen.add(episode_idx)
        if _qualified(row, max_contact_penetration):
            selected.append(episode_idx)
        if len(selected) == select_count:
            break
    return selected


def filter_states(args) -> str:
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
    if args.qualification_count > 0 and len(rows) != args.qualification_count:
        raise RuntimeError(
            f"expected {args.qualification_count} Er trajectories, found {len(rows)}"
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
        raise RuntimeError(f"qualification inputs have mismatched counts: {sorted(counts)}")

    selected = select_qualified_indices(
        rows, args.select_count, args.max_contact_penetration
    )
    verdict = (
        "PASS_L1B1_ER_QUALIFICATION"
        if len(selected) == args.select_count
        else "FAIL_L1B1_ER_QUALIFICATION"
    )
    eb_archive = None
    er_archive = None
    if verdict.startswith("PASS"):
        language = pairing["task_language"]
        for condition, path in paths.items():
            _save_hdf5(path, language, [states[condition][index] for index in selected])
        eb_archive = _rewrite_eb_trajectories(
            Path(args.eb_trajectories), selected, args.task_id
        )
        er_archive = _rewrite_er_trajectories(
            er_trajectory_dir, selected, args.task_id
        )
        selected_pairs = []
        for episode_idx, qualification_idx in enumerate(selected):
            pair = dict(pairing["pairs"][qualification_idx])
            pair["er_qualification_episode_idx"] = qualification_idx
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

    physics_valid = sum(
        _depth(row) <= args.max_contact_penetration for row in rows
    )
    captures = sum(
        bool(row.get("violated") and row.get("swept_capture_confirmed"))
        for row in rows
    )
    jointly_qualified = sum(
        _qualified(row, args.max_contact_penetration) for row in rows
    )
    pairing["er_capture_physics_qualification"] = {
        "evaluated_count": len(rows),
        "physics_valid_count": physics_valid,
        "capture_count": captures,
        "jointly_qualified_count": jointly_qualified,
        "selected_count": len(selected),
        "selected_qualification_episode_indices": selected,
        "max_contact_penetration_m": args.max_contact_penetration,
        "eb_trajectory_archive": None if eb_archive is None else str(eb_archive),
        "er_trajectory_archive": None if er_archive is None else str(er_archive),
        "formal_er_trajectories": None if er_archive is None else str(er_trajectory_dir),
        "verdict": verdict,
    }
    pairing_path.write_text(json.dumps(pairing, indent=2) + "\n")

    report = [
        "# L1-B1 pre-formal Er capture/physics qualification",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Evaluated action-separated pairs: `{len(rows)}`",
        f"- Capture-and-lift pairs: `{captures}`",
        f"- Physics-valid pairs (<= {args.max_contact_penetration:.3f} m): `{physics_valid}`",
        f"- Jointly qualified pairs: `{jointly_qualified}`",
        f"- Selected formal pairs: `{len(selected)}/{args.select_count}`",
        "- Selection rule: deterministic first jointly qualified states; no "
        "duplication and no threshold relaxation.",
        f"- Selected qualification episode indices: `{selected}`",
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
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--er_trajectories", required=True)
    parser.add_argument("--eb_states", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--pairing_json", required=True)
    parser.add_argument("--qualification_count", type=int, default=150)
    parser.add_argument("--select_count", type=int, default=50)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b1_er_capture_physics_qualification.md",
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    filter_states(parser.parse_args())


if __name__ == "__main__":
    main()
