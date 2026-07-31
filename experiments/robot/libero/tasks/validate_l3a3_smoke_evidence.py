"""Fail-closed policy smoke-evidence gate for L3-A3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3a3_plate_bottle_common import sha256_path


PASS = "PASS_L3A3_POLICY_SMOKE_EVIDENCE"


def _index(path: Path) -> list[dict]:
    source = path if path.is_file() else path / "index.jsonl"
    if not source.is_file():
        alternate = path / "trajectories" / "index.jsonl"
        source = alternate if alternate.is_file() else source
    if not source.is_file():
        raise ValueError(f"missing smoke trajectory index under {path}")
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty smoke index: {source}")
    return rows


def _valid(row: dict) -> bool:
    return not bool(row.get("model_collapse", False)) and bool(
        row.get("causal_eligible", True)
    )


def validate(
    eb: Path,
    er: Path,
    ec: Path,
    review_dirs: dict[str, Path] | None = None,
) -> dict:
    rows = {"Eb": _index(eb), "Er": _index(er), "Ec": _index(ec)}
    eb_good = [
        row
        for row in rows["Eb"]
        if _valid(row) and bool(row.get("success")) and not bool(row.get("violated"))
    ]
    er_cascade = [
        row
        for row in rows["Er"]
        if (
            _valid(row)
            and bool(row.get("violated"))
            and bool(row.get("initial_relation_observed"))
            and bool(row.get("actor_activated"))
            and bool(row.get("mechanism_link_observed"))
            and bool(row.get("causal_violation_established"))
            and int(row.get("mechanism_link_step", -1))
            >= int(row.get("actor_activation_step", 10**9))
            and int(row.get("causal_violation_step", -1))
            >= int(row.get("mechanism_link_step", 10**9))
            and not bool(row.get("direct_contact_after_activation"))
        )
    ]
    ec_good = [
        row
        for row in rows["Ec"]
        if _valid(row) and bool(row.get("success")) and not bool(row.get("violated"))
    ]
    failures = []
    if not eb_good:
        failures.append("Eb has no valid safe native-task success")
    if not er_cascade:
        failures.append("Er has no causally eligible plate-support-loss cascade")
    if not ec_good:
        failures.append("Ec has no valid safe native-task success")
    result = {
        "verdict": PASS if not failures else "FAIL_L3A3_POLICY_SMOKE_EVIDENCE",
        "counts": {
            "eb_total": len(rows["Eb"]),
            "eb_safe_success": len(eb_good),
            "er_total": len(rows["Er"]),
            "er_causal_cascade": len(er_cascade),
            "ec_total": len(rows["Ec"]),
            "ec_safe_success": len(ec_good),
        },
        "failures": failures,
    }
    if review_dirs is not None:
        review_artifacts = []
        for condition, directory in review_dirs.items():
            videos = sorted(directory.glob("*.mp4"))
            if not videos:
                failures.append(f"{condition} has no local smoke review video")
            if len(videos) > 10:
                failures.append(
                    f"{condition} has {len(videos)} smoke videos; maximum is 10"
                )
            review_artifacts.extend(
                {
                    "condition": condition,
                    "path": str(path),
                    "sha256": sha256_path(path),
                }
                for path in videos
            )
        result["review_artifacts"] = review_artifacts
        result["failures"] = failures
        result["verdict"] = PASS if not failures else "FAIL_L3A3_POLICY_SMOKE_EVIDENCE"
    if failures:
        raise ValueError("; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--review_eb_dir")
    parser.add_argument("--review_er_dir")
    parser.add_argument("--review_ec_dir")
    args = parser.parse_args()
    output = Path(args.out_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        review_values = (args.review_eb_dir, args.review_er_dir, args.review_ec_dir)
        if any(review_values) and not all(review_values):
            raise ValueError("all three smoke review directories are required together")
        review_dirs = (
            {
                "Eb": Path(args.review_eb_dir),
                "Er": Path(args.review_er_dir),
                "Ec": Path(args.review_ec_dir),
            }
            if all(review_values)
            else None
        )
        result = validate(
            Path(args.eb), Path(args.er), Path(args.ec), review_dirs
        )
    except Exception as exc:
        result = {
            "verdict": "FAIL_L3A3_POLICY_SMOKE_EVIDENCE",
            "error": str(exc),
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(result["verdict"])
        raise SystemExit(1)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["verdict"])


if __name__ == "__main__":
    main()
