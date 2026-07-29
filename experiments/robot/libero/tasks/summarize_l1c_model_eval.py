"""Summarize a registered L1-C model evaluation from trajectory indexes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


CONDITIONS = ("eb", "er", "ec")
STANDARD_FIELDS = {
    "episode_idx",
    "success",
    "violated",
    "model_collapse",
}


def _assignments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or key not in CONDITIONS:
            raise ValueError(f"Expected CONDITION=PATH, got {value!r}")
        result[key] = Path(raw)
    missing = set(CONDITIONS) - set(result)
    if missing:
        raise ValueError(f"Missing condition assignments: {sorted(missing)}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_index(path: Path, expected_episodes: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_episodes:
        raise ValueError(
            f"{path}: expected {expected_episodes} episodes, found {len(rows)}"
        )
    episode_indices = [int(row["episode_idx"]) for row in rows]
    if episode_indices != list(range(expected_episodes)):
        raise ValueError(
            f"{path}: episode indexes are not the exact ordered range "
            f"0..{expected_episodes - 1}"
        )
    return rows


def _condition_metrics(rows: list[dict]) -> dict:
    episodes = len(rows)
    successes = sum(bool(row.get("success")) for row in rows)
    violations = sum(bool(row.get("violated")) for row in rows)
    safe_successes = sum(
        bool(row.get("success")) and not bool(row.get("violated"))
        for row in rows
    )
    collapses = sum(bool(row.get("model_collapse")) for row in rows)
    metric_maxima: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if (
                key in STANDARD_FIELDS
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value is None
            ):
                continue
            if any(
                marker in key
                for marker in (
                    "displacement",
                    "tilt",
                    "offset",
                    "clearance",
                    "drift",
                    "drop",
                )
            ):
                metric_maxima[key] = max(metric_maxima.get(key, float("-inf")), float(value))
    return {
        "episodes": episodes,
        "task_successes": successes,
        "task_success_rate": successes / episodes,
        "safety_violations": violations,
        "safety_violation_rate": violations / episodes,
        "safe_successes": safe_successes,
        "safe_success_rate": safe_successes / episodes,
        "model_collapses": collapses,
        "oracle_metric_maxima": dict(sorted(metric_maxima.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("l1c1", "l1c2", "l1c3"), required=True)
    parser.add_argument(
        "--model_family", choices=("pi05", "cosmos", "gr00t_n16"), required=True
    )
    parser.add_argument("--evaluation_kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--index", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument("--state", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument("--model_revision", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--calibration_report", type=Path, required=True)
    parser.add_argument("--safe_reference_report", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    parser.add_argument("--out_manifest", type=Path, required=True)
    args = parser.parse_args()

    indexes = _assignments(args.index)
    states = _assignments(args.state)
    results = {
        "scenario": args.scenario,
        "model_family": args.model_family,
        "evaluation_kind": args.evaluation_kind,
        "checkpoint": str(args.checkpoint),
        "episodes_per_condition": args.episodes,
        "conditions": {
            condition: _condition_metrics(
                _load_index(indexes[condition], args.episodes)
            )
            for condition in CONDITIONS
        },
    }
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenario": args.scenario,
        "model_family": args.model_family,
        "evaluation_kind": args.evaluation_kind,
        "checkpoint": str(args.checkpoint),
        "model_revision": args.model_revision,
        "source_revision": args.source_revision,
        "episodes_per_condition": args.episodes,
        "state_sha256": {
            condition: _sha256(states[condition]) for condition in CONDITIONS
        },
        "index_sha256": {
            condition: _sha256(indexes[condition]) for condition in CONDITIONS
        },
        "calibration_report": str(args.calibration_report),
        "calibration_report_sha256": _sha256(args.calibration_report),
        "safe_reference_report": str(args.safe_reference_report),
        "safe_reference_report_sha256": _sha256(args.safe_reference_report),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.model_family} {args.scenario.upper()} {args.evaluation_kind} evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Episodes per condition: `{args.episodes}`",
        "- Scene calibration: `PASS`",
        "- Dynamic safe reference: `PASS`",
        "",
        "| Condition | Task success | Violations | Safe success | Collapse |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = results["conditions"][condition]
        n = item["episodes"]
        lines.append(
            f"| {condition.upper()} | {item['task_successes']}/{n} | "
            f"{item['safety_violations']}/{n} | {item['safe_successes']}/{n} | "
            f"{item['model_collapses']}/{n} |"
        )
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
