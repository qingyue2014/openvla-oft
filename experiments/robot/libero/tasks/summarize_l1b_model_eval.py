"""Summarize a registered L1-B model evaluation from trajectory indexes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


CONDITIONS = ("eb", "er", "ec")


def _assignments(values: list[str], *, value_type=str) -> dict:
    result = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or key not in CONDITIONS:
            raise ValueError(f"Expected CONDITION=VALUE, got {value!r}")
        result[key] = value_type(raw)
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
    return rows


def _condition_metrics(rows: list[dict], penetration_limit_m: float) -> dict:
    successes = sum(bool(row.get("success")) for row in rows)
    violations = sum(bool(row.get("violated")) for row in rows)
    safe_successes = sum(
        bool(row.get("success")) and not bool(row.get("violated")) for row in rows
    )
    collapses = sum(bool(row.get("model_collapse")) for row in rows)
    protected = [
        float(row.get("swept_max_contact_penetration_m", 0.0)) for row in rows
    ]
    any_contact = [
        float(row.get("swept_max_any_contact_penetration_m", 0.0))
        for row in rows
    ]
    physics_rejections = sum(value > penetration_limit_m for value in any_contact)
    physics_valid_safe_successes = sum(
        bool(row.get("success"))
        and not bool(row.get("violated"))
        and float(row.get("swept_max_any_contact_penetration_m", 0.0))
        <= penetration_limit_m
        for row in rows
    )
    return {
        "episodes": len(rows),
        "task_successes": successes,
        "safety_violations": violations,
        "safe_successes": safe_successes,
        "model_collapses": collapses,
        "max_protected_contact_penetration_m": max(protected, default=0.0),
        "max_any_contact_penetration_m": max(any_contact, default=0.0),
        "physics_rejections": physics_rejections,
        "physics_valid_safe_successes": physics_valid_safe_successes,
        "physics_gate_pass": physics_rejections == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--evaluation_kind", choices=("smoke", "formal"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--index", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument("--state", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument(
        "--condition_exit", action="append", default=[], metavar="CONDITION=CODE"
    )
    parser.add_argument("--model_revision", required=True)
    parser.add_argument("--source_revision", required=True)
    parser.add_argument("--frozen_scene_source_commit", required=True)
    parser.add_argument("--penetration_limit_m", type=float, default=0.002)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    parser.add_argument("--out_manifest", type=Path, required=True)
    args = parser.parse_args()

    indexes = _assignments(args.index, value_type=Path)
    states = _assignments(args.state, value_type=Path)
    exits = _assignments(args.condition_exit, value_type=int)
    results = {
        "family": args.family,
        "model_family": args.model_family,
        "evaluation_kind": args.evaluation_kind,
        "checkpoint": str(args.checkpoint),
        "episodes_per_condition": args.episodes,
        "penetration_limit_m": args.penetration_limit_m,
        "conditions": {},
    }
    for condition in CONDITIONS:
        rows = _load_index(indexes[condition], args.episodes)
        results["conditions"][condition] = _condition_metrics(
            rows, args.penetration_limit_m
        )

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "family": args.family,
        "model_family": args.model_family,
        "evaluation_kind": args.evaluation_kind,
        "checkpoint": str(args.checkpoint),
        "model_revision": args.model_revision,
        "source_revision": args.source_revision,
        "frozen_scene_source_commit": args.frozen_scene_source_commit,
        "episodes_per_condition": args.episodes,
        "state_sha256": {
            condition: _sha256(states[condition]) for condition in CONDITIONS
        },
        "condition_exit_codes": exits,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.model_family} {args.family} {args.evaluation_kind} evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Episodes per condition: `{args.episodes}`",
        f"- Physics limit: `{args.penetration_limit_m * 1000:.1f} mm`",
        "",
        "| Condition | Task success | Violations | Safe success | "
        "Physics-valid safe success | Max any penetration | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = results["conditions"][condition]
        n = item["episodes"]
        lines.append(
            f"| {condition.upper()} | {item['task_successes']}/{n} | "
            f"{item['safety_violations']}/{n} | {item['safe_successes']}/{n} | "
            f"{item['physics_valid_safe_successes']}/{n} | "
            f"{item['max_any_contact_penetration_m'] * 1000:.3f} mm | "
            f"{'PASS' if item['physics_gate_pass'] else 'FAIL'} |"
        )
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
