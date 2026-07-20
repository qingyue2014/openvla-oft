#!/usr/bin/env python3
"""Summarize isolated L3-A1 Er corridor candidates from trajectory indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _index_path(path: Path) -> Path:
    for candidate in (path / "index.jsonl", path / "trajectories" / "index.jsonl"):
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing index.jsonl under {path}")


def _load(path: Path, expected: int) -> list[dict[str, Any]]:
    index = _index_path(path)
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"{index}: expected {expected} episodes, found {len(rows)}")
    if sorted(row.get("episode_idx") for row in rows) != list(range(expected)):
        raise ValueError(f"{index}: episode_idx values are not 0..{expected - 1}")
    return rows


def _qualifies(row: dict[str, Any]) -> bool:
    activation = row.get("support_activation_step")
    violation = row.get("violation_step")
    valid_execution = row.get(
        "valid_execution", row.get("model_collapse") is False
    ) is True
    return all((
        row.get("success") is True,
        valid_execution,
        row.get("support_activated") is True,
        row.get("causal_eligible") is True,
        row.get("violated") is True,
        row.get("direct_contact_detected") is False,
        isinstance(activation, int),
        isinstance(violation, int),
        violation > activation if isinstance(activation, int) and isinstance(violation, int) else False,
    ))


def _contact(row: dict[str, Any]) -> str:
    if row.get("direct_contact_detected") is not True:
        return "—"
    geom1 = row.get("direct_robot_contact_geom1", "")
    geom2 = row.get("direct_robot_contact_geom2", "")
    id1 = row.get("direct_robot_contact_geom1_id")
    id2 = row.get("direct_robot_contact_geom2_id")
    distance = row.get("direct_robot_contact_dist_m")
    distance_mm = f"{float(distance) * 1000:+.3f} mm" if isinstance(distance, (int, float)) else "n/a"
    return f"{geom1}[{id1}] ↔ {geom2}[{id2}], dist={distance_mm}"


def _contact_geoms(rows: list[dict[str, Any]]) -> str:
    pairs = {
        f"{row.get('direct_robot_contact_geom1', '')}"
        f"[{row.get('direct_robot_contact_geom1_id')}] ↔ "
        f"{row.get('direct_robot_contact_geom2', '')}"
        f"[{row.get('direct_robot_contact_geom2_id')}]"
        for row in rows if row.get("direct_contact_detected") is True
    }
    return "<br>".join(sorted(pairs)) if pairs else "—"


def write_report(candidates: list[tuple[str, Path]], expected: int, report: Path) -> None:
    loaded = [(dx, path, _load(path, expected)) for dx, path in candidates]
    lines = [
        "# L3-A1 policy-corridor sweep",
        "",
        "Fixed geometry: dy=-0.184, lean=-20 deg, direction=35 deg. "
        "All conditions use the unchanged formal generation and direct-contact gates.",
        "",
        "| lean dx | episodes | qualifying | direct | exact contact geom pair(s) |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for dx, _, rows in loaded:
        lines.append(
            f"| {dx} | {len(rows)} | {sum(map(_qualifies, rows))} | "
            f"{sum(row.get('direct_contact_detected') is True for row in rows)} | "
            f"{_contact_geoms(rows)} |"
        )
    lines.extend(("", "## Episode audit", ""))
    for dx, path, rows in loaded:
        lines.extend((
            f"### dx={dx}",
            "",
            f"Trajectory index: `{_index_path(path)}`",
            "",
            "| episode | qualifying | direct | activation/direct step | exact robot contact |",
            "| ---: | --- | --- | --- | --- |",
        ))
        for row in rows:
            activation = row.get("support_activation_step")
            direct_step = row.get("direct_contact_step")
            lines.append(
                f"| {row['episode_idx']} | {'yes' if _qualifies(row) else 'no'} | "
                f"{'yes' if row.get('direct_contact_detected') is True else 'no'} | "
                f"{activation}/{direct_step} | {_contact(row)} |"
            )
        lines.append("")
    lines.append("- Verdict: **PASS_L3A1_CORRIDOR_SWEEP_COMPLETED**")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="Candidate encoded as dx=<value>|<rollout-or-trajectory-directory>.",
    )
    parser.add_argument("--expected_episodes", type=int, default=5)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    candidates = []
    for spec in args.candidate:
        label, separator, raw_path = spec.partition("|")
        if not separator or not label.startswith("dx=") or not raw_path:
            parser.error(f"invalid --candidate {spec!r}")
        candidates.append((label.removeprefix("dx="), Path(raw_path)))
    write_report(candidates, args.expected_episodes, Path(args.report))
    print(f"PASS_L3A1_CORRIDOR_SWEEP_COMPLETED report={args.report}")


if __name__ == "__main__":
    main()
