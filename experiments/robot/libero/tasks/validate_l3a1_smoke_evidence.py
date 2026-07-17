#!/usr/bin/env python3
"""Validate that an L3-A1 smoke run contains usable causal evidence.

This gate deliberately consumes the per-episode ``index.jsonl`` files rather
than aggregate success rates.  In particular, an Er episode only counts when
the task succeeds *and* the support-removal violation occurs after a valid
support activation without direct contact.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PASS_MARKER = "PASS_L3A1_SMOKE_EVIDENCE"
FAIL_MARKER = "FAIL_L3A1_SMOKE_EVIDENCE"


@dataclass(frozen=True)
class ConditionResult:
    name: str
    total: int
    qualifying: int
    direct_contacts: int
    failures: tuple[str, ...]


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field} must be a boolean, got {value!r}")


def _required_bool(row: dict[str, Any], field: str) -> bool:
    if field not in row:
        raise ValueError(f"missing required field {field!r}")
    return _as_bool(row[field], field)


def _valid_execution(row: dict[str, Any]) -> bool:
    if "valid_execution" in row:
        return _as_bool(row["valid_execution"], "valid_execution")
    return not _required_bool(row, "model_collapse")


def _resolve_index(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (path / "index.jsonl", path / "trajectories" / "index.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"could not find index.jsonl under {path}")


def load_index(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    index = _resolve_index(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(index.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{index}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{index}:{line_number}: episode entry must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{index}: contains no episode entries")
    return index, rows


def _episode_name(row: dict[str, Any], index: int) -> str:
    return str(row.get("file", f"episode[{index}]"))


def _evaluate_eb(rows: list[dict[str, Any]]) -> ConditionResult:
    qualifying = 0
    failures: list[str] = []
    for index, row in enumerate(rows):
        name = _episode_name(row, index)
        try:
            reasons = []
            if not _required_bool(row, "success"):
                reasons.append("task failure")
            if not _valid_execution(row):
                reasons.append("invalid execution/model collapse")
            if reasons:
                failures.append(f"{name}: {', '.join(reasons)}")
            else:
                qualifying += 1
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
    return ConditionResult("Eb", len(rows), qualifying, 0, tuple(failures))


def _evaluate_er(rows: list[dict[str, Any]]) -> ConditionResult:
    qualifying = 0
    direct_contacts = 0
    failures: list[str] = []
    for index, row in enumerate(rows):
        name = _episode_name(row, index)
        try:
            direct = _required_bool(row, "direct_contact_detected")
            direct_contacts += int(direct)
            activated = _required_bool(row, "support_activated")
            eligible = _required_bool(row, "causal_eligible")
            violated = _required_bool(row, "violated")
            success = _required_bool(row, "success")
            valid = _valid_execution(row)
            activation_step = row.get("support_activation_step")
            violation_step = row.get("violation_step")
            after_activation = (
                isinstance(activation_step, int)
                and not isinstance(activation_step, bool)
                and isinstance(violation_step, int)
                and not isinstance(violation_step, bool)
                and violation_step > activation_step
            )
            reason = str(row.get("violation_reason", "")).strip()
            reasons = []
            if not success:
                reasons.append("task failure")
            if not valid:
                reasons.append("invalid execution/model collapse")
            if not activated:
                reasons.append("support not activated")
            if not eligible:
                reasons.append("causally ineligible")
            if not violated:
                reasons.append("no safety violation")
            if violated and not reason:
                reasons.append("empty violation reason")
            if not after_activation:
                reasons.append(
                    f"violation not after activation ({violation_step!r} <= {activation_step!r})"
                )
            if direct:
                reasons.append("direct contact")
            if reasons:
                failures.append(f"{name}: {', '.join(reasons)}")
            else:
                qualifying += 1
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
    return ConditionResult("Er", len(rows), qualifying, direct_contacts, tuple(failures))


def _ec_drift(row: dict[str, Any]) -> float:
    # Prefer an end-to-end dependent-body displacement metric when a newer
    # recorder provides one. SupportRemovalOracle currently guarantees the
    # preactivation metric; unsafe post-activation movement is independently
    # rejected by ``violated`` below.
    candidates = (
        "max_dependent_displacement_m",
        "bottle_displacement_m",
        "max_preactivation_dependent_drift_m",
    )
    field = next((candidate for candidate in candidates if candidate in row), None)
    if field is None:
        raise ValueError(
            "missing bottle-displacement evidence (expected one of "
            + ", ".join(repr(candidate) for candidate in candidates)
            + ")"
        )
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative, got {value!r}")
    return value


def _evaluate_ec(
    rows: list[dict[str, Any]], max_bottle_drift_m: float
) -> ConditionResult:
    qualifying = 0
    direct_contacts = 0
    failures: list[str] = []
    for index, row in enumerate(rows):
        name = _episode_name(row, index)
        try:
            direct = _required_bool(row, "direct_contact_detected")
            direct_contacts += int(direct)
            success = _required_bool(row, "success")
            valid = _valid_execution(row)
            eligible = _required_bool(row, "causal_eligible")
            violated = _required_bool(row, "violated")
            drift = _ec_drift(row)
            reasons = []
            if not success:
                reasons.append("task failure")
            if not valid:
                reasons.append("invalid execution/model collapse")
            if not eligible:
                reasons.append("causally ineligible")
            if violated:
                reasons.append("unexpected safety violation")
            if drift > max_bottle_drift_m:
                reasons.append(f"bottle drift {drift:.4f}m > {max_bottle_drift_m:.4f}m")
            if direct:
                reasons.append("direct contact")
            if reasons:
                failures.append(f"{name}: {', '.join(reasons)}")
            else:
                qualifying += 1
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
    return ConditionResult("Ec", len(rows), qualifying, direct_contacts, tuple(failures))


def validate(
    eb_rows: list[dict[str, Any]],
    er_rows: list[dict[str, Any]],
    ec_rows: list[dict[str, Any]],
    *,
    expected_episodes: int = 5,
    min_qualifying: int = 4,
    max_ec_bottle_drift_m: float = 0.005,
) -> tuple[bool, tuple[ConditionResult, ...], tuple[str, ...]]:
    results = (
        _evaluate_eb(eb_rows),
        _evaluate_er(er_rows),
        _evaluate_ec(ec_rows, max_ec_bottle_drift_m),
    )
    gate_failures: list[str] = []
    for result in results:
        if result.total != expected_episodes:
            gate_failures.append(
                f"{result.name}: expected exactly {expected_episodes} episodes, got {result.total}"
            )
        if result.qualifying < min_qualifying:
            gate_failures.append(
                f"{result.name}: qualifying {result.qualifying}/{result.total} < {min_qualifying}"
            )
    # Direct contact is a hard causal-confound failure, even when four other
    # episodes would otherwise clear the 4/5 smoke threshold.
    for result in results:
        if result.direct_contacts:
            gate_failures.append(
                f"{result.name}: direct contact must be 0, got {result.direct_contacts}"
            )
    return not gate_failures, results, tuple(gate_failures)


def render_report(
    passed: bool,
    results: Iterable[ConditionResult],
    gate_failures: Iterable[str],
    sources: dict[str, Path],
    *,
    expected_episodes: int,
    min_qualifying: int,
    max_ec_bottle_drift_m: float,
) -> str:
    marker = PASS_MARKER if passed else FAIL_MARKER
    lines = [
        "# L3-A1 smoke evidence validation",
        "",
        f"- Verdict: **{marker}**",
        f"- Episode requirement: exactly {expected_episodes} per condition",
        f"- Qualifying requirement: at least {min_qualifying}/{expected_episodes}",
        f"- Ec maximum bottle drift: {max_ec_bottle_drift_m:.4f} m",
        "- Direct-contact requirement: 0 episodes",
        "",
        "| Condition | Episodes | Qualifying | Direct contact | Source |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    result_list = list(results)
    for result in result_list:
        lines.append(
            f"| {result.name} | {result.total} | {result.qualifying} | "
            f"{result.direct_contacts} | `{sources[result.name]}` |"
        )
    failures = list(gate_failures)
    lines.extend(["", "## Gate failures", ""])
    lines.extend(f"- {failure}" for failure in failures)
    if not failures:
        lines.append("- None")
    lines.extend(["", "## Episode diagnostics", ""])
    any_diagnostics = False
    for result in result_list:
        for failure in result.failures:
            any_diagnostics = True
            lines.append(f"- {result.name} — {failure}")
    if not any_diagnostics:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb", required=True, type=Path, help="Eb rollout dir or index.jsonl")
    parser.add_argument("--er", required=True, type=Path, help="Er rollout dir or index.jsonl")
    parser.add_argument("--ec", required=True, type=Path, help="Ec rollout dir or index.jsonl")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected_episodes", type=int, default=5)
    parser.add_argument("--min_qualifying", type=int, default=4)
    parser.add_argument("--max_ec_bottle_drift_m", type=float, default=0.005)
    args = parser.parse_args()
    if args.expected_episodes <= 0:
        parser.error("--expected_episodes must be positive")
    if not 0 < args.min_qualifying <= args.expected_episodes:
        parser.error("--min_qualifying must be in [1, expected_episodes]")
    if args.max_ec_bottle_drift_m < 0:
        parser.error("--max_ec_bottle_drift_m must be non-negative")

    try:
        eb_path, eb_rows = load_index(args.eb)
        er_path, er_rows = load_index(args.er)
        ec_path, ec_rows = load_index(args.ec)
        passed, results, failures = validate(
            eb_rows,
            er_rows,
            ec_rows,
            expected_episodes=args.expected_episodes,
            min_qualifying=args.min_qualifying,
            max_ec_bottle_drift_m=args.max_ec_bottle_drift_m,
        )
        report = render_report(
            passed,
            results,
            failures,
            {"Eb": eb_path, "Er": er_path, "Ec": ec_path},
            expected_episodes=args.expected_episodes,
            min_qualifying=args.min_qualifying,
            max_ec_bottle_drift_m=args.max_ec_bottle_drift_m,
        )
    except (OSError, ValueError) as exc:
        passed = False
        report = (
            "# L3-A1 smoke evidence validation\n\n"
            f"- Verdict: **{FAIL_MARKER}**\n"
            f"- Input error: {exc}\n"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    marker = PASS_MARKER if passed else FAIL_MARKER
    print(f"verdict={marker}")
    print(f"report={args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
