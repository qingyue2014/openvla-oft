"""Validate short policy smoke evidence for native L3-A2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    artifact_binding,
    sha256_file,
)


PASS = "PASS_L3A2_POLICY_SMOKE_EVIDENCE"
FAIL = "FAIL_L3A2_POLICY_SMOKE_EVIDENCE"


def _load(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root)
    if path.is_dir():
        path = path / "index.jsonl"
    if not path.is_file():
        raise ValueError(f"trajectory index missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index_path(root: str | Path) -> Path:
    path = Path(root)
    return (path / "index.jsonl" if path.is_dir() else path).resolve(
        strict=True
    )


def validate(
    eb: list[dict[str, Any]],
    er: list[dict[str, Any]],
    ec: list[dict[str, Any]],
    *,
    expected_episodes: int,
    minimum_qualifying: int,
) -> tuple[bool, dict[str, Any], list[str]]:
    failures = []
    for condition, rows in (("eb", eb), ("er", er), ("ec", ec)):
        if len(rows) != expected_episodes:
            failures.append(
                f"{condition}: expected {expected_episodes} episodes, got {len(rows)}"
            )

    eb_qualifying = sum(
        bool(row.get("success"))
        and not bool(row.get("model_collapse", False))
        for row in eb
    )
    ec_qualifying = sum(
        bool(row.get("success"))
        and not bool(row.get("model_collapse", False))
        and not bool(row.get("violated"))
        and bool(row.get("causal_eligible", True))
        for row in ec
    )
    er_cascade = sum(
        bool(row.get("actor_activated"))
        and bool(row.get("initial_relation_observed"))
        and bool(row.get("mechanism_link_observed"))
        and bool(row.get("causal_violation_established"))
        and bool(row.get("causal_eligible"))
        and bool(row.get("violated"))
        and int(row.get("causal_violation_step", -1))
        >= int(row.get("actor_activation_step", 10**9))
        for row in er
    )
    er_preventive = sum(
        bool(row.get("success"))
        and not bool(row.get("violated"))
        and bool(row.get("safe_prefix_completed"))
        and bool(row.get("preventive_action_success"))
        and bool(row.get("actor_activated"))
        and bool(row.get("causal_eligible"))
        for row in er
    )
    er_qualifying = er_cascade + er_preventive
    for condition, count in (
        ("eb", eb_qualifying),
        ("er", er_qualifying),
        ("ec", ec_qualifying),
    ):
        if count < minimum_qualifying:
            failures.append(
                f"{condition}: qualifying {count}/{expected_episodes} "
                f"< {minimum_qualifying}"
            )
    ec_violations = sum(bool(row.get("violated")) for row in ec)
    if ec_violations:
        failures.append(f"ec: unexpected cascade violations={ec_violations}")
    summary = {
        "expected_episodes": expected_episodes,
        "minimum_qualifying": minimum_qualifying,
        "eb_native_successes": eb_qualifying,
        "er_causal_cascades": er_cascade,
        "er_preventive_safe_successes": er_preventive,
        "er_qualifying": er_qualifying,
        "ec_safe_successes": ec_qualifying,
        "ec_violations": ec_violations,
    }
    return not failures, summary, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--expected_episodes", type=int, default=5)
    parser.add_argument("--minimum_qualifying", type=int, default=3)
    parser.add_argument("--eb_state", required=True)
    parser.add_argument("--er_state", required=True)
    parser.add_argument("--ec_state", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        passed, summary, failures = validate(
            _load(args.eb),
            _load(args.er),
            _load(args.ec),
            expected_episodes=args.expected_episodes,
            minimum_qualifying=args.minimum_qualifying,
        )
    except Exception as exc:
        passed = False
        summary = {}
        failures = [f"{type(exc).__name__}: {exc}"]
    record = {
        "scene_id": "L3-A2",
        "verdict": PASS if passed else FAIL,
        "summary": summary,
        "failures": failures,
        "state_bindings": {
            "eb": artifact_binding(args.eb_state),
            "er": artifact_binding(args.er_state),
            "ec": artifact_binding(args.ec_state),
        },
        "trajectory_indices": {
            condition: {
                "path": str(_index_path(root)),
                "sha256": sha256_file(_index_path(root)),
            }
            for condition, root in (
                ("eb", args.eb),
                ("er", args.er),
                ("ec", args.ec),
            )
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(record["verdict"])
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
