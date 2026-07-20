#!/usr/bin/env python3
"""Create or verify the frozen L2-A Native policy-view approval manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_DIR = Path("experiments/robot/libero/tasks")
BDDL = TASK_DIR / "PHYSCOG_L2A_NATIVE_bowl_stove_choice.bddl"
PAIRING = TASK_DIR / "l2a_native_pairing.json"
EC_STATES = TASK_DIR / "l2a_native_ec_stove_off.hdf5"
ER_STATES = TASK_DIR / "l2a_native_er_stove_on.hdf5"
PREVIEW_DIR = TASK_DIR / "l2a_native_preview"
APPROVAL = TASK_DIR / "l2a_native_policy_view_approval.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence_paths() -> list[Path]:
    return [BDDL, PAIRING, EC_STATES, ER_STATES, *sorted(PREVIEW_DIR.glob("*.png"))]


def create(args: argparse.Namespace) -> None:
    pairing = json.loads(PAIRING.read_text(encoding="utf-8"))
    episodes = int(pairing["num_states"])
    ec_views = sorted(PREVIEW_DIR.glob("episode_*_Ec_stove_off_policy224.png"))
    er_views = sorted(PREVIEW_DIR.glob("episode_*_Er_stove_on_policy224.png"))
    if len(ec_views) != episodes or len(er_views) != episodes:
        raise SystemExit(
            f"Expected {episodes} Ec and Er policy views, found {len(ec_views)}/{len(er_views)}"
        )
    paths = _evidence_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing frozen evidence: {missing}")
    payload = {
        "verdict": "APPROVED_L2A_NATIVE_POLICY_VIEW",
        "reviewer": args.reviewer,
        "reviewed_episode_count": episodes,
        "review_scope": (
            "All frozen Ec/Er 224x224 policy views; both bowls and plate visible; "
            "Er native red burner cue visible; no unintended Ec/Er layout change."
        ),
        "notes": args.notes,
        "sha256": {str(path): _sha256(path) for path in paths},
    }
    APPROVAL.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Verdict: APPROVED_L2A_NATIVE_POLICY_VIEW")
    print(f"Approval: {APPROVAL}")


def verify(args: argparse.Namespace) -> None:
    payload = json.loads(APPROVAL.read_text(encoding="utf-8"))
    pairing = json.loads(PAIRING.read_text(encoding="utf-8"))
    failures = []
    if payload.get("verdict") != "APPROVED_L2A_NATIVE_POLICY_VIEW":
        failures.append("approval verdict is absent")
    episodes = int(pairing["num_states"])
    if episodes != args.expected_states:
        failures.append(f"frozen state count is {episodes}, expected {args.expected_states}")
    if int(payload.get("reviewed_episode_count", -1)) != episodes:
        failures.append("approval episode count does not match pairing manifest")
    expected_hashes = payload.get("sha256", {})
    current_paths = {str(path): path for path in _evidence_paths()}
    if set(expected_hashes) != set(current_paths):
        failures.append("approval evidence file set does not match frozen evidence")
    for name, path in current_paths.items():
        if not path.is_file():
            failures.append(f"missing frozen evidence: {name}")
        elif expected_hashes.get(name) != _sha256(path):
            failures.append(f"hash mismatch: {name}")
    if failures:
        print("Verdict: FAIL_L2A_NATIVE_FROZEN_APPROVAL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Verdict: PASS_L2A_NATIVE_FROZEN_APPROVAL")
    print(f"Frozen states and {episodes * 2} policy views match the approval manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--reviewer", required=True)
    create_parser.add_argument("--notes", default="")
    create_parser.set_defaults(func=create)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--expected-states", type=int, required=True)
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
