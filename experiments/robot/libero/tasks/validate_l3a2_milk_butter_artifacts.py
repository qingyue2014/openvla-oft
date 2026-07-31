"""CLI for L3-A2 paired-state and explicit human-review gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    TASK_KEY,
    validate_generation_manifest,
    validate_human_approval,
    validate_state_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--native_bddl", required=True)
    parser.add_argument("--minimum_count", type=int, default=1)
    parser.add_argument("--manifest")
    parser.add_argument("--review_evidence")
    parser.add_argument("--human_approval")
    parser.add_argument("--out")
    parser.add_argument(
        "--print_floor_support_bodies",
        action="store_true",
        help="Print the compiled floor contact body names persisted by generation.",
    )
    args = parser.parse_args()

    result = validate_state_artifacts(
        args.eb,
        args.er,
        args.ec,
        native_bddl=args.native_bddl,
        minimum_count=args.minimum_count,
    )
    result["state_verdict"] = "PASS_L3A2_PAIRED_FORMAL_STATE_GATES"
    if args.manifest:
        result["generation_manifest"] = validate_generation_manifest(
            args.manifest,
            eb_path=args.eb,
            er_path=args.er,
            ec_path=args.ec,
            minimum_count=args.minimum_count,
        )
        result["generation_manifest_verdict"] = (
            "PASS_L3A2_GENERATION_MANIFEST_BINDING"
        )
    if args.print_floor_support_bodies:
        with h5py.File(args.eb, "r") as handle:
            encoded = handle[TASK_KEY].attrs.get("floor_support_bodies", "[]")
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        bodies = json.loads(str(encoded))
        if not bodies:
            raise ValueError("compiled floor support body inventory is empty")
        print(",".join(bodies))
        return
    if args.human_approval:
        if not args.manifest or not args.review_evidence:
            parser.error(
                "--manifest and --review_evidence are required with "
                "--human_approval"
            )
        result["human_review"] = validate_human_approval(
            args.human_approval,
            eb_path=args.eb,
            er_path=args.er,
            ec_path=args.ec,
            manifest_path=args.manifest,
            review_evidence_path=args.review_evidence,
        )
        result["human_review_verdict"] = "PASS_L3A2_HUMAN_REVIEW"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(result["state_verdict"])
    if "human_review_verdict" in result:
        print(result["human_review_verdict"])


if __name__ == "__main__":
    main()
