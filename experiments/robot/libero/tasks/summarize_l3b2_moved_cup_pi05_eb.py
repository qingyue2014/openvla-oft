"""Summarize the bounded pi0.5 Eb diagnostic for a moved L3-B2 mug.

The validated geometry currently carries the L3-B3 scene identifier because
that calibration found the common target relocation.  This report is
explicitly diagnostic-only: it may motivate a fresh L3-B2 preregistration but
cannot be pooled with L3-B3 or treated as L3-B2 smoke/formal evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    PROJECT_TARGET_WORLD_XY,
    SCENE_ID as SOURCE_SCENE_ID,
)


DIAGNOSTIC_ID = "L3-B2-MICROWAVE-ORDER-MOVED-CUP-PI05-EB-DIAGNOSTIC"


def summarize(directory: str | Path, *, expected_count: int) -> dict[str, object]:
    root = Path(directory).resolve(strict=True)
    index = root / "index.jsonl"
    rows = [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_count:
        raise ValueError(f"diagnostic has {len(rows)} rows; expected {expected_count}")
    if [int(row.get("episode_idx", -1)) for row in rows] != list(
        range(expected_count)
    ):
        raise ValueError("diagnostic episode indices are not contiguous")

    for row in rows:
        if row.get("model_family") != "pi05":
            raise ValueError("diagnostic contains a non-pi0.5 episode")
        if row.get("task_suite_name") != "libero_10" or int(
            row.get("task_id", -1)
        ) != 9:
            raise ValueError("diagnostic native task identity mismatch")
        gate = row.get("runtime_initial_gate", {})
        if (
            gate.get("scenario") != SOURCE_SCENE_ID
            or gate.get("condition") != "native"
            or gate.get("physical_gate_pass") is not True
        ):
            raise ValueError("diagnostic lacks the exact Eb first-policy gate")
        trace = row.get("l3b3_microwave_sequence", {})
        if (
            trace.get("scenario") != SOURCE_SCENE_ID
            or trace.get("condition") != "native"
        ):
            raise ValueError("diagnostic lacks the native microwave event trace")

    successes = sum(bool(row.get("success")) for row in rows)
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_geometry_scene": SOURCE_SCENE_ID,
        "source_geometry_evidence_reused_as_l3b2_result": False,
        "candidate_target_world_xy": list(PROJECT_TARGET_WORLD_XY),
        "model_family": "pi05",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "condition": "Eb/native",
        "count": len(rows),
        "native_successes": successes,
        "native_success_rate": successes / len(rows),
        "at_least_one_native_success": successes > 0,
        "candidate_should_be_preregistered_as_l3b2": successes > 0,
        "safe_reference_complete": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "interpretation": (
            "capability diagnostic only; a success supports fresh L3-B2 "
            "preregistration and regeneration, while a failure does not "
            "constitute a formal model metric"
        ),
        "verdict": (
            "PASS_L3B2_MOVED_CUP_PI05_EB_DIAGNOSTIC"
            if successes > 0
            else "FAIL_L3B2_MOVED_CUP_PI05_EB_DIAGNOSTIC"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.trajectory_dir, expected_count=args.expected_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{result['verdict']} successes={result['native_successes']}/"
        f"{result['count']} formal_authorized=false"
    )


if __name__ == "__main__":
    main()
