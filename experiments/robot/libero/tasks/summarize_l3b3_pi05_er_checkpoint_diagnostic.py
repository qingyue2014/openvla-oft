"""Summarize the one-episode pi0.5 exact-Er checkpoint diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    SCENE_ID,
)


DIAGNOSTIC_ID = "L3-B3-PI05-EXACT-ER-CHECKPOINT-DIAGNOSTIC"
RUN_NOTE = "L3-B3-pi05-Er-checkpoint-diagnostic"


def summarize(directory: str | Path, *, expected_count: int) -> dict[str, object]:
    root = Path(directory).resolve(strict=True)
    rows = [
        json.loads(line)
        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_count:
        raise ValueError(f"diagnostic has {len(rows)} rows; expected {expected_count}")
    if expected_count != 1:
        raise ValueError("L3-B3 pi0.5 Er diagnostic is restricted to one episode")

    for row in rows:
        if row.get("model_family") != "pi05":
            raise ValueError("diagnostic contains a non-pi0.5 episode")
        if row.get("run_id_note") != RUN_NOTE:
            raise ValueError("diagnostic run-note mismatch")
        if row.get("task_suite_name") != "libero_10" or int(
            row.get("task_id", -1)
        ) != 9:
            raise ValueError("diagnostic native task identity mismatch")
        gate = row.get("runtime_initial_gate", {})
        if (
            gate.get("scenario") != SCENE_ID
            or gate.get("condition") != "closed_microwave"
            or gate.get("physical_gate_pass") is not True
        ):
            raise ValueError("diagnostic lacks the exact Er first-policy gate")
        trace = row.get("l3b3_microwave_sequence", {})
        if (
            trace.get("scenario") != SCENE_ID
            or trace.get("condition") != "closed_microwave"
        ):
            raise ValueError("diagnostic lacks the exact Er microwave event trace")

    row = rows[0]
    trace = row["l3b3_microwave_sequence"]
    settle = trace.get("post_success_settle", {})
    task_success = bool(row.get("success"))
    settle_count = int(settle.get("sample_count", 0))
    strict_stability = bool(settle.get("strict_target_stability_pass"))

    if not task_success:
        checkpoint_test = "INCONCLUSIVE_PI05_DID_NOT_COMPLETE_PLACEMENT"
        interpretation = (
            "pi0.5 did not reach native task success, so it produced no "
            "post-success floor-settle window to compare with OpenVLA-OFT."
        )
    elif settle_count != 100:
        checkpoint_test = "INVALID_MISSING_EXACT_SETTLE_WINDOW"
        interpretation = (
            "pi0.5 reached task success, but the required 100-step no-op "
            "settle window is incomplete."
        )
    elif strict_stability:
        checkpoint_test = "PI05_PLACEMENT_STABLE"
        interpretation = (
            "This pi0.5 placement passed the strict settle checks. The result "
            "can indicate release-state sensitivity, but it does not prove the "
            "checkpoint directly changes MuJoCo contact dynamics."
        )
    else:
        checkpoint_test = "MICROBOUNCE_OR_INSTABILITY_PERSISTS_UNDER_PI05"
        interpretation = (
            "The exact post-success no-op window also failed under pi0.5, "
            "which rules out the OpenVLA-OFT checkpoint as the sole cause."
        )

    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "scenario": SCENE_ID,
        "model_family": "pi05",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "condition": "closed_microwave",
        "count": 1,
        "native_task_success": task_success,
        "door_opened": bool(trace.get("door_opened")),
        "insertion_after_open": bool(trace.get("insertion_after_open")),
        "reclose_after_insertion": bool(trace.get("reclose_after_insertion")),
        "failure_stage": trace.get("failure_stage"),
        "post_success_settle": settle,
        "checkpoint_cause_test": checkpoint_test,
        "interpretation": interpretation,
        "diagnostic_only": True,
        "do_not_pool_as_formal_model_evidence": True,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": "PASS_L3B3_PI05_ER_CHECKPOINT_DIAGNOSTIC_EXECUTED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.trajectory_dir, expected_count=args.expected_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{result['verdict']} outcome={result['checkpoint_cause_test']} "
        "formal_authorized=false"
    )


if __name__ == "__main__":
    main()
