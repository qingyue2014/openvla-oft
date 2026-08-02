"""Summarize L3-B bowl rollouts using predicate history, not collisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    BOWL_BODY,
    CONDITION_LABEL,
    FORMAL_WAIT_STEPS,
    MAX_BOWL_TILT_DEG,
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)
from experiments.robot.libero.tasks.validate_l3b_bowl_design import validate_spec


VERDICT = "PASS_L3B_BOWL_POLICY_DIAGNOSTIC_SUMMARY"
TERMINAL_WINDOW_STEPS = 30
TERMINAL_MAX_DRIFT_M = 0.003
WILSON_Z_95 = 1.959963984540054


def _wilson_interval(successes: int, total: int) -> dict:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial count")
    z2 = WILSON_Z_95**2
    rate = successes / total
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    half_width = (
        WILSON_Z_95
        * ((rate * (1.0 - rate) / total + z2 / (4.0 * total**2)) ** 0.5)
        / denominator
    )
    return {
        "method": "Wilson score",
        "confidence": 0.95,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def _tilt_deg(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if q.ndim != 2 or q.shape[1] != 4 or np.any(norms <= 0):
        raise ValueError("invalid bowl quaternion trajectory")
    q = q / norms
    rzz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(rzz, -1.0, 1.0)))


def _files(directory: str | Path) -> list[Path]:
    directory = Path(directory).resolve(strict=True)
    files = sorted(directory.glob("task3_ep*.npz"))
    if not files:
        files = sorted(directory.glob("*.npz"))
    if not files:
        raise ValueError(f"no trajectories under {directory}")
    return files


def _terminal_stability(trajectory: dict, success: bool) -> dict:
    key_pos = f"body_pos__{BOWL_BODY}"
    key_quat = f"body_quat__{BOWL_BODY}"
    if key_pos not in trajectory or key_quat not in trajectory:
        raise ValueError("trajectory omits the black bowl")
    phases = np.asarray(trajectory["phases"]).astype(str)
    positions = np.asarray(trajectory[key_pos], dtype=float)
    tilts = _tilt_deg(np.asarray(trajectory[key_quat], dtype=float))
    settle = np.flatnonzero(phases == "settle")
    terminal = settle[-TERMINAL_WINDOW_STEPS:] if len(settle) else np.arange(max(0, len(phases)-TERMINAL_WINDOW_STEPS), len(phases))
    if len(terminal) == 0:
        raise ValueError("empty trajectory")
    drift = float(np.max(np.linalg.norm(positions[terminal] - positions[terminal[0]], axis=1)))
    max_tilt = float(np.max(tilts[terminal]))
    failures = []
    if success and len(settle) < TERMINAL_WINDOW_STEPS:
        failures.append("insufficient_post_success_settle")
    if success and max_tilt > MAX_BOWL_TILT_DEG:
        failures.append("terminal_bowl_tilt")
    if success and drift > TERMINAL_MAX_DRIFT_M:
        failures.append("terminal_bowl_drift")
    return {
        "passed": not failures,
        "failures": failures,
        "settle_steps": int(len(settle)),
        "terminal_max_tilt_deg": max_tilt,
        "terminal_max_drift_m": drift,
    }


def _episode(
    path: Path,
    condition: str,
    *,
    expected_model_family: str,
    expected_checkpoint: str,
    require_model_identity: bool,
) -> dict:
    trajectory = load_trajectory(str(path))
    meta = trajectory["metadata"]
    if require_model_identity and (
        meta.get("model_family") != expected_model_family
        or meta.get("pretrained_checkpoint") != expected_checkpoint
    ):
        raise ValueError(f"{path} model/checkpoint identity mismatch")
    if (
        meta.get("task_suite_name") != SUITE
        or int(meta.get("task_id", -1)) != TASK_ID
        or meta.get("task_description") != TASK_PROMPT
        or meta.get("safety_oracle") != "none"
        or int(meta.get("num_steps_wait", -1)) != FORMAL_WAIT_STEPS
    ):
        raise ValueError(f"{path} evaluator identity mismatch")
    runtime = meta.get("runtime_initial_gate", {})
    if (
        runtime.get("condition") != condition
        or runtime.get("condition_label") != CONDITION_LABEL[condition]
        or runtime.get("physical_gate_pass") is not True
        or runtime.get("failures")
    ):
        raise ValueError(f"{path} lacks exact runtime physical PASS")
    sequence = meta.get("l3b_bowl_sequence", {})
    if (
        sequence.get("condition") != condition
        or sequence.get("defines_task_success") is not False
        or sequence.get("collision_oracle_used") is not False
    ):
        raise ValueError(f"{path} lacks predicate sequence metrics")
    success = bool(meta.get("success", False))
    stability = _terminal_stability(trajectory, success)
    strict_success = bool(success and stability["passed"])
    if condition == "premature_close":
        if sequence.get("full_ordered_repair") and strict_success:
            outcome = "full_ordered_repair"
        else:
            outcome = str(sequence.get("failure_stage", "unknown"))
    elif condition == "native":
        outcome = "native_success" if strict_success else "native_failure"
    else:
        outcome = "close_only_success" if strict_success else "close_only_failure"
    return {
        "file": str(path),
        "sha256": sha256_path(path),
        "episode_idx": int(meta.get("episode_idx", -1)),
        "native_init_state_index": int(runtime.get("native_init_state_index", -1)),
        "success": success,
        "strict_success": strict_success,
        "terminal_stability": stability,
        "sequence": sequence,
        "outcome": outcome,
    }


def _condition(
    condition: str,
    directory: str | Path,
    expected_count: int,
    expected_native_indices: list[int],
    expected_model_family: str,
    expected_checkpoint: str,
    require_model_identity: bool,
) -> dict:
    files = _files(directory)
    if len(files) != expected_count:
        raise ValueError(f"{condition} has {len(files)} trajectories, expected {expected_count}")
    episodes = [
        _episode(
            path,
            condition,
            expected_model_family=expected_model_family,
            expected_checkpoint=expected_checkpoint,
            require_model_identity=require_model_identity,
        )
        for path in files
    ]
    if len({item["episode_idx"] for item in episodes}) != len(episodes):
        raise ValueError(f"{condition} duplicate episode indices")
    if any(item["native_init_state_index"] < 0 for item in episodes):
        raise ValueError(f"{condition} missing official state index")
    actual_native_indices = [
        item["native_init_state_index"]
        for item in sorted(episodes, key=lambda item: item["episode_idx"])
    ]
    if actual_native_indices != expected_native_indices:
        raise ValueError(
            f"{condition} official state indices {actual_native_indices} "
            f"!= registered {expected_native_indices}"
        )
    strict_successes = sum(item["strict_success"] for item in episodes)
    result = {
        "condition": condition,
        "condition_label": CONDITION_LABEL[condition],
        "count": len(episodes),
        "strict_successes": strict_successes,
        "strict_success_rate": strict_successes / len(episodes),
        "strict_success_ci95": _wilson_interval(strict_successes, len(episodes)),
        "outcomes": {},
        "episodes": episodes,
    }
    for item in episodes:
        result["outcomes"][item["outcome"]] = result["outcomes"].get(item["outcome"], 0) + 1
    if condition == "premature_close":
        result.update(
            {
                "rollback_recognition_count": sum(item["sequence"]["rollback_recognized"] for item in episodes),
                "insertion_after_rollback_count": sum(item["sequence"]["insertion_after_rollback"] for item in episodes),
                "reclose_after_insertion_count": sum(item["sequence"]["reclose_after_insertion"] for item in episodes),
                "full_ordered_repair_count": sum(item["sequence"]["full_ordered_repair"] and item["strict_success"] for item in episodes),
            }
        )
        result["full_ordered_repair_rate"] = result["full_ordered_repair_count"] / len(episodes)
        result["full_ordered_repair_ci95"] = _wilson_interval(
            result["full_ordered_repair_count"], len(episodes)
        )
    return result


def summarize(
    eb,
    er,
    ec,
    *,
    expected_count: int,
    formal_approval: dict | None = None,
    design_preregistration: str | Path | None = None,
    model_label: str = "pi05",
    model_family: str = "pi05",
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero",
) -> dict:
    if design_preregistration is None:
        design_preregistration = Path(__file__).with_name(
            "l3b_bowl_v1_design_prereg.json"
        )
    design = validate_spec(design_preregistration)
    require_model_identity = int(design.get("evaluation_version", 1)) >= 2
    if require_model_identity:
        registered_checkpoint = design.get("model_matrix", {}).get(model_label)
        if registered_checkpoint != checkpoint:
            raise ValueError(
                f"model {model_label!r} checkpoint {checkpoint!r} is not registered"
            )
    if expected_count > design["count"]:
        raise ValueError("expected rollout count exceeds registered native pool")
    expected_native_indices = design["official_state_indices"][:expected_count]
    conditions = {
        "native": _condition(
            "native", eb, expected_count, expected_native_indices,
            model_family, checkpoint, require_model_identity,
        ),
        "premature_close": _condition(
            "premature_close", er, expected_count, expected_native_indices,
            model_family, checkpoint, require_model_identity,
        ),
        "prerequisite_done": _condition(
            "prerequisite_done", ec, expected_count, expected_native_indices,
            model_family, checkpoint, require_model_identity,
        ),
    }
    human_review = None
    if formal_approval is not None:
        human_review = {
            "verdict": formal_approval["verdict"],
            "reviewer": formal_approval["reviewer"],
            "approved_at_utc": formal_approval["approved_at_utc"],
            "smoke_report_sha256": formal_approval["smoke_report_sha256"],
            "approval_sha256": formal_approval["approval_sha256"],
        }
    return {
        "scenario": SCENE_ID,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "expected_count_per_condition": expected_count,
        "evaluation_design": design,
        "model": {
            "label": model_label,
            "family": model_family,
            "checkpoint": checkpoint,
            "trajectory_identity_verified": require_model_identity,
        },
        "primary_metric": "Er full_ordered_repair_rate",
        "primary_value": conditions["premature_close"]["full_ordered_repair_rate"],
        "conditions": conditions,
        "interpretation_limit": (
            "Raw Er/Ec success rates are workload-asymmetric; the primary diagnostic "
            "is the event-based Er rollback-and-repair trace."
        ),
        "collision_oracle_used": False,
        "human_review": human_review,
        "formal_authorized": formal_approval is not None,
        "verdict": VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--design-preregistration",
        default=str(Path(__file__).with_name("l3b_bowl_v1_design_prereg.json")),
    )
    parser.add_argument("--human-approval")
    parser.add_argument("--smoke-report")
    parser.add_argument("--model-label", default="pi05")
    parser.add_argument("--model-family", default="pi05")
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi05_libero",
    )
    args = parser.parse_args()
    if bool(args.human_approval) != bool(args.smoke_report):
        parser.error("--human-approval and --smoke-report must be provided together")
    formal_approval = None
    if args.human_approval:
        # Imported lazily because the approval module imports this module's
        # smoke verdict when validating the hash-bound review record.
        from experiments.robot.libero.tasks.validate_l3b_bowl_human_review import verify

        formal_approval = dict(
            verify(args.human_approval, smoke_report=args.smoke_report)
        )
        formal_approval["approval_sha256"] = sha256_path(args.human_approval)
    result = summarize(
        args.eb,
        args.er,
        args.ec,
        expected_count=args.expected_count,
        formal_approval=formal_approval,
        design_preregistration=args.design_preregistration,
        model_label=args.model_label,
        model_family=args.model_family,
        checkpoint=args.checkpoint,
    )
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{VERDICT} primary={result['primary_value']:.3f}")


if __name__ == "__main__":
    main()
