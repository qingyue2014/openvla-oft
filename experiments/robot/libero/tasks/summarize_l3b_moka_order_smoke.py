"""Summarize π0.5 moka-sequence smoke runs without a collision oracle.

Eb (native) is descriptive.  Ec (far-first) is the matched capability
control because it leaves the same moka pot 1 manipulation as Er while
placing moka pot 2 in the less obstructive far slot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITION_INTERVENTION_BODY,
    CONDITION_REMAINING_BODY,
    CONDITIONS,
    FORMAL_WAIT_STEPS,
    MAX_RECEPTACLE_TILT_DEG,
    POT_BODIES,
    SCENE_ID,
    STOVE_BODY,
    SUITE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)
from experiments.robot.libero.physcog_trajectory import load_trajectory


PASS_NATIVE = "PASS_L3B_MOKA_NATIVE_CAPABILITY_SMOKE"
PASS_CONTROL = "PASS_L3B_MOKA_EC_CAPABILITY_CONTROL"
PASS_SMOKE = "PASS_L3B_MOKA_POLICY_SMOKE_DIAGNOSTIC"
MIN_TARGET_MOTION_M = 0.03
DIRECT_OCCUPANT_MAX_DISPLACEMENT_M = 0.015
TERMINAL_WINDOW_STEPS = 30
TERMINAL_MAX_DRIFT_M = 0.003


def _tilt_deg(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"invalid quaternion array: {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero quaternion in trajectory")
    q = quaternions / norms
    rzz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(rzz, -1.0, 1.0)))


def _trajectory_files(directory: str | Path) -> list[Path]:
    directory = Path(directory).resolve(strict=True)
    files = sorted(directory.glob("task8_ep*.npz"))
    if not files:
        files = sorted(directory.glob("*.npz"))
    if not files:
        raise ValueError(f"no smoke trajectories under {directory}")
    return files


def _body_metrics(
    trajectory: dict,
    body: str,
    first_policy_record: dict,
) -> dict:
    pos_key = f"body_pos__{body}"
    quat_key = f"body_quat__{body}"
    if pos_key not in trajectory or quat_key not in trajectory:
        raise ValueError(f"trajectory omits {body}")
    positions = np.asarray(trajectory[pos_key], dtype=float)
    quaternions = np.asarray(trajectory[quat_key], dtype=float)
    phases = np.asarray(trajectory["phases"]).astype(str)
    if len(positions) != len(phases) or len(positions) == 0:
        raise ValueError(f"invalid trajectory length for {body}")
    policy_mask = phases != "wait"
    if not np.any(policy_mask):
        raise ValueError("trajectory contains no policy or settle samples")
    policy_positions = positions[policy_mask]
    baseline = np.asarray(first_policy_record[body]["position"], dtype=float)
    max_displacement = float(
        np.max(np.linalg.norm(policy_positions - baseline, axis=1))
    )
    tilts = _tilt_deg(quaternions)
    settle_indices = np.flatnonzero(phases == "settle")
    terminal_indices = (
        settle_indices[-TERMINAL_WINDOW_STEPS:]
        if len(settle_indices)
        else np.flatnonzero(policy_mask)[-TERMINAL_WINDOW_STEPS:]
    )
    terminal_positions = positions[terminal_indices]
    terminal_tilts = tilts[terminal_indices]
    terminal_origin = terminal_positions[0]
    terminal_drift = float(
        np.max(
            np.linalg.norm(
                terminal_positions - terminal_origin,
                axis=1,
            )
        )
    )
    return {
        "max_displacement_m": max_displacement,
        "final_position": positions[-1].tolist(),
        "final_tilt_deg": float(tilts[-1]),
        "terminal_max_tilt_deg": float(np.max(terminal_tilts)),
        "terminal_max_drift_m": terminal_drift,
        "settle_steps_recorded": int(len(settle_indices)),
    }


def _episode(path: Path, expected_condition: str) -> dict:
    trajectory = load_trajectory(str(path))
    meta = trajectory["metadata"]
    if (
        meta.get("task_suite_name") != SUITE
        or int(meta.get("task_id", -1)) != TASK_ID
        or meta.get("task_description") != TASK_PROMPT
        or meta.get("safety_oracle") != "none"
        or int(meta.get("num_steps_wait", -1)) != FORMAL_WAIT_STEPS
    ):
        raise ValueError(f"{path} task/evaluator identity mismatch")
    runtime = meta.get("runtime_initial_gate")
    if (
        not isinstance(runtime, dict)
        or runtime.get("condition") != expected_condition
        or runtime.get("physical_gate_pass") is not True
        or runtime.get("failures")
    ):
        raise ValueError(f"{path} lacks exact first-policy physical PASS")
    body = {
        name: _body_metrics(trajectory, name, runtime["first_policy"])
        for name in (*POT_BODIES, STOVE_BODY)
    }
    success = bool(meta.get("success", False))
    terminal_stable = True
    terminal_failures = []
    if success:
        for pot in POT_BODIES:
            if body[pot]["settle_steps_recorded"] < TERMINAL_WINDOW_STEPS:
                terminal_failures.append(f"{pot}:insufficient_settle_samples")
            if body[pot]["terminal_max_tilt_deg"] > MAX_RECEPTACLE_TILT_DEG:
                terminal_failures.append(f"{pot}:terminal_tilt")
            if body[pot]["terminal_max_drift_m"] > TERMINAL_MAX_DRIFT_M:
                terminal_failures.append(f"{pot}:terminal_drift")
        terminal_stable = not terminal_failures

    if expected_condition == "native":
        target_motion = min(
            body[pot]["max_displacement_m"] for pot in POT_BODIES
        )
        occupant_displacement = None
        if success and terminal_stable:
            outcome = "native_complete"
        elif target_motion < MIN_TARGET_MOTION_M:
            outcome = "no_two_pot_capability"
        elif success:
            outcome = "native_complete_unstable"
        else:
            outcome = "native_incomplete"
    else:
        target = CONDITION_REMAINING_BODY[expected_condition]
        occupant = CONDITION_INTERVENTION_BODY[expected_condition]
        target_motion = body[target]["max_displacement_m"]
        occupant_displacement = body[occupant]["max_displacement_m"]
        if success and terminal_stable:
            outcome = (
                "direct_completion"
                if occupant_displacement
                <= DIRECT_OCCUPANT_MAX_DISPLACEMENT_M
                else "repair_then_complete"
            )
        elif success:
            outcome = "completion_unstable"
        elif target_motion < MIN_TARGET_MOTION_M:
            outcome = "no_target_manipulation"
        elif occupant_displacement > DIRECT_OCCUPANT_MAX_DISPLACEMENT_M:
            outcome = "repair_attempt_incomplete"
        else:
            outcome = "target_attempt_incomplete"
    return {
        "file": str(path),
        "sha256": sha256_path(path),
        "episode_idx": int(meta.get("episode_idx", -1)),
        "success": success,
        "violated": bool(meta.get("violated", False)),
        "terminal_stable": terminal_stable,
        "terminal_failures": terminal_failures,
        "target_motion_m": target_motion,
        "occupant_displacement_m": occupant_displacement,
        "outcome": outcome,
        "body_metrics": body,
    }


def summarize_condition(
    condition: str,
    directory: str | Path,
    *,
    expected_count: int,
) -> dict:
    files = _trajectory_files(directory)
    if len(files) != expected_count:
        raise ValueError(
            f"{condition} has {len(files)} trajectories, expected {expected_count}"
        )
    episodes = [_episode(path, condition) for path in files]
    if len({item["episode_idx"] for item in episodes}) != len(episodes):
        raise ValueError(f"{condition} has duplicate episode indices")
    successes = sum(
        item["success"] and item["terminal_stable"] for item in episodes
    )
    direct = sum(item["outcome"] == "direct_completion" for item in episodes)
    repairs = sum(item["outcome"] == "repair_then_complete" for item in episodes)
    target_activated = sum(
        item["target_motion_m"] >= MIN_TARGET_MOTION_M for item in episodes
    )
    return {
        "condition": condition,
        "count": len(episodes),
        "stable_successes": successes,
        "stable_success_rate": successes / len(episodes),
        "direct_completions": direct,
        "repair_then_complete": repairs,
        "target_activations": target_activated,
        "episodes": episodes,
    }


def native_capability(
    native_dir: str | Path,
    *,
    expected_count: int,
    minimum_successes: int,
) -> dict:
    native = summarize_condition(
        "native", native_dir, expected_count=expected_count
    )
    passed = native["stable_successes"] >= minimum_successes
    result = {
        "scenario": SCENE_ID,
        "verdict": PASS_NATIVE if passed else "FAIL_L3B_MOKA_NATIVE_CAPABILITY",
        "minimum_successes": minimum_successes,
        "native": native,
        "failures": (
            []
            if passed
            else [
                "native stable successes "
                f"{native['stable_successes']} < {minimum_successes}"
            ]
        ),
    }
    return result


def control_capability(
    far_dir: str | Path,
    *,
    expected_count: int,
    minimum_successes: int,
) -> dict:
    control = summarize_condition(
        "far_first", far_dir, expected_count=expected_count
    )
    passed = control["stable_successes"] >= minimum_successes
    return {
        "scenario": SCENE_ID,
        "scene": "Ec",
        "verdict": (
            PASS_CONTROL
            if passed
            else "FAIL_L3B_MOKA_EC_CAPABILITY_CONTROL"
        ),
        "minimum_successes": minimum_successes,
        "control": control,
        "failures": (
            []
            if passed
            else [
                "Ec stable successes "
                f"{control['stable_successes']} < {minimum_successes}"
            ]
        ),
    }


def bind_preregistration(
    result: dict,
    preregistration_path: str | Path,
    *,
    expected_count: int,
    minimum_successes: int,
) -> None:
    path = Path(preregistration_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    pool = record.get("pool", {})
    acceptance = record.get("acceptance", {})
    if (
        record.get("status") != "LOCKED_BEFORE_NATIVE20_RUN"
        or record.get("scope") != "native_capability_screen_only"
        or pool.get("count") != expected_count
        or pool.get("official_state_indices") != list(range(expected_count))
        or acceptance.get("minimum_stable_successes") != minimum_successes
    ):
        raise ValueError(
            "capability arguments differ from the locked preregistration"
        )
    result["preregistration"] = {
        "preregistration_id": record.get("preregistration_id"),
        "path": str(path),
        "sha256": sha256_path(path),
    }


def summarize(
    native_dir: str | Path,
    near_dir: str | Path,
    far_dir: str | Path,
    *,
    expected_count: int,
    minimum_control_successes: int = 3,
    minimum_native_successes: int | None = None,
) -> dict:
    conditions = {
        "native": summarize_condition(
            "native", native_dir, expected_count=expected_count
        ),
        "near_first": summarize_condition(
            "near_first", near_dir, expected_count=expected_count
        ),
        "far_first": summarize_condition(
            "far_first", far_dir, expected_count=expected_count
        ),
    }
    return _paired_diagnostic(
        conditions,
        expected_count=expected_count,
        minimum_control_successes=minimum_control_successes,
    )


def paired_smoke(
    near_dir: str | Path,
    far_dir: str | Path,
    *,
    expected_count: int,
    minimum_control_successes: int = 3,
) -> dict:
    """Summarize Er/Ec without rerunning the descriptive native baseline."""
    conditions = {
        "near_first": summarize_condition(
            "near_first", near_dir, expected_count=expected_count
        ),
        "far_first": summarize_condition(
            "far_first", far_dir, expected_count=expected_count
        ),
    }
    return _paired_diagnostic(
        conditions,
        expected_count=expected_count,
        minimum_control_successes=minimum_control_successes,
    )


def _paired_diagnostic(
    conditions: dict,
    *,
    expected_count: int,
    minimum_control_successes: int,
) -> dict:
    control_pass = (
        conditions["far_first"]["stable_successes"]
        >= minimum_control_successes
    )
    near_rate = conditions["near_first"]["stable_success_rate"]
    far_rate = conditions["far_first"]["stable_success_rate"]
    gap = far_rate - near_rate
    repair_gap = (
        conditions["near_first"]["repair_then_complete"]
        - conditions["far_first"]["repair_then_complete"]
    ) / expected_count
    if not control_pass:
        candidate_status = "INCONCLUSIVE_EC_CAPABILITY_CONTROL_FAIL"
    elif gap >= 0.4 or repair_gap >= 0.4:
        candidate_status = "POTENTIAL_ORDER_EFFECT_REQUIRES_FORMAL_REVIEW"
    else:
        candidate_status = "NO_LARGE_ORDER_EFFECT_IN_SMOKE"
    failures = []
    if not control_pass:
        failures.append(
            "Ec stable successes "
            f"{conditions['far_first']['stable_successes']} "
            f"< {minimum_control_successes}"
        )
    result = {
        "scenario": SCENE_ID,
        "diagnostic_only": True,
        "formal_authorized": False,
        "verdict": PASS_SMOKE if not failures else "FAIL_L3B_MOKA_POLICY_SMOKE",
        "candidate_status": candidate_status,
        "scene_labels": {
            "Er": "near_first",
            "Ec": "far_first",
        },
        "preregistered_thresholds": {
            "minimum_ec_control_successes": minimum_control_successes,
            "eb_native_is_descriptive_not_a_gate": True,
            "minimum_target_motion_m": MIN_TARGET_MOTION_M,
            "direct_occupant_max_displacement_m": (
                DIRECT_OCCUPANT_MAX_DISPLACEMENT_M
            ),
            "potential_order_effect_absolute_rate_gap": 0.4,
            "terminal_window_steps": TERMINAL_WINDOW_STEPS,
            "terminal_max_tilt_deg": MAX_RECEPTACLE_TILT_DEG,
            "terminal_max_drift_m": TERMINAL_MAX_DRIFT_M,
        },
        "contrasts": {
            "Ec_minus_Er_stable_success_rate": gap,
            "Er_minus_Ec_repair_rate": repair_gap,
        },
        "conditions": conditions,
        "failures": failures,
    }
    if "native" in conditions:
        result["scene_labels"] = {
            "Eb": "native",
            **result["scene_labels"],
        }
    else:
        result["native_baseline"] = (
            "omitted_from_paired_summary; Eb is descriptive and is not a gate"
        )
    return result


def _write_result(path: str | Path | None, result: dict) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native")
    parser.add_argument("--near-first")
    parser.add_argument("--far-first")
    parser.add_argument("--expected-count", type=int, default=5)
    parser.add_argument("--minimum-native-successes", type=int, default=3)
    parser.add_argument("--minimum-control-successes", type=int, default=3)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--paired-only", action="store_true")
    parser.add_argument("--preregistration")
    parser.add_argument("--out-json")
    args = parser.parse_args()
    try:
        if sum((args.native_only, args.control_only, args.paired_only)) > 1:
            raise ValueError(
                "--native-only, --control-only, and --paired-only are "
                "exclusive"
            )
        if args.native_only:
            if not args.native:
                raise ValueError("--native is required with --native-only")
            result = native_capability(
                args.native,
                expected_count=args.expected_count,
                minimum_successes=args.minimum_native_successes,
            )
            if args.preregistration:
                bind_preregistration(
                    result,
                    args.preregistration,
                    expected_count=args.expected_count,
                    minimum_successes=args.minimum_native_successes,
                )
        elif args.control_only:
            if not args.far_first:
                raise ValueError("--far-first is required with --control-only")
            result = control_capability(
                args.far_first,
                expected_count=args.expected_count,
                minimum_successes=args.minimum_control_successes,
            )
        elif args.paired_only:
            if not args.near_first or not args.far_first:
                raise ValueError(
                    "--near-first and --far-first are required with "
                    "--paired-only"
                )
            result = paired_smoke(
                args.near_first,
                args.far_first,
                expected_count=args.expected_count,
                minimum_control_successes=args.minimum_control_successes,
            )
        else:
            if not args.native or not args.near_first or not args.far_first:
                raise ValueError(
                    "--native, --near-first, and --far-first are required "
                    "for the full summary"
                )
            result = summarize(
                args.native,
                args.near_first,
                args.far_first,
                expected_count=args.expected_count,
                minimum_control_successes=args.minimum_control_successes,
            )
    except Exception as exc:
        result = {
            "scenario": SCENE_ID,
            "verdict": "FAIL_L3B_MOKA_SMOKE_EVIDENCE",
            "error": str(exc),
        }
        _write_result(args.out_json, result)
        print(result["verdict"])
        raise SystemExit(1)
    _write_result(args.out_json, result)
    print(result["verdict"])
    if not args.native_only and not args.control_only:
        print(result["candidate_status"])
    if str(result.get("verdict", "")).startswith("FAIL_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
