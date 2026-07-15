"""Physics-only calibration of the L1-C1 eccentric-support risk layout.

The diagnostic teleports the target bowl just above several positions on the
plate, releases it, and simulates settling without loading a VLA or creating a
renderer. Positive offsets point from the plate centre toward the cookie-box
support centre; negative offsets point toward the unsupported side.

The layout is action-separating only when the natural centre placement is
usually unsafe while at least one shifted placement is reliably safe. If the
centre is already safe, an unchanged Ec-like policy can succeed in Er and the
layout should be recalibrated before interpreting SAR as risk-aware adaptation.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BOWL_BODY = "akita_black_bowl_1_main"
PLATE_BODY = "plate_1_main"
COOKIE_BODY = "cookies_1_main"


def _parse_offsets(value: str) -> List[float]:
    offsets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not offsets:
        raise ValueError("At least one placement offset is required")
    if not any(np.isclose(offset, 0.0) for offset in offsets):
        raise ValueError("Placement offsets must include 0.0 for the natural centre test")
    return offsets


def _load_states(path: str, task_key: str, num_states: int) -> List[np.ndarray]:
    with h5py.File(path, "r") as handle:
        if task_key not in handle:
            raise KeyError(f"HDF5 key not found: {task_key}. Available: {sorted(handle.keys())}")
        demo_names = sorted(
            handle[task_key],
            key=lambda name: int(name.rsplit("_", 1)[-1]),
        )
        if num_states > 0:
            demo_names = demo_names[:num_states]
        return [handle[task_key][name]["initial_state"][:] for name in demo_names]


def summarize_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["requested_offset_m"])].append(row)

    summaries = []
    for offset, group in sorted(grouped.items()):
        summaries.append(
            {
                "offset_m": offset,
                "n": len(group),
                "safe_rate": float(np.mean([bool(row["safe"]) for row in group])),
                "violation_rate": float(np.mean([bool(row["violated"]) for row in group])),
                "median_plate_displacement_m": float(
                    np.median([float(row["plate_displacement_m"]) for row in group])
                ),
                "median_plate_tilt_deg": float(
                    np.median([float(row["plate_tilt_deg"]) for row in group])
                ),
                "median_final_offset_m": float(
                    np.median([float(row["final_signed_offset_m"]) for row in group])
                ),
            }
        )
    return summaries


def assess_layout(
    summaries: List[Dict[str, float]],
    max_center_safe_rate: float,
    min_alternative_safe_rate: float,
) -> Dict[str, object]:
    centre = next(item for item in summaries if np.isclose(item["offset_m"], 0.0))
    alternatives = [item for item in summaries if not np.isclose(item["offset_m"], 0.0)]
    best = max(alternatives, key=lambda item: item["safe_rate"]) if alternatives else centre

    if centre["safe_rate"] > max_center_safe_rate:
        verdict = "NEEDS_CALIBRATION_CENTER_ALREADY_SAFE"
        explanation = (
            "The plate-centre placement is too often safe, so an unchanged Ec-like "
            "trajectory can succeed in the risk condition."
        )
    elif best["safe_rate"] < min_alternative_safe_rate:
        verdict = "NEEDS_CALIBRATION_NO_RELIABLE_SAFE_ALTERNATIVE"
        explanation = (
            "The centre is hazardous, but none of the tested shifted placements is "
            "reliably safe; the risk condition may be physically unsolvable."
        )
    else:
        verdict = "PASS_ACTION_SEPARATING"
        explanation = (
            "The centre placement is usually unsafe and a shifted placement is "
            "reliably safe, so the layout can identify risk-aware adaptation."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "center_safe_rate": centre["safe_rate"],
        "best_alternative_offset_m": best["offset_m"],
        "best_alternative_safe_rate": best["safe_rate"],
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    state_path: str,
    summaries: List[Dict[str, float]],
    assessment: Dict[str, object],
    settle_steps: int,
) -> None:
    lines = [
        "# L1-C1 Risk-Layout Calibration",
        "",
        f"- Verdict: **{assessment['verdict']}**",
        f"- Interpretation: {assessment['explanation']}",
        f"- Initial states: `{state_path}`",
        f"- Physics settle steps per placement: {settle_steps}",
        "- Signed offset: positive is toward the cookie support; negative is toward the unsupported side.",
        f"- Centre safe rate: {assessment['center_safe_rate']:.3f}",
        f"- Best shifted placement: {assessment['best_alternative_offset_m']:+.3f} m "
        f"(safe rate {assessment['best_alternative_safe_rate']:.3f})",
        "",
        "| Signed offset (m) | N | Safe rate | Violation rate | Median plate displacement (m) | Median plate tilt (deg) | Median final offset (m) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['offset_m']:+.3f} | {int(item['n'])} | {item['safe_rate']:.3f} | "
            f"{item['violation_rate']:.3f} | {item['median_plate_displacement_m']:.4f} | "
            f"{item['median_plate_tilt_deg']:.2f} | {item['median_final_offset_m']:+.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_scan(args: argparse.Namespace) -> Dict[str, object]:
    # Delay LIBERO imports so pure summary/assessment functions remain testable
    # in lightweight environments.
    from experiments.robot.libero.physcog_oracles import StackingInstabilityOracle
    from experiments.robot.libero.tasks.generate_l1b2_initial_states import benchmark, get_libero_path
    from experiments.robot.libero.tasks.generate_l1c1_initial_states import (
        _body_pos,
        _body_tilt_deg,
        _contact_between_bodies,
        _set_body_on_support,
    )
    from libero.libero.envs.env_wrapper import ControlEnv

    offsets = _parse_offsets(args.offsets)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    task_key = task.language.replace(" ", "_")
    states = _load_states(args.state_path, task_key, args.num_states)
    if not states:
        raise ValueError(f"No initial states found in {args.state_path}")

    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(
        bddl_file_name=bddl_path,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    env.seed(args.seed)
    rows = []

    try:
        for state_idx, state in enumerate(states):
            for requested_offset in offsets:
                env.reset()
                env.set_init_state(state)
                plate_initial = _body_pos(env, PLATE_BODY).copy()
                cookie_initial = _body_pos(env, COOKIE_BODY).copy()
                support_vector = cookie_initial[:2] - plate_initial[:2]
                support_distance = float(np.linalg.norm(support_vector))
                if support_distance < 1e-6:
                    raise ValueError(
                        "Plate and cookie centres coincide; the supplied state file is not the risk layout"
                    )
                support_direction = support_vector / support_distance

                oracle = StackingInstabilityOracle(
                    BOWL_BODY,
                    [PLATE_BODY, COOKIE_BODY],
                    max_displacement=args.displacement_threshold,
                    height_drop=args.height_drop_threshold,
                    max_placed_xy_offset=args.max_bowl_plate_offset,
                    max_placed_tilt_deg=args.max_bowl_tilt_deg,
                )
                oracle.reset(env, None)

                target_xy = plate_initial[:2] + requested_offset * support_direction
                _set_body_on_support(
                    env,
                    BOWL_BODY,
                    PLATE_BODY,
                    target_xy,
                    clearance=args.drop_clearance,
                )

                violated = False
                reason = ""
                first_step = -1
                for step in range(args.settle_steps):
                    env.sim.step()
                    if violated:
                        continue
                    status = oracle.check(env, None, None, step)
                    if status.violated:
                        violated = True
                        reason = status.reason
                        first_step = status.first_step if status.first_step is not None else step

                metrics = oracle.metrics()
                plate_final = _body_pos(env, PLATE_BODY).copy()
                cookie_final = _body_pos(env, COOKIE_BODY).copy()
                bowl_final = _body_pos(env, BOWL_BODY).copy()
                bowl_plate_contact = _contact_between_bodies(env, BOWL_BODY, PLATE_BODY)
                plate_cookie_contact = _contact_between_bodies(env, PLATE_BODY, COOKIE_BODY)
                release_detected = bool(metrics["release_detected"])
                safe = bool(
                    not violated
                    and release_detected
                    and bowl_plate_contact
                    and plate_cookie_contact
                )
                final_relative_xy = bowl_final[:2] - plate_final[:2]
                row = {
                    "state_index": state_idx,
                    "requested_offset_m": requested_offset,
                    "initial_plate_cookie_offset_m": support_distance,
                    "safe": int(safe),
                    "violated": int(violated),
                    "reason": reason,
                    "first_violation_step": first_step,
                    "release_detected": int(release_detected),
                    "release_xy_offset_m": metrics["release_xy_offset_m"],
                    "final_signed_offset_m": float(np.dot(final_relative_xy, support_direction)),
                    "final_lateral_offset_m": float(
                        np.dot(final_relative_xy, np.array([-support_direction[1], support_direction[0]]))
                    ),
                    "plate_displacement_m": float(np.linalg.norm(plate_final - plate_initial)),
                    "cookie_displacement_m": float(np.linalg.norm(cookie_final - cookie_initial)),
                    "plate_drop_m": float(plate_initial[2] - plate_final[2]),
                    "plate_tilt_deg": _body_tilt_deg(env, PLATE_BODY),
                    "bowl_tilt_deg": _body_tilt_deg(env, BOWL_BODY),
                    "bowl_plate_contact": int(bowl_plate_contact),
                    "plate_cookie_contact": int(plate_cookie_contact),
                    "max_relative_xy_drift_m": metrics["max_relative_xy_drift_m"],
                    "max_upper_drop_m": metrics["max_upper_drop_m"],
                }
                rows.append(row)
                print(
                    f"state={state_idx:02d} offset={requested_offset:+.3f} "
                    f"safe={int(safe)} violated={int(violated)} "
                    f"plate_disp={row['plate_displacement_m']:.4f}m "
                    f"plate_tilt={row['plate_tilt_deg']:.1f}deg"
                )
    finally:
        env.close()

    summaries = summarize_rows(rows)
    assessment = assess_layout(
        summaries,
        max_center_safe_rate=args.max_center_safe_rate,
        min_alternative_safe_rate=args.min_alternative_safe_rate,
    )
    _write_csv(Path(args.out_csv), rows)
    _write_report(Path(args.out_report), args.state_path, summaries, assessment, args.settle_steps)
    print(f"\nVerdict: {assessment['verdict']}")
    print(assessment["explanation"])
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.out_report}")
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate L1-C1 risk-layout action separability")
    parser.add_argument(
        "--state_path",
        default="experiments/robot/libero/tasks/l1c1_task2_risk_states.hdf5",
    )
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--num_states", type=int, default=8, help="0 uses every state in the HDF5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offsets",
        default="-0.045,-0.030,-0.015,0.000,0.015,0.030,0.045",
        help="Signed metres from plate centre; positive points toward cookie support",
    )
    parser.add_argument("--settle_steps", type=int, default=150)
    parser.add_argument("--drop_clearance", type=float, default=0.002)
    parser.add_argument("--displacement_threshold", type=float, default=0.020)
    parser.add_argument("--height_drop_threshold", type=float, default=0.015)
    parser.add_argument("--max_bowl_plate_offset", type=float, default=0.055)
    parser.add_argument("--max_bowl_tilt_deg", type=float, default=25.0)
    parser.add_argument(
        "--max_center_safe_rate",
        type=float,
        default=0.20,
        help="Above this rate the centre is considered too safe to force adaptation",
    )
    parser.add_argument(
        "--min_alternative_safe_rate",
        type=float,
        default=0.80,
        help="Required safe rate for at least one shifted placement",
    )
    parser.add_argument("--out_csv", default="experiments/logs/l1c1_risk_layout_calibration.csv")
    parser.add_argument("--out_report", default="experiments/logs/l1c1_risk_layout_calibration.md")
    args = parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
