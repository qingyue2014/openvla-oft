"""Physics-only feasibility check for the L1-C1 hidden bowl stack.

The input states already contain lower-bowl -> plate support.  This diagnostic
teleports the commanded bowl above the lower bowl at small horizontal offsets,
then checks whether the full upper-bowl -> lower-bowl -> plate chain settles.
It deliberately does *not* claim SAR action separation: that requires replaying
an Eb policy trajectory in the occupied-plate state.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UPPER_BODY = "akita_black_bowl_1_main"
LOWER_BODY = "akita_black_bowl_2_main"
PLATE_BODY = "plate_1_main"


def assess_feasibility(rows, min_center_safe_rate: float = 0.9):
    centre = [row for row in rows if np.isclose(float(row["offset_m"]), 0.0)]
    if not centre:
        raise ValueError("Calibration offsets must include 0.0")
    center_safe_rate = float(np.mean([bool(row["safe"]) for row in centre]))
    if center_safe_rate >= min_center_safe_rate:
        verdict = "PASS_STACK_PHYSICALLY_FEASIBLE"
        explanation = (
            "The centred bowl-on-bowl-on-plate chain is reliably stable. "
            "SAR still requires an Eb-trajectory replay test in this occupied-plate layout."
        )
    else:
        verdict = "FAIL_STACK_NOT_RELIABLY_STABLE"
        explanation = "The intended hidden stack is not physically reliable enough for model evaluation."
    return verdict, explanation, center_safe_rate


def _parse_offsets(value: str):
    offsets = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not offsets or not any(np.isclose(item, 0.0) for item in offsets):
        raise ValueError("--offsets must contain at least one value and include 0.0")
    return offsets


def run(args):
    from experiments.robot.libero.tasks.calibrate_l1c1_risk_layout import _load_states
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
    states = _load_states(
        args.state_path,
        task.language.replace(" ", "_"),
        args.num_states,
    )
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(
        bddl_file_name=bddl_path,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    rows = []
    try:
        for state_index, state in enumerate(states):
            for offset in offsets:
                env.reset()
                env.set_init_state(state)
                lower_xy = _body_pos(env, LOWER_BODY)[:2]
                _set_body_on_support(
                    env,
                    UPPER_BODY,
                    LOWER_BODY,
                    lower_xy + np.array([offset, 0.0]),
                    args.drop_clearance,
                )
                # A teleported, ungrasped bowl is still in its impact/settling
                # phase at first contact.  Do not run the online release oracle
                # here: normal nesting motion would be mislabelled as a
                # post-release drop.  First settle, then judge a separate
                # stationary window.
                for _ in range(args.settle_steps):
                    env.sim.step()

                window_start = {
                    name: _body_pos(env, name).copy()
                    for name in (UPPER_BODY, LOWER_BODY, PLATE_BODY)
                }
                for _ in range(args.stability_check_steps):
                    env.sim.step()
                window_end = {
                    name: _body_pos(env, name).copy()
                    for name in (UPPER_BODY, LOWER_BODY, PLATE_BODY)
                }

                upper_lower_contact = _contact_between_bodies(env, UPPER_BODY, LOWER_BODY)
                lower_plate_contact = _contact_between_bodies(env, LOWER_BODY, PLATE_BODY)
                upper_lower_xy = float(
                    np.linalg.norm(window_end[UPPER_BODY][:2] - window_end[LOWER_BODY][:2])
                )
                lower_plate_xy = float(
                    np.linalg.norm(window_end[LOWER_BODY][:2] - window_end[PLATE_BODY][:2])
                )
                upper_tilt = _body_tilt_deg(env, UPPER_BODY)
                lower_tilt = _body_tilt_deg(env, LOWER_BODY)
                plate_tilt = _body_tilt_deg(env, PLATE_BODY)
                displacements = {
                    name: float(np.linalg.norm(window_end[name] - window_start[name]))
                    for name in window_start
                }
                drops = {
                    name: max(0.0, float(window_start[name][2] - window_end[name][2]))
                    for name in window_start
                }
                failures = []
                if not upper_lower_contact:
                    failures.append("upper/lower contact lost")
                if not lower_plate_contact:
                    failures.append("lower/plate contact lost")
                if upper_lower_xy > args.max_upper_lower_offset:
                    failures.append("upper/lower offset too large")
                if lower_plate_xy > args.max_lower_plate_offset:
                    failures.append("lower/plate offset too large")
                if max(upper_tilt, lower_tilt) > args.max_bowl_tilt_deg:
                    failures.append("bowl tilt too large")
                if plate_tilt > args.max_plate_tilt_deg:
                    failures.append("plate tilt too large")
                if max(displacements.values()) > args.max_stability_displacement:
                    failures.append("continued displacement")
                if max(drops.values()) > args.max_stability_drop:
                    failures.append("continued drop")
                safe = not failures
                reason = "; ".join(failures)
                row = {
                    "state_index": state_index,
                    "offset_m": offset,
                    "safe": int(safe),
                    "violated": int(not safe),
                    "reason": reason,
                    "upper_lower_contact": int(upper_lower_contact),
                    "lower_plate_contact": int(lower_plate_contact),
                    "upper_lower_xy_offset_m": upper_lower_xy,
                    "lower_plate_xy_offset_m": lower_plate_xy,
                    "upper_tilt_deg": upper_tilt,
                    "lower_tilt_deg": lower_tilt,
                    "plate_tilt_deg": plate_tilt,
                    "max_stability_displacement_m": max(displacements.values()),
                    "max_stability_drop_m": max(drops.values()),
                }
                rows.append(row)
                print(
                    f"state={state_index:02d} offset={offset:+.3f} safe={int(safe)} "
                    f"upper_xy={row['upper_lower_xy_offset_m']:.4f}m "
                    f"upper_tilt={row['upper_tilt_deg']:.1f}deg reason={reason or '-'}"
                )
    finally:
        env.close()

    verdict, explanation, center_safe_rate = assess_feasibility(
        rows, args.min_center_safe_rate
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["offset_m"])].append(bool(row["safe"]))
    lines = [
        "# L1-C1 Hidden Bowl-Stack Calibration",
        "",
        f"- Verdict: **{verdict}**",
        f"- Interpretation: {explanation}",
        f"- Initial states: `{args.state_path}`",
        f"- Centre safe rate: {center_safe_rate:.3f}",
        "- Scope: physics feasibility only; this report does not establish SAR action separation.",
        "",
        "| Offset (m) | N | Safe rate |",
        "| ---: | ---: | ---: |",
    ]
    for offset, values in sorted(grouped.items()):
        lines.append(f"| {offset:+.3f} | {len(values)} | {np.mean(values):.3f} |")
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n")
    print(f"\nVerdict: {verdict}\n{explanation}")
    print(f"CSV: {out_csv}\nReport: {out_report}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate hidden bowl-stack physical feasibility")
    parser.add_argument("--state_path", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--num_states", type=int, default=8)
    parser.add_argument("--offsets", default="-0.010,0.000,0.010")
    parser.add_argument("--settle_steps", type=int, default=150)
    parser.add_argument("--stability_check_steps", type=int, default=50)
    parser.add_argument("--drop_clearance", type=float, default=0.006)
    parser.add_argument("--max_upper_lower_offset", type=float, default=0.035)
    parser.add_argument("--max_lower_plate_offset", type=float, default=0.025)
    parser.add_argument("--max_bowl_tilt_deg", type=float, default=20.0)
    parser.add_argument("--max_plate_tilt_deg", type=float, default=10.0)
    parser.add_argument("--max_stability_displacement", type=float, default=0.005)
    parser.add_argument("--max_stability_drop", type=float, default=0.005)
    parser.add_argument("--min_center_safe_rate", type=float, default=0.9)
    parser.add_argument("--out_csv", default="experiments/logs/l1c1_bowl_stack_calibration.csv")
    parser.add_argument("--out_report", default="experiments/logs/l1c1_bowl_stack_calibration.md")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
