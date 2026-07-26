#!/usr/bin/env python3
"""Sweep L3-A4 A/B/C geometry against the compiled native drawer front.

This calibration is raw MuJoCo only: no policy model and no camera rendering.
Each candidate must pass Er's ordered drawer->A->B->C response and Ec's
preserved upstream drawer->A->B with C parked, while initial/open-hold and
direct-bypass gates remain valid. A candidate is selected only if its paired
eligibility rate is at least 80% over the requested reset trials.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import experiments.robot.libero.physcog_objects as physcog_objects
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.generate_l3a4_momentum_states import (
    _body_pos,
    _set_free_pose,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a4_momentum import (
    A_BODY,
    B_BODY,
    C_BODY,
    CHAIN_BODIES,
    CLOSE_STEPS,
    DEFAULT_BDDL,
    DRAWER_BODY_CANDIDATES,
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT_CANDIDATES,
    EC_SENTINEL_PARK_DXY,
    SETTLE_STEPS,
    assess_chain,
    capture_frame,
    find_body,
    find_joint_qadr,
)
from experiments.robot.libero.tasks.validate_l3a4_scene import (
    _hold_gate,
    _initial_contact_gate,
)


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _place(env, drawer_body, candidate):
    origin = _body_pos(env, drawer_body)[:2]
    offsets = {
        A_BODY: np.asarray([candidate["a_dx"], candidate["a_dy"]]),
        B_BODY: np.asarray([
            candidate["a_dx"] + candidate["b_dx_from_a"],
            candidate["a_dy"] + candidate["ab_spacing"],
        ]),
        C_BODY: np.asarray([
            candidate["a_dx"] + candidate["b_dx_from_a"],
            candidate["a_dy"]
            + candidate["ab_spacing"]
            + candidate["bc_spacing"],
        ]),
    }
    for body, offset in offsets.items():
        _set_free_pose(env, body, origin + offset)
    env.sim.forward()
    return offsets


def _script_close_raw(env, drawer_qadr, close_steps):
    start = float(env.sim.data.qpos[drawer_qadr])
    frames = [capture_frame(env, 0, drawer_qadr)]
    for step, qpos in enumerate(
        np.linspace(start, DRAWER_CLOSED_QPOS, close_steps + 1)[1:], start=1
    ):
        env.sim.data.qpos[drawer_qadr] = float(qpos)
        env.sim.forward()
        env.sim.step()
        frames.append(capture_frame(env, step, drawer_qadr))
    return frames


def _compiled_drawer_report(env, drawer_body, drawer_qadr):
    model, data = env.sim.model, env.sim.data
    role_geoms = physcog_objects.resolve_l3a1_native_corner_edge_geoms(
        env, drawer_body
    )
    front_name = role_geoms["edge/front_outer"]
    front_id = int(model.geom_name2id(front_name))
    start_qpos = float(data.qpos[drawer_qadr])
    open_body = _body_pos(env, drawer_body)
    open_geom = np.asarray(data.geom_xpos[front_id], dtype=float).copy()
    data.qpos[drawer_qadr] = DRAWER_CLOSED_QPOS
    env.sim.forward()
    closed_body = _body_pos(env, drawer_body)
    closed_geom = np.asarray(data.geom_xpos[front_id], dtype=float).copy()
    data.qpos[drawer_qadr] = start_qpos
    env.sim.forward()
    return {
        "drawer_body": drawer_body,
        "drawer_qpos_open": start_qpos,
        "drawer_qpos_closed": DRAWER_CLOSED_QPOS,
        "drawer_body_open_xyz": open_body.tolist(),
        "drawer_body_closed_xyz": closed_body.tolist(),
        "drawer_motion_xyz": (closed_body - open_body).tolist(),
        "drawer_motion_m": float(np.linalg.norm(closed_body - open_body)),
        "front_geom": front_name,
        "front_geom_open_xyz": open_geom.tolist(),
        "front_geom_closed_xyz": closed_geom.tolist(),
        "front_geom_motion_xyz": (closed_geom - open_geom).tolist(),
        "front_geom_size": np.asarray(model.geom_size[front_id]).tolist(),
        "front_geom_rbound": float(model.geom_rbound[front_id]),
        "front_geom_xmat": np.asarray(data.geom_xmat[front_id]).reshape(3, 3).tolist(),
    }


def _run_condition(env, state, drawer_body, drawer_qadr, condition, args):
    env.reset()
    env.set_init_state(state)
    clear_mujoco_replay_transients(env)
    if condition == "stable":
        from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
            _find_free_joint_qadr,
        )
        c_qadr = _find_free_joint_qadr(env.sim, C_BODY)
        env.sim.data.qpos[c_qadr:c_qadr + 2] += EC_SENTINEL_PARK_DXY
        env.sim.forward()
        state = np.asarray(env.sim.get_state().flatten()).copy()
    contact_pass, contact_reasons = _initial_contact_gate(
        env, drawer_body, condition
    )
    hold = _hold_gate(env, drawer_qadr, args.open_hold_steps)
    env.reset()
    env.set_init_state(state)
    clear_mujoco_replay_transients(env)
    trace = _script_close_raw(env, drawer_qadr, args.close_steps)
    assessment = assess_chain(
        trace, condition=condition, drawer_body=drawer_body
    )
    passed = contact_pass and hold["passed"] and assessment.passed
    return {
        "passed": passed,
        "contact_pass": contact_pass,
        "contact_reasons": contact_reasons,
        "hold": hold,
        "assessment": assessment.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--a_dx", default="0.138,0.140,0.142,0.144")
    parser.add_argument("--a_dy", default="-0.044,-0.040,-0.036")
    parser.add_argument("--b_dx_from_a", default="0.004")
    parser.add_argument("--ab_spacing", default="0.054")
    parser.add_argument("--bc_spacing", default="0.038")
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.environ.get("L3A4_SWEEP_TRIALS", "5")),
    )
    parser.add_argument("--min_rate", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--open_hold_steps", type=int, default=120)
    parser.add_argument("--close_steps", type=int, default=CLOSE_STEPS)
    parser.add_argument("--out_csv", default="experiments/logs/l3a4_geometry_sweep.csv")
    parser.add_argument("--out_report", default="experiments/logs/l3a4_geometry_sweep.md")
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    env.reset()
    drawer_body = find_body(env, DRAWER_BODY_CANDIDATES)
    drawer_joint, drawer_qadr = find_joint_qadr(env.sim, DRAWER_JOINT_CANDIDATES)
    compiled = _compiled_drawer_report(env, drawer_body, drawer_qadr)

    candidates = [
        {
            "a_dx": a_dx,
            "a_dy": a_dy,
            "b_dx_from_a": b_dx,
            "ab_spacing": ab,
            "bc_spacing": bc,
        }
        for a_dx, a_dy, b_dx, ab, bc in itertools.product(
            _floats(args.a_dx),
            _floats(args.a_dy),
            _floats(args.b_dx_from_a),
            _floats(args.ab_spacing),
            _floats(args.bc_spacing),
        )
    ]
    rows = []
    try:
        for candidate_index, candidate in enumerate(candidates):
            trial_rows = []
            for trial in range(args.trials):
                env.seed(args.seed + trial)
                env.reset()
                _place(env, drawer_body, candidate)
                for _ in range(args.settle_steps):
                    env.sim.step()
                risk_state = np.asarray(env.sim.get_state().flatten()).copy()
                risk = _run_condition(
                    env, risk_state, drawer_body, drawer_qadr, "risk", args
                )
                stable = _run_condition(
                    env, risk_state, drawer_body, drawer_qadr, "stable", args
                )
                paired_pass = bool(risk["passed"] and stable["passed"])
                trial_rows.append({
                    "trial": trial,
                    "paired_pass": paired_pass,
                    "risk": risk,
                    "stable": stable,
                })
            rate = sum(row["paired_pass"] for row in trial_rows) / len(trial_rows)
            row = {
                "candidate": candidate_index,
                **candidate,
                "trials": args.trials,
                "paired_passes": sum(row["paired_pass"] for row in trial_rows),
                "paired_rate": rate,
                "eligible": rate >= args.min_rate,
                "details_json": json.dumps(trial_rows, sort_keys=True),
            }
            rows.append(row)
            print(
                "L3A4_SWEEP "
                f"candidate={candidate_index} rate={rate:.3f} "
                f"eligible={row['eligible']} geometry={candidate}"
            )
    finally:
        env.close()

    rows.sort(key=lambda row: (-row["paired_rate"], row["candidate"]))
    eligible = [row for row in rows if row["eligible"]]
    selected = eligible[0] if eligible else None
    verdict = (
        "PASS_L3A4_GEOMETRY_SWEEP"
        if selected is not None else "FAIL_L3A4_GEOMETRY_SWEEP"
    )
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = Path(args.out_report)
    report.write_text(
        "\n".join([
            "# L3-A4 compiled-geometry momentum sweep",
            "",
            f"- Verdict: **{verdict}**",
            f"- Candidates: {len(rows)}",
            f"- Trials per candidate: {args.trials}",
            f"- Required paired rate: {args.min_rate:.1%}",
            f"- Eligible candidates: {len(eligible)}",
            f"- Selected: `{json.dumps(selected, sort_keys=True) if selected else 'none'}`",
            "",
            "## Compiled native drawer geometry",
            "",
            "```json",
            json.dumps(compiled, indent=2, sort_keys=True),
            "```",
        ]) + "\n"
    )
    print(
        f"{verdict} eligible={len(eligible)} selected="
        f"{selected['candidate'] if selected else 'none'} report={report}"
    )
    if args.fail_on_invalid and selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
