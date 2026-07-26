#!/usr/bin/env python3
"""Sweep L3-A4 A/B/C geometry against the compiled native drawer front.

This calibration is raw MuJoCo only: no policy model and no camera rendering.
Each candidate must pass Er's ordered drawer->A->B->C response, Ec's preserved
upstream drawer->A->B with C parked, and rA's absent endpoint response after A
is parked. Initial/open-hold and direct-bypass gates must also remain valid. A
candidate is selected only if its three-condition eligibility rate is at least
80% over the requested reset trials.
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
    DRAWER_TARGET_QPOS,
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


def _script_required_motion_raw(env, drawer_qadr, motion_steps):
    start = float(env.sim.data.qpos[drawer_qadr])
    frames = [capture_frame(env, 0, drawer_qadr)]
    for step, qpos in enumerate(
        np.linspace(start, DRAWER_TARGET_QPOS, motion_steps + 1)[1:], start=1
    ):
        env.sim.data.qpos[drawer_qadr] = float(qpos)
        env.sim.forward()
        env.sim.step()
        frames.append(capture_frame(env, step, drawer_qadr))
    return frames


def _compiled_drawer_report(env, drawer_body, drawer_qadr):
    model, data = env.sim.model, env.sim.data
    start_qpos = float(data.qpos[drawer_qadr])
    start_body = _body_pos(env, drawer_body)
    data.qpos[drawer_qadr] = DRAWER_TARGET_QPOS
    env.sim.forward()
    target_body = _body_pos(env, drawer_body)
    data.qpos[drawer_qadr] = start_qpos
    env.sim.forward()
    motion = target_body - start_body
    motion_norm = float(np.linalg.norm(motion))
    if motion_norm <= 1e-9:
        raise RuntimeError("native drawer target qpos produces no body motion")
    direction = motion / motion_norm
    drawer_body_id = int(model.body_name2id(drawer_body))
    candidates = []
    for geom_id in range(int(model.ngeom)):
        if (
            int(model.geom_bodyid[geom_id]) != drawer_body_id
            or int(model.geom_group[geom_id]) != 0
            or int(model.geom_contype[geom_id]) == 0
            or int(model.geom_conaffinity[geom_id]) == 0
            or int(model.geom_type[geom_id]) != 6  # mjGEOM_BOX
        ):
            continue
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        half_extent = float(
            np.sum(np.abs(rotation.T @ direction) * model.geom_size[geom_id])
        )
        center_projection = float(
            np.dot(np.asarray(data.geom_xpos[geom_id]) - start_body, direction)
        )
        candidates.append((center_projection + half_extent, geom_id))
    if not candidates:
        raise RuntimeError(f"no collidable box geoms found on {drawer_body}")
    _, front_id = max(candidates)
    front_name = str(model.geom_id2name(front_id))
    start_geom = np.asarray(data.geom_xpos[front_id], dtype=float).copy()
    start_xmat = np.asarray(data.geom_xmat[front_id], dtype=float).reshape(3, 3)
    data.qpos[drawer_qadr] = DRAWER_TARGET_QPOS
    env.sim.forward()
    target_geom = np.asarray(data.geom_xpos[front_id], dtype=float).copy()
    data.qpos[drawer_qadr] = start_qpos
    env.sim.forward()
    return {
        "drawer_body": drawer_body,
        "drawer_qpos_start": start_qpos,
        "drawer_qpos_target": DRAWER_TARGET_QPOS,
        "drawer_body_start_xyz": start_body.tolist(),
        "drawer_body_target_xyz": target_body.tolist(),
        "drawer_motion_xyz": motion.tolist(),
        "drawer_motion_m": motion_norm,
        "front_geom": front_name,
        "front_geom_start_xyz": start_geom.tolist(),
        "front_geom_target_xyz": target_geom.tolist(),
        "front_geom_motion_xyz": (target_geom - start_geom).tolist(),
        "front_geom_size": np.asarray(model.geom_size[front_id]).tolist(),
        "front_geom_rbound": float(model.geom_rbound[front_id]),
        "front_geom_xmat": start_xmat.tolist(),
    }


def _trajectory_diagnostics(trace):
    """Summarize directions without weakening the contact-based causal gate."""
    initial = trace[0]
    result = {}
    for body in CHAIN_BODIES:
        p0 = np.asarray(initial.positions[body], dtype=float)
        displacements = [
            np.asarray(frame.positions[body], dtype=float) - p0 for frame in trace
        ]
        max_displacement_index = int(
            np.argmax([np.linalg.norm(delta) for delta in displacements])
        )
        max_speed_index = int(
            np.argmax([float(frame.speeds_m_s[body]) for frame in trace])
        )
        result[body] = {
            "max_displacement_step": int(trace[max_displacement_index].step),
            "max_displacement_xyz": displacements[
                max_displacement_index
            ].tolist(),
            "max_speed_step": int(trace[max_speed_index].step),
            "position_at_max_speed_xyz": np.asarray(
                trace[max_speed_index].positions[body], dtype=float
            ).tolist(),
            "displacement_at_max_speed_xyz": displacements[
                max_speed_index
            ].tolist(),
        }
    return result


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
    elif condition == "a_removed":
        # Causal ablation rA: remove the drawer->A / A->B link by parking A
        # laterally while leaving drawer, B, C, robot, and all native objects
        # in the exact Er state. C must remain below both response thresholds.
        from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
            _find_free_joint_qadr,
        )
        a_qadr = _find_free_joint_qadr(env.sim, A_BODY)
        env.sim.data.qpos[a_qadr] += 0.120
        env.sim.forward()
        state = np.asarray(env.sim.get_state().flatten()).copy()
    contact_pass, contact_reasons = _initial_contact_gate(
        env, drawer_body, condition
    )
    hold = _hold_gate(env, drawer_qadr, args.open_hold_steps)
    env.reset()
    env.set_init_state(state)
    clear_mujoco_replay_transients(env)
    trace = _script_required_motion_raw(env, drawer_qadr, args.close_steps)
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
        "trajectory": _trajectory_diagnostics(trace),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    # The opening drawer moves outward toward -y. Place the chain on the clear
    # table outside the closed front, with explicit passive gaps at each link.
    parser.add_argument("--a_dx", default="-0.015,0.000,0.015")
    parser.add_argument("--a_dy", default="0.035,0.038,0.041")
    parser.add_argument("--b_dx_from_a", default="0.000")
    parser.add_argument("--ab_spacing", default="-0.060,-0.065,-0.070")
    parser.add_argument("--bc_spacing", default="-0.042,-0.046,-0.050")
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
                passive_valid = bool(
                    risk["contact_pass"] and risk["hold"]["passed"]
                )
                if passive_valid:
                    stable = _run_condition(
                        env, risk_state, drawer_body, drawer_qadr, "stable", args
                    )
                    a_removed = _run_condition(
                        env, risk_state, drawer_body, drawer_qadr, "a_removed", args
                    )
                else:
                    skipped = {
                        "passed": False,
                        "skipped": True,
                        "reason": "risk_passive_contact_or_open_hold_gate_failed",
                    }
                    stable = dict(skipped)
                    a_removed = dict(skipped)
                paired_pass = bool(
                    risk["passed"] and stable["passed"] and a_removed["passed"]
                )
                trial_rows.append({
                    "trial": trial,
                    "paired_pass": paired_pass,
                    "risk": risk,
                    "stable": stable,
                    "a_removed": a_removed,
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
                f"eligible={row['eligible']} "
                f"risk_contact={trial_rows[0]['risk']['contact_pass']} "
                f"risk_hold={trial_rows[0]['risk']['hold']['passed']} "
                f"risk_steps="
                f"{trial_rows[0]['risk']['assessment']['drawer_a_step']},"
                f"{trial_rows[0]['risk']['assessment']['a_b_step']},"
                f"{trial_rows[0]['risk']['assessment']['b_c_step']},"
                f"{trial_rows[0]['risk']['assessment']['c_response_step']} "
                f"geometry={candidate}"
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
