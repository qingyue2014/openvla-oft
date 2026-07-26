#!/usr/bin/env python3
"""Export a full raw-MuJoCo trace for one L3-A4 geometry candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a4_momentum import (
    A_BODY,
    B_BODY,
    C_BODY,
    DEFAULT_BDDL,
    DRAWER_BODY_CANDIDATES,
    DRAWER_JOINT_CANDIDATES,
    assess_chain,
    find_body,
    find_joint_qadr,
)
from experiments.robot.libero.tasks.sweep_l3a4_momentum_geometry import (
    _compiled_drawer_report,
    _place,
    _script_required_motion_raw,
)
from experiments.robot.libero.tasks.validate_l3a4_scene import (
    _hold_gate,
    _initial_contact_gate,
)


def _pair_step(trace, left, right):
    target = frozenset((left, right))
    return next(
        (
            int(frame.step)
            for frame in trace
            if target
            in {
                frozenset((str(pair[0]), str(pair[1])))
                for pair in frame.contacts
            }
        ),
        -1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--a_dx", type=float, default=0.0)
    parser.add_argument("--a_dy", type=float, default=0.142)
    parser.add_argument("--b_dx_from_a", type=float, default=0.0)
    parser.add_argument("--ab_spacing", type=float, default=0.056)
    parser.add_argument("--bc_spacing", type=float, default=0.038)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle_steps", type=int, default=400)
    parser.add_argument("--open_hold_steps", type=int, default=120)
    parser.add_argument("--motion_steps", type=int, default=240)
    parser.add_argument(
        "--out_json", default="experiments/logs/l3a4_candidate_trace.json"
    )
    parser.add_argument(
        "--out_report", default="experiments/logs/l3a4_candidate_trace.md"
    )
    args = parser.parse_args()

    candidate = {
        "a_dx": args.a_dx,
        "a_dy": args.a_dy,
        "b_dx_from_a": args.b_dx_from_a,
        "ab_spacing": args.ab_spacing,
        "bc_spacing": args.bc_spacing,
    }
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    try:
        env.seed(args.seed)
        env.reset()
        drawer_body = find_body(env, DRAWER_BODY_CANDIDATES)
        _, drawer_qadr = find_joint_qadr(env.sim, DRAWER_JOINT_CANDIDATES)
        compiled = _compiled_drawer_report(env, drawer_body, drawer_qadr)
        _place(env, drawer_body, candidate)
        for _ in range(args.settle_steps):
            env.sim.step()
        state = np.asarray(env.sim.get_state().flatten()).copy()

        env.reset()
        env.set_init_state(state)
        clear_mujoco_replay_transients(env)
        contact_pass, contact_reasons = _initial_contact_gate(
            env, drawer_body, "risk"
        )
        hold = _hold_gate(env, drawer_qadr, args.open_hold_steps)

        env.reset()
        env.set_init_state(state)
        clear_mujoco_replay_transients(env)
        trace = _script_required_motion_raw(
            env, drawer_qadr, args.motion_steps
        )
        assessment = assess_chain(
            trace, condition="risk", drawer_body=drawer_body
        )
        direct_steps = {
            "drawer_B": _pair_step(trace, drawer_body, B_BODY),
            "drawer_C": _pair_step(trace, drawer_body, C_BODY),
            "A_C": _pair_step(trace, A_BODY, C_BODY),
        }
        payload = {
            "candidate": candidate,
            "compiled": compiled,
            "initial_contact_pass": contact_pass,
            "initial_contact_reasons": contact_reasons,
            "hold": hold,
            "assessment": assessment.to_dict(),
            "direct_contact_steps": direct_steps,
            "trace": [
                {
                    "step": int(frame.step),
                    "drawer_qpos": float(frame.drawer_qpos),
                    "positions": {
                        name: list(map(float, position))
                        for name, position in frame.positions.items()
                    },
                    "speeds_m_s": {
                        name: float(speed)
                        for name, speed in frame.speeds_m_s.items()
                    },
                    "velocities_m_s": {
                        name: list(map(float, velocity))
                        for name, velocity in frame.velocities_m_s.items()
                    },
                    "tilts_deg": {
                        name: float(tilt)
                        for name, tilt in frame.tilts_deg.items()
                    },
                    "contacts": [list(map(str, pair)) for pair in frame.contacts],
                }
                for frame in trace
            ],
        }
    finally:
        env.close()

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    event_steps = {
        name: payload["assessment"][name]
        for name in (
            "drawer_a_step",
            "a_b_step",
            "b_c_step",
            "c_response_step",
        )
    }
    report = Path(args.out_report)
    report.write_text(
        "\n".join(
            [
                "# L3-A4 single-candidate full trace",
                "",
                f"- Candidate: `{json.dumps(candidate, sort_keys=True)}`",
                f"- Initial contact pass: **{contact_pass}**",
                f"- Closed hold pass: **{hold['passed']}**",
                f"- Risk assessment pass: **{assessment.passed}**",
                f"- Ordered event steps: `{json.dumps(event_steps, sort_keys=True)}`",
                f"- Direct contact steps: `{json.dumps(direct_steps, sort_keys=True)}`",
                f"- Full trace: `{out_json}`",
                "",
            ]
        )
    )
    print(
        "PASS_L3A4_CANDIDATE_TRACE_EXPORTED "
        f"risk_pass={assessment.passed} report={report}"
    )


if __name__ == "__main__":
    main()
