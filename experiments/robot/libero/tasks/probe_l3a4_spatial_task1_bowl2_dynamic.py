#!/usr/bin/env python3
"""One-shot no-VLA dynamic gate for the frozen 24 bowl2 relay poses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_l3a4_spatial_task1_bowl2_feasibility import (
    B_GAPS_M,
    DIRECTIONS,
    EXPECTED_BDDL_SHA256,
    PROMPT,
    ROLES,
    SA_OVERLAPS_M,
    TASK_ID,
)
from audit_l3a4_spatial_task1_native import (
    DUMMY_ACTION,
    EVALUATOR_NUM_STEPS_WAIT,
    descendant_geoms,
    resolve_body,
)
from probe_l3a4_spatial_task1_momentum import (
    _candidate,
    _contact,
    _free_pose,
    _hold,
    _jsonable,
    _policy_image,
    _replace_pose,
    _restore,
    _robot_geoms,
    _set_pose,
    _sha,
    _write_video,
)


FROZEN_FEASIBLE_POSES = tuple(
    (direction, tilt, overlap, gap)
    for direction in DIRECTIONS
    for tilt, overlaps in ((20.0, (-0.004, -0.006)), (30.0, (-0.006,)))
    for overlap in overlaps
    for gap in B_GAPS_M
)


def _ab_force(env, a_geoms: set[int], b_geoms: set[int]) -> float:
    """Maximum positive normal constraint force over current A/B contacts."""
    maximum = 0.0
    for index in range(int(env.sim.data.ncon)):
        item = env.sim.data.contact[index]
        if not (
            (
                int(item.geom1) in a_geoms
                and int(item.geom2) in b_geoms
            )
            or (
                int(item.geom2) in a_geoms
                and int(item.geom1) in b_geoms
            )
        ):
            continue
        address = int(item.efc_address)
        if address >= 0:
            maximum = max(
                maximum, abs(float(env.sim.data.efc_force[address]))
            )
    return maximum


def _pose(env, body: str) -> tuple[np.ndarray, float]:
    body_id = int(env.sim.model.body_name2id(body))
    pos = np.asarray(env.sim.data.body_xpos[body_id]).copy()
    mat = np.asarray(env.sim.data.body_xmat[body_id]).reshape(3, 3)
    tilt = float(
        np.degrees(np.arccos(np.clip(mat[2, 2], -1.0, 1.0)))
    )
    return pos, tilt


def _projected_speed(env, body: str, axis_xy: np.ndarray) -> float:
    velocity = np.asarray(env.sim.data.get_body_xvelp(body), dtype=float)
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0])
    return float(velocity @ axis)


def _release(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    fall_direction: np.ndarray,
    *,
    lift_m: float,
    move_steps: int,
    settle_steps: int,
    freeze_a: bool = False,
    capture: bool = False,
) -> dict:
    _restore(env, state)
    starts = {role: _pose(env, bodies[role]) for role in ("A", "B")}
    s_pose = _free_pose(env, bodies["S"])
    a_pose = _free_pose(env, bodies["A"])
    events = {
        "S_release": None,
        "A_motion": None,
        "A_B_positive_force": None,
        "B_response": None,
    }
    forbidden = {"S_B": False, "robot_A": False, "robot_B": False}
    max_ab_force = 0.0
    max_b_projected_speed = 0.0
    frames = [_policy_image(env)] if capture else []
    trace = []
    for step in range(1, move_steps + settle_steps + 1):
        moved = s_pose.copy()
        moved[2] += lift_m * min(1.0, step / move_steps)
        _set_pose(env, bodies["S"], moved)
        if freeze_a:
            _set_pose(env, bodies["A"], a_pose)
        env.sim.forward()
        env.sim.step()
        if freeze_a:
            _set_pose(env, bodies["A"], a_pose)
            env.sim.forward()
        sa = _contact(env, geoms["S"], geoms["A"])
        ab_force = _ab_force(env, geoms["A"], geoms["B"])
        max_ab_force = max(max_ab_force, ab_force)
        b_speed = _projected_speed(env, bodies["B"], fall_direction)
        max_b_projected_speed = max(max_b_projected_speed, b_speed)
        forbidden["S_B"] |= _contact(env, geoms["S"], geoms["B"])
        forbidden["robot_A"] |= _contact(env, robot, geoms["A"])
        forbidden["robot_B"] |= _contact(env, robot, geoms["B"])
        delta, tilt = {}, {}
        for role in ("A", "B"):
            pos, angle = _pose(env, bodies[role])
            delta[role] = float(np.linalg.norm(pos - starts[role][0]))
            tilt[role] = abs(angle - starts[role][1])
        if events["S_release"] is None and not sa:
            events["S_release"] = step
        if (
            events["A_motion"] is None
            and (delta["A"] >= 0.003 or tilt["A"] >= 3.0)
        ):
            events["A_motion"] = step
        if events["A_B_positive_force"] is None and ab_force > 1e-6:
            events["A_B_positive_force"] = step
        if (
            events["A_B_positive_force"] is not None
            and events["B_response"] is None
            and (delta["B"] >= 0.005 or tilt["B"] >= 5.0)
            and max_b_projected_speed >= 0.03
        ):
            events["B_response"] = step
        if step == 1 or step % 5 == 0 or step in events.values():
            trace.append(
                {
                    "step": step,
                    "S_A": sa,
                    "A_B_force_N": ab_force,
                    "B_projected_speed_m_s": b_speed,
                    "displacement_m": delta,
                    "tilt_change_deg": tilt,
                }
            )
        if capture and step % 4 == 0:
            frames.append(_policy_image(env))
    sequence = list(events.values())
    ordered = bool(
        all(value is not None for value in sequence)
        and int(sequence[0]) < int(sequence[1])
        < int(sequence[2]) < int(sequence[3])
    )
    return {
        "events": events,
        "strictly_ordered": ordered,
        "forbidden_seen": forbidden,
        "max_A_B_force_N": max_ab_force,
        "max_B_projected_speed_m_s": max_b_projected_speed,
        "passed": bool(ordered and not any(forbidden.values())),
        "frames": frames,
        "trace": trace,
    }


def _without_frames(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in ("frames", "trace")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_spatial_task1_bowl2_dynamic",
    )
    parser.add_argument("--hold_steps", type=int, default=100)
    parser.add_argument("--scratch_settle_steps", type=int, default=50)
    parser.add_argument("--lift_m", type=float, default=0.12)
    parser.add_argument("--move_steps", type=int, default=30)
    parser.add_argument("--settle_steps", type=int, default=220)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID))
    if task.language != PROMPT:
        raise RuntimeError("native prompt drift")
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != EXPECTED_BDDL_SHA256:
        raise RuntimeError("native BDDL drift")
    raw = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(42)
        env.reset()
        env.set_init_state(raw)
        for _ in range(EVALUATOR_NUM_STEPS_WAIT):
            env.step(DUMMY_ACTION)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        bodies = {
            role: resolve_body(env.sim, stem) for role, stem in ROLES.items()
        }
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        robot = _robot_geoms(env)
        rows, eligible = [], []
        for direction_name, tilt, overlap, gap in FROZEN_FEASIBLE_POSES:
            direction = DIRECTIONS[direction_name]
            state = _candidate(
                env, base, bodies, direction, tilt, overlap, gap,
                args.scratch_settle_steps,
            )
            hold = _hold(
                env, state, bodies, geoms, robot, args.hold_steps
            )
            row = {
                "direction": direction_name,
                "tilt_deg": tilt,
                "SA_overlap_m": overlap,
                "B_gap_m": gap,
                "state_sha256": _sha(state),
                "hold": hold,
                "single_candidate_pass": False,
            }
            rows.append(row)
            if not hold["passed"]:
                continue
            fall = -direction
            release = _release(
                env, state, bodies, geoms, robot, fall,
                lift_m=args.lift_m, move_steps=args.move_steps,
                settle_steps=args.settle_steps,
            )
            row["release"] = _without_frames(release)
            if not release["passed"]:
                continue
            frozen = _release(
                env, state, bodies, geoms, robot, fall,
                lift_m=args.lift_m, move_steps=args.move_steps,
                settle_steps=args.settle_steps, freeze_a=True,
            )
            a_parked_state = _replace_pose(
                env, state, base, bodies["A"]
            )
            a_parked = _release(
                env, a_parked_state, bodies, geoms, robot, fall,
                lift_m=args.lift_m, move_steps=args.move_steps,
                settle_steps=args.settle_steps,
            )
            b_parked_state = _replace_pose(
                env, state, base, bodies["B"]
            )
            b_parked = _release(
                env, b_parked_state, bodies, geoms, robot, fall,
                lift_m=args.lift_m, move_steps=args.move_steps,
                settle_steps=args.settle_steps,
            )
            controls_pass = bool(
                frozen["events"]["A_B_positive_force"] is None
                and frozen["events"]["B_response"] is None
                and a_parked["events"]["A_B_positive_force"] is None
                and a_parked["events"]["B_response"] is None
                and b_parked["events"]["S_release"] is not None
                and b_parked["events"]["A_motion"] is not None
                and b_parked["events"]["A_B_positive_force"] is None
                and not any(frozen["forbidden_seen"].values())
                and not any(a_parked["forbidden_seen"].values())
                and not any(b_parked["forbidden_seen"].values())
            )
            row["A_frozen_ablation"] = _without_frames(frozen)
            row["A_parked_ablation"] = _without_frames(a_parked)
            row["B_parked_ablation"] = _without_frames(b_parked)
            row["single_candidate_pass"] = controls_pass
            if controls_pass:
                eligible.append((row, state, fall))

        param_index = {
            key: {value: i for i, value in enumerate(values)}
            for key, values in {
                "direction": list(DIRECTIONS),
                "tilt_deg": (20.0, 30.0),
                "SA_overlap_m": (-0.004, -0.006),
                "B_gap_m": B_GAPS_M,
            }.items()
        }
        winner = None
        for item in eligible:
            row = item[0]
            witnesses = []
            for other in eligible:
                candidate = other[0]
                if (
                    candidate is row
                    or candidate["direction"] != row["direction"]
                ):
                    continue
                distance = sum(
                    abs(param_index[key][candidate[key]]
                        - param_index[key][row[key]])
                    for key in ("tilt_deg", "SA_overlap_m", "B_gap_m")
                )
                if distance == 1:
                    witnesses.append(candidate["state_sha256"])
            row["adjacent_witnesses"] = witnesses
            row["passed"] = bool(witnesses)
            if winner is None and witnesses:
                winner = item

        artifacts, selected = {}, None
        if winner is not None:
            row, state, fall = winner
            selected = {
                key: row[key]
                for key in (
                    "direction", "tilt_deg", "SA_overlap_m",
                    "B_gap_m", "state_sha256", "adjacent_witnesses",
                )
            }
            _restore(env, state)
            png = out / "er_policy_entry_wait0_agentview.png"
            imageio.imwrite(png, _policy_image(env))
            release = _release(
                env, state, bodies, geoms, robot, fall,
                lift_m=args.lift_m, move_steps=args.move_steps,
                settle_steps=args.settle_steps, capture=True,
            )
            mp4 = out / "er_bowl2_relay.mp4"
            _write_video(mp4, release["frames"])
            trace = out / "er_bowl2_relay_trace.json"
            trace.write_text(
                json.dumps(_jsonable(release["trace"]), indent=2) + "\n"
            )
            states = out / "selected_policy_entry_wait0_state.npz"
            np.savez_compressed(
                states, raw_native_state=raw,
                policy_entry_base_state=base, er_state=state,
            )
            artifacts = {
                "policy_png": str(png),
                "video": str(mp4),
                "trace": str(trace),
                "states": str(states),
            }

        passed = winner is not None
        report = {
            "verdict": (
                "PASS_L3A4_SPATIAL_TASK1_BOWL2_DYNAMIC"
                if passed else
                "FAIL_L3A4_SPATIAL_TASK1_BOWL2_DYNAMIC"
            ),
            "task_suite": "libero_spatial",
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "bddl_sha256": EXPECTED_BDDL_SHA256,
            "policy_entry": {
                "wait_applied_exactly_once": EVALUATOR_NUM_STEPS_WAIT,
                "base_state_sha256": _sha(base),
                "future_evaluator_num_steps_wait": 0,
            },
            "frozen_candidate_count": len(FROZEN_FEASIBLE_POSES),
            "hold_pass_count": sum(row["hold"]["passed"] for row in rows),
            "single_candidate_pass_count": len(eligible),
            "robust_candidate_count": sum(
                bool(row.get("passed")) for row in rows
            ),
            "selected": selected,
            "candidates": rows,
            "artifacts": artifacts,
            "manual_policy_rgb_review": (
                "PENDING" if passed else "NOT_REVIEWABLE_PHYSICS_FAILED"
            ),
            "custom_assets": False,
            "vla_run": False,
        }
        (out / "probe.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        print(report["verdict"])
        if args.fail_on_invalid and not passed:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
