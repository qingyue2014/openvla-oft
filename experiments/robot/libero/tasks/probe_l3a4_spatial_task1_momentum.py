#!/usr/bin/env python3
"""Bounded one-state, native-only S-release -> A -> B mechanics probe.

This script never imports or runs a VLA.  It reproduces the evaluator's ten
dummy actions once, freezes that policy-entry state, and changes only the
native ramekin (A) and cookies (B) free-joint poses.  S is lifted vertically
to model removal by the native pickup; no scripted force or contact is applied
to A or B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_l3a4_spatial_task1_native import (
    DUMMY_ACTION,
    EVALUATOR_NUM_STEPS_WAIT,
    PROMPT,
    ROLES,
    TASK_ID,
    descendant_geoms,
    resolve_body,
)


EXPECTED_BDDL_SHA256 = (
    "53a7516571412a2f46a27cbf8482d3b76dbad4221858c8f6b565d506c274e61d"
)
DIRECTIONS = {
    "+y": np.asarray([0.0, 1.0]),
    "+x": np.asarray([1.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0]),
    "-y": np.asarray([0.0, -1.0]),
}
LEAN_DEGREES = (12.0, 18.0, 24.0, 30.0)
SA_OVERLAPS_M = (-0.001, -0.003, -0.005)
B_GAPS_M = (0.003, 0.008, 0.013)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _free_joint(env, body: str) -> tuple[int, int]:
    body_id = int(env.sim.model.body_name2id(body))
    for joint_id in range(env.sim.model.njnt):
        if (
            int(env.sim.model.jnt_bodyid[joint_id]) == body_id
            and int(env.sim.model.jnt_type[joint_id]) == 0
        ):
            return (
                int(env.sim.model.jnt_qposadr[joint_id]),
                int(env.sim.model.jnt_dofadr[joint_id]),
            )
    raise RuntimeError(f"no free joint for {body}")


def _restore(env, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state).copy())
    for name in ("qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        buffer = getattr(env.sim.data, name, None)
        if buffer is not None:
            buffer[...] = 0
    env.sim.forward()


def _pose(env, body: str) -> tuple[np.ndarray, float]:
    body_id = int(env.sim.model.body_name2id(body))
    pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()
    mat = np.asarray(
        env.sim.data.body_xmat[body_id], dtype=float
    ).reshape(3, 3)
    tilt = float(
        np.degrees(np.arccos(np.clip(mat[2, 2], -1.0, 1.0)))
    )
    return pos, tilt


def _set_pose(
    env, body: str, pose: np.ndarray, *, zero_velocity: bool = True
) -> None:
    qadr, vadr = _free_joint(env, body)
    env.sim.data.qpos[qadr:qadr + 7] = np.asarray(pose, dtype=float)
    if zero_velocity:
        env.sim.data.qvel[vadr:vadr + 6] = 0.0


def _free_pose(env, body: str) -> np.ndarray:
    qadr, _ = _free_joint(env, body)
    return np.asarray(env.sim.data.qpos[qadr:qadr + 7]).copy()


def _quat_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = np.radians(degrees) / 2.0
    return np.asarray([np.cos(half), *(np.sin(half) * axis)])


def _geom_vertices(env, geom_id: int) -> np.ndarray:
    model = env.sim.model
    pos = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
    mat = np.asarray(env.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    kind = int(model.geom_type[geom_id])
    if kind == 7:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        return local @ mat.T + pos
    if kind == 2:
        half = np.repeat(float(size[0]), 3)
    elif kind == 3:
        half = np.abs(mat) @ np.asarray(
            [size[0], size[0], size[1] + size[0]]
        )
    elif kind == 5:
        half = np.abs(mat) @ np.asarray([size[0], size[0], size[1]])
    elif kind in (4, 6):
        half = np.abs(mat) @ size[:3]
    else:
        half = np.repeat(float(model.geom_rbound[geom_id]), 3)
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=float,
    )
    return pos + signs * half


def _bounds(env, body: str) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for geom_id in descendant_geoms(env.sim, body):
        if int(env.sim.model.geom_group[geom_id]) != 0:
            continue
        vertices = _geom_vertices(env, geom_id)
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))
    if not np.isfinite(lo).all():
        raise RuntimeError(f"no collision bounds for {body}")
    return lo, hi


def _extent(env, body: str, axis_xy: np.ndarray) -> float:
    center, _ = _pose(env, body)
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0])
    return max(
        float(np.max((_geom_vertices(env, geom) - center) @ axis))
        for geom in descendant_geoms(env.sim, body)
        if int(env.sim.model.geom_group[geom]) == 0
    )


def _contact(env, left: set[int], right: set[int]) -> bool:
    for index in range(int(env.sim.data.ncon)):
        item = env.sim.data.contact[index]
        pair = {int(item.geom1), int(item.geom2)}
        if pair & left and pair & right:
            return True
    return False


def _robot_geoms(env) -> set[int]:
    result = set()
    for geom in range(env.sim.model.ngeom):
        body = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[geom])
        ) or ""
        if (
            body.startswith("robot")
            or "gripper" in body.lower()
            or "panda" in body.lower()
        ):
            result.add(geom)
    return result


def _policy_image(env) -> np.ndarray:
    state = np.asarray(env.sim.get_state().flatten()).copy()
    obs = env.regenerate_obs_from_state(state)
    return np.ascontiguousarray(
        np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
    )


def _candidate(
    env,
    base: np.ndarray,
    bodies: dict[str, str],
    direction: np.ndarray,
    lean: float,
    overlap: float,
    b_gap: float,
    settle_steps: int,
) -> np.ndarray:
    _restore(env, base)
    s_pose = _free_pose(env, bodies["S"])
    s_xy = s_pose[:2].copy()
    table_z = float(_bounds(env, bodies["S"])[0][2])

    a_pose = _free_pose(env, bodies["A"])
    # A's local +z leans toward S (-direction).
    a_pose[3:7] = _quat_axis_angle(
        np.asarray([direction[1], -direction[0], 0.0]), lean
    )
    _set_pose(env, bodies["A"], a_pose)
    env.sim.forward()
    sep = (
        _extent(env, bodies["S"], direction)
        + _extent(env, bodies["A"], -direction)
        + overlap
    )
    a_pose[:2] = s_xy + direction * sep
    _set_pose(env, bodies["A"], a_pose)
    env.sim.forward()
    a_pose[2] += table_z - float(_bounds(env, bodies["A"])[0][2])
    _set_pose(env, bodies["A"], a_pose)
    env.sim.forward()

    # B is beyond S along A's fall direction, but starts outside S contact.
    b_pose = _free_pose(env, bodies["B"])
    b_pose[:2] = s_xy - direction * (
        _extent(env, bodies["S"], -direction)
        + _extent(env, bodies["B"], direction)
        + b_gap
    )
    _set_pose(env, bodies["B"], b_pose)
    env.sim.forward()
    b_pose[2] += table_z - float(_bounds(env, bodies["B"])[0][2])
    _set_pose(env, bodies["B"], b_pose)
    env.sim.forward()

    for _ in range(settle_steps):
        env.sim.step()
        _set_pose(env, bodies["S"], s_pose)
        env.sim.forward()
    settled = {
        role: _free_pose(env, bodies[role]) for role in ("A", "B")
    }
    _restore(env, base)
    for role in ("A", "B"):
        _set_pose(
            env, bodies[role], settled[role], zero_velocity=False
        )
    env.sim.forward()
    state = np.asarray(env.sim.get_state().flatten()).copy()
    allowed = set()
    for role in ("A", "B"):
        qadr, _ = _free_joint(env, bodies[role])
        allowed.update(range(1 + qadr, 1 + qadr + 7))
    changed = set(
        np.flatnonzero(~np.isclose(state, base, rtol=0, atol=1e-12))
    )
    if not changed or not changed <= allowed:
        raise RuntimeError("candidate changed bytes outside A/B free qpos")
    return state


def _hold(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    steps: int,
) -> dict:
    _restore(env, state)
    starts = {role: _pose(env, body) for role, body in bodies.items()}
    counts = {"S_A": 0}
    forbidden = {"A_B": False, "S_B": False, "robot_A": False, "robot_B": False}
    max_drift = {role: 0.0 for role in ("S", "A", "B")}
    max_tilt = {role: 0.0 for role in ("S", "A", "B")}
    for step in range(steps + 1):
        counts["S_A"] += int(_contact(env, geoms["S"], geoms["A"]))
        forbidden["A_B"] |= _contact(env, geoms["A"], geoms["B"])
        forbidden["S_B"] |= _contact(env, geoms["S"], geoms["B"])
        forbidden["robot_A"] |= _contact(env, robot, geoms["A"])
        forbidden["robot_B"] |= _contact(env, robot, geoms["B"])
        for role in ("S", "A", "B"):
            pos, tilt = _pose(env, bodies[role])
            max_drift[role] = max(
                max_drift[role], float(np.linalg.norm(pos - starts[role][0]))
            )
            max_tilt[role] = max(
                max_tilt[role], abs(tilt - starts[role][1])
            )
        if step < steps:
            env.sim.step()
    occupancy = counts["S_A"] / float(steps + 1)
    passed = bool(
        occupancy >= 0.90
        and not any(forbidden.values())
        and max(max_drift.values()) <= 0.003
        and max(max_tilt.values()) <= 5.0
    )
    return {
        "passed": passed,
        "S_A_contact_occupancy": occupancy,
        "forbidden_seen": forbidden,
        "max_drift_m": max_drift,
        "max_tilt_change_deg": max_tilt,
    }


def _release(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    robot: set[int],
    *,
    lift_m: float,
    move_steps: int,
    settle_steps: int,
    freeze_a: bool = False,
    capture: bool = False,
) -> dict:
    _restore(env, state)
    starts = {role: _pose(env, body) for role, body in bodies.items()}
    s_pose, a_pose = _free_pose(env, bodies["S"]), _free_pose(env, bodies["A"])
    events = {
        "S_release": None,
        "A_motion": None,
        "A_B_contact": None,
        "B_response": None,
    }
    forbidden = {"S_B": False, "robot_A": False, "robot_B": False}
    frames = [_policy_image(env)] if capture else []
    trace = []
    max_disp = {role: 0.0 for role in ("A", "B")}
    max_tilt = {role: 0.0 for role in ("A", "B")}
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
        ab = _contact(env, geoms["A"], geoms["B"])
        forbidden["S_B"] |= _contact(env, geoms["S"], geoms["B"])
        forbidden["robot_A"] |= _contact(env, robot, geoms["A"])
        forbidden["robot_B"] |= _contact(env, robot, geoms["B"])
        deltas, tilts = {}, {}
        for role in ("A", "B"):
            pos, tilt = _pose(env, bodies[role])
            deltas[role] = float(np.linalg.norm(pos - starts[role][0]))
            tilts[role] = abs(tilt - starts[role][1])
            max_disp[role] = max(max_disp[role], deltas[role])
            max_tilt[role] = max(max_tilt[role], tilts[role])
        if events["S_release"] is None and not sa:
            events["S_release"] = step
        if (
            events["A_motion"] is None
            and (deltas["A"] >= 0.003 or tilts["A"] >= 3.0)
        ):
            events["A_motion"] = step
        if events["A_B_contact"] is None and ab:
            events["A_B_contact"] = step
        if (
            events["B_response"] is None
            and (deltas["B"] >= 0.005 or tilts["B"] >= 5.0)
        ):
            events["B_response"] = step
        if step == 1 or step % 5 == 0 or step in events.values():
            trace.append(
                {
                    "step": step,
                    "S_A": sa,
                    "A_B": ab,
                    "displacement_m": deltas,
                    "tilt_change_deg": tilts,
                }
            )
        if capture and (step % 4 == 0):
            frames.append(_policy_image(env))
    values = list(events.values())
    ordered = bool(
        all(value is not None for value in values)
        and int(values[0]) < int(values[1]) < int(values[2]) < int(values[3])
    )
    return {
        "events": events,
        "strictly_ordered": ordered,
        "forbidden_seen": forbidden,
        "max_displacement_m": max_disp,
        "max_tilt_change_deg": max_tilt,
        "passed": bool(ordered and not any(forbidden.values())),
        "frames": frames,
        "trace": trace,
    }


def _replace_pose(
    env, state: np.ndarray, base: np.ndarray, body: str
) -> np.ndarray:
    _restore(env, base)
    native = _free_pose(env, body)
    _restore(env, state)
    _set_pose(env, body, native, zero_velocity=False)
    env.sim.forward()
    return np.asarray(env.sim.get_state().flatten()).copy()


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    with imageio.get_writer(path, fps=20, format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_spatial_task1_momentum",
    )
    parser.add_argument("--hold_steps", type=int, default=70)
    parser.add_argument("--scratch_settle_steps", type=int, default=50)
    parser.add_argument("--lift_m", type=float, default=0.12)
    parser.add_argument("--move_steps", type=int, default=30)
    parser.add_argument("--settle_steps", type=int, default=170)
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
        bodies = {role: resolve_body(env.sim, stem) for role, stem in ROLES.items()}
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        robot = _robot_geoms(env)

        rows, eligible = [], []
        for direction_name, direction in DIRECTIONS.items():
            print(f"[spatial-task1] direction={direction_name}", flush=True)
            for lean in LEAN_DEGREES:
                for overlap in SA_OVERLAPS_M:
                    for b_gap in B_GAPS_M:
                        try:
                            state = _candidate(
                                env, base, bodies, direction, lean, overlap,
                                b_gap, args.scratch_settle_steps,
                            )
                            hold = _hold(
                                env, state, bodies, geoms, robot, args.hold_steps
                            )
                            row = {
                                "direction": direction_name,
                                "lean_deg": lean,
                                "SA_overlap_m": overlap,
                                "B_gap_m": b_gap,
                                "state_sha256": _sha(state),
                                "hold": hold,
                                "single_candidate_pass": False,
                            }
                            rows.append(row)
                            if not hold["passed"]:
                                continue
                            run = _release(
                                env, state, bodies, geoms, robot,
                                lift_m=args.lift_m,
                                move_steps=args.move_steps,
                                settle_steps=args.settle_steps,
                            )
                            row["release"] = {
                                k: v for k, v in run.items()
                                if k not in ("frames", "trace")
                            }
                            if not run["passed"]:
                                continue
                            frozen = _release(
                                env, state, bodies, geoms, robot,
                                lift_m=args.lift_m,
                                move_steps=args.move_steps,
                                settle_steps=args.settle_steps,
                                freeze_a=True,
                            )
                            a_parked = _release(
                                env,
                                _replace_pose(env, state, base, bodies["A"]),
                                bodies, geoms, robot,
                                lift_m=args.lift_m,
                                move_steps=args.move_steps,
                                settle_steps=args.settle_steps,
                            )
                            controls_pass = bool(
                                frozen["events"]["A_B_contact"] is None
                                and frozen["events"]["B_response"] is None
                                and a_parked["events"]["A_B_contact"] is None
                                and a_parked["events"]["B_response"] is None
                                and not any(frozen["forbidden_seen"].values())
                                and not any(a_parked["forbidden_seen"].values())
                            )
                            row["A_frozen_ablation"] = {
                                k: v for k, v in frozen.items()
                                if k not in ("frames", "trace")
                            }
                            row["A_parked_ablation"] = {
                                k: v for k, v in a_parked.items()
                                if k not in ("frames", "trace")
                            }
                            row["single_candidate_pass"] = controls_pass
                            if controls_pass:
                                eligible.append((row, state, direction))
                        except Exception as exc:
                            rows.append(
                                {
                                    "direction": direction_name,
                                    "lean_deg": lean,
                                    "SA_overlap_m": overlap,
                                    "B_gap_m": b_gap,
                                    "error": repr(exc),
                                    "single_candidate_pass": False,
                                }
                            )

        index = {
            "lean": {v: i for i, v in enumerate(LEAN_DEGREES)},
            "overlap": {v: i for i, v in enumerate(SA_OVERLAPS_M)},
            "gap": {v: i for i, v in enumerate(B_GAPS_M)},
        }
        winner = None
        for item in eligible:
            row = item[0]
            witnesses = []
            for other in eligible:
                candidate = other[0]
                if candidate is row or candidate["direction"] != row["direction"]:
                    continue
                distance = (
                    abs(index["lean"][candidate["lean_deg"]] - index["lean"][row["lean_deg"]])
                    + abs(index["overlap"][candidate["SA_overlap_m"]] - index["overlap"][row["SA_overlap_m"]])
                    + abs(index["gap"][candidate["B_gap_m"]] - index["gap"][row["B_gap_m"]])
                )
                if distance == 1:
                    witnesses.append(candidate["state_sha256"])
            row["adjacent_witnesses"] = witnesses
            row["passed"] = bool(witnesses)
            if winner is None and witnesses:
                winner = item

        artifacts = {}
        selected = None
        if winner is not None:
            row, state, _ = winner
            selected = {
                "parameters": {
                    key: row[key]
                    for key in ("direction", "lean_deg", "SA_overlap_m", "B_gap_m")
                },
                "state_sha256": row["state_sha256"],
                "adjacent_witnesses": row["adjacent_witnesses"],
            }
            _restore(env, state)
            png = out / "er_policy_entry_wait0_agentview.png"
            imageio.imwrite(png, _policy_image(env))
            run = _release(
                env, state, bodies, geoms, robot,
                lift_m=args.lift_m, move_steps=args.move_steps,
                settle_steps=args.settle_steps, capture=True,
            )
            mp4 = out / "er_release_chain.mp4"
            _write_video(mp4, run["frames"])
            trace = out / "er_release_trace.json"
            trace.write_text(json.dumps(_jsonable(run["trace"]), indent=2) + "\n")
            np.savez_compressed(
                out / "selected_policy_entry_wait0_state.npz",
                raw_native_state=raw,
                policy_entry_base_state=base,
                er_state=state,
            )
            artifacts = {
                "policy_png": str(png),
                "release_video": str(mp4),
                "trace": str(trace),
                "states": str(out / "selected_policy_entry_wait0_state.npz"),
            }

        passed = winner is not None
        report = {
            "verdict": (
                "PASS_L3A4_SPATIAL_TASK1_ONE_STATE_MOMENTUM"
                if passed else
                "FAIL_L3A4_SPATIAL_TASK1_ONE_STATE_MOMENTUM"
            ),
            "task_suite": "libero_spatial",
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "native_bddl_sha256": EXPECTED_BDDL_SHA256,
            "policy_entry": {
                "wait_applied_exactly_once": EVALUATOR_NUM_STEPS_WAIT,
                "base_state_sha256": _sha(base),
                "future_evaluator_num_steps_wait": 0,
            },
            "roles": bodies,
            "grid": {
                "directions": list(DIRECTIONS),
                "lean_degrees": LEAN_DEGREES,
                "SA_overlaps_m": SA_OVERLAPS_M,
                "B_gaps_m": B_GAPS_M,
                "candidate_count": len(rows),
                "single_candidate_pass_count": len(eligible),
                "robust_candidate_count": sum(
                    bool(row.get("passed")) for row in rows
                ),
            },
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
