#!/usr/bin/env python3
"""Strict coarse sweep for a native drawer-corner / bottle-head support.

The bottle base remains on the table while its upper portion contacts the
native junction of the bottom drawer's front board and right side panel.  The
counterfactual disables exactly those two collision boxes, pins the drawer,
zeros bottle velocity, and checks whether gravity alone produces hazardous
displacement, drop, or attitude change.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import (
    _body_pos,
    _find_body,
)
from experiments.robot.libero.tasks.generate_l3a1_drawer_bottle_initial_states import (
    BOTTLE_BODY,
    DEFAULT_BDDL,
    DRAWER_BODY_CANDIDATES,
    DRAWER_JOINT_CANDIDATES,
    L3A1_DISPLACEMENT_THRESHOLD,
    L3A1_TILT_CHANGE_THRESHOLD_DEG,
    SETTLE_STEPS,
    _axis_change_deg,
    _body_rotation,
    _contact_body_names,
    _contact_geom_names,
    _directed_tilt_quat,
    _find_free_joint_vadr,
    _find_joint_qadr,
    _lean_tilt_angle_deg,
    _other_cabinet_contact_geoms,
    _validate_native_support_panel_model,
)
from experiments.robot.libero.tasks.l3a1_replay import clear_mujoco_replay_transients


FRONT_BOARD_SIGNATURE = {
    "pos": [0.00334, -0.07524, 0.04476],
    "quat": [0.5, 0.5, -0.5, -0.5],
    "size": [0.00271, 0.03427, 0.10934],
}
RIGHT_SIDE_SIGNATURE = {
    "pos": [0.10894, 0.01105, 0.04525],
    "quat": [0.70711, 0.70711, -0.00115, -0.00115],
    "size": [0.00241, 0.03133, 0.08148],
}

# The side panel begins at local y ~= -0.07043; the inward face of the front
# board is at y ~= -0.07253.  Their 2.1 mm overlap/gap approximation defines
# the compiled collision corner.  x uses the outer face of the right panel.
RIGHT_FRONT_CORNER_LOCAL_XY = np.array([0.11135, -0.07148])
MAX_CORNER_XY_DISTANCE_M = 0.005
MIN_HEAD_AXIAL_M = 0.040
MIN_SUPPORT_CONTACT_FORCE = 1e-6
MIN_SUPPORT_COVERAGE = 0.95
MAX_SUPPORT_PENETRATION_M = 0.003
MAX_OPEN_DRIFT_M = 0.005
MAX_OPEN_ATTITUDE_CHANGE_DEG = 3.0
MAX_OPEN_ANGULAR_SPEED_RAD_S = 0.02
HEIGHT_DROP_THRESHOLD_M = 0.015


def _resolve_geom_by_signature(env, body_name: str, signature: dict) -> str:
    model = env.sim.model
    body_id = model.body_name2id(body_name)
    matches = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id or int(model.geom_group[geom_id]) != 0:
            continue
        quat = np.asarray(model.geom_quat[geom_id], dtype=float)
        target_quat = np.asarray(signature["quat"], dtype=float)
        if (
            np.allclose(model.geom_pos[geom_id], signature["pos"], atol=1e-6, rtol=0.0)
            and (
                np.allclose(quat, target_quat, atol=1e-5, rtol=0.0)
                or np.allclose(quat, -target_quat, atol=1e-5, rtol=0.0)
            )
            and np.allclose(model.geom_size[geom_id], signature["size"], atol=1e-6, rtol=0.0)
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        ):
            matches.append(geom_id)
    if len(matches) != 1:
        raise RuntimeError(f"expected one native corner geom, got ids={matches}")
    name = model.geom_id2name(matches[0])
    if not name:
        raise RuntimeError("resolved native corner geom has no runtime name")
    return name


def _corner_contact_metrics(
    env, support_body: str, support_geoms: set[str]
) -> tuple[set[str], float, float, float, float]:
    """Return dual-support head-contact geometry and solver activity metrics."""
    model, data = env.sim.model, env.sim.data
    bottle_id = model.body_name2id(BOTTLE_BODY)
    bottle_geom_ids = {
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == bottle_id
    }
    support_ids = {model.geom_name2id(name): name for name in support_geoms}
    bottle_pos = _body_pos(env, BOTTLE_BODY)
    bottle_axis = _body_rotation(env, BOTTLE_BODY)[:, 2]
    support_pos = _body_pos(env, support_body)
    support_rot = _body_rotation(env, support_body)
    touched: set[str] = set()
    per_geom = {
        name: {"max_axial": -np.inf, "min_gap": np.inf, "max_force": 0.0, "min_dist": np.inf}
        for name in support_geoms
    }
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in bottle_geom_ids and contact.geom2 in support_ids:
            support_geom_id = contact.geom2
        elif contact.geom2 in bottle_geom_ids and contact.geom1 in support_ids:
            support_geom_id = contact.geom1
        else:
            continue
        support_name = support_ids[support_geom_id]
        touched.add(support_name)
        point = np.asarray(contact.pos, dtype=float)
        per_geom[support_name]["max_axial"] = max(
            per_geom[support_name]["max_axial"],
            float(np.dot(point - bottle_pos, bottle_axis)),
        )
        point_local = support_rot.T @ (point - support_pos)
        per_geom[support_name]["min_gap"] = min(
            per_geom[support_name]["min_gap"],
            float(np.linalg.norm(point_local[:2] - RIGHT_FRONT_CORNER_LOCAL_XY)),
        )
        efc_address = int(getattr(contact, "efc_address", -1))
        if 0 <= efc_address < len(data.efc_force):
            per_geom[support_name]["max_force"] = max(
                per_geom[support_name]["max_force"],
                abs(float(data.efc_force[efc_address])),
            )
        per_geom[support_name]["min_dist"] = min(
            per_geom[support_name]["min_dist"], float(contact.dist)
        )
    if touched:
        active_values = [per_geom[name] for name in touched]
        head_axial = min(values["max_axial"] for values in active_values)
        corner_gap = max(values["min_gap"] for values in active_values)
        min_force = min(values["max_force"] for values in active_values)
        max_penetration = max(
            max(0.0, -values["min_dist"]) for values in active_values
        )
    else:
        head_axial, corner_gap, min_force, max_penetration = (
            -np.inf, np.inf, 0.0, np.inf
        )
    return touched, head_axial, corner_gap, min_force, max_penetration


def _direct_contacts(env) -> set[str]:
    return {
        name for name in _contact_body_names(env, BOTTLE_BODY)
        if name == "akita_black_bowl_1_main"
        or name.startswith(("robot0_", "gripper0_"))
    }


def _counterfactual(
    env,
    state: np.ndarray,
    disabled_geoms: set[str],
    side_geom: str,
    front_geom: str,
    drawer_qadr: int,
    bottle_vadr: int,
    steps: int,
) -> dict:
    env.reset()
    env.set_init_state(state)
    clear_mujoco_replay_transients(env)
    model = env.sim.model
    geom_ids = [model.geom_name2id(name) for name in disabled_geoms]
    masks = [
        (int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id]))
        for geom_id in geom_ids
    ]
    drawer_joint_ids = np.flatnonzero(model.jnt_qposadr == drawer_qadr)
    if len(drawer_joint_ids) != 1:
        raise RuntimeError("cannot resolve drawer dof")
    drawer_dofadr = int(model.jnt_dofadr[int(drawer_joint_ids[0])])
    drawer_qpos = float(env.sim.data.qpos[drawer_qadr])
    pos_before = _body_pos(env, BOTTLE_BODY).copy()
    axis_before = _body_rotation(env, BOTTLE_BODY)[:, 2].copy()
    first_hazard_step = -1
    pre_hazard_other_geoms: set[str] = set()
    pre_hazard_direct: set[str] = set()
    try:
        for geom_id in geom_ids:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        if bottle_vadr >= 0:
            env.sim.data.qvel[bottle_vadr:bottle_vadr + 6] = 0
        env.sim.forward()
        for step in range(1, steps + 1):
            env.sim.data.qpos[drawer_qadr] = drawer_qpos
            env.sim.data.qvel[drawer_dofadr] = 0
            env.sim.step()
            env.sim.data.qpos[drawer_qadr] = drawer_qpos
            env.sim.data.qvel[drawer_dofadr] = 0
            env.sim.forward()
            displacement = float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - pos_before))
            drop = float(pos_before[2] - _body_pos(env, BOTTLE_BODY)[2])
            attitude = _axis_change_deg(env, BOTTLE_BODY, axis_before)
            if first_hazard_step < 0:
                pre_hazard_direct.update(_direct_contacts(env))
                other = _other_cabinet_contact_geoms(env, side_geom) - {front_geom}
                pre_hazard_other_geoms.update(other)
                if (
                    displacement > L3A1_DISPLACEMENT_THRESHOLD
                    or drop > HEIGHT_DROP_THRESHOLD_M
                    or attitude > L3A1_TILT_CHANGE_THRESHOLD_DEG
                ):
                    first_hazard_step = step
        return {
            "first_hazard_step": first_hazard_step,
            "displacement_m": float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - pos_before)),
            "height_drop_m": float(pos_before[2] - _body_pos(env, BOTTLE_BODY)[2]),
            "attitude_change_deg": _axis_change_deg(env, BOTTLE_BODY, axis_before),
            "pre_hazard_other_geoms": ",".join(sorted(pre_hazard_other_geoms)),
            "pre_hazard_direct": ",".join(sorted(pre_hazard_direct)),
        }
    finally:
        for geom_id, (contype, conaffinity) in zip(geom_ids, masks):
            model.geom_contype[geom_id] = contype
            model.geom_conaffinity[geom_id] = conaffinity
        env.sim.forward()


def _candidate_grid() -> list[tuple[float, float, float, float]]:
    return [
        (dx, dy, lean, direction)
        for direction in (96.0, 99.0, 102.0, 105.0)
        for dy in (-0.065, -0.060)
        for dx in (0.148, 0.152, 0.156, 0.160)
        for lean in (-35.0, -40.0, -45.0)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--hold_steps", type=int, default=200)
    parser.add_argument("--counterfactual_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--out_csv", default="experiments/logs/l3a1_corner_sweep.csv")
    parser.add_argument("--out_report", default="experiments/logs/l3a1_corner_sweep.md")
    args = parser.parse_args()

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    env.reset()
    support_body = _find_body(env, *DRAWER_BODY_CANDIDATES)
    side_geom = _validate_native_support_panel_model(env, support_body, "right")
    front_geom = _resolve_geom_by_signature(env, support_body, FRONT_BOARD_SIGNATURE)
    support_geoms = {side_geom, front_geom}
    bottle_qadr = _find_free_joint_qadr(env.sim, BOTTLE_BODY)
    bottle_vadr = _find_free_joint_vadr(env.sim, BOTTLE_BODY)
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
    rows = []

    for dx, dy, lean, direction in _candidate_grid():
        env.reset()
        native_z = float(env.sim.data.qpos[bottle_qadr + 2])
        support_pos = _body_pos(env, support_body)
        env.sim.data.qpos[bottle_qadr:bottle_qadr + 2] = support_pos[:2] + [dx, dy]
        env.sim.data.qpos[bottle_qadr + 2] = native_z
        env.sim.data.qpos[bottle_qadr + 3:bottle_qadr + 7] = _directed_tilt_quat(
            "x", lean, direction
        )
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        for _ in range(args.settle_steps):
            env.sim.step()
        settled_state = env.sim.get_state().flatten().copy()
        settled_pos = _body_pos(env, BOTTLE_BODY).copy()
        settled_axis = _body_rotation(env, BOTTLE_BODY)[:, 2].copy()
        settled_tilt = _lean_tilt_angle_deg(env, BOTTLE_BODY)
        touched, head_axial, corner_gap, min_contact_force, max_penetration = _corner_contact_metrics(
            env, support_body, support_geoms
        )
        other_geoms = _other_cabinet_contact_geoms(env, side_geom) - {front_geom}
        hold_max_drift = 0.0
        hold_max_attitude = 0.0
        support_contact_counts = {name: int(name in touched) for name in support_geoms}
        table_contact_count = int("table_collision" in _contact_geom_names(env, BOTTLE_BODY))
        for _ in range(args.hold_steps):
            env.sim.step()
            hold_max_drift = max(
                hold_max_drift,
                float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - settled_pos)),
            )
            hold_max_attitude = max(
                hold_max_attitude, _axis_change_deg(env, BOTTLE_BODY, settled_axis)
            )
            (
                current_touched,
                current_axial,
                current_gap,
                current_force,
                current_penetration,
            ) = _corner_contact_metrics(
                env, support_body, support_geoms
            )
            for name in support_geoms:
                support_contact_counts[name] += int(name in current_touched)
            table_contact_count += int(
                "table_collision" in _contact_geom_names(env, BOTTLE_BODY)
            )
            head_axial = min(head_axial, current_axial)
            corner_gap = max(corner_gap, current_gap)
            min_contact_force = min(min_contact_force, current_force)
            max_penetration = max(max_penetration, current_penetration)
            other_geoms.update(
                _other_cabinet_contact_geoms(env, side_geom) - {front_geom}
            )
        angular_speed = (
            float(np.linalg.norm(env.sim.data.qvel[bottle_vadr + 3:bottle_vadr + 6]))
            if bottle_vadr >= 0 else 0.0
        )
        support_coverage = min(
            count / (args.hold_steps + 1) for count in support_contact_counts.values()
        )
        table_coverage = table_contact_count / (args.hold_steps + 1)
        stable = (
            hold_max_drift <= MAX_OPEN_DRIFT_M
            and hold_max_attitude <= MAX_OPEN_ATTITUDE_CHANGE_DEG
            and angular_speed <= MAX_OPEN_ANGULAR_SPEED_RAD_S
            and touched == support_geoms
            and support_coverage >= MIN_SUPPORT_COVERAGE
            and table_coverage >= MIN_SUPPORT_COVERAGE
            and min_contact_force >= MIN_SUPPORT_CONTACT_FORCE
            and max_penetration <= MAX_SUPPORT_PENETRATION_M
            and not other_geoms
            and not _direct_contacts(env)
        )
        head_corner = (
            np.isfinite(head_axial)
            and head_axial >= MIN_HEAD_AXIAL_M
            and np.isfinite(corner_gap)
            and corner_gap <= MAX_CORNER_XY_DISTANCE_M
        )
        empty_response = {
            "first_hazard_step": -1,
            "displacement_m": 0.0,
            "height_drop_m": 0.0,
            "attitude_change_deg": 0.0,
            "pre_hazard_other_geoms": "",
            "pre_hazard_direct": "",
        }
        counterfactual = dict(empty_response)
        front_only = dict(empty_response)
        side_only = dict(empty_response)
        if stable and head_corner:
            counterfactual = _counterfactual(
                env, settled_state, support_geoms, side_geom, front_geom,
                drawer_qadr, bottle_vadr, args.counterfactual_steps,
            )
            front_only = _counterfactual(
                env, settled_state, {front_geom}, side_geom, front_geom,
                drawer_qadr, bottle_vadr, args.counterfactual_steps,
            )
            side_only = _counterfactual(
                env, settled_state, {side_geom}, side_geom, front_geom,
                drawer_qadr, bottle_vadr, args.counterfactual_steps,
            )
        passed = (
            stable
            and head_corner
            and counterfactual["first_hazard_step"] >= 1
            and not counterfactual["pre_hazard_other_geoms"]
            and not counterfactual["pre_hazard_direct"]
        )
        rows.append({
            "dx": dx,
            "dy": dy,
            "lean_deg": lean,
            "direction_deg": direction,
            "settled_x_m": settled_pos[0],
            "settled_y_m": settled_pos[1],
            "settled_z_m": settled_pos[2],
            "settled_tilt_deg": settled_tilt,
            "support_geoms": ",".join(sorted(touched)),
            "head_axial_m": head_axial,
            "corner_xy_gap_m": corner_gap,
            "min_support_force": min_contact_force,
            "max_penetration_m": max_penetration,
            "support_coverage": support_coverage,
            "table_coverage": table_coverage,
            "hold_max_drift_m": hold_max_drift,
            "hold_max_attitude_deg": hold_max_attitude,
            "angular_speed_rad_s": angular_speed,
            "other_cabinet_geoms": ",".join(sorted(other_geoms)),
            "stable": stable,
            "head_corner": head_corner,
            **counterfactual,
            "front_only_first_hazard_step": front_only["first_hazard_step"],
            "front_only_displacement_m": front_only["displacement_m"],
            "front_only_attitude_change_deg": front_only["attitude_change_deg"],
            "side_only_first_hazard_step": side_only["first_hazard_step"],
            "side_only_displacement_m": side_only["displacement_m"],
            "side_only_attitude_change_deg": side_only["attitude_change_deg"],
            "verdict": "PASS" if passed else "FAIL",
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed_rows = [row for row in rows if row["verdict"] == "PASS"]
    out_report = Path(args.out_report)
    lines = [
        "# L3-A1 native corner geometry sweep",
        "",
        f"- Front geom: `{front_geom}`",
        f"- Right-side geom: `{side_geom}`",
        f"- Candidates: {len(rows)}",
        f"- Strict passes: {len(passed_rows)}",
        "",
        "| dx | dy | lean | dir | support | head axial | corner gap | dpos | datt | verdict |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dx']:.3f} | {row['dy']:.3f} | {row['lean_deg']:.1f} | "
            f"{row['direction_deg']:.1f} | {row['support_geoms'] or '—'} | "
            f"{row['head_axial_m']:.4f} | {row['corner_xy_gap_m']:.4f} | "
            f"{row['displacement_m']:.4f} | {row['attitude_change_deg']:.2f} | "
            f"{row['verdict']} |"
        )
    verdict = "PASS_L3A1_CORNER_SWEEP" if passed_rows else "FAIL_L3A1_CORNER_SWEEP"
    lines.extend(["", f"- Verdict: **{verdict}**", ""])
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"{verdict} candidates={len(rows)} passes={len(passed_rows)}")
    return 0 if passed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
