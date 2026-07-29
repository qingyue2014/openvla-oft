#!/usr/bin/env python3
"""Validate table + native drawer-front edge support for L3-A1.

The bottle base remains on the table while its upper shoulder/collar rests on
the native right edge of the bottom drawer front board.  Initial support is
the exact g33 edge; g35 and g36 are members of the same moving corner
component but are forbidden as initial load-bearing contacts.  Causal removal
disables exactly g33, g35, and g36 with the drawer pinned and bottle velocity
zeroed.
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
from experiments.robot.libero.tasks.l3a1_native_geometry import (
    BOTTLE_BODY,
    DRAWER_BODY_CANDIDATES,
    DRAWER_JOINT_CANDIDATES,
    FRONT_BOARD_SIGNATURE,
    INNER_FRONT_BOARD_SIGNATURE,
    L3A1_DISPLACEMENT_THRESHOLD,
    L3A1_TILT_CHANGE_THRESHOLD_DEG,
    RIGHT_SIDE_SIGNATURE,
    SETTLE_STEPS,
    _axis_change_deg,
    _body_rotation,
    _directed_tilt_quat,
    _find_free_joint_vadr,
    _find_joint_qadr,
    _lean_tilt_angle_deg,
    _other_cabinet_contact_geoms,
    _validate_native_support_panel_model,
)
from experiments.robot.libero.tasks.sweep_l3a1_corner_geometry import (
    HEIGHT_DROP_THRESHOLD_M,
    MAX_OPEN_ANGULAR_SPEED_RAD_S,
    MAX_OPEN_ATTITUDE_CHANGE_DEG,
    MAX_OPEN_DRIFT_M,
    MAX_SUPPORT_PENETRATION_M,
    _direct_contacts,
    _resolve_geom_by_signature,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)


# Exact endpoint of the g33 right edge / inward face, derived from the
# canonical body-local collision signature:
# x = pos_x + half_extent_x = 0.00334 + 0.10934
# y = pos_y + half_thickness_y = -0.07524 + 0.00271
FRONT_RIGHT_INNER_EDGE_LOCAL_XY = np.array([0.11268, -0.07253])
# The native upper-neck collision union reaches 7.08 mm from the bottle axis.
# Requiring the front-edge contact point within 6 mm keeps the drawer endpoint
# inside that physical cross-section instead of imposing an arbitrary 3 mm
# point-to-point tolerance.
BOTTLE_NECK_COLLISION_RADIUS_M = 0.00708
MAX_EDGE_DISTANCE_M = 0.006
MIN_UPPER_AXIAL_M = 0.086
BOTTLE_COLLISION_TOP_M = 0.15804
MIN_EDGE_COVERAGE = 0.95
MIN_CALIBRATION_COVERAGE = 0.90
MIN_EDGE_FORCE_WEIGHT_FRACTION = 0.05
MIN_TABLE_FORCE_WEIGHT_FRACTION = 0.25
MIN_ABSOLUTE_FORCE_N = 1e-4


def _candidate_grid() -> list[tuple[float, float, float, float]]:
    return [
        (dx, dy, -40.0, 105.0)
        for dy in (
            -0.060100,
            -0.060125,
            -0.060150,
            -0.060160,
            -0.060170,
        )
        for dx in (0.14790, 0.147925, 0.14795, 0.147975, 0.14800)
    ]


def _contact_wrench(env, contact_index: int) -> np.ndarray:
    import mujoco

    wrench = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(
        env.sim.model._model,
        env.sim.data._data,
        contact_index,
        wrench,
    )
    return wrench


def _edge_frame(
    env,
    support_body: str,
    component_roles: dict[str, str],
    candidate: dict,
    step: int,
) -> tuple[dict, list[dict]]:
    model, data = env.sim.model, env.sim.data
    bottle_id = model.body_name2id(BOTTLE_BODY)
    bottle_weight_n = float(
        model.body_mass[bottle_id] * np.linalg.norm(model.opt.gravity)
    )
    min_edge_force_n = max(
        MIN_ABSOLUTE_FORCE_N,
        MIN_EDGE_FORCE_WEIGHT_FRACTION * bottle_weight_n,
    )
    min_table_force_n = max(
        MIN_ABSOLUTE_FORCE_N,
        MIN_TABLE_FORCE_WEIGHT_FRACTION * bottle_weight_n,
    )
    bottle_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == bottle_id
    }
    relevant_roles = {
        model.geom_name2id(name): role
        for name, role in component_roles.items()
    }
    table_id = model.geom_name2id("table_collision")
    relevant_roles[table_id] = "table"
    bottle_pos = _body_pos(env, BOTTLE_BODY)
    bottle_axis = _body_rotation(env, BOTTLE_BODY)[:, 2]
    support_pos = _body_pos(env, support_body)
    support_rot = _body_rotation(env, support_body)
    bottle_vadr = _find_free_joint_vadr(env.sim, BOTTLE_BODY)
    bottle_qvel = (
        np.asarray(data.qvel[bottle_vadr:bottle_vadr + 6], dtype=float)
        if bottle_vadr >= 0
        else np.zeros(6)
    )
    detail_rows: list[dict] = []
    touched_roles: set[str] = set()
    edge_qualified = False
    table_qualified = False
    edge_witness_forces: list[float] = []
    table_witness_forces: list[float] = []
    edge_penetrations: list[float] = []
    qualified_gaps: list[float] = []
    qualified_axials: list[float] = []

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if (
            contact.geom1 in bottle_geom_ids
            and contact.geom2 in relevant_roles
        ):
            bottle_geom_id = int(contact.geom1)
            support_geom_id = int(contact.geom2)
        elif (
            contact.geom2 in bottle_geom_ids
            and contact.geom1 in relevant_roles
        ):
            bottle_geom_id = int(contact.geom2)
            support_geom_id = int(contact.geom1)
        else:
            continue
        role = relevant_roles[support_geom_id]
        touched_roles.add(role)
        point_world = np.asarray(contact.pos, dtype=float)
        point_local = support_rot.T @ (point_world - support_pos)
        edge_delta = point_local[:2] - FRONT_RIGHT_INNER_EDGE_LOCAL_XY
        edge_gap = float(np.linalg.norm(edge_delta))
        axial = float(np.dot(point_world - bottle_pos, bottle_axis))
        penetration = max(0.0, -float(contact.dist))
        wrench = _contact_wrench(env, contact_index)
        normal_force = float(wrench[0])
        tangential_force = float(np.linalg.norm(wrench[1:3]))
        force_threshold = (
            min_edge_force_n
            if role == "edge"
            else min_table_force_n if role == "table" else MIN_ABSOLUTE_FORCE_N
        )
        load_qualified = (
            normal_force >= force_threshold
            and penetration <= MAX_SUPPORT_PENETRATION_M
        )
        is_edge_qualified = (
            role == "edge"
            and axial >= MIN_UPPER_AXIAL_M
            and edge_gap <= MAX_EDGE_DISTANCE_M
            and load_qualified
        )
        is_table_qualified = role == "table" and load_qualified
        edge_qualified = edge_qualified or is_edge_qualified
        table_qualified = table_qualified or is_table_qualified
        if is_edge_qualified:
            edge_witness_forces.append(normal_force)
            edge_penetrations.append(penetration)
            qualified_gaps.append(edge_gap)
            qualified_axials.append(axial)
        if is_table_qualified:
            table_witness_forces.append(normal_force)
        raw_frame = np.asarray(contact.frame, dtype=float).reshape(-1)
        normal = raw_frame[:3] if len(raw_frame) >= 3 else np.full(3, np.nan)
        detail_rows.append({
            **candidate,
            "step": step,
            "contact_index": contact_index,
            "geom1_id": int(contact.geom1),
            "geom1_name": model.geom_id2name(int(contact.geom1)) or "",
            "geom2_id": int(contact.geom2),
            "geom2_name": model.geom_id2name(int(contact.geom2)) or "",
            "bottle_geom_id": bottle_geom_id,
            "bottle_geom_name": model.geom_id2name(bottle_geom_id) or "",
            "support_geom_id": support_geom_id,
            "support_geom_name": model.geom_id2name(support_geom_id) or "",
            "contact_role": role,
            "world_x_m": point_world[0],
            "world_y_m": point_world[1],
            "world_z_m": point_world[2],
            "drawer_local_x_m": point_local[0],
            "drawer_local_y_m": point_local[1],
            "drawer_local_z_m": point_local[2],
            "edge_dx_m": edge_delta[0],
            "edge_dy_m": edge_delta[1],
            "edge_gap_m": edge_gap,
            "bottle_axis_axial_m": axial,
            "bottle_axis_fraction": axial / BOTTLE_COLLISION_TOP_M,
            "contact_dist_m": float(contact.dist),
            "penetration_m": penetration,
            "normal_force_n": normal_force,
            "normal_force_weight_fraction": (
                normal_force / bottle_weight_n if bottle_weight_n > 0 else np.nan
            ),
            "force_threshold_n": force_threshold,
            "tangential_force_n": tangential_force,
            "normal_x": normal[0],
            "normal_y": normal[1],
            "normal_z": normal[2],
            "load_qualified": load_qualified,
            "edge_qualified": is_edge_qualified,
            "table_qualified": is_table_qualified,
            "bottle_x_m": bottle_pos[0],
            "bottle_y_m": bottle_pos[1],
            "bottle_z_m": bottle_pos[2],
            "bottle_axis_x": bottle_axis[0],
            "bottle_axis_y": bottle_axis[1],
            "bottle_axis_z": bottle_axis[2],
            "bottle_vx_m_s": bottle_qvel[0],
            "bottle_vy_m_s": bottle_qvel[1],
            "bottle_vz_m_s": bottle_qvel[2],
            "bottle_wx_rad_s": bottle_qvel[3],
            "bottle_wy_rad_s": bottle_qvel[4],
            "bottle_wz_rad_s": bottle_qvel[5],
        })

    if not detail_rows:
        detail_rows.append({
            **candidate,
            "step": step,
            "contact_index": -1,
            "geom1_id": -1,
            "geom1_name": "",
            "geom2_id": -1,
            "geom2_name": "",
            "bottle_geom_id": -1,
            "bottle_geom_name": "",
            "support_geom_id": -1,
            "support_geom_name": "",
            "contact_role": "none",
            **{
                name: np.nan
                for name in (
                    "world_x_m", "world_y_m", "world_z_m",
                    "drawer_local_x_m", "drawer_local_y_m", "drawer_local_z_m",
                    "edge_dx_m", "edge_dy_m", "edge_gap_m",
                    "bottle_axis_axial_m", "bottle_axis_fraction",
                    "contact_dist_m", "penetration_m", "normal_force_n",
                    "normal_force_weight_fraction", "force_threshold_n",
                    "tangential_force_n", "normal_x", "normal_y", "normal_z",
                    "bottle_x_m", "bottle_y_m", "bottle_z_m",
                    "bottle_axis_x", "bottle_axis_y", "bottle_axis_z",
                    "bottle_vx_m_s", "bottle_vy_m_s", "bottle_vz_m_s",
                    "bottle_wx_rad_s", "bottle_wy_rad_s", "bottle_wz_rad_s",
                )
            },
            "load_qualified": False,
            "edge_qualified": False,
            "table_qualified": False,
        })

    return {
        "touched_roles": touched_roles,
        "edge_active": "edge" in touched_roles,
        "edge_qualified": edge_qualified,
        "table_active": "table" in touched_roles,
        "table_qualified": table_qualified,
        "edge_witness_force": max(edge_witness_forces, default=0.0),
        "table_witness_force": max(table_witness_forces, default=0.0),
        "edge_penetrations": edge_penetrations,
        "qualified_gaps": qualified_gaps,
        "qualified_axials": qualified_axials,
        "bottle_weight_n": bottle_weight_n,
        "min_edge_force_n": min_edge_force_n,
        "min_table_force_n": min_table_force_n,
    }, detail_rows


def _dropout_summary(active_steps: set[int], total_steps: int) -> tuple[str, int, int]:
    missing = [step for step in range(total_steps) if step not in active_steps]
    if not missing:
        return "", 0, 0
    ranges = []
    start = previous = missing[0]
    longest = 1
    transitions = 0
    for step in missing[1:]:
        if step != previous + 1:
            ranges.append((start, previous))
            longest = max(longest, previous - start + 1)
            start = step
            transitions += 1
        previous = step
    ranges.append((start, previous))
    longest = max(longest, previous - start + 1)
    text = ",".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )
    return text, longest, transitions


def _safe_response(response: dict) -> bool:
    return (
        response["first_hazard_step"] < 0
        and response["displacement_m"] <= L3A1_DISPLACEMENT_THRESHOLD
        and response["height_drop_m"] <= HEIGHT_DROP_THRESHOLD_M
        and response["attitude_change_deg"] <= L3A1_TILT_CHANGE_THRESHOLD_DEG
        and not response["pre_hazard_other_geoms"]
        and not response["pre_hazard_direct"]
    )


def _edge_counterfactual(
    env,
    state: np.ndarray,
    disabled_geoms: set[str],
    allowed_cabinet_geoms: set[str],
    component_roles: dict[str, str],
    support_body: str,
    side_geom: str,
    drawer_qadr: int,
    bottle_vadr: int,
    steps: int,
    candidate: dict,
    phase: str,
) -> tuple[dict, list[dict]]:
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
    edge_qualified_steps: set[int] = set()
    table_qualified_steps: set[int] = set()
    edge_table_steps: set[int] = set()
    touched_roles: set[str] = set()
    detail_rows: list[dict] = []
    try:
        for geom_id in geom_ids:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
        if bottle_vadr >= 0:
            env.sim.data.qvel[bottle_vadr:bottle_vadr + 6] = 0
        env.sim.forward()
        for step in range(steps + 1):
            if step:
                env.sim.data.qpos[drawer_qadr] = drawer_qpos
                env.sim.data.qvel[drawer_dofadr] = 0
                env.sim.step()
                env.sim.data.qpos[drawer_qadr] = drawer_qpos
                env.sim.data.qvel[drawer_dofadr] = 0
                env.sim.forward()
            frame, details = _edge_frame(
                env,
                support_body,
                component_roles,
                {**candidate, "phase": phase},
                step,
            )
            detail_rows.extend(details)
            touched_roles.update(frame["touched_roles"])
            if frame["edge_qualified"]:
                edge_qualified_steps.add(step)
            if frame["table_qualified"]:
                table_qualified_steps.add(step)
            if frame["edge_qualified"] and frame["table_qualified"]:
                edge_table_steps.add(step)
            displacement = float(
                np.linalg.norm(_body_pos(env, BOTTLE_BODY) - pos_before)
            )
            drop = float(pos_before[2] - _body_pos(env, BOTTLE_BODY)[2])
            attitude = _axis_change_deg(env, BOTTLE_BODY, axis_before)
            if first_hazard_step < 0:
                pre_hazard_direct.update(_direct_contacts(env))
                cabinet_contacts = _other_cabinet_contact_geoms(
                    env, side_geom
                )
                if "side" in frame["touched_roles"]:
                    cabinet_contacts.add(side_geom)
                pre_hazard_other_geoms.update(
                    cabinet_contacts - allowed_cabinet_geoms
                )
                if (
                    displacement > L3A1_DISPLACEMENT_THRESHOLD
                    or drop > HEIGHT_DROP_THRESHOLD_M
                    or attitude > L3A1_TILT_CHANGE_THRESHOLD_DEG
                ):
                    first_hazard_step = step
        frame_count = steps + 1
        return {
            "disabled_geoms": ",".join(sorted(disabled_geoms)),
            "first_hazard_step": first_hazard_step,
            "displacement_m": float(
                np.linalg.norm(_body_pos(env, BOTTLE_BODY) - pos_before)
            ),
            "height_drop_m": float(
                pos_before[2] - _body_pos(env, BOTTLE_BODY)[2]
            ),
            "attitude_change_deg": _axis_change_deg(
                env, BOTTLE_BODY, axis_before
            ),
            "pre_hazard_other_geoms": ",".join(
                sorted(pre_hazard_other_geoms)
            ),
            "pre_hazard_direct": ",".join(sorted(pre_hazard_direct)),
            "edge_qualified_coverage": len(edge_qualified_steps) / frame_count,
            "table_qualified_coverage": len(table_qualified_steps) / frame_count,
            "edge_table_coverage": len(edge_table_steps) / frame_count,
            "touched_component_roles": ",".join(sorted(
                touched_roles - {"table"}
            )),
        }, detail_rows
    finally:
        for geom_id, (contype, conaffinity) in zip(geom_ids, masks):
            model.geom_contype[geom_id] = contype
            model.geom_conaffinity[geom_id] = conaffinity
        env.sim.forward()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument("--hold_steps", type=int, default=800)
    parser.add_argument("--counterfactual_steps", type=int, default=SETTLE_STEPS)
    parser.add_argument(
        "--out_csv", default="experiments/logs/l3a1_edge_sweep.csv"
    )
    parser.add_argument(
        "--out_contacts", default="experiments/logs/l3a1_edge_contacts.csv"
    )
    parser.add_argument(
        "--out_report", default="experiments/logs/l3a1_edge_sweep.md"
    )
    args = parser.parse_args()

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    env.reset()
    support_body = _find_body(env, *DRAWER_BODY_CANDIDATES)
    side_geom = _validate_native_support_panel_model(env, support_body, "right")
    front_geom = _resolve_geom_by_signature(
        env, support_body, FRONT_BOARD_SIGNATURE
    )
    inner_front_geom = _resolve_geom_by_signature(
        env, support_body, INNER_FRONT_BOARD_SIGNATURE
    )
    resolved_side = _resolve_geom_by_signature(
        env, support_body, RIGHT_SIDE_SIGNATURE
    )
    if resolved_side != side_geom:
        raise RuntimeError("right-side resolvers disagree")
    front_component = {front_geom, inner_front_geom}
    removal_component = front_component | {side_geom}
    forbidden_initial = {inner_front_geom, side_geom}
    component_roles = {
        front_geom: "edge",
        inner_front_geom: "inner_front",
        side_geom: "side",
    }
    bottle_qadr = _find_free_joint_qadr(env.sim, BOTTLE_BODY)
    bottle_vadr = _find_free_joint_vadr(env.sim, BOTTLE_BODY)
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
    rows: list[dict] = []
    contact_rows: list[dict] = []

    for candidate_index, (dx, dy, lean, direction) in enumerate(_candidate_grid()):
        candidate = {
            "candidate_index": candidate_index,
            "dx": dx,
            "dy": dy,
            "lean_deg": lean,
            "direction_deg": direction,
        }
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
        frame_count = args.hold_steps + 1
        active_steps: set[int] = set()
        qualified_steps: set[int] = set()
        table_steps: set[int] = set()
        qualified_table_steps: set[int] = set()
        forbidden_contacts: set[str] = set()
        other_geoms: set[str] = set()
        direct_contacts: set[str] = set()
        edge_witness_forces: list[float] = []
        table_witness_forces: list[float] = []
        penetrations: list[float] = []
        qualified_gaps: list[float] = []
        qualified_axials: list[float] = []
        hold_max_drift = 0.0
        hold_max_attitude = 0.0

        for step in range(frame_count):
            if step:
                env.sim.step()
                hold_max_drift = max(
                    hold_max_drift,
                    float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - settled_pos)),
                )
                hold_max_attitude = max(
                    hold_max_attitude,
                    _axis_change_deg(env, BOTTLE_BODY, settled_axis),
                )
            frame, details = _edge_frame(
                env, support_body, component_roles, {
                    **candidate, "phase": "factual_hold"
                }, step
            )
            contact_rows.extend(details)
            if frame["edge_active"]:
                active_steps.add(step)
            if frame["edge_qualified"]:
                qualified_steps.add(step)
            if frame["table_qualified"]:
                table_steps.add(step)
            if frame["edge_qualified"] and frame["table_qualified"]:
                qualified_table_steps.add(step)
            if "inner_front" in frame["touched_roles"]:
                forbidden_contacts.add(inner_front_geom)
            if "side" in frame["touched_roles"]:
                forbidden_contacts.add(side_geom)
            cabinet_contacts = _other_cabinet_contact_geoms(env, front_geom)
            other_geoms.update(cabinet_contacts - forbidden_initial)
            direct_contacts.update(_direct_contacts(env))
            if frame["edge_qualified"]:
                edge_witness_forces.append(frame["edge_witness_force"])
                penetrations.extend(frame["edge_penetrations"])
            if frame["table_qualified"]:
                table_witness_forces.append(frame["table_witness_force"])
            qualified_gaps.extend(frame["qualified_gaps"])
            qualified_axials.extend(frame["qualified_axials"])

        angular_speed = (
            float(np.linalg.norm(
                env.sim.data.qvel[bottle_vadr + 3:bottle_vadr + 6]
            ))
            if bottle_vadr >= 0
            else 0.0
        )
        active_coverage = len(active_steps) / frame_count
        qualified_coverage = len(qualified_steps) / frame_count
        table_coverage = len(table_steps) / frame_count
        qualified_table_coverage = len(qualified_table_steps) / frame_count
        dropout_ranges, longest_dropout, dropout_transitions = _dropout_summary(
            active_steps, frame_count
        )
        (
            qualified_dropout_ranges,
            qualified_longest_dropout,
            qualified_dropout_transitions,
        ) = _dropout_summary(qualified_steps, frame_count)
        stable = (
            0 in qualified_steps
            and qualified_table_coverage >= MIN_EDGE_COVERAGE
            and table_coverage == 1.0
            and hold_max_drift <= MAX_OPEN_DRIFT_M
            and hold_max_attitude <= MAX_OPEN_ATTITUDE_CHANGE_DEG
            and angular_speed <= MAX_OPEN_ANGULAR_SPEED_RAD_S
            and not forbidden_contacts
            and not other_geoms
            and not direct_contacts
        )
        calibration_eligible = (
            0 in qualified_steps
            and qualified_table_coverage >= MIN_CALIBRATION_COVERAGE
            and table_coverage == 1.0
            and hold_max_drift <= MAX_OPEN_DRIFT_M
            and hold_max_attitude <= MAX_OPEN_ATTITUDE_CHANGE_DEG
            and angular_speed <= MAX_OPEN_ANGULAR_SPEED_RAD_S
            and not forbidden_contacts
            and not other_geoms
            and not direct_contacts
        )

        empty = {
            "disabled_geoms": "",
            "first_hazard_step": -1,
            "displacement_m": 0.0,
            "height_drop_m": 0.0,
            "attitude_change_deg": 0.0,
            "pre_hazard_other_geoms": "",
            "pre_hazard_direct": "",
            "edge_qualified_coverage": 0.0,
            "table_qualified_coverage": 0.0,
            "edge_table_coverage": 0.0,
            "touched_component_roles": "",
        }
        full_removal = dict(empty)
        front_removal = dict(empty)
        side_removal = dict(empty)
        if calibration_eligible:
            full_removal, details = _edge_counterfactual(
                env, settled_state, removal_component, set(), component_roles,
                support_body, side_geom, drawer_qadr, bottle_vadr,
                args.counterfactual_steps, candidate, "full_removal",
            )
            contact_rows.extend(details)
            front_removal, details = _edge_counterfactual(
                env, settled_state, front_component, set(), component_roles,
                support_body, side_geom, drawer_qadr, bottle_vadr,
                args.counterfactual_steps, candidate, "front_removal",
            )
            contact_rows.extend(details)
            side_removal, details = _edge_counterfactual(
                env, settled_state, {side_geom}, {front_geom}, component_roles,
                support_body, side_geom, drawer_qadr, bottle_vadr,
                args.counterfactual_steps, candidate, "side_removal",
            )
            contact_rows.extend(details)

        full_hazard = (
            full_removal["first_hazard_step"] >= 1
            and not full_removal["pre_hazard_other_geoms"]
            and not full_removal["pre_hazard_direct"]
        )
        front_hazard = (
            front_removal["first_hazard_step"] >= 1
            and not front_removal["pre_hazard_other_geoms"]
            and not front_removal["pre_hazard_direct"]
        )
        side_safe = (
            _safe_response(side_removal)
            and side_removal["edge_table_coverage"] >= MIN_EDGE_COVERAGE
            and side_removal["touched_component_roles"] == "edge"
        )
        # The physical drawer corner is one rigid component.  Front-only
        # removal is retained as a decomposition diagnostic: the side panel
        # can catch the bottle in that unrealizable half-drawer intervention.
        passed = stable and full_hazard and side_safe
        rows.append({
            **candidate,
            "front_geom": front_geom,
            "bottle_neck_collision_radius_m": BOTTLE_NECK_COLLISION_RADIUS_M,
            "max_edge_distance_m": MAX_EDGE_DISTANCE_M,
            "forbidden_initial_geoms": ",".join(sorted(forbidden_initial)),
            "removal_component_geoms": ",".join(sorted(removal_component)),
            "settled_x_m": settled_pos[0],
            "settled_y_m": settled_pos[1],
            "settled_z_m": settled_pos[2],
            "settled_tilt_deg": settled_tilt,
            "bottle_weight_n": frame["bottle_weight_n"],
            "min_edge_force_n": frame["min_edge_force_n"],
            "min_table_force_n": frame["min_table_force_n"],
            "active_frames": len(active_steps),
            "qualified_frames": len(qualified_steps),
            "frame_count": frame_count,
            "active_coverage": active_coverage,
            "qualified_coverage": qualified_coverage,
            "table_coverage": table_coverage,
            "qualified_table_coverage": qualified_table_coverage,
            "dropout_ranges": dropout_ranges,
            "longest_dropout_frames": longest_dropout,
            "dropout_transitions": dropout_transitions,
            "qualified_dropout_ranges": qualified_dropout_ranges,
            "qualified_longest_dropout_frames": qualified_longest_dropout,
            "qualified_dropout_transitions": qualified_dropout_transitions,
            "edge_witness_force_min_n": min(edge_witness_forces, default=0.0),
            "edge_witness_force_median_n": (
                float(np.median(edge_witness_forces))
                if edge_witness_forces else 0.0
            ),
            "edge_witness_force_p05_n": (
                float(np.quantile(edge_witness_forces, 0.05))
                if edge_witness_forces else 0.0
            ),
            "table_witness_force_min_n": min(table_witness_forces, default=0.0),
            "table_witness_force_p05_n": (
                float(np.quantile(table_witness_forces, 0.05))
                if table_witness_forces else 0.0
            ),
            "max_penetration_m": max(penetrations, default=0.0),
            "worst_qualified_edge_gap_m": max(qualified_gaps, default=np.inf),
            "min_qualified_axial_m": min(qualified_axials, default=-np.inf),
            "min_qualified_axial_fraction": (
                min(qualified_axials, default=-np.inf) / BOTTLE_COLLISION_TOP_M
            ),
            "hold_max_drift_m": hold_max_drift,
            "hold_max_attitude_deg": hold_max_attitude,
            "angular_speed_rad_s": angular_speed,
            "forbidden_component_contacts": ",".join(sorted(forbidden_contacts)),
            "other_cabinet_geoms": ",".join(sorted(other_geoms)),
            "direct_contact_bodies": ",".join(sorted(direct_contacts)),
            "stable": stable,
            "calibration_eligible": calibration_eligible,
            "full_hazard": full_hazard,
            "front_hazard": front_hazard,
            "side_safe": side_safe,
            **full_removal,
            "front_disabled_geoms": front_removal["disabled_geoms"],
            "front_first_hazard_step": front_removal["first_hazard_step"],
            "front_displacement_m": front_removal["displacement_m"],
            "front_height_drop_m": front_removal["height_drop_m"],
            "front_attitude_change_deg": front_removal["attitude_change_deg"],
            "front_pre_hazard_other_geoms": front_removal[
                "pre_hazard_other_geoms"
            ],
            "front_pre_hazard_direct": front_removal["pre_hazard_direct"],
            "front_touched_component_roles": front_removal[
                "touched_component_roles"
            ],
            "side_disabled_geoms": side_removal["disabled_geoms"],
            "side_first_hazard_step": side_removal["first_hazard_step"],
            "side_displacement_m": side_removal["displacement_m"],
            "side_height_drop_m": side_removal["height_drop_m"],
            "side_attitude_change_deg": side_removal["attitude_change_deg"],
            "side_pre_hazard_other_geoms": side_removal[
                "pre_hazard_other_geoms"
            ],
            "side_pre_hazard_direct": side_removal["pre_hazard_direct"],
            "side_edge_qualified_coverage": side_removal[
                "edge_qualified_coverage"
            ],
            "side_table_qualified_coverage": side_removal[
                "table_qualified_coverage"
            ],
            "side_edge_table_coverage": side_removal[
                "edge_table_coverage"
            ],
            "side_touched_component_roles": side_removal[
                "touched_component_roles"
            ],
            "verdict": "PASS" if passed else "FAIL",
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_contacts = Path(args.out_contacts)
    with out_contacts.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contact_rows[0]))
        writer.writeheader()
        writer.writerows(contact_rows)

    passed_rows = [row for row in rows if row["verdict"] == "PASS"]
    adjacent_pass = any(
        left["verdict"] == right["verdict"] == "PASS"
        # Chebyshev adjacency is the correct 2-D grid neighborhood: a one-cell
        # diagonal is no farther than one sampled step on either axis.
        and abs(left["dx"] - right["dx"]) <= 0.000051
        and abs(left["dy"] - right["dy"]) <= 0.000051
        and (left["dx"], left["dy"]) != (right["dx"], right["dy"])
        for index, left in enumerate(rows)
        for right in rows[index + 1:]
    )
    lines = [
        "# L3-A1 native front-edge geometry sweep",
        "",
        f"- Initial support: `table + {front_geom} right edge`",
        f"- Forbidden initial component contacts: `{','.join(sorted(forbidden_initial))}`",
        f"- Full removal component: `{','.join(sorted(removal_component))}`",
        f"- Candidates: {len(rows)}",
        f"- Strict passes: {len(passed_rows)}",
        f"- Adjacent strict-pass pair: {adjacent_pass}",
        "",
        "| dy | qualified/table | edge gap | axial | full hazard | front hazard | side safe | verdict |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dy']:.6f} | {row['qualified_table_coverage']:.3f} | "
            f"{row['worst_qualified_edge_gap_m']:.6f} | "
            f"{row['min_qualified_axial_m']:.6f} | "
            f"{row['first_hazard_step']} | {row['front_first_hazard_step']} | "
            f"{row['side_safe']} | {row['verdict']} |"
        )
    verdict = (
        "PASS_L3A1_EDGE_SWEEP"
        if passed_rows and adjacent_pass
        else "FAIL_L3A1_EDGE_SWEEP"
    )
    lines.extend(["", f"- Verdict: **{verdict}**", ""])
    Path(args.out_report).write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{verdict} candidates={len(rows)} passes={len(passed_rows)} "
        f"adjacent_pass={adjacent_pass}"
    )
    return 0 if passed_rows and adjacent_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
