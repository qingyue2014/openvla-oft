#!/usr/bin/env python3
"""Export a small policy-view video for the validated native L3-A1 edge pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
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
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT_CANDIDATES,
    FRONT_BOARD_SIGNATURE,
    INNER_FRONT_BOARD_SIGNATURE,
    L3A1_DISPLACEMENT_THRESHOLD,
    L3A1_TILT_CHANGE_THRESHOLD_DEG,
    RIGHT_SIDE_SIGNATURE,
    SETTLE_STEPS,
    _axis_change_deg,
    _body_rotation,
    _contact_geom_names,
    _directed_tilt_quat,
    _find_joint_qadr,
    _lean_tilt_angle_deg,
    _other_cabinet_contact_geoms,
)
from experiments.robot.libero.tasks.sweep_l3a1_corner_geometry import (
    _resolve_geom_by_signature,
)
from experiments.robot.libero.tasks.sweep_l3a1_edge_geometry import (
    _edge_frame,
)


SELECTED_DX = 0.147925
SELECTED_DY = -0.060125
SELECTED_LEAN_DEG = -40.0
SELECTED_DIRECTION_DEG = 105.0
HEIGHT_DROP_THRESHOLD_M = 0.015
MIN_FINAL_TABLE_FALL_TILT_DEG = 80.0


def policy_agentview(obs: dict) -> np.ndarray:
    """Apply the exact orientation used by the LIBERO policy evaluator."""
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(
            f"expected 256x256 RGB agentview_image, got {image.shape}"
        )
    return np.ascontiguousarray(image[::-1, ::-1])


def _render_policy_frame(env) -> np.ndarray:
    state = np.asarray(env.sim.get_state().flatten()).copy()
    obs = env.regenerate_obs_from_state(state)
    refreshed = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(state, refreshed):
        raise RuntimeError("policy observation refresh changed simulator state")
    return policy_agentview(obs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a1_edge_preview"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--hold_frames", type=int, default=30)
    parser.add_argument("--close_frames", type=int, default=60)
    parser.add_argument("--post_frames", type=int, default=120)
    parser.add_argument("--post_steps_per_frame", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    env.reset()
    support_body = _find_body(env, *DRAWER_BODY_CANDIDATES)
    front_geom = _resolve_geom_by_signature(
        env, support_body, FRONT_BOARD_SIGNATURE
    )
    inner_front_geom = _resolve_geom_by_signature(
        env, support_body, INNER_FRONT_BOARD_SIGNATURE
    )
    side_geom = _resolve_geom_by_signature(
        env, support_body, RIGHT_SIDE_SIGNATURE
    )
    component_roles = {
        front_geom: "edge",
        inner_front_geom: "inner_front",
        side_geom: "side",
    }
    clearance_component = set(component_roles)
    bottle_qadr = _find_free_joint_qadr(env.sim, BOTTLE_BODY)
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
    drawer_joint_ids = np.flatnonzero(env.sim.model.jnt_qposadr == drawer_qadr)
    if len(drawer_joint_ids) != 1:
        raise RuntimeError("cannot resolve drawer dof")
    drawer_dofadr = int(
        env.sim.model.jnt_dofadr[int(drawer_joint_ids[0])]
    )
    support_pos = _body_pos(env, support_body)
    native_z = float(env.sim.data.qpos[bottle_qadr + 2])
    env.sim.data.qpos[bottle_qadr:bottle_qadr + 2] = (
        support_pos[:2] + [SELECTED_DX, SELECTED_DY]
    )
    env.sim.data.qpos[bottle_qadr + 2] = native_z
    env.sim.data.qpos[bottle_qadr + 3:bottle_qadr + 7] = _directed_tilt_quat(
        "x", SELECTED_LEAN_DEG, SELECTED_DIRECTION_DEG
    )
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    candidate_state = np.asarray(env.sim.get_state().flatten()).copy()
    initial_pos = _body_pos(env, BOTTLE_BODY).copy()
    initial_axis = _body_rotation(env, BOTTLE_BODY)[:, 2].copy()
    initial_frame, initial_details = _edge_frame(
        env,
        support_body,
        component_roles,
        {
            "candidate_index": 0,
            "dx": SELECTED_DX,
            "dy": SELECTED_DY,
            "lean_deg": SELECTED_LEAN_DEG,
            "direction_deg": SELECTED_DIRECTION_DEG,
            "phase": "preview_initial",
        },
        0,
    )
    if (
        not initial_frame["edge_qualified"]
        or not initial_frame["table_qualified"]
        or initial_frame["touched_roles"] - {"edge", "table"}
    ):
        raise RuntimeError(
            f"selected preview state lost validated topology: {initial_frame}"
        )

    frames: list[np.ndarray] = []
    frames.append(_render_policy_frame(env))
    imageio.imwrite(out_dir / "initial_policy_agentview.png", frames[0])
    for _ in range(max(0, args.hold_frames - 1)):
        env.sim.step()
        frames.append(_render_policy_frame(env))

    # Restore the exact validated state so the scripted close is not affected
    # by the display hold.
    env.sim.set_state_from_flattened(candidate_state)
    env.sim.forward()
    start_qpos = float(env.sim.data.qpos[drawer_qadr])
    first_oracle_step = -1
    timeline = []

    def record(step: int, phase: str) -> None:
        nonlocal first_oracle_step
        contacts = _contact_geom_names(env, BOTTLE_BODY)
        active_component = sorted(contacts & clearance_component)
        displacement = float(
            np.linalg.norm(_body_pos(env, BOTTLE_BODY) - initial_pos)
        )
        height_drop = float(
            initial_pos[2] - _body_pos(env, BOTTLE_BODY)[2]
        )
        attitude = _axis_change_deg(env, BOTTLE_BODY, initial_axis)
        if first_oracle_step < 0 and (
            displacement > L3A1_DISPLACEMENT_THRESHOLD
            or height_drop > HEIGHT_DROP_THRESHOLD_M
            or attitude > L3A1_TILT_CHANGE_THRESHOLD_DEG
        ):
            first_oracle_step = step
        timeline.append({
            "step": step,
            "phase": phase,
            "drawer_qpos": float(env.sim.data.qpos[drawer_qadr]),
            "component_contacts": active_component,
            "displacement_m": displacement,
            "height_drop_m": height_drop,
            "attitude_change_deg": attitude,
        })

    step = 0
    for index in range(args.close_frames):
        frac = (index + 1) / args.close_frames
        env.sim.data.qpos[drawer_qadr] = (
            start_qpos + frac * (DRAWER_CLOSED_QPOS - start_qpos)
        )
        env.sim.data.qvel[drawer_dofadr] = 0
        env.sim.forward()
        env.sim.step()
        env.sim.data.qpos[drawer_qadr] = (
            start_qpos + frac * (DRAWER_CLOSED_QPOS - start_qpos)
        )
        env.sim.data.qvel[drawer_dofadr] = 0
        env.sim.forward()
        step += 1
        record(step, "close")
        frames.append(_render_policy_frame(env))
    for _ in range(args.post_frames):
        for _ in range(args.post_steps_per_frame):
            env.sim.data.qpos[drawer_qadr] = DRAWER_CLOSED_QPOS
            env.sim.data.qvel[drawer_dofadr] = 0
            env.sim.step()
            env.sim.data.qpos[drawer_qadr] = DRAWER_CLOSED_QPOS
            env.sim.data.qvel[drawer_dofadr] = 0
            env.sim.forward()
            step += 1
            record(step, "post_release")
        frames.append(_render_policy_frame(env))

    component_contact_steps = [
        row["step"] for row in timeline if row["component_contacts"]
    ]
    release_step = (
        max(component_contact_steps) + 1
        if component_contact_steps
        and max(component_contact_steps) < timeline[-1]["step"]
        else -1
    )
    recontact_after_release = (
        release_step >= 1
        and any(
            row["step"] >= release_step and row["component_contacts"]
            for row in timeline
        )
    )
    pre_release_rows = [
        row for row in timeline
        if release_step < 0 or row["step"] <= release_step
    ]
    max_pre_release_displacement = max(
        (row["displacement_m"] for row in pre_release_rows), default=0.0
    )
    max_pre_release_attitude = max(
        (row["attitude_change_deg"] for row in pre_release_rows), default=0.0
    )
    video_path = out_dir / "native_edge_policy_agentview.mp4"
    imageio.mimsave(
        video_path,
        frames,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    imageio.imwrite(out_dir / "final_policy_agentview.png", frames[-1])
    final_displacement = float(
        np.linalg.norm(_body_pos(env, BOTTLE_BODY) - initial_pos)
    )
    final_attitude = _axis_change_deg(env, BOTTLE_BODY, initial_axis)
    final_tilt = _lean_tilt_angle_deg(env, BOTTLE_BODY)
    final_contacts = _contact_geom_names(env, BOTTLE_BODY)
    final_component_contacts = final_contacts & clearance_component
    final_cabinet_contacts = _other_cabinet_contact_geoms(
        env, front_geom
    ) | final_component_contacts
    final_table_contact = "table_collision" in final_contacts
    final_table_fall = (
        final_tilt >= MIN_FINAL_TABLE_FALL_TILT_DEG
        and final_table_contact
        and not final_cabinet_contacts
    )
    report = {
        "verdict": (
            "PASS_L3A1_EDGE_PREVIEW"
            if (
                release_step >= 1
                and first_oracle_step > release_step
                and not recontact_after_release
                and final_table_fall
            )
            else "FAIL_L3A1_EDGE_PREVIEW"
        ),
        "policy_view_transform": "agentview_image[::-1, ::-1]",
        "video": str(video_path),
        "frames": len(frames),
        "fps": args.fps,
        "candidate": {
            "dx": SELECTED_DX,
            "dy": SELECTED_DY,
            "lean_deg": SELECTED_LEAN_DEG,
            "direction_deg": SELECTED_DIRECTION_DEG,
        },
        "support_body": support_body,
        "initial_support_geom": front_geom,
        "clearance_component_geoms": sorted(clearance_component),
        "initial_contact_rows": initial_details,
        "release_step": release_step,
        "first_oracle_step": first_oracle_step,
        "factual_close_modified_bottle_velocity": False,
        "recontact_after_release": recontact_after_release,
        "max_pre_release_displacement_m": max_pre_release_displacement,
        "max_pre_release_attitude_change_deg": max_pre_release_attitude,
        "final_displacement_m": final_displacement,
        "final_attitude_change_deg": final_attitude,
        "final_tilt_deg": final_tilt,
        "min_final_table_fall_tilt_deg": MIN_FINAL_TABLE_FALL_TILT_DEG,
        "final_contacts": sorted(final_contacts),
        "final_table_contact": final_table_contact,
        "final_cabinet_contacts": sorted(final_cabinet_contacts),
        "final_table_fall": final_table_fall,
        "timeline": timeline,
    }
    (out_dir / "preview_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "\n".join([
            "# L3-A1 native edge policy-view preview",
            "",
            f"- Verdict: **{report['verdict']}**",
            f"- Video: `{video_path.name}`",
            f"- Release step: {release_step}",
            f"- First oracle step: {first_oracle_step}",
            f"- Final displacement: {final_displacement:.4f} m",
            f"- Final attitude change: {final_attitude:.2f} deg",
            f"- Final absolute tilt: {final_tilt:.2f} deg",
            f"- Final table contact: {final_table_contact}",
            f"- Final cabinet contacts: `{','.join(sorted(final_cabinet_contacts))}`",
            f"- Final table-fall gate: {final_table_fall}",
            "",
        ]),
        encoding="utf-8",
    )
    env.close()
    # Keep the verdict machine-readable for physcog_remote_agent.py.  A bare
    # PASS token is useful to a human but is intentionally not accepted by the
    # remote classifier because unrelated subprocess output may contain one.
    print(f"verdict={report['verdict']} video={video_path}")
    return 0 if report["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
