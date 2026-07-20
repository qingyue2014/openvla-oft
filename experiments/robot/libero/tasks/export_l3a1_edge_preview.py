#!/usr/bin/env python3
"""Export a small policy-view video for the validated native L3-A1 edge pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.export_l3a1_init_evidence import (
    policy_agentview,
)
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
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT_CANDIDATES,
    L3A1_DISPLACEMENT_THRESHOLD,
    L3A1_TILT_CHANGE_THRESHOLD_DEG,
    SETTLE_STEPS,
    _axis_change_deg,
    _body_rotation,
    _contact_geom_names,
    _directed_tilt_quat,
    _find_free_joint_vadr,
    _find_joint_qadr,
)
from experiments.robot.libero.tasks.sweep_l3a1_corner_geometry import (
    FRONT_BOARD_SIGNATURE,
    INNER_FRONT_BOARD_SIGNATURE,
    RIGHT_SIDE_SIGNATURE,
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


def _render_policy_frame(env) -> np.ndarray:
    state = np.asarray(env.sim.get_state().flatten()).copy()
    obs = env.regenerate_obs_from_state(state)
    refreshed = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(state, refreshed):
        raise RuntimeError("policy observation refresh changed simulator state")
    return policy_agentview(obs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
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
    bottle_vadr = _find_free_joint_vadr(env.sim, BOTTLE_BODY)
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
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
    release_step = -1
    first_oracle_step = -1
    recontact_after_release = False
    release_velocity_zeroed = False
    max_pre_release_displacement = 0.0
    max_pre_release_attitude = 0.0
    timeline = []

    def record(step: int, phase: str) -> None:
        nonlocal release_step, first_oracle_step, recontact_after_release
        nonlocal release_velocity_zeroed
        nonlocal max_pre_release_displacement, max_pre_release_attitude
        contacts = _contact_geom_names(env, BOTTLE_BODY)
        active_component = sorted(contacts & clearance_component)
        displacement = float(
            np.linalg.norm(_body_pos(env, BOTTLE_BODY) - initial_pos)
        )
        height_drop = float(
            initial_pos[2] - _body_pos(env, BOTTLE_BODY)[2]
        )
        attitude = _axis_change_deg(env, BOTTLE_BODY, initial_axis)
        if release_step < 0:
            max_pre_release_displacement = max(
                max_pre_release_displacement, displacement
            )
            max_pre_release_attitude = max(
                max_pre_release_attitude, attitude
            )
            if not active_component:
                release_step = step
                if bottle_vadr >= 0:
                    env.sim.data.qvel[bottle_vadr:bottle_vadr + 6] = 0
                    env.sim.forward()
                    release_velocity_zeroed = True
        elif active_component:
            recontact_after_release = True
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
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        env.sim.step()
        step += 1
        record(step, "close")
        frames.append(_render_policy_frame(env))
    for _ in range(args.post_frames):
        for _ in range(args.post_steps_per_frame):
            env.sim.step()
            step += 1
            record(step, "post_release")
        frames.append(_render_policy_frame(env))

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
    report = {
        "verdict": (
            "PASS_L3A1_EDGE_PREVIEW"
            if (
                release_step >= 1
                and first_oracle_step > release_step
                and release_velocity_zeroed
                and not recontact_after_release
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
        "release_velocity_zeroed": release_velocity_zeroed,
        "recontact_after_release": recontact_after_release,
        "max_pre_release_displacement_m": max_pre_release_displacement,
        "max_pre_release_attitude_change_deg": max_pre_release_attitude,
        "final_displacement_m": final_displacement,
        "final_attitude_change_deg": final_attitude,
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
            "",
        ]),
        encoding="utf-8",
    )
    env.close()
    print(report["verdict"], video_path)
    return 0 if report["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
