#!/usr/bin/env python3
"""Validate exact L3-A4 states, causal physics, and policy-view artifacts.

The script emits separate physical and visual verdicts. Generated PNG/MP4 files
are not a visual PASS until ``manual_review.json`` explicitly approves every
object in every Eb/Er/Ec condition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import h5py
import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import experiments.robot.libero.physcog_objects  # noqa: F401
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
    CHAIN_BODIES,
    CLOSE_STEPS,
    DEFAULT_BDDL,
    DRAWER_BODY_CANDIDATES,
    DRAWER_TARGET_QPOS,
    DRAWER_JOINT_CANDIDATES,
    MAX_OPEN_HOLD_DRIFT_M,
    MAX_OPEN_HOLD_TILT_DEG,
    OPEN_HOLD_STEPS,
    SCHEMA_VERSION,
    TASK_DESCRIPTION,
    TOPOLOGY_ID,
    assess_chain,
    body_tilt_deg,
    capture_frame,
    contact_body_pairs,
    find_body,
    find_joint_qadr,
    runtime_visibility_audit,
    write_trace_json,
)


def _refresh_policy_obs(env):
    """Use the same agentview observation path consumed by the VLA."""
    # LIBERO's ControlEnv wrapper intentionally does not expose robosuite's
    # private _get_observations(). Its public regenerate path restores the
    # current flattened state, forwards (without stepping), and returns the
    # exact observation dict used by evaluation.
    current = np.asarray(env.sim.get_state().flatten()).copy()
    return env.regenerate_obs_from_state(current)


def _policy_image(env) -> np.ndarray:
    obs = _refresh_policy_obs(env)
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)
    if image.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 policy RGB, got {image.shape}")
    # Exact get_libero_image preprocessing without importing TensorFlow.
    return np.ascontiguousarray(image[::-1, ::-1])


def _pose(env, body_name: str) -> tuple[np.ndarray, float]:
    body_id = env.sim.model.body_name2id(body_name)
    return (
        np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy(),
        body_tilt_deg(env.sim, body_id),
    )


def _artifact(path: str, expected_variant: str):
    key = TASK_DESCRIPTION.replace(" ", "_")
    handle = h5py.File(path, "r")
    group = handle[key]
    checks = {
        "schema": int(group.attrs.get("l3a4_schema_version", -1)) == SCHEMA_VERSION,
        "topology": str(group.attrs.get("l3a4_topology_id", "")) == TOPOLOGY_ID,
        "variant": str(group.attrs.get("l3a4_variant", "")) == expected_variant,
        "task": str(group.attrs.get("task_description", "")) == TASK_DESCRIPTION,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        handle.close()
        raise ValueError(f"{path}: stale/mismatched L3-A4 artifact: {failed}")
    return handle, group


def _body_role(name: str) -> str:
    if name == A_BODY:
        return "A"
    if name == B_BODY:
        return "B"
    if name == C_BODY:
        return "C"
    return name


def _initial_contact_gate(env, drawer_body: str, condition: str) -> tuple[bool, list[str]]:
    contacts = {frozenset(pair) for pair in contact_body_pairs(env)}
    reasons = []
    robot_tokens = ("robot", "gripper", "finger", "hand")
    for pair in contacts:
        if any(body in CHAIN_BODIES for body in pair) and any(
            any(token in body.lower() for token in robot_tokens) for body in pair
        ):
            reasons.append(f"robot_initial_contact:{sorted(pair)}")
        chain_members = [body for body in pair if body in CHAIN_BODIES]
        cabinet_members = [
            body for body in pair
            if "cabinet" in body.lower() and body not in CHAIN_BODIES
        ]
        if chain_members and cabinet_members:
            allowed_target_contact = (
                condition != "baseline"
                and chain_members == [A_BODY]
                and cabinet_members == [drawer_body]
            )
            if not allowed_target_contact:
                reasons.append(f"cabinet_initial_contact:{sorted(pair)}")
    # Only A may touch the moving drawer in Er/Ec. Parked Eb must not touch it.
    for body in (B_BODY, C_BODY):
        if frozenset((drawer_body, body)) in contacts:
            reasons.append(f"direct_drawer_{_body_role(body)}_initial_contact")
    if frozenset((A_BODY, C_BODY)) in contacts:
        reasons.append("direct_A_C_initial_contact")
    if condition == "baseline" and frozenset((drawer_body, A_BODY)) in contacts:
        reasons.append("baseline_drawer_A_initial_contact")
    reasons = list(dict.fromkeys(reasons))
    return not reasons, reasons


def _hold_gate(env, drawer_qadr: int, steps: int) -> dict:
    initial = {body: _pose(env, body) for body in CHAIN_BODIES}
    max_drift = {body: 0.0 for body in CHAIN_BODIES}
    max_tilt = {body: 0.0 for body in CHAIN_BODIES}
    for _ in range(steps):
        env.sim.step()
        for body in CHAIN_BODIES:
            pose, tilt = _pose(env, body)
            max_drift[body] = max(
                max_drift[body], float(np.linalg.norm(pose - initial[body][0]))
            )
            max_tilt[body] = max(
                max_tilt[body], abs(float(tilt) - initial[body][1])
            )
    passed = (
        max(max_drift.values()) <= MAX_OPEN_HOLD_DRIFT_M
        and max(max_tilt.values()) <= MAX_OPEN_HOLD_TILT_DEG
    )
    return {
        "passed": passed,
        "max_drift_m": max_drift,
        "max_tilt_change_deg": max_tilt,
        "drawer_qpos": float(env.sim.data.qpos[drawer_qadr]),
    }


def _script_required_motion(
    env, drawer_qadr: int, motion_steps: int, video_stride: int
):
    start = float(env.sim.data.qpos[drawer_qadr])
    frames = [capture_frame(env, 0, drawer_qadr)]
    images = [_policy_image(env)]
    for step, qpos in enumerate(
        np.linspace(start, DRAWER_TARGET_QPOS, motion_steps + 1)[1:], start=1
    ):
        env.sim.data.qpos[drawer_qadr] = float(qpos)
        env.sim.forward()
        env.sim.step()
        frames.append(capture_frame(env, step, drawer_qadr))
        if step % max(1, video_stride) == 0 or step == motion_steps:
            images.append(_policy_image(env))
    return frames, images


def _write_video(path: Path, frames, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def _load_manual_review(path: Path, required: set[tuple[str, int]]) -> dict:
    if not path.is_file():
        template = {
            "reviewer": "",
            "policy_camera": "agentview 256x256 through get_libero_image",
            "captures": [
                {
                    "condition": condition,
                    "episode": episode,
                    "A_recognizable": None,
                    "B_recognizable": None,
                    "C_recognizable": None,
                    "not_occluded": None,
                    "inside_frame": None,
                    "visible_early_enough": None,
                    "approved": None,
                    "notes": "",
                }
                for condition, episode in sorted(required)
            ],
        }
        path.write_text(json.dumps(template, indent=2) + "\n")
        return {"passed": False, "reason": "manual_review_missing_or_incomplete"}
    review = json.loads(path.read_text())
    entries = {
        (str(item["condition"]), int(item["episode"])): item
        for item in review.get("captures", [])
    }
    if set(entries) != required:
        return {"passed": False, "reason": "manual_review_capture_set_mismatch"}
    fields = (
        "A_recognizable",
        "B_recognizable",
        "C_recognizable",
        "not_occluded",
        "inside_frame",
        "visible_early_enough",
        "approved",
    )
    passed = bool(review.get("reviewer")) and all(
        all(entry.get(field) is True for field in fields)
        for entry in entries.values()
    )
    return {
        "passed": passed,
        "reason": "" if passed else "manual_review_missing_or_rejected",
        "reviewer": review.get("reviewer", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--eb", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--open_hold_steps", type=int, default=OPEN_HOLD_STEPS)
    parser.add_argument("--close_steps", type=int, default=CLOSE_STEPS)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out_dir", default="experiments/logs/l3a4_scene")
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(42)
    env.reset()
    drawer_body = find_body(env, DRAWER_BODY_CANDIDATES)
    drawer_joint, drawer_qadr = find_joint_qadr(env.sim, DRAWER_JOINT_CANDIDATES)
    visibility_asset_gate = runtime_visibility_audit(env)

    artifact_paths = {
        "baseline": args.eb,
        "risk": args.er,
        "stable": args.ec,
    }
    handles = []
    rows = []
    required_review = set()
    try:
        groups = {}
        for condition, path in artifact_paths.items():
            handle, group = _artifact(path, condition)
            handles.append(handle)
            groups[condition] = group
        available = min(len(group) for group in groups.values())
        if args.episodes > available:
            raise ValueError(f"requested {args.episodes}, artifacts have {available}")

        for episode in range(args.episodes):
            for condition in ("baseline", "risk", "stable"):
                state = np.asarray(
                    groups[condition][f"demo_{episode}"]["initial_state"][:]
                )
                env.reset()
                env.set_init_state(state)
                clear_mujoco_replay_transients(env)
                env.sim.forward()
                initial_image = _policy_image(env)
                image_path = out_dir / f"{condition}_ep{episode:03d}_init.png"
                imageio.imwrite(image_path, initial_image)
                required_review.add((condition, episode))

                contact_pass, contact_reasons = _initial_contact_gate(
                    env, drawer_body, condition
                )
                hold = _hold_gate(env, drawer_qadr, args.open_hold_steps)
                # Re-restore the exact serialized state before the dynamic gate.
                env.reset()
                env.set_init_state(state)
                clear_mujoco_replay_transients(env)
                env.sim.forward()
                trace, video_frames = _script_required_motion(
                    env, drawer_qadr, args.close_steps, args.video_stride
                )
                assessment = assess_chain(
                    trace, condition=condition, drawer_body=drawer_body
                )
                trace_path = out_dir / f"{condition}_ep{episode:03d}_trace.json"
                video_path = out_dir / f"{condition}_ep{episode:03d}.mp4"
                write_trace_json(trace_path, trace, assessment)
                _write_video(video_path, video_frames, args.fps)
                rows.append(
                    {
                        "condition": condition,
                        "episode": episode,
                        "initial_state_sha256": hashlib.sha256(
                            state.tobytes()
                        ).hexdigest(),
                        "initial_contact_pass": contact_pass,
                        "initial_contact_reasons": contact_reasons,
                        "open_hold": hold,
                        "chain": assessment.to_dict(),
                        "physical_pass": bool(
                            contact_pass and hold["passed"] and assessment.passed
                        ),
                        "init_image": str(image_path),
                        "video": str(video_path),
                        "trace": str(trace_path),
                    }
                )
    finally:
        env.close()
        for handle in handles:
            handle.close()

    physical_pass = (
        all(item["pass"] for item in visibility_asset_gate.values())
        and all(row["physical_pass"] for row in rows)
    )
    manual = _load_manual_review(
        out_dir / "manual_review.json", required_review
    )
    visual_pass = (
        all(item["pass"] for item in visibility_asset_gate.values())
        and manual["passed"]
    )
    overall_pass = physical_pass and visual_pass
    payload = {
        "verdict": (
            "PASS_L3A4_SCENE_GATE"
            if overall_pass
            else (
                "PENDING_L3A4_POLICY_VIEW_REVIEW"
                if physical_pass and not visual_pass
                else "FAIL_L3A4_SCENE_GATE"
            )
        ),
        "physical_verdict": (
            "PASS_L3A4_PHYSICAL_SCENE_GATE"
            if physical_pass
            else "FAIL_L3A4_PHYSICAL_SCENE_GATE"
        ),
        "visual_verdict": (
            "PASS_L3A4_POLICY_VIEW_REVIEWED"
            if visual_pass
            else "PENDING_L3A4_POLICY_VIEW_REVIEW"
        ),
        "drawer_body": drawer_body,
        "drawer_joint": drawer_joint,
        "runtime_asset_audit": visibility_asset_gate,
        "manual_review": manual,
        "rows": rows,
    }
    (out_dir / "scene_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# L3-A4 momentum-chain scene validation",
        "",
        f"- Verdict: **{payload['verdict']}**",
        f"- Physical: **{payload['physical_verdict']}**",
        f"- Policy view: **{payload['visual_verdict']}**",
        f"- Conditions × episodes: 3 × {args.episodes}",
        f"- Drawer binding: `{drawer_body}` / `{drawer_joint}`",
        "",
        "The policy-view verdict requires explicit human review of all exact-state "
        "256×256 agentview PNGs and rollout videos.",
    ]
    (out_dir / "scene_validation.md").write_text("\n".join(report) + "\n")
    print(
        f"{payload['verdict']} physical={payload['physical_verdict']} "
        f"visual={payload['visual_verdict']} report={out_dir / 'scene_validation.json'}"
    )
    if args.fail_on_invalid and not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
