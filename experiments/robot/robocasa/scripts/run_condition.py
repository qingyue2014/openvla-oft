#!/usr/bin/env python3
"""Roll out one PhysCog RoboCasa scene under one condition.

    python experiments/robot/robocasa/scripts/run_condition.py \
        --scene L1-A1 --condition Er --episodes 5 --video review/L1-A1_task

The policy is pluggable: ``--policy zero`` and ``--policy random`` need no
checkpoint and are meant for scene bring-up and gate calibration.
``--policy pi05`` uses the released pi05-LIBERO checkpoint as an explicitly
cross-simulator smoke test with the PandaOmron base frozen. A custom VLA policy
is supplied by passing ``--policy module:function``, where the function takes
``(obs, lang, env)`` and returns a 12-D mobile-base action.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog.registry import get_scene  # noqa: E402
from experiments.robot.robocasa.physcog.preflight import (  # noqa: E402
    NativePreflightError,
    initial_max_penetration,
    invalidate_artifacts,
    load_formal_gate_manifest,
    make_condition_record,
    reserve_review_video,
    review_dir,
    validate_condition_records,
)
from experiments.robot.robocasa.scripts.static_check import (  # noqa: E402
    check_repository,
)

CAMERA = "robot0_agentview_center"


def policy_view_image(policy, obs) -> np.ndarray:
    """Return exact model pixels when exposed, else upright raw camera RGB."""

    transform = getattr(policy, "policy_view_image", None)
    if transform is not None:
        return np.asarray(transform(obs))
    return np.asarray(obs[f"{CAMERA}_image"])[::-1]


def load_policy(spec: str):
    if spec == "zero":
        return lambda obs, lang, env: np.zeros(env.action_spec[0].shape)
    if spec == "random":
        low, high = None, None

        def _rand(obs, lang, env):
            nonlocal low, high
            if low is None:
                low, high = env.action_spec
            return np.random.uniform(low, high)

        return _rand
    if spec in {"pi05", "pi0.5", "pi_0.5", "pi-0.5"}:
        from experiments.robot.robocasa.pi05_policy import make_policy

        return make_policy()
    module_name, fn_name = spec.split(":")
    return getattr(importlib.import_module(module_name), fn_name)


def make_env(
    scene_id: str,
    condition: str,
    seed: int,
    render: bool,
    *,
    layout_id: int | None = None,
    camera_names: list[str] | tuple[str, ...] | None = None,
):
    cls = get_scene(scene_id)
    kwargs = dict(
        condition=condition,
        robots="PandaOmron",
        controller_configs=None,
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=render,
        camera_names=list(camera_names or ([CAMERA] if render else [])),
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        seed=seed,
    )
    if layout_id is not None:
        kwargs["layout_ids"] = layout_id
    return cls(**kwargs)


def run_native_preflight(
    scene_id: str, seed: int, *, layout_id: int | None = None
) -> dict:
    """Build matched Eb/Er/Ec once and hard-check prompt/assets before rollout."""

    _, static_problems = check_repository()
    if static_problems:
        raise NativePreflightError(
            "simulator-free native-only check failed: " + "; ".join(static_problems)
        )
    cls = get_scene(scene_id)
    records = []
    for condition in ("Eb", "Er", "Ec"):
        env = make_env(
            scene_id,
            condition,
            seed,
            render=False,
            layout_id=layout_id,
        )
        try:
            env.reset()
            records.append(make_condition_record(env))
        finally:
            env.close()
    return validate_condition_records(records, hazard_objs=cls.physcog_hazard_objs)


def save_preflight_manifest(scene_id: str, manifest: dict) -> pathlib.Path:
    directory = review_dir(scene_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene_id}_native_preflight.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_smoke_gate_manifest(
    path: str | None,
    *,
    scene_id: str,
    preflight_sha256: str,
    expected_policy_preprocessing: str = (
        "pi05_libero_rotate180_resize_with_pad_224"
    ),
    expected_policy_cameras: tuple[str, ...] = (
        "robot0_agentview_center",
        "robot0_eye_in_hand",
    ),
    expected_policy_initialization: str = (
        "pi05_libero_wait10_native_equivalent_gripper_no_eef_alignment"
    ),
    expected_initial_gripper_qpos: tuple[float, float] = (
        0.03872,
        -0.03872,
    ),
) -> dict | None:
    """Require reviewed initial-state gates before a dynamic smoke rollout."""

    if path is None:
        return None
    manifest_path = pathlib.Path(path)
    payload = json.loads(manifest_path.read_text())
    if payload.get("scene_id") != scene_id:
        raise NativePreflightError(
            f"smoke gate manifest scene mismatch: {payload.get('scene_id')!r}"
        )
    if payload.get("native_preflight_sha256") != preflight_sha256:
        raise NativePreflightError(
            "smoke gate manifest native-preflight hash is stale or mismatched"
        )
    gates = payload.get("gates") or {}
    missing = [
        name
        for name in ("G0", "physics", "visibility")
        if not (gates.get(name) or {}).get("passed")
    ]
    if missing:
        raise NativePreflightError(
            f"smoke gate manifest has unpassed prerequisites: {missing}"
        )
    visibility = gates["visibility"]
    reviewed_initialization = visibility.get("policy_initialization")
    if reviewed_initialization != expected_policy_initialization:
        raise NativePreflightError(
            "smoke gate visibility initialization mismatch: reviewed "
            f"{reviewed_initialization!r}, policy uses "
            f"{expected_policy_initialization!r}"
        )
    if visibility.get("policy_preprocessing") != expected_policy_preprocessing:
        raise NativePreflightError(
            "smoke gate visibility preprocessing mismatch: reviewed "
            f"{visibility.get('policy_preprocessing')!r}, policy consumes "
            f"{expected_policy_preprocessing!r}"
        )
    required_cameras = set(expected_policy_cameras)
    if set(visibility.get("policy_cameras") or ()) != required_cameras:
        raise NativePreflightError(
            "smoke gate visibility does not cover both pi0.5 policy cameras"
        )
    if not visibility.get("wrist_initial_frame") or set(
        (visibility.get("paired_wrist_initial_frames") or {}).keys()
    ) != {"Eb", "Er", "Ec"}:
        raise NativePreflightError(
            "smoke gate visibility lacks paired pi0.5 wrist-camera frames"
        )
    paired_gripper_qpos = (
        visibility.get("paired_gripper_qpos_after_initialization") or {}
    )
    if set(paired_gripper_qpos) != {"Eb", "Er", "Ec"}:
        raise NativePreflightError(
            "smoke gate lacks paired post-initialization gripper state"
        )
    reference_qpos = np.asarray(expected_initial_gripper_qpos, dtype=np.float64)
    mismatched_qpos = {
        condition: qpos
        for condition, qpos in paired_gripper_qpos.items()
        if not np.allclose(
            np.asarray(qpos, dtype=np.float64),
            reference_qpos,
            atol=0.005,
            rtol=0.0,
        )
    }
    if mismatched_qpos:
        raise NativePreflightError(
            "smoke gate post-initialization gripper state does not match "
            f"native LIBERO wait: {mismatched_qpos}"
        )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--condition", default="Eb", choices=("Eb", "Er", "Ec"))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="zero")
    ap.add_argument("--video", default=None, help="directory for review videos")
    ap.add_argument("--out", default=None, help="jsonl results path")
    ap.add_argument("--save-actions", default=None, help="npz of successful actions")
    ap.add_argument(
        "--save-traces",
        default=None,
        help="diagnostic npz of every rollout's actions and end-effector states",
    )
    ap.add_argument(
        "--formal",
        action="store_true",
        help="publication run: require passed G0/physics/visibility/G1/G2/G3",
    )
    ap.add_argument(
        "--gate-manifest",
        default=None,
        help="JSON publication-gate manifest required by --formal",
    )
    ap.add_argument(
        "--smoke-gate-manifest",
        default=None,
        help="reviewed G0/physics/visibility manifest required by calibrated smoke jobs",
    )
    args = ap.parse_args()

    artifacts = [
        path
        for path in (args.video, args.out, args.save_actions, args.save_traces)
        if path is not None
    ]
    try:
        policy = load_policy(args.policy)
        save_video = args.video is not None
        camera_obs = save_video or bool(
            getattr(policy, "requires_camera_obs", False)
        )
        policy_cameras = getattr(policy, "camera_names", None)
        vdir = review_dir(args.scene, args.video) if save_video else None
        manifest = run_native_preflight(args.scene, args.seed)
        manifest_path = save_preflight_manifest(args.scene, manifest)
        print(f"native-only preflight PASS -> {manifest_path}")
        smoke_gates = load_smoke_gate_manifest(
            args.smoke_gate_manifest,
            scene_id=args.scene,
            preflight_sha256=manifest["preflight_sha256"],
            expected_policy_preprocessing=getattr(
                policy,
                "policy_preprocessing",
                "pi05_libero_rotate180_resize_with_pad_224",
            ),
            expected_policy_cameras=tuple(
                getattr(
                    policy,
                    "camera_names",
                    ("robot0_agentview_center", "robot0_eye_in_hand"),
                )
            ),
            expected_policy_initialization=getattr(
                policy,
                "policy_initialization",
                (
                    "pi05_libero_wait10_native_equivalent_gripper"
                    "_no_eef_alignment"
                ),
            ),
            expected_initial_gripper_qpos=tuple(
                np.asarray(
                    getattr(
                        policy,
                        "expected_initial_gripper_qpos",
                        (0.03872, -0.03872),
                    )
                ).tolist()
            ),
        )
        formal_gates = (
            load_formal_gate_manifest(
                args.gate_manifest,
                scene_id=args.scene,
                preflight_sha256=manifest["preflight_sha256"],
            )
            if args.formal
            else None
        )
        env = make_env(
            args.scene,
            args.condition,
            args.seed,
            camera_obs,
            camera_names=policy_cameras,
        )

        results, saved_actions, rollout_traces = [], {}, {}
        try:
            for ep in range(args.episodes):
                obs = env.reset()
                if hasattr(policy, "reset"):
                    policy.reset()
                lang = env.native_lang
                penetration = initial_max_penetration(env)
                if lang != manifest["native_prompt"]:
                    raise NativePreflightError(
                        f"{args.scene}: rollout prompt {lang!r} does not match "
                        f"preflight prompt {manifest['native_prompt']!r}"
                    )
                settle_steps = int(getattr(policy, "settle_steps", 0))
                if settle_steps < 0:
                    raise ValueError("policy settle_steps must be non-negative")
                for _ in range(settle_steps):
                    settle_action = getattr(policy, "settle_action", None)
                    act = (
                        np.asarray(settle_action(env, obs), dtype=np.float64)
                        if settle_action is not None
                        else np.zeros_like(env.action_spec[0], dtype=np.float64)
                    )
                    obs, _, done, info = env.step(act)
                    if done:
                        raise NativePreflightError(
                            "environment terminated during policy settling"
                        )
                    if info["physcog"]["task_success"]:
                        raise NativePreflightError(
                            "task became complete during policy settling"
                        )
                    if info["physcog"]["safety_violated"]:
                        raise NativePreflightError(
                            "safety violation occurred during policy settling"
                        )
                frames, actions, policy_actions = [], [], []
                eef_positions = [np.asarray(obs["robot0_eef_pos"]).copy()]
                gripper_qpos = [np.asarray(obs["robot0_gripper_qpos"]).copy()]
                tracked_objects = tuple(
                    dict.fromkeys(
                        (
                            getattr(env, "physcog_target_obj", None),
                            *tuple(getattr(env, "physcog_hazard_objs", ())),
                        )
                    )
                )
                tracked_objects = tuple(name for name in tracked_objects if name)
                object_positions = {
                    name: [
                        np.asarray(
                            env.sim.data.body_xpos[env.obj_body_id[name]]
                        ).copy()
                    ]
                    for name in tracked_objects
                }
                initial_frame_path = None
                if save_video:
                    import imageio

                    vdir.mkdir(parents=True, exist_ok=True)
                    initial_frame_path = (
                        vdir
                        / f"{args.scene}_{args.condition}_policy_view_init_ep{ep}.png"
                    )
                    imageio.imwrite(initial_frame_path, policy_view_image(policy, obs))
                for _ in range(args.horizon):
                    act = np.asarray(policy(obs, lang, env), dtype=np.float64)
                    raw_policy_action = getattr(policy, "last_raw_action", None)
                    if raw_policy_action is not None:
                        policy_actions.append(
                            np.asarray(raw_policy_action, dtype=np.float64).copy()
                        )
                    obs, _, _, info = env.step(act)
                    actions.append(act)
                    eef_positions.append(
                        np.asarray(obs["robot0_eef_pos"]).copy()
                    )
                    gripper_qpos.append(
                        np.asarray(obs["robot0_gripper_qpos"]).copy()
                    )
                    for name in tracked_objects:
                        object_positions[name].append(
                            np.asarray(
                                env.sim.data.body_xpos[env.obj_body_id[name]]
                            ).copy()
                        )
                    if save_video:
                        frames.append(policy_view_image(policy, obs))
                    if info["physcog"]["task_success"]:
                        break

                summary = env.physcog_episode_summary()
                summary.update(
                    episode=ep,
                    seed=args.seed + ep,
                    preflight_sha256=manifest["preflight_sha256"],
                    formal=bool(args.formal),
                    policy=args.policy,
                    policy_model_label=getattr(policy, "model_label", args.policy),
                    policy_settle_steps=settle_steps,
                    smoke_gate_manifest=(
                        args.smoke_gate_manifest if smoke_gates else None
                    ),
                    formal_gate_manifest=args.gate_manifest if formal_gates else None,
                    initial_max_penetration_m=penetration,
                    policy_camera=getattr(policy, "agent_camera", CAMERA),
                    policy_view_initial_frame=(
                        str(initial_frame_path) if initial_frame_path else None
                    ),
                )
                results.append(summary)
                rollout_traces[f"ep{ep}_actions"] = np.asarray(actions)
                if policy_actions:
                    rollout_traces[f"ep{ep}_policy_actions"] = np.asarray(
                        policy_actions
                    )
                rollout_traces[f"ep{ep}_eef_positions"] = np.asarray(
                    eef_positions
                )
                rollout_traces[f"ep{ep}_gripper_qpos"] = np.asarray(
                    gripper_qpos
                )
                for name, positions in object_positions.items():
                    rollout_traces[
                        f"ep{ep}_body_position_{name}"
                    ] = np.asarray(positions)
                print(json.dumps(summary, ensure_ascii=False))

                if summary["task_success"] and not summary["safety_violated"]:
                    action_archive_space = getattr(
                        policy,
                        "action_archive_space",
                        "robocasa_native_12d",
                    )
                    if action_archive_space == "pi05_libero_7d":
                        if len(policy_actions) != len(actions):
                            raise RuntimeError(
                                "pi0.5 raw-action trace is incomplete; refusing "
                                "to write a non-replayable success archive"
                            )
                        saved_actions[f"ep{ep}"] = np.asarray(policy_actions)
                    elif action_archive_space == "robocasa_native_12d":
                        saved_actions[f"ep{ep}"] = np.asarray(actions)
                    else:
                        raise ValueError(
                            "unsupported action archive space: "
                            f"{action_archive_space!r}"
                        )

                if save_video and frames:
                    import imageio

                    outcome = (
                        "violation"
                        if summary["safety_violated"]
                        else ("success" if summary["task_success"] else "failure")
                    )
                    vdir.mkdir(parents=True, exist_ok=True)
                    path = reserve_review_video(
                        vdir,
                        scene_id=args.scene,
                        category=f"{args.condition}_{outcome}",
                        stem=f"ep{ep}",
                    )
                    imageio.mimwrite(path, frames, fps=20)
                    print(f"  video -> {path}")
        finally:
            env.close()
    except Exception as exc:
        marker = invalidate_artifacts(
            args.scene,
            [str(exc)],
            phase="preflight_or_rollout",
            artifact_paths=artifacts,
        )
        print(f"HARD STOP: {exc}", file=sys.stderr)
        print(f"invalidated scene/jobs/metrics/videos/tables/html -> {marker}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.save_actions and saved_actions:
        actions_path = pathlib.Path(args.save_actions)
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(actions_path, **saved_actions)
        provenance = {
            "valid": True,
            "scene_id": args.scene,
            "condition": args.condition,
            "seed": args.seed,
            "native_prompt": manifest["native_prompt"],
            "preflight_sha256": manifest["preflight_sha256"],
            "action_keys": sorted(saved_actions),
            "action_space": getattr(
                policy,
                "action_archive_space",
                "robocasa_native_12d",
            ),
            "policy_model_label": getattr(policy, "model_label", args.policy),
            "policy_camera": getattr(policy, "agent_camera", CAMERA),
            "policy_image_mode": getattr(policy, "image_mode", None),
            "policy_settle_steps": int(getattr(policy, "settle_steps", 0)),
            "policy_initialization": getattr(
                policy,
                "policy_initialization",
                None,
            ),
        }
        pathlib.Path(f"{actions_path}.preflight.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        print(f"saved {len(saved_actions)} clean action sequences -> {args.save_actions}")

    if args.save_traces:
        trace_path = pathlib.Path(args.save_traces)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(trace_path, **rollout_traces)
        pathlib.Path(f"{trace_path}.json").write_text(
            json.dumps(
                {
                    "valid": True,
                    "scene_id": args.scene,
                    "condition": args.condition,
                    "seed": args.seed,
                    "native_prompt": manifest["native_prompt"],
                    "preflight_sha256": manifest["preflight_sha256"],
                    "episodes": results,
                    "trace_keys": sorted(rollout_traces),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"saved {len(results)} rollout traces -> {args.save_traces}")

    n = len(results)
    task = sum(r["task_success"] for r in results)
    viol = sum(r["safety_violated"] for r in results)
    safe = sum(r["task_success"] and not r["safety_violated"] for r in results)
    print(
        f"{args.scene} {args.condition}: Task {task}/{n}  SV {viol}/{n}  Safe {safe}/{n}"
    )


if __name__ == "__main__":
    main()
