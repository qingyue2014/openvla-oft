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
    path: str | None, *, scene_id: str, preflight_sha256: str
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
    if visibility.get("policy_preprocessing") != (
        "pi05_libero_rotate180_resize_with_pad_224"
    ):
        raise NativePreflightError(
            "smoke gate visibility was not reviewed after exact pi0.5 "
            "preprocessing"
        )
    required_cameras = {
        "robot0_agentview_center",
        "robot0_eye_in_hand",
    }
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
                frames, actions = [], []
                eef_positions = [np.asarray(obs["robot0_eef_pos"]).copy()]
                gripper_qpos = [np.asarray(obs["robot0_gripper_qpos"]).copy()]
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
                    obs, _, _, info = env.step(act)
                    actions.append(act)
                    eef_positions.append(
                        np.asarray(obs["robot0_eef_pos"]).copy()
                    )
                    gripper_qpos.append(
                        np.asarray(obs["robot0_gripper_qpos"]).copy()
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
                    smoke_gate_manifest=(
                        args.smoke_gate_manifest if smoke_gates else None
                    ),
                    formal_gate_manifest=args.gate_manifest if formal_gates else None,
                    initial_max_penetration_m=penetration,
                    policy_camera=CAMERA,
                    policy_view_initial_frame=(
                        str(initial_frame_path) if initial_frame_path else None
                    ),
                )
                results.append(summary)
                rollout_traces[f"ep{ep}_actions"] = np.asarray(actions)
                rollout_traces[f"ep{ep}_eef_positions"] = np.asarray(
                    eef_positions
                )
                rollout_traces[f"ep{ep}_gripper_qpos"] = np.asarray(
                    gripper_qpos
                )
                print(json.dumps(summary, ensure_ascii=False))

                if summary["task_success"] and not summary["safety_violated"]:
                    saved_actions[f"ep{ep}"] = np.asarray(actions)

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
