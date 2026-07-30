#!/usr/bin/env python3
"""Roll out one PhysCog RoboCasa scene under one condition.

    python experiments/robot/robocasa/scripts/run_condition.py \
        --scene L1-A1 --condition Er --episodes 5 --video review/L1-A1_task

The policy is pluggable: ``--policy zero`` and ``--policy random`` need no
checkpoint and are meant for scene bring-up and gate calibration. A VLA policy
is supplied by passing ``--policy module:function``, where the function takes
``(obs, lang)`` and returns a 7-D (or 12-D mobile-base) action.
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
    module_name, fn_name = spec.split(":")
    return getattr(importlib.import_module(module_name), fn_name)


def make_env(
    scene_id: str,
    condition: str,
    seed: int,
    render: bool,
    *,
    layout_id: int | None = None,
):
    cls = get_scene(scene_id)
    kwargs = dict(
        condition=condition,
        robots="PandaOmron",
        controller_configs=None,
        has_renderer=False,
        has_offscreen_renderer=render,
        use_camera_obs=render,
        camera_names=[CAMERA] if render else [],
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
        "--formal",
        action="store_true",
        help="publication run: require passed G0/physics/visibility/G1/G2/G3",
    )
    ap.add_argument(
        "--gate-manifest",
        default=None,
        help="JSON publication-gate manifest required by --formal",
    )
    args = ap.parse_args()

    artifacts = [
        path
        for path in (args.video, args.out, args.save_actions)
        if path is not None
    ]
    try:
        render = args.video is not None
        vdir = review_dir(args.scene, args.video) if render else None
        manifest = run_native_preflight(args.scene, args.seed)
        manifest_path = save_preflight_manifest(args.scene, manifest)
        print(f"native-only preflight PASS -> {manifest_path}")
        formal_gates = (
            load_formal_gate_manifest(
                args.gate_manifest,
                scene_id=args.scene,
                preflight_sha256=manifest["preflight_sha256"],
            )
            if args.formal
            else None
        )
        env = make_env(args.scene, args.condition, args.seed, render)
        policy = load_policy(args.policy)

        results, saved_actions = [], {}
        try:
            for ep in range(args.episodes):
                obs = env.reset()
                lang = env.native_lang
                penetration = initial_max_penetration(env)
                if lang != manifest["native_prompt"]:
                    raise NativePreflightError(
                        f"{args.scene}: rollout prompt {lang!r} does not match "
                        f"preflight prompt {manifest['native_prompt']!r}"
                    )
                frames, actions = [], []
                initial_frame_path = None
                if render:
                    import imageio

                    vdir.mkdir(parents=True, exist_ok=True)
                    initial_frame_path = (
                        vdir
                        / f"{args.scene}_{args.condition}_policy_view_init_ep{ep}.png"
                    )
                    imageio.imwrite(
                        initial_frame_path, obs[f"{CAMERA}_image"][::-1]
                    )
                for _ in range(args.horizon):
                    act = np.asarray(policy(obs, lang, env), dtype=np.float64)
                    obs, _, _, info = env.step(act)
                    actions.append(act)
                    if render:
                        frames.append(obs[f"{CAMERA}_image"][::-1])
                    if info["physcog"]["task_success"]:
                        break

                summary = env.physcog_episode_summary()
                summary.update(
                    episode=ep,
                    seed=args.seed + ep,
                    preflight_sha256=manifest["preflight_sha256"],
                    formal=bool(args.formal),
                    formal_gate_manifest=args.gate_manifest if formal_gates else None,
                    initial_max_penetration_m=penetration,
                    policy_camera=CAMERA,
                    policy_view_initial_frame=(
                        str(initial_frame_path) if initial_frame_path else None
                    ),
                )
                results.append(summary)
                print(json.dumps(summary, ensure_ascii=False))

                if summary["task_success"] and not summary["safety_violated"]:
                    saved_actions[f"ep{ep}"] = np.asarray(actions)

                if render and frames:
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

    n = len(results)
    task = sum(r["task_success"] for r in results)
    viol = sum(r["safety_violated"] for r in results)
    safe = sum(r["task_success"] and not r["safety_violated"] for r in results)
    print(
        f"{args.scene} {args.condition}: Task {task}/{n}  SV {viol}/{n}  Safe {safe}/{n}"
    )


if __name__ == "__main__":
    main()
