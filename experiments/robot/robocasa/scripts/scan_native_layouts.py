#!/usr/bin/env python3
"""Scan native RoboCasa layouts without changing task assets or prompts.

This is a development diagnostic, never formal evidence.  Every candidate
uses the registered native task class and its unchanged Eb/Er/Ec object
inventory.  It records exact pi0.5 center/wrist pixels plus initial task,
safety, and penetration state so a fixed native layout can be selected before
the full 200-step gates are run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog.preflight import (  # noqa: E402
    initial_max_penetration,
)
from experiments.robot.robocasa.pi05_policy import (  # noqa: E402
    AGENT_CAMERA,
    WRIST_CAMERA,
    preprocess_camera_image,
)
from experiments.robot.robocasa.scripts.run_condition import (  # noqa: E402
    make_env,
    run_native_preflight,
)


def scan_layout(
    scene_id: str,
    layout_id: int,
    seed: int,
    output: pathlib.Path,
) -> dict:
    layout_dir = output / f"layout_{layout_id}"
    layout_dir.mkdir(parents=True, exist_ok=True)
    preflight = run_native_preflight(scene_id, seed, layout_id=layout_id)

    rows = []
    for condition in ("Eb", "Er", "Ec"):
        env = make_env(
            scene_id,
            condition,
            seed,
            render=True,
            layout_id=layout_id,
            camera_names=(AGENT_CAMERA, WRIST_CAMERA),
        )
        try:
            obs = env.reset()
            if env.native_lang != preflight["native_prompt"]:
                raise RuntimeError(
                    f"{condition}: runtime prompt differs from native preflight"
                )
            frames = {}
            for camera in (AGENT_CAMERA, WRIST_CAMERA):
                path = layout_dir / (
                    f"{scene_id}_{condition}_layout{layout_id}_{camera}_pi05.png"
                )
                imageio.imwrite(
                    path,
                    preprocess_camera_image(obs[f"{camera}_image"]),
                )
                frames[camera] = str(path)
            rows.append(
                {
                    "condition": condition,
                    "frames": frames,
                    "initial_max_penetration_m": initial_max_penetration(env),
                    "task_success_initial": bool(env._check_success()),
                    "safety_violated_initial": bool(env.safety_violated),
                    "object_positions": {
                        name: env.sim.data.body_xpos[body_id].tolist()
                        for name, body_id in env.obj_body_id.items()
                    },
                }
            )
        finally:
            env.close()
    return {
        "valid_native_preflight": True,
        "layout_id": layout_id,
        "native_preflight_sha256": preflight["preflight_sha256"],
        "native_prompt": preflight["native_prompt"],
        "native_task": preflight["native_task"],
        "conditions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layouts", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for layout_id in args.layouts:
        try:
            row = scan_layout(args.scene, layout_id, args.seed, args.out)
        except Exception as exc:
            row = {
                "valid_native_preflight": False,
                "layout_id": layout_id,
                "error": str(exc),
            }
        results.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    summary = {
        "scene_id": args.scene,
        "seed": args.seed,
        "diagnostic_only": True,
        "layouts": results,
    }
    path = args.out / f"{args.scene}_native_layout_scan.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not any(row["valid_native_preflight"] for row in results):
        raise SystemExit("no native layout passed preflight")
    print(f"native layout diagnostic -> {path}")


if __name__ == "__main__":
    main()
