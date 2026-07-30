#!/usr/bin/env python3
"""Scan native SetupFrying layouts for L2-A1 policy-view visibility.

This is a development diagnostic, not experiment evidence. It changes only
the native ``layout_ids`` constructor argument, runs the full matched native
preflight separately for every candidate, and records the actual policy-camera
initial views and native collision-geometry clearances.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog import oracles as OR  # noqa: E402
from experiments.robot.robocasa.scripts.run_condition import (  # noqa: E402
    CAMERA,
    make_env,
    run_native_preflight,
)


def scan_layout(scene_id: str, layout_id: int, seed: int, output: pathlib.Path):
    layout_dir = output / f"layout_{layout_id}"
    layout_dir.mkdir(parents=True, exist_ok=True)
    preflight = run_native_preflight(scene_id, seed, layout_id=layout_id)
    preflight_path = layout_dir / f"{scene_id}_native_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    )

    rows = []
    for condition in ("Eb", "Er", "Ec"):
        env = make_env(
            scene_id,
            condition,
            seed,
            render=True,
            layout_id=layout_id,
        )
        try:
            obs = env.reset()
            if env.native_lang != preflight["native_prompt"]:
                raise RuntimeError(
                    f"{condition}: runtime prompt differs from native preflight"
                )
            frame = layout_dir / (
                f"{scene_id}_{condition}_layout{layout_id}_policy_view.png"
            )
            imageio.imwrite(frame, obs[f"{CAMERA}_image"][::-1])
            hazard_geoms = OR.obj_geoms(env, "distr_counter_0")
            stove_geoms = list(env.stove.contact_geoms)
            rows.append(
                {
                    "condition": condition,
                    "frame": str(frame),
                    "hazard_to_stove_geom_min_signed_distance_m": (
                        OR.min_signed_geom_distance(
                            env,
                            hazard_geoms,
                            stove_geoms,
                            distmax_m=1.0,
                        )
                    ),
                    "initial_hazard_contacts_stove": OR.contact(
                        env, hazard_geoms, stove_geoms
                    ),
                    "native_prompt": env.native_lang,
                    "policy_camera": CAMERA,
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
    parser.add_argument("--scene", default="L2-A1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layouts", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.scene != "L2-A1":
        raise SystemExit("this diagnostic is registered only for L2-A1")

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
    summary_path = args.out / f"{args.scene}_layout_scan.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not any(row["valid_native_preflight"] for row in results):
        raise SystemExit("no native layout passed preflight")
    print(f"layout scan -> {summary_path}")


if __name__ == "__main__":
    main()
