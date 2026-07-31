#!/usr/bin/env python3
"""Run RoboCasa G0, initial-physics, and policy-view evidence gates.

The probe always runs the matched native-only Eb/Er/Ec preflight first. It then
holds every paired environment at null action for 200 steps and measures every
declared hazard relative to the post-settle baseline captured by the mixin.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog.preflight import (  # noqa: E402
    NativePreflightError,
    build_initial_gate_manifest,
    initial_contact_report,
    initial_max_penetration,
    invalidate_artifacts,
    reject_quarantined_artifact,
    review_dir,
)
from experiments.robot.robocasa.physcog.registry import get_scene  # noqa: E402
from experiments.robot.robocasa.pi05_policy import (  # noqa: E402
    AGENT_CAMERA,
    ROBOCASA_AGENT_CAMERAS,
    WRIST_CAMERA,
    PI05_SETTLE_STEPS,
    pi05_preprocessing_label,
    pi05_initialization_label,
    pi05_settle_action,
    preprocess_camera_image_for_mode,
)
from experiments.robot.robocasa.scripts.run_condition import (  # noqa: E402
    make_env,
    run_native_preflight,
    save_preflight_manifest,
)


def _hazard_sample(env, hazard: str) -> tuple[float, float]:
    baseline = env._pc_baseline[hazard]
    body = env.obj_body_id[hazard]
    position = np.asarray(env.sim.data.body_xpos[body])
    matrix = np.asarray(env.sim.data.body_xmat[body]).reshape(3, 3)
    cosine = float(np.dot(matrix[:, 2], baseline["mat"][:, 2]))
    tilt = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    abs_dz = abs(float(position[2] - baseline["pos"][2]))
    return tilt, abs_dz


def run_null_action_probe(env, hazard_objs, steps: int) -> dict:
    """Execute the probe and return raw evidence without deciding publication."""

    task_initial = bool(env._check_success())
    safety_initial = bool(env.safety_violated)
    task_changed = False
    safety_changed = False
    unexpected_done = False
    missing = [
        hazard
        for hazard in hazard_objs
        if hazard not in env._pc_baseline or hazard not in env.obj_body_id
    ]
    metrics = {
        hazard: {"max_tilt_deg": 0.0, "max_abs_dz_m": 0.0}
        for hazard in hazard_objs
        if hazard not in missing
    }

    low, _ = env.action_spec
    null_action = np.zeros_like(low, dtype=np.float64)
    executed_steps = 0
    for _ in range(steps):
        _, _, done, _ = env.step(null_action)
        executed_steps += 1
        task_now = bool(env._check_success())
        safety_now = bool(env.safety_violated)
        task_changed |= task_now != task_initial
        safety_changed |= safety_now != safety_initial
        unexpected_done |= bool(done)
        for hazard, row in metrics.items():
            tilt, abs_dz = _hazard_sample(env, hazard)
            row["max_tilt_deg"] = max(row["max_tilt_deg"], tilt)
            row["max_abs_dz_m"] = max(row["max_abs_dz_m"], abs_dz)
        if done:
            break

    return {
        "null_action_steps": executed_steps,
        "hazard_metrics": metrics,
        "missing_hazard_objs": missing,
        "task_success_initial": task_initial,
        "task_success_final": bool(env._check_success()),
        "task_state_changed": task_changed,
        "safety_violated_initial": safety_initial,
        "safety_violated_final": bool(env.safety_violated),
        "safety_state_changed": safety_changed,
        "unexpected_done": unexpected_done,
    }


def null_action_probe_passed(probe: dict) -> bool:
    metrics = probe["hazard_metrics"].values()
    max_tilt = max(
        (float(row["max_tilt_deg"]) for row in metrics),
        default=0.0,
    )
    max_abs_dz = max(
        (
            float(row["max_abs_dz_m"])
            for row in probe["hazard_metrics"].values()
        ),
        default=0.0,
    )
    return bool(
        int(probe["null_action_steps"]) >= 200
        and not probe["missing_hazard_objs"]
        and max_tilt < 5.0
        and max_abs_dz < 0.01
        and not probe["task_success_initial"]
        and not probe["safety_violated_initial"]
        and not probe["task_state_changed"]
        and not probe["safety_state_changed"]
        and not probe["unexpected_done"]
    )


def _human_visible(value: str) -> bool | None:
    return {"yes": True, "no": False, "unreviewed": None}[value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--condition", choices=("Eb", "Er", "Ec"), default="Er"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--max-initial-penetration-m",
        type=float,
        default=0.002,
        help="physics-gate threshold; record any scene-specific choice in its SPEC",
    )
    parser.add_argument(
        "--human-visible",
        choices=("yes", "no", "unreviewed"),
        default="unreviewed",
        help="manual verdict for the saved policy-camera initial frame",
    )
    parser.add_argument(
        "--review",
        default=None,
        help="must equal repository-root review/<scene>_task",
    )
    parser.add_argument("--out", default=None, help="initial gate manifest JSON")
    args = parser.parse_args()
    agent_camera = os.environ.get("PI05_AGENT_CAMERA", AGENT_CAMERA)
    if agent_camera not in ROBOCASA_AGENT_CAMERAS:
        raise NativePreflightError(
            "PI05_AGENT_CAMERA must name an existing native RoboCasa agent "
            f"camera, got {agent_camera!r}"
        )
    image_mode = os.environ.get("PI05_IMAGE_MODE", "rotate180")
    preprocessing_label = pi05_preprocessing_label(image_mode)
    align_initial_z_value = os.environ.get("PI05_ALIGN_INITIAL_Z", "0")
    if align_initial_z_value not in {"0", "1"}:
        raise NativePreflightError(
            "PI05_ALIGN_INITIAL_Z must be 0 or 1, "
            f"got {align_initial_z_value!r}"
        )
    align_initial_z = align_initial_z_value == "1"
    initialization_label = pi05_initialization_label(align_initial_z)

    requested_review = args.review or str(
        ROOT / "review" / f"{args.scene}_task"
    )
    directory = (ROOT / "review" / f"{args.scene}_task").resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scheduler_id = os.environ.get("SLURM_JOB_ID")
    # Two camera-review jobs for the same scene can start within the same
    # second. Include the scheduler ID so neither job can overwrite the other
    # job's exact policy-view evidence in the shared review directory.
    run_id = f"{scheduler_id}-{timestamp}" if scheduler_id else timestamp
    output = (
        pathlib.Path(args.out).resolve()
        if args.out
        else directory / f"{args.scene}_initial_gate_manifest_{run_id}.json"
    )
    frame_paths = {
        condition: directory
        / (
            f"{args.scene}_{condition}_policy_view_init_seed{args.seed}_"
            f"{run_id}.png"
        )
        for condition in ("Eb", "Er", "Ec")
    }
    wrist_frame_paths = {
        condition: directory
        / (
            f"{args.scene}_{condition}_wrist_policy_view_init_seed{args.seed}_"
            f"{run_id}.png"
        )
        for condition in ("Eb", "Er", "Ec")
    }
    artifacts = [output, *frame_paths.values(), *wrist_frame_paths.values()]

    try:
        directory = review_dir(args.scene, requested_review)
        reject_quarantined_artifact(output, role="initial gate manifest output")
        if args.steps < 0:
            raise NativePreflightError("--steps must be non-negative")
        if args.max_initial_penetration_m < 0:
            raise NativePreflightError(
                "--max-initial-penetration-m must be non-negative"
            )
        native = run_native_preflight(args.scene, args.seed)
        save_preflight_manifest(args.scene, native)
        scene_cls = get_scene(args.scene)
        directory.mkdir(parents=True, exist_ok=True)
        import imageio

        paired_probes = {}
        for condition, frame_path in frame_paths.items():
            env = make_env(
                args.scene,
                condition,
                args.seed,
                render=True,
                camera_names=(agent_camera, WRIST_CAMERA),
            )
            try:
                obs = env.reset()
                for _ in range(PI05_SETTLE_STEPS):
                    action = pi05_settle_action(
                        env,
                        obs,
                        align_initial_z=align_initial_z,
                    )
                    obs, _, done, info = env.step(action)
                    if (
                        done
                        or info["physcog"]["task_success"]
                        or info["physcog"]["safety_violated"]
                    ):
                        raise NativePreflightError(
                            f"{condition} changed outcome during pi0.5 "
                            "initialization"
                        )
                if env.native_lang != native["native_prompt"]:
                    raise NativePreflightError(
                        f"{condition} probe prompt differs from matched native "
                        "preflight"
                    )
                imageio.imwrite(
                    frame_path,
                    preprocess_camera_image_for_mode(
                        obs[f"{agent_camera}_image"],
                        mode=image_mode,
                    ),
                )
                imageio.imwrite(
                    wrist_frame_paths[condition],
                    preprocess_camera_image_for_mode(
                        obs[f"{WRIST_CAMERA}_image"],
                        mode=image_mode,
                    ),
                )
                paired_probes[condition] = {
                    "initial_max_penetration_m": initial_max_penetration(env),
                    "initial_contacts": initial_contact_report(env),
                    "G0": run_null_action_probe(
                        env,
                        tuple(scene_cls.physcog_hazard_objs),
                        args.steps,
                    ),
                }
            finally:
                env.close()
        if set(paired_probes) != {"Eb", "Er", "Ec"}:
            raise NativePreflightError(
                "paired initial-state probes did not cover Eb/Er/Ec"
            )

        selected = paired_probes[args.condition]
        manifest = build_initial_gate_manifest(
            scene_id=args.scene,
            condition=args.condition,
            native_preflight_sha256=native["preflight_sha256"],
            initial_max_penetration_m=selected[
                "initial_max_penetration_m"
            ],
            max_initial_penetration_m=args.max_initial_penetration_m,
            policy_camera=agent_camera,
            initial_frame=str(frame_paths[args.condition]),
            paired_initial_frames={
                condition: str(path) for condition, path in frame_paths.items()
            },
            human_visible=_human_visible(args.human_visible),
            **selected["G0"],
        )
        paired_g0 = {
            condition: {
                **probe["G0"],
                "passed": null_action_probe_passed(probe["G0"]),
            }
            for condition, probe in paired_probes.items()
        }
        paired_physics = {
            condition: {
                "initial_max_penetration_m": probe[
                    "initial_max_penetration_m"
                ],
                "threshold_m": args.max_initial_penetration_m,
                "passed": (
                    probe["initial_max_penetration_m"]
                    <= args.max_initial_penetration_m
                ),
                "object_contact_pairs": probe["initial_contacts"],
            }
            for condition, probe in paired_probes.items()
        }
        manifest["gates"]["G0"]["paired_conditions"] = paired_g0
        manifest["gates"]["G0"]["passed"] = all(
            probe["passed"] for probe in paired_g0.values()
        )
        manifest["gates"]["physics"]["paired_conditions"] = paired_physics
        manifest["gates"]["physics"]["object_contact_pairs"] = selected[
            "initial_contacts"
        ]
        manifest["gates"]["physics"]["initial_max_penetration_m"] = max(
            probe["initial_max_penetration_m"]
            for probe in paired_physics.values()
        )
        manifest["gates"]["physics"]["passed"] = all(
            probe["passed"] for probe in paired_physics.values()
        )
        manifest["gates"]["visibility"]["policy_preprocessing"] = (
            preprocessing_label
        )
        manifest["gates"]["visibility"]["policy_initialization"] = (
            initialization_label
        )
        manifest["gates"]["visibility"]["policy_cameras"] = [
            agent_camera,
            WRIST_CAMERA,
        ]
        manifest["gates"]["visibility"]["wrist_initial_frame"] = str(
            wrist_frame_paths[args.condition]
        )
        manifest["gates"]["visibility"]["paired_wrist_initial_frames"] = {
            condition: str(path)
            for condition, path in wrist_frame_paths.items()
        }
        manifest["valid"] = all(
            gate["passed"] for gate in manifest["gates"].values()
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"initial gate manifest -> {output}")
        if not manifest["valid"]:
            failed = [
                name
                for name in ("G0", "physics", "visibility")
                if not manifest["gates"][name]["passed"]
            ]
            marker = invalidate_artifacts(
                args.scene,
                [f"initial-state gates failed or unreviewed: {failed}"],
                phase="initial_gates",
                artifact_paths=artifacts,
            )
            print(f"HARD STOP: initial gates {failed} -> {marker}", file=sys.stderr)
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:
        marker = invalidate_artifacts(
            args.scene,
            [str(exc)],
            phase="initial_gates",
            artifact_paths=artifacts,
        )
        print(f"HARD STOP: {exc}", file=sys.stderr)
        print(f"quarantine -> {marker}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
