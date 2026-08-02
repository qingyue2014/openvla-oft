"""Pre-policy geometry calibration for the native L1-A1 counterfactual.

This diagnostic never writes evaluated HDF5 files and never runs a learned
policy.  It applies a preregistered finite candidate list to one native task-1
state and runs the same exact reset, no-op wait, support/contact, stability,
and policy-view gates used by the paired-state generator.  Passing candidates
remain calibration evidence only; one must be frozen in a new scene version
before generating any formal state bundle.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks import l1a1_native_pipeline as scene


CANDIDATES = (
    {"id": "C01", "target_xy": (-0.080, 0.070), "landmark_xy": (-0.245, 0.070), "ec_lure_xy": (0.220, -0.150)},
    {"id": "C02", "target_xy": (-0.060, 0.040), "landmark_xy": (-0.225, 0.040), "ec_lure_xy": (0.220, -0.150)},
    {"id": "C03", "target_xy": (-0.040, -0.040), "landmark_xy": (-0.205, -0.040), "ec_lure_xy": (0.220, -0.150)},
    {"id": "C04", "target_xy": (-0.100, 0.020), "landmark_xy": (-0.265, 0.020), "ec_lure_xy": (0.220, -0.150)},
    {"id": "C05", "target_xy": (-0.050, 0.100), "landmark_xy": (-0.215, 0.100), "ec_lure_xy": (0.220, -0.150)},
    {"id": "C06", "target_xy": (-0.100, 0.120), "landmark_xy": (-0.265, 0.120), "ec_lure_xy": (0.220, -0.150)},
    {"id": "C07", "target_xy": (0.100, -0.100), "landmark_xy": (-0.065, -0.100), "ec_lure_xy": (0.260, 0.060)},
    {"id": "C08", "target_xy": (0.120, -0.060), "landmark_xy": (-0.045, -0.060), "ec_lure_xy": (0.270, 0.100)},
)


def _paired_states(env, native_state, candidate):
    p = scene.pipeline
    env.reset()
    env.set_init_state(native_state)
    env.sim.forward()
    eb_state = env.sim.get_state().flatten()
    eb_target_xy = p._body_pos(env, scene.TARGET)[:2]
    positions = {
        scene.TARGET: np.asarray(candidate["target_xy"], dtype=float),
        scene.LANDMARK: np.asarray(candidate["landmark_xy"], dtype=float),
    }
    er_state, _ = p._settled_variant(
        env,
        eb_state,
        {**positions, scene.LURE: eb_target_xy},
    )
    ec_candidate, _ = p._settled_variant(
        env,
        eb_state,
        {
            **positions,
            scene.LURE: np.asarray(candidate["ec_lure_xy"], dtype=float),
        },
    )
    env.set_init_state(er_state)
    shared = {
        scene.TARGET: p._capture_free_joint(env.sim, scene.TARGET),
        scene.LANDMARK: p._capture_free_joint(env.sim, scene.LANDMARK),
    }
    env.set_init_state(ec_candidate)
    shared[scene.LURE] = p._capture_free_joint(env.sim, scene.LURE)
    ec_state = p._transplant_bowls(env, eb_state, shared)
    return {"Eb": eb_state, "Er": er_state, "Ec": ec_state}


def run(args) -> list[dict[str, object]]:
    p = scene.pipeline
    p.MAX_TARGET_LANDMARK_DISTANCE = 0.190
    p.MIN_BOWL_DISTANCE = 0.160
    suite, task, bddl = p._task_and_suite()
    del suite
    native_states = p._load_native_init_states(task)
    env = p._env(bddl, render=True)
    out_dir = Path(args.preview_dir)
    rows: list[dict[str, object]] = []
    try:
        candidates = tuple(
            candidate
            for candidate in CANDIDATES
            if not args.candidate or candidate["id"] == args.candidate
        )
        if not candidates:
            raise ValueError(f"unknown candidate: {args.candidate}")
        state_indices = (
            range(len(native_states))
            if args.all_native_states
            else (args.native_state_index,)
        )
        for state_index in state_indices:
            native_state = native_states[state_index]
            for candidate in candidates:
                row: dict[str, object] = {
                    **candidate,
                    "native_state_index": state_index,
                }
                try:
                    states = _paired_states(env, native_state, candidate)
                    diagnostics = {
                        condition: p._validate_condition(env, state, condition)
                        for condition, state in states.items()
                    }
                    er_ec_qpos, er_ec_qvel = p._purity_error(
                        env, states["Er"], states["Ec"], (scene.LURE,)
                    )
                    if max(er_ec_qpos, er_ec_qvel) > p.PAIR_TOLERANCE:
                        raise RuntimeError(
                            "Er/Ec differs outside the lure joint: "
                            f"qpos={er_ec_qpos:.3e}, qvel={er_ec_qvel:.3e}"
                        )
                    row.update(
                        {
                            "verdict": "PASS_L1A1_GEOMETRY_CANDIDATE",
                            "er_ec_unallowed_qpos_error": er_ec_qpos,
                            "er_ec_unallowed_qvel_error": er_ec_qvel,
                            "diagnostics": diagnostics,
                        }
                    )
                    if state_index < args.preview_count:
                        candidate_dir = out_dir / str(candidate["id"])
                        for condition, state in states.items():
                            p._save_preview(
                                env, state, candidate_dir, condition, state_index
                            )
                except Exception as exc:  # fail each pair independently
                    row.update(
                        {
                            "verdict": "FAIL_L1A1_GEOMETRY_CANDIDATE",
                            "failure_type": type(exc).__name__,
                            "failure": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                rows.append(row)
                print(
                    f"candidate={candidate['id']} state={state_index} "
                    f"verdict={row['verdict']} reason={row.get('failure', '--')}"
                )
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native_state_index", type=int, default=0)
    parser.add_argument("--all_native_states", action="store_true")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--preview_count", type=int, default=3)
    parser.add_argument("--preview_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = run(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene": "L1-A1 pre-policy native geometry calibration",
                "native_task": "libero_spatial/1",
                "learned_policy_rollouts": 0,
                "candidate_count": len(rows),
                "pass_count": sum(row["verdict"].startswith("PASS") for row in rows),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not any(row["verdict"].startswith("PASS") for row in rows):
        print("verdict=FAIL_L1A1_GEOMETRY_CALIBRATION")
        raise SystemExit(2)
    print("verdict=PASS_L1A1_GEOMETRY_CALIBRATION")


if __name__ == "__main__":
    main()
