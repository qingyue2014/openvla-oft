#!/usr/bin/env python3
"""Calibrate terminal panel B for a robust L3-A2 A->B cascade."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import _find_free_joint_vadr
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.l3a2_cascade_artifacts import (
    DEFAULT_EB,
    DEFAULT_EC,
    DEFAULT_ER,
    TASK_KEY,
    extract_eb,
    validate_pairing,
    write_report,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a2_cascade_logic import (
    trajectory_candidates,
)
from experiments.robot.libero.tasks.validate_l3a2_cascade_scene import (
    TERMINAL_BODY,
    _scripted_close,
    audit_terminal_asset_geoms,
    initial_terminal_clearance_gate,
    passive_terminal_gate,
    validate_episode,
)


def _load(path: Path) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        names = sorted(
            (name for name in group if name.startswith("demo_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        return [np.asarray(group[name]["initial_state"][:]) for name in names]


def _patch_state(
    state: np.ndarray,
    qpos_flat: int,
    qvel_flat: int,
    x: float,
    y: float,
    yaw_deg: float,
    equilibrium_slices: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    result = np.asarray(state).copy()
    if equilibrium_slices is not None:
        result[qpos_flat:qpos_flat + 7] = equilibrium_slices[0]
        result[qvel_flat:qvel_flat + 6] = equilibrium_slices[1]
    result[qpos_flat:qpos_flat + 2] = (x, y)
    yaw = np.radians(yaw_deg)
    result[qpos_flat + 3:qpos_flat + 7] = (
        np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
    )
    result[qvel_flat:qvel_flat + 6] = 0.0
    return result


def _settle_terminal_slices(
    env,
    state: np.ndarray,
    qpos_flat: int,
    qvel_flat: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Settle B once at a safe staging pose before transplanting it."""
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(state))
    clear_mujoco_replay_transients(env)
    env.sim.forward()
    initial = np.asarray(env.sim.get_state().flatten()).copy()
    for _ in range(steps):
        env.sim.step()
    settled = np.asarray(env.sim.get_state().flatten()).copy()
    qpos = settled[qpos_flat:qpos_flat + 7].copy()
    qvel = settled[qvel_flat:qvel_flat + 6].copy()
    initial_pos = initial[qpos_flat:qpos_flat + 3]
    xy_displacement = float(np.linalg.norm(qpos[:2] - initial_pos[:2]))
    vertical_settle = float(abs(qpos[2] - initial_pos[2]))
    _, qx, qy, _ = qpos[3:7]
    up_z = float(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))
    tilt = float(np.degrees(np.arccos(up_z)))
    speed = float(np.linalg.norm(qvel))
    if (
        xy_displacement > 0.005
        or vertical_settle > 0.100
        or tilt > 3.0
        or speed > 0.01
    ):
        raise RuntimeError(
            "terminal B staging equilibrium is invalid: "
            f"xy_displacement={xy_displacement:.4f}m "
            f"vertical_settle={vertical_settle:.4f}m "
            f"tilt={tilt:.2f}deg speed={speed:.4f}"
        )
    return qpos, qvel


def _write_candidate(
    source: Path,
    destination: Path,
    qpos_flat: int,
    qvel_flat: int,
    x: float,
    y: float,
    yaw_deg: float,
    equilibrium_slices: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with h5py.File(destination, "a") as handle:
        group = handle[TASK_KEY]
        group.attrs["l3a2_terminal_x_m"] = x
        group.attrs["l3a2_terminal_y_m"] = y
        group.attrs["l3a2_terminal_yaw_deg"] = yaw_deg
        group.attrs["l3a2_topology"] = (
            "drawer_S_supports_A_then_A_impacts_terminal_B"
        )
        names = sorted(
            (key for key in group if key.startswith("demo_")),
            key=lambda key: int(key.split("_")[-1]),
        )
        if len(names) != len(equilibrium_slices):
            raise ValueError("terminal equilibrium count does not match demos")
        for index, name in enumerate(names):
            demo = group[name]
            for dataset in ("initial_state", "base_reset_state"):
                if dataset not in demo:
                    continue
                original = np.asarray(demo[dataset][:])
                updated = _patch_state(
                    original,
                    qpos_flat,
                    qvel_flat,
                    x,
                    y,
                    yaw_deg,
                    equilibrium_slices[index],
                )
                demo[dataset][...] = updated
                if dataset == "base_reset_state":
                    demo.attrs["base_state_sha256"] = hashlib.sha256(
                        updated.tobytes()
                    ).hexdigest()


def _yaw_distance(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _cluster(
    rows: list[dict], radius: float, max_yaw_distance_deg: float
) -> list[dict]:
    passed = [row for row in rows if row["pass_rate"] >= 0.8]
    for row in passed:
        witnesses = [
            other for other in passed
            if other is not row
            and np.hypot(row["x"] - other["x"], row["y"] - other["y"])
            > 1e-6
            and np.hypot(row["x"] - other["x"], row["y"] - other["y"])
            <= radius
            and _yaw_distance(row["yaw_deg"], other["yaw_deg"])
            <= max_yaw_distance_deg
            and abs(row["mass_scale"] - other["mass_scale"]) <= 1e-9
        ]
        if witnesses:
            return [row, min(
                witnesses,
                key=lambda other: np.hypot(
                    row["x"] - other["x"], row["y"] - other["y"]
                ),
            )]
    return []


def run(args: argparse.Namespace) -> str:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base_er, base_ec, base_eb = (
        work / "base_er.hdf5",
        work / "base_ec.hdf5",
        work / "base_eb.hdf5",
    )
    env = os.environ.copy()
    env.update({
        "NUM_TRIALS": str(args.num_states),
        "MAX_ATTEMPTS": str(args.max_attempts),
        "ER": str(base_er),
        "EC": str(base_ec),
        "EB": str(base_eb),
        "LOG_DIR": str(work / "base_logs"),
        "SKIP_CASCADE_GATE": "1",
        "BDDL": args.bddl,
    })
    subprocess.run(
        [
            "bash",
            "experiments/robot/libero/tasks/run_l3a2_cascade.sh",
            "all",
            "prepare",
        ],
        check=True,
        env=env,
    )
    eb_states, er_states, ec_states = (
        _load(base_eb), _load(base_er), _load(base_ec)
    )
    sim_env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    sim_env.reset()
    qadr = _find_free_joint_qadr(sim_env.sim, TERMINAL_BODY)
    vadr = _find_free_joint_vadr(sim_env.sim, TERMINAL_BODY)
    if qadr < 0 or vadr < 0:
        raise RuntimeError("terminal panel free joint not found")
    audit_terminal_asset_geoms(sim_env)
    qpos_flat = 1 + qadr
    qvel_flat = 1 + int(sim_env.sim.model.nq) + vadr
    terminal_equilibria = [
        _settle_terminal_slices(
            sim_env,
            state,
            qpos_flat,
            qvel_flat,
            args.terminal_settle_steps,
        )
        for state in eb_states
    ]
    diagnostic_responses = []
    for state, equilibrium in zip(er_states, terminal_equilibria):
        staged = _patch_state(
            state,
            qpos_flat,
            qvel_flat,
            float(equilibrium[0][0]),
            float(equilibrium[0][1]),
            0.0,
            equilibrium,
        )
        diagnostic_responses.append(_scripted_close(
            sim_env,
            staged,
            steps=args.close_steps,
            disable_terminal_collision=True,
        ))
    if args.x is not None or args.y is not None:
        if args.x is None or args.y is None:
            raise ValueError("--x and --y must be supplied together")
        yaw_values = args.yaw_deg if args.yaw_deg is not None else [0.0]
        candidates = [
            (x, y, yaw, mass_scale)
            for x in args.x
            for y in args.y
            for yaw in yaw_values
            for mass_scale in args.mass_scales
        ]
        trace_rows = []
        candidate_source = "explicit_xy_grid"
    else:
        panel_poses, trace_rows = trajectory_candidates(
            diagnostic_responses,
            axial_stations=tuple(args.link_axial_stations),
            normal_offsets=tuple(args.path_normal_offsets),
            tangent_offset=args.path_tangent_offset,
            cabinet_outward_y_offsets=tuple(
                args.cabinet_outward_y_offsets
            ),
            panel_yaws_deg=tuple(args.panel_yaws_deg),
            quantization=args.path_quantization,
            yaw_quantization_deg=args.yaw_quantization_deg,
            limit=args.max_candidates,
        )
        candidates = [
            (*pose, mass_scale)
            for pose in panel_poses
            for mass_scale in args.mass_scales
        ]
        candidate_source = "measured_post_release_panel_pose_sweep"
    if not candidates:
        raise RuntimeError("no terminal-B candidates were derived")
    trace_path = Path(args.trace_json)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_payload = {
        "candidate_source": candidate_source,
        "candidates": [
            {"x": x, "y": y, "yaw_deg": yaw, "mass_scale": mass_scale}
            for x, y, yaw, mass_scale in candidates
        ],
        "link_endpoint_trace": trace_rows,
        "diagnostic_responses": [{
            key: value for key, value in response.items()
            if key != "timeline"
        } for response in diagnostic_responses],
    }
    rows = []
    episode_evidence = []
    try:
        for x, y, yaw_deg, mass_scale in candidates:
            passed = 0
            passive_passed = 0
            reasons = []
            max_b_disp = 0.0
            max_disabled_b_disp = 0.0
            max_disabled_b_tilt = 0.0
            direct_component_contact = False
            min_initial_center_distance = float("inf")
            reset_contacts = set()
            for episode, (eb, er, ec) in enumerate(
                zip(eb_states, er_states, ec_states)
            ):
                patched = [
                    _patch_state(
                        state,
                        qpos_flat,
                        qvel_flat,
                        x,
                        y,
                        yaw_deg,
                        terminal_equilibria[episode],
                    )
                    for state in (eb, er, ec)
                ]
                passive = [
                    passive_terminal_gate(
                        sim_env,
                        state,
                        terminal_mass_scale=mass_scale,
                    )
                    for state in (patched[0], patched[2])
                ]
                risk_reset = initial_terminal_clearance_gate(
                    sim_env,
                    patched[1],
                    terminal_mass_scale=mass_scale,
                )
                min_initial_center_distance = min(
                    min_initial_center_distance,
                    *(row["initial_link_terminal_center_distance_m"]
                      for row in passive),
                )
                for row in passive:
                    reset_contacts.update(row["initial_contacts"])
                if (
                    not all(row["passed"] for row in passive)
                    or not risk_reset["passed"]
                ):
                    reasons.append(
                        "PRE_FILTER_STATIC_CLEARANCE_OR_STABILITY_FAIL"
                    )
                    episode_evidence.append({
                        "x": x,
                        "y": y,
                        "yaw_deg": yaw_deg,
                        "mass_scale": mass_scale,
                        "episode": episode,
                        "prefilter_passed": False,
                        "baseline_passive": passive[0],
                        "stable_passive": passive[1],
                        "risk_reset_clearance": risk_reset,
                    })
                    continue
                passive_passed += 1
                result = validate_episode(
                    sim_env,
                    *patched,
                    args.close_steps,
                    baseline_passive=passive[0],
                    stable_passive=passive[1],
                    terminal_mass_scale=mass_scale,
                )
                risk_timeline = result["risk"]["timeline"]
                collision_disabled = result["collision_disabled"]
                episode_evidence.append({
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                    "mass_scale": mass_scale,
                    "episode": episode,
                    "prefilter_passed": True,
                    "risk_reset_clearance": risk_reset,
                    "cascade_passed": result["passed"],
                    "failures": result["failures"],
                    "initial_link_terminal_center_distance_m": result[
                        "risk"
                    ]["initial_link_terminal_center_distance_m"],
                    "min_link_terminal_center_distance_m": min(
                        event["link_terminal_center_distance_m"]
                        for event in risk_timeline
                    ),
                    "terminal_contact_bodies": sorted({
                        body
                        for event in risk_timeline
                        for body in event["terminal_contact_bodies"]
                    }),
                    "forbidden_direct_component_contact": result[
                        "risk"
                    ]["terminal_component_contact"],
                    "risk_terminal_displacement_m": result[
                        "risk"
                    ]["max_terminal_displacement_m"],
                    "risk_terminal_tilt_change_deg": result[
                        "risk"
                    ]["max_terminal_tilt_change_deg"],
                    "collision_disable_applied": collision_disabled[
                        "collision_intervention_applied"
                    ],
                    "collision_disabled_terminal_displacement_m": (
                        collision_disabled["max_terminal_displacement_m"]
                    ),
                    "collision_disabled_terminal_tilt_change_deg": (
                        collision_disabled["max_terminal_tilt_change_deg"]
                    ),
                    "support_release_step": result["risk"].get(
                        "support_release_step"
                    ),
                    "impact_step": result["risk"].get("impact_step"),
                    "terminal_hazard_step": result["risk"].get(
                        "terminal_hazard_step"
                    ),
                })
                passed += int(result["passed"])
                reasons.extend(result["failures"])
                max_b_disp = max(
                    max_b_disp,
                    result["risk"]["max_terminal_displacement_m"],
                )
                max_disabled_b_disp = max(
                    max_disabled_b_disp,
                    collision_disabled["max_terminal_displacement_m"],
                )
                max_disabled_b_tilt = max(
                    max_disabled_b_tilt,
                    collision_disabled["max_terminal_tilt_change_deg"],
                )
                direct_component_contact |= result[
                    "risk"
                ]["terminal_component_contact"]
            rate = passed / len(er_states)
            passive_rate = passive_passed / len(er_states)
            row = {
                "x": x,
                "y": y,
                "yaw_deg": yaw_deg,
                "mass_scale": mass_scale,
                "passed": passed,
                "passive_passed": passive_passed,
                "episodes": len(er_states),
                "pass_rate": rate,
                "passive_pass_rate": passive_rate,
                "min_initial_link_terminal_center_distance_m": (
                    min_initial_center_distance
                ),
                "initial_terminal_contacts": ",".join(
                    sorted(reset_contacts)
                ),
                "max_terminal_displacement_m": max_b_disp,
                "max_collision_disabled_terminal_displacement_m": (
                    max_disabled_b_disp
                ),
                "max_collision_disabled_terminal_tilt_change_deg": (
                    max_disabled_b_tilt
                ),
                "forbidden_direct_component_contact": int(
                    direct_component_contact
                ),
                "failures": " | ".join(sorted(set(reasons))),
            }
            rows.append(row)
            print(
                f"x={x:+.3f} y={y:+.3f} yaw={yaw_deg:+.1f} "
                f"mass_scale={mass_scale:.3f} "
                f"passive={passive_passed}/{len(er_states)} "
                f"cascade={passed}/{len(er_states)} "
                f"min_AB={min_initial_center_distance:.4f} "
                f"contacts={row['initial_terminal_contacts'] or '-'} "
                f"failure={row['failures'] or '-'}"
            )
    finally:
        sim_env.close()
    trace_payload["candidate_evaluations"] = episode_evidence
    trace_path.write_text(
        json.dumps(trace_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cluster = _cluster(
        rows, args.adjacent_radius, args.adjacent_yaw_distance_deg
    )
    selected = (
        max(
            cluster,
            key=lambda row: (
                row["pass_rate"], row["max_terminal_displacement_m"]
            ),
        )
        if cluster else None
    )
    if args.calibration_only:
        verdict = (
            "PASS_L3A2_PANEL_MASS_SWEEP"
            if cluster else "FAIL_L3A2_PANEL_MASS_SWEEP"
        )
    else:
        verdict = (
            "PASS_L3A2_CASCADE_GEOMETRY_SWEEP"
            if cluster else "FAIL_L3A2_CASCADE_GEOMETRY_SWEEP"
        )
    if cluster and not args.calibration_only:
        assert selected is not None
        if abs(selected["mass_scale"] - 1.0) > 1e-9:
            raise RuntimeError(
                "non-unit mass calibration must be committed to XML and "
                "revalidated before writing paired artifacts"
            )
        _write_candidate(
            base_er, Path(args.er), qpos_flat, qvel_flat,
            selected["x"], selected["y"], selected["yaw_deg"],
            terminal_equilibria,
        )
        _write_candidate(
            base_ec, Path(args.ec), qpos_flat, qvel_flat,
            selected["x"], selected["y"], selected["yaw_deg"],
            terminal_equilibria,
        )
        extract_eb(args.er, args.eb)
        pairing = validate_pairing(args.eb, args.er, args.ec)
        write_report(pairing, args.pairing_report)
        if pairing["verdict"].startswith("FAIL"):
            raise RuntimeError("selected geometry destroyed episode pairing")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# L3-A2 cascade geometry sweep",
        "",
        f"- Verdict: **{verdict}**",
        f"- Episodes per candidate: {args.num_states}",
        f"- Candidate source: `{candidate_source}` ({len(candidates)} poses).",
        "- Every pose was prefiltered for Eb/Ec passive stability, table-only "
        "contact, and no initial A-B contact.",
        "- Candidate acceptance: physical cascade family pass rate ≥0.80.",
        f"- Robustness witness radius: {args.adjacent_radius:.3f} m.",
        "- No formal evaluation is authorized by this sweep alone.",
        "",
    ]
    if cluster:
        report += [
            f"- Selected B xy: ({selected['x']:+.4f}, {selected['y']:+.4f}) m",
            f"- Selected B yaw: {selected['yaw_deg']:+.1f}°",
            f"- Selected terminal mass scale: {selected['mass_scale']:.4f}",
            f"- Selected pass rate: {selected['pass_rate']:.3f}",
            f"- Adjacent witness: ({cluster[1]['x']:+.4f}, "
            f"{cluster[1]['y']:+.4f}) m, yaw="
            f"{cluster[1]['yaw_deg']:+.1f}°; "
            f"rate={cluster[1]['pass_rate']:.3f}",
        ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl",
        default="experiments/robot/libero/tasks/"
        "PHYSCOG_L3A2_drawer_bottle_cascade.bddl",
    )
    parser.add_argument("--x", nargs="+", type=float)
    parser.add_argument("--y", nargs="+", type=float)
    parser.add_argument("--yaw-deg", nargs="+", type=float)
    parser.add_argument(
        "--mass-scales", nargs="+", type=float, default=[1.0]
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=int(os.environ.get("NUM_TRIALS", "5")),
    )
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--close-steps", type=int, default=120)
    parser.add_argument("--terminal-settle-steps", type=int, default=800)
    parser.add_argument(
        "--link-axial-stations",
        nargs="+",
        type=float,
        default=[0.040, 0.065, 0.090],
    )
    parser.add_argument(
        "--path-normal-offsets",
        nargs="+",
        type=float,
        default=[0.000, 0.008, 0.016],
    )
    parser.add_argument("--path-tangent-offset", type=float, default=0.010)
    parser.add_argument(
        "--cabinet-outward-y-offsets",
        nargs="+",
        type=float,
        default=[0.030, 0.040, 0.050],
    )
    parser.add_argument(
        "--panel-yaws-deg", nargs="+", type=float, default=[90, 95]
    )
    parser.add_argument("--path-quantization", type=float, default=0.005)
    parser.add_argument("--yaw-quantization-deg", type=float, default=5.0)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--adjacent-radius", type=float, default=0.021)
    parser.add_argument("--adjacent-yaw-distance-deg", type=float, default=15.0)
    parser.add_argument(
        "--work-dir", default="experiments/logs/l3a2_geometry_work"
    )
    parser.add_argument("--er", default=DEFAULT_ER)
    parser.add_argument("--ec", default=DEFAULT_EC)
    parser.add_argument("--eb", default=DEFAULT_EB)
    parser.add_argument(
        "--pairing-report", default="experiments/logs/l3a2_pairing.md"
    )
    parser.add_argument(
        "--out-report", default="experiments/logs/l3a2_geometry_sweep.md"
    )
    parser.add_argument(
        "--out-csv", default="experiments/logs/l3a2_geometry_sweep.csv"
    )
    parser.add_argument(
        "--trace-json", default="experiments/logs/l3a2_link_fall_trace.json"
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.mass_scales):
        parser.error("--mass-scales must all be positive")
    verdict = run(args)
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
