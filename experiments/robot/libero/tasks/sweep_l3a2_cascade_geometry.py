#!/usr/bin/env python3
"""Calibrate terminal bottle B for a robust L3-A2 A->B cascade."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from experiments.robot.libero.tasks.validate_l3a2_cascade_scene import (
    TERMINAL_BODY,
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
) -> np.ndarray:
    result = np.asarray(state).copy()
    result[qpos_flat:qpos_flat + 2] = (x, y)
    result[qpos_flat + 3:qpos_flat + 7] = (1.0, 0.0, 0.0, 0.0)
    result[qvel_flat:qvel_flat + 6] = 0.0
    return result


def _write_candidate(
    source: Path,
    destination: Path,
    qpos_flat: int,
    qvel_flat: int,
    x: float,
    y: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with h5py.File(destination, "a") as handle:
        group = handle[TASK_KEY]
        group.attrs["l3a2_terminal_x_m"] = x
        group.attrs["l3a2_terminal_y_m"] = y
        group.attrs["l3a2_topology"] = (
            "drawer_S_supports_A_then_A_impacts_terminal_B"
        )
        for name in sorted(
            (key for key in group if key.startswith("demo_")),
            key=lambda key: int(key.split("_")[-1]),
        ):
            demo = group[name]
            for dataset in ("initial_state", "base_reset_state"):
                if dataset not in demo:
                    continue
                original = np.asarray(demo[dataset][:])
                updated = _patch_state(
                    original, qpos_flat, qvel_flat, x, y
                )
                demo[dataset][...] = updated
                if dataset == "base_reset_state":
                    demo.attrs["base_state_sha256"] = hashlib.sha256(
                        updated.tobytes()
                    ).hexdigest()


def _cluster(rows: list[dict], radius: float) -> list[dict]:
    passed = [row for row in rows if row["pass_rate"] >= 0.8]
    for row in passed:
        witnesses = [
            other for other in passed
            if other is not row
            and np.hypot(row["x"] - other["x"], row["y"] - other["y"])
            <= radius
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
        raise RuntimeError("terminal bottle free joint not found")
    qpos_flat = 1 + qadr
    qvel_flat = 1 + int(sim_env.sim.model.nq) + vadr
    rows = []
    try:
        for x in args.x:
            for y in args.y:
                passed = 0
                reasons = []
                max_b_disp = 0.0
                for eb, er, ec in zip(eb_states, er_states, ec_states):
                    patched = [
                        _patch_state(state, qpos_flat, qvel_flat, x, y)
                        for state in (eb, er, ec)
                    ]
                    result = validate_episode(
                        sim_env, *patched, args.close_steps
                    )
                    passed += int(result["passed"])
                    reasons.extend(result["failures"])
                    max_b_disp = max(
                        max_b_disp,
                        result["risk"]["max_terminal_displacement_m"],
                    )
                rate = passed / len(er_states)
                row = {
                    "x": x,
                    "y": y,
                    "passed": passed,
                    "episodes": len(er_states),
                    "pass_rate": rate,
                    "max_terminal_displacement_m": max_b_disp,
                    "failures": " | ".join(sorted(set(reasons))),
                }
                rows.append(row)
                print(
                    f"x={x:+.3f} y={y:+.3f} pass={passed}/{len(er_states)} "
                    f"B_disp_max={max_b_disp:.4f}"
                )
    finally:
        sim_env.close()
    cluster = _cluster(rows, args.adjacent_radius)
    verdict = (
        "PASS_L3A2_CASCADE_GEOMETRY_SWEEP"
        if cluster else "FAIL_L3A2_CASCADE_GEOMETRY_SWEEP"
    )
    if cluster:
        selected = max(
            cluster,
            key=lambda row: (
                row["pass_rate"], row["max_terminal_displacement_m"]
            ),
        )
        _write_candidate(
            base_er, Path(args.er), qpos_flat, qvel_flat,
            selected["x"], selected["y"],
        )
        _write_candidate(
            base_ec, Path(args.ec), qpos_flat, qvel_flat,
            selected["x"], selected["y"],
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
        "- Candidate acceptance: physical cascade family pass rate ≥0.80.",
        f"- Robustness witness radius: {args.adjacent_radius:.3f} m.",
        "- No formal evaluation is authorized by this sweep alone.",
        "",
    ]
    if cluster:
        report += [
            f"- Selected B xy: ({selected['x']:+.4f}, {selected['y']:+.4f}) m",
            f"- Selected pass rate: {selected['pass_rate']:.3f}",
            f"- Adjacent witness: ({cluster[1]['x']:+.4f}, "
            f"{cluster[1]['y']:+.4f}) m; rate={cluster[1]['pass_rate']:.3f}",
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
    parser.add_argument("--x", nargs="+", type=float, default=[0.04, 0.055, 0.07, 0.085, 0.10])
    parser.add_argument("--y", nargs="+", type=float, default=[0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--close-steps", type=int, default=120)
    parser.add_argument("--adjacent-radius", type=float, default=0.021)
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
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
