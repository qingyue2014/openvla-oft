"""
Generate paired initial states for the L1-A3/L1-A4 contact-boundary scenes.

Both scenes reuse libero_spatial task 1 with the native prompt:
    "pick up the black bowl next to the ramekin and place it on the plate"

L1-A3 (grasp-side boundary): the ramekin — the prompt's own landmark — is the
only moved object. It is parked a gated rim gap away from the target bowl so
one grasp arc is blocked; the policy must rotate its grasp axis into the free
arc.

L1-A4 (placement-side boundary): the bystander bowl is the only moved object.
It is parked a gated rim gap outside the plate so the free placement region
collapses to a crescent; the policy must steer the release point into the free
sector.

Unlike the L1-A2 generator, this one follows the strict official-state
transplant discipline: the mover is settled in a scratch pass, its free-joint
qpos/qvel are captured, the official benchmark state is restored, and only the
mover joint is transplanted. Every non-mover qpos/qvel element must match the
official state to <= 1e-10, so Eb (the shared L1-A1 native baseline), Er, and
Ec differ in exactly one free joint.
"""

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1a2_initial_states import (  # noqa: E402
    _body_pos,
    _find_free_joint_qadr,
    _geom_ids_for_body,
    _import_libero_modules,
    _min_contact_distance_between_bodies,
    _settle,
    _visible_pixel_count,
    _world_aabb,
    _zero_free_joint_velocity,
)

TASK_ID = 1
TARGET_BODY = "akita_black_bowl_1_main"
PLATE_BODY = "plate_1_main"
RAMEKIN_BODY = "glazed_rim_porcelain_ramekin_1_main"
BYSTANDER_BODY = "akita_black_bowl_2_main"
COOKIE_BODY = "cookies_1_main"
ALL_BODIES = (TARGET_BODY, PLATE_BODY, RAMEKIN_BODY, BYSTANDER_BODY, COOKIE_BODY)

SCENARIOS = {
    # L1-A3: ramekin crowds the target bowl's grasp circle.
    "l1a3": {
        "mover_body": RAMEKIN_BODY,
        "anchor_body": TARGET_BODY,
        # Er rim gap between mover and anchor outer radii (sampled per demo).
        "er_gap_range": (0.008, 0.020),
        # Ec parks the mover at a matched-safe distance along the same bearing
        # so the transplant/settle history is identical; only the gap differs.
        "ec_gap": 0.100,
        "min_ec_gap": 0.060,
        # Angular span of the anchor's grasp circle blocked by the mover
        # (computed from asset AABB radii + finger clearance, so the window is
        # generous; the behavioral calibrate gate is the authoritative check).
        "blocked_arc_window_deg": (55.0, 170.0),
        "min_free_arc_deg": 190.0,
        "gap_attr": "rim_gap_m",
    },
    # L1-A4: bystander bowl crowds the plate's placement boundary.
    "l1a4": {
        "mover_body": BYSTANDER_BODY,
        "anchor_body": PLATE_BODY,
        "er_gap_range": (0.005, 0.015),
        "ec_gap": 0.100,
        "min_ec_gap": 0.060,
        "blocked_arc_window_deg": (30.0, 170.0),
        "min_free_arc_deg": 120.0,
        "gap_attr": "plate_rim_gap_m",
    },
}

# Gripper finger clearance added to the mover radius when computing the
# blocked arc: a rim grasp needs finger room, not just body separation.
FINGER_CLEARANCE = 0.030

SETTLE_STEPS = 60
STABILITY_CHECK_STEPS = 40
NUM_STEPS_WAIT = 10  # matches the evaluator's pre-policy no-op window
MAX_MOVER_SETTLE_DRIFT = 0.010
MAX_MOVER_WAIT_DRIFT = 0.010
MAX_PROTECTED_DISPLACEMENT = 0.005
MAX_NON_MOVER_ERROR = 1e-10
MAX_MOVER_RESIDUAL_SPEED = 0.010

VIS_CAMERAS = ("agentview", "robot0_eye_in_hand")
VIS_RESOLUTION = 512
MIN_MOVER_PIXELS = 30

BEARING_SWEEP_DEG = (0.0, 12.0, -12.0, 25.0, -25.0)


def _xy_radius(env, body_name: str) -> float:
    lo, hi = _world_aabb(env, body_name)
    return float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2.0)


def _free_joint_slices(sim, body_name: str):
    qadr = _find_free_joint_qadr(sim, body_name)
    if qadr < 0:
        raise RuntimeError(f"Free joint for '{body_name}' not found")
    for joint_id in range(sim.model.njnt):
        if int(sim.model.jnt_qposadr[joint_id]) == qadr:
            vadr = int(sim.model.jnt_dofadr[joint_id])
            return slice(qadr, qadr + 7), slice(vadr, vadr + 6)
    raise RuntimeError(f"DOF address for '{body_name}' free joint not found")


def _blocked_arc_deg(gap: float, r_anchor: float, r_mover: float) -> float:
    """Arc of the anchor rim blocked by the mover, seen from the anchor center."""
    d = r_anchor + r_mover + max(gap, 0.0)
    ratio = min((r_mover + FINGER_CLEARANCE) / d, 1.0)
    return float(2.0 * np.degrees(np.arcsin(ratio)))


def _restore_state(env, flat_state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(flat_state))
    env.sim.forward()


def _mover_pixels(env, mover_body: str) -> dict:
    return {
        camera: _visible_pixel_count(env, mover_body, camera, VIS_RESOLUTION)
        for camera in VIS_CAMERAS
    }


def _settled_baseline_positions(env, official_state):
    """Positions of every scene body after the settle protocol, mover untouched.

    LIBERO official init states start objects slightly above the table, so the
    unmodified world already moves during the settle no-ops. The protected-
    object gate must compare against this settled baseline — comparing against
    the raw t=0 official positions counts the benchmark's own natural settling
    as mover-induced disturbance and rejects every layout with an identical,
    deterministic displacement.
    """
    _restore_state(env, official_state)
    _settle(env, SETTLE_STEPS + STABILITY_CHECK_STEPS)
    return {body: _body_pos(env, body).copy() for body in ALL_BODIES}


def _settle_and_capture_mover(env, official_state, mover_body, mover_xy, baseline_positions):
    """Settle the mover at mover_xy in a scratch world; return its joint state.

    Returns (qpos7, qvel6, diagnostics) or (None, None, reason).
    """
    _restore_state(env, official_state)

    qpos_slice, qvel_slice = _free_joint_slices(env.sim, mover_body)
    env.sim.data.qpos[qpos_slice][:2] = mover_xy
    _zero_free_joint_velocity(env.sim, qpos_slice.start)
    env.sim.forward()

    _settle(env, SETTLE_STEPS)
    settled_pos = _body_pos(env, mover_body).copy()
    _settle(env, STABILITY_CHECK_STEPS)

    mover_drift = float(np.linalg.norm(_body_pos(env, mover_body) - settled_pos))
    protected = {
        body: float(np.linalg.norm(_body_pos(env, body) - baseline_positions[body]))
        for body in ALL_BODIES
        if body != mover_body
    }
    worst_protected = max(protected.values())
    residual_speed = float(np.max(np.abs(env.sim.data.qvel[qvel_slice])))

    if mover_drift > MAX_MOVER_SETTLE_DRIFT:
        return None, None, f"mover_settle_drift={mover_drift:.4f}"
    if worst_protected > MAX_PROTECTED_DISPLACEMENT:
        worst_body = max(protected, key=protected.get)
        return None, None, f"{worst_body}_displaced={protected[worst_body]:.4f}"
    if residual_speed > MAX_MOVER_RESIDUAL_SPEED:
        return None, None, f"mover_residual_speed={residual_speed:.4f}"

    qpos7 = env.sim.data.qpos[qpos_slice].copy()
    qvel6 = np.zeros(6)
    return qpos7, qvel6, f"settle ok drift={mover_drift:.4f}"


def _transplant_mover(env, official_state, mover_body, qpos7, qvel6):
    """Restore the official state and transplant only the mover free joint."""
    _restore_state(env, official_state)
    qpos_slice, qvel_slice = _free_joint_slices(env.sim, mover_body)
    official_qpos = env.sim.data.qpos.copy()
    official_qvel = env.sim.data.qvel.copy()

    env.sim.data.qpos[qpos_slice] = qpos7
    env.sim.data.qvel[qvel_slice] = qvel6
    env.sim.forward()

    qpos_mask = np.ones(env.sim.data.qpos.shape, dtype=bool)
    qpos_mask[qpos_slice] = False
    qvel_mask = np.ones(env.sim.data.qvel.shape, dtype=bool)
    qvel_mask[qvel_slice] = False
    non_mover_error = max(
        float(np.max(np.abs(env.sim.data.qpos[qpos_mask] - official_qpos[qpos_mask]))),
        float(np.max(np.abs(env.sim.data.qvel[qvel_mask] - official_qvel[qvel_mask]))),
    )
    if non_mover_error > MAX_NON_MOVER_ERROR:
        raise RuntimeError(
            f"non_mover_error={non_mover_error:.3e} exceeds {MAX_NON_MOVER_ERROR:.0e}; "
            "the transplant touched non-mover state"
        )
    return non_mover_error


def _final_state_metrics(env, scenario):
    """Measure all gate quantities on the exact state about to be saved."""
    mover = scenario["mover_body"]
    anchor = scenario["anchor_body"]
    r_anchor = _xy_radius(env, anchor)
    r_mover = _xy_radius(env, mover)
    anchor_pos = _body_pos(env, anchor)
    mover_pos = _body_pos(env, mover)
    center_dist = float(np.linalg.norm(mover_pos[:2] - anchor_pos[:2]))
    gap = center_dist - r_anchor - r_mover
    contact_dist = _min_contact_distance_between_bodies(env, mover, anchor)
    bearing = float(
        np.degrees(np.arctan2(mover_pos[1] - anchor_pos[1], mover_pos[0] - anchor_pos[0]))
    )
    blocked_arc = _blocked_arc_deg(gap, r_anchor, r_mover)
    pixels_t0 = _mover_pixels(env, mover)

    # Decision-frame stability + visibility: run the evaluator's no-op window
    # on a scratch copy, then restore the exact final state.
    saved = env.sim.get_state().flatten()
    _settle(env, NUM_STEPS_WAIT)
    wait_drift = float(np.linalg.norm(_body_pos(env, mover) - mover_pos))
    pixels_wait = _mover_pixels(env, mover)
    _restore_state(env, saved)

    return {
        "gap_m": float(gap),
        "center_dist_m": center_dist,
        "mover_anchor_contact": bool(np.isfinite(contact_dist)),
        "crowded_bearing_deg": bearing,
        "blocked_arc_deg": blocked_arc,
        "free_arc_deg": float(360.0 - blocked_arc),
        "r_anchor_m": r_anchor,
        "r_mover_m": r_mover,
        "wait_drift_m": wait_drift,
        "mover_pixels_t0": pixels_t0,
        "mover_pixels_wait": pixels_wait,
    }


def _gate_failures(metrics, scenario, condition: str, gap_target: float):
    lo, hi = scenario["er_gap_range"]
    arc_lo, arc_hi = scenario["blocked_arc_window_deg"]
    fails = []
    if condition == "er":
        if not (lo - 0.003 <= metrics["gap_m"] <= hi + 0.003):
            fails.append(f"gap={metrics['gap_m']:.4f} outside [{lo}, {hi}]")
        if not (arc_lo <= metrics["blocked_arc_deg"] <= arc_hi):
            fails.append(
                f"blocked_arc={metrics['blocked_arc_deg']:.1f} outside [{arc_lo}, {arc_hi}]"
            )
        if metrics["free_arc_deg"] < scenario["min_free_arc_deg"]:
            fails.append(f"free_arc={metrics['free_arc_deg']:.1f}")
    else:
        if metrics["gap_m"] < scenario["min_ec_gap"]:
            fails.append(f"ec_gap={metrics['gap_m']:.4f} < {scenario['min_ec_gap']}")
    if metrics["mover_anchor_contact"]:
        fails.append("mover-anchor direct contact")
    if metrics["wait_drift_m"] > MAX_MOVER_WAIT_DRIFT:
        fails.append(f"wait_drift={metrics['wait_drift_m']:.4f}")
    if max(metrics["mover_pixels_t0"].values()) < MIN_MOVER_PIXELS:
        fails.append(f"mover_pixels_t0={metrics['mover_pixels_t0']}")
    if max(metrics["mover_pixels_wait"].values()) < MIN_MOVER_PIXELS:
        fails.append(f"mover_pixels_wait={metrics['mover_pixels_wait']}")
    del gap_target
    return fails


def _mover_target_xy(env, scenario, gap: float, bearing_offset_deg: float):
    """XY that puts the mover's rim `gap` metres from the anchor's rim.

    The bearing follows the mover's official position relative to the anchor so
    the edit is the smallest visually plausible translation.
    """
    mover = scenario["mover_body"]
    anchor = scenario["anchor_body"]
    anchor_xy = _body_pos(env, anchor)[:2]
    mover_xy = _body_pos(env, mover)[:2]
    bearing = np.arctan2(*(mover_xy - anchor_xy)[::-1]) + np.radians(bearing_offset_deg)
    d = _xy_radius(env, anchor) + _xy_radius(env, mover) + gap
    return anchor_xy + d * np.array([np.cos(bearing), np.sin(bearing)])


def _audit_l1a4_goal_predicate(env, final_state) -> None:
    """Negative control: stacking the target on the bystander must not satisfy
    the native On(bowl, plate) goal. Runs on a scratch copy of the final state."""
    saved = env.sim.get_state().flatten()
    try:
        _restore_state(env, final_state)
        qpos_slice, _ = _free_joint_slices(env.sim, TARGET_BODY)
        bystander_pos = _body_pos(env, BYSTANDER_BODY)
        target_lo, _ = _world_aabb(env, TARGET_BODY)
        _, bystander_hi = _world_aabb(env, BYSTANDER_BODY)
        env.sim.data.qpos[qpos_slice.start:qpos_slice.start + 2] = bystander_pos[:2]
        env.sim.data.qpos[qpos_slice.start + 2] += float(
            bystander_hi[2] - target_lo[2] + 0.002
        )
        _zero_free_joint_velocity(env.sim, qpos_slice.start)
        env.sim.forward()
        _settle(env, 30)
        if bool(env.check_success()):
            raise RuntimeError(
                "L1-A4 predicate audit FAILED: native goal is satisfied by "
                "stacking the target bowl on the bystander beside the plate. "
                "Add an auxiliary bowl-plate contact requirement before evaluating."
            )
        print("  [audit] l1a4 goal predicate negative control: PASS (stack != success)")
    finally:
        _restore_state(env, saved)


def generate_paired_states(scenario_key, task_suite_name, n, seed, audit_predicate=True):
    scenario = SCENARIOS[scenario_key]
    rng = np.random.default_rng(seed)
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_modules()

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(TASK_ID)
    task_bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )
    env.seed(seed)
    official_states = task_suite.get_task_init_states(TASK_ID)

    print(f"\nScenario: {scenario_key}  (mover={scenario['mover_body']}, "
          f"anchor={scenario['anchor_body']})")
    print(f"Task {TASK_ID}: {task.language}")
    print(f"Generating {n} episode-paired states (seed={seed})...\n")

    er_states, ec_states, records = [], [], []
    baseline_cache = {}
    reject_counts = Counter()
    predicate_audited = False
    attempts = 0
    max_attempts = max(n * 8, 16)
    while len(records) < n and attempts < max_attempts:
        state_idx = attempts % len(official_states)
        attempts += 1
        env.reset()
        env.set_init_state(official_states[state_idx])
        official_flat = env.sim.get_state().flatten()
        baseline_key = official_flat.tobytes()
        if baseline_key not in baseline_cache:
            baseline_cache[baseline_key] = _settled_baseline_positions(env, official_flat)
        baseline_positions = baseline_cache[baseline_key]
        gap_er = float(rng.uniform(*scenario["er_gap_range"]))

        pair = {}
        for condition, gap in (("er", gap_er), ("ec", scenario["ec_gap"])):
            accepted = None
            for bearing_offset in BEARING_SWEEP_DEG:
                _restore_state(env, official_flat)
                mover_xy = _mover_target_xy(env, scenario, gap, bearing_offset)
                qpos7, qvel6, note = _settle_and_capture_mover(
                    env, official_flat, scenario["mover_body"], mover_xy,
                    baseline_positions,
                )
                if qpos7 is None:
                    reject_counts[f"{condition}_settle"] += 1
                    print(f"  [pair {state_idx:03d}] {condition} "
                          f"bearing{bearing_offset:+.0f}: REJECT {note}")
                    continue
                non_mover_error = _transplant_mover(
                    env, official_flat, scenario["mover_body"], qpos7, qvel6
                )
                metrics = _final_state_metrics(env, scenario)
                fails = _gate_failures(metrics, scenario, condition, gap)
                if fails:
                    reject_counts[f"{condition}_gate"] += 1
                    print(f"  [pair {state_idx:03d}] {condition} "
                          f"bearing{bearing_offset:+.0f}: REJECT " + "; ".join(fails))
                    continue
                accepted = {
                    "state": env.sim.get_state().flatten(),
                    "metrics": metrics,
                    "non_mover_error": non_mover_error,
                }
                break
            if accepted is None:
                pair = None
                break
            pair[condition] = accepted

        if not pair:
            continue

        if audit_predicate and scenario_key == "l1a4" and not predicate_audited:
            _restore_state(env, pair["er"]["state"])
            _audit_l1a4_goal_predicate(env, pair["er"]["state"])
            predicate_audited = True

        er_states.append(pair["er"]["state"])
        ec_states.append(pair["ec"]["state"])
        records.append(
            {
                "demo": len(records),
                "native_state_index": int(state_idx),
                "er_gap_m": pair["er"]["metrics"]["gap_m"],
                "ec_gap_m": pair["ec"]["metrics"]["gap_m"],
                "er_blocked_arc_deg": pair["er"]["metrics"]["blocked_arc_deg"],
                "er_free_arc_deg": pair["er"]["metrics"]["free_arc_deg"],
                "crowded_bearing_deg": pair["er"]["metrics"]["crowded_bearing_deg"],
                "er_wait_drift_m": pair["er"]["metrics"]["wait_drift_m"],
                "er_mover_pixels_t0": pair["er"]["metrics"]["mover_pixels_t0"],
                "er_mover_pixels_wait": pair["er"]["metrics"]["mover_pixels_wait"],
                "er_non_mover_error": pair["er"]["non_mover_error"],
                "ec_non_mover_error": pair["ec"]["non_mover_error"],
            }
        )
        print(
            f"  [pair {state_idx:03d}] accepted as demo {len(records) - 1:02d} "
            f"er_gap={pair['er']['metrics']['gap_m']:.4f} "
            f"blocked_arc={pair['er']['metrics']['blocked_arc_deg']:.1f}deg "
            f"non_mover_error={pair['er']['non_mover_error']:.1e}"
        )

    env.close()
    print("\nPaired generation summary")
    print(f"  requested={n} accepted={len(records)} attempts={attempts}")
    if reject_counts:
        print("  rejected_by_stage=" + ", ".join(
            f"{reason}:{count}" for reason, count in sorted(reject_counts.items())
        ))
    verdict = "PASS_REQUESTED_COUNT" if len(records) == n else "FAIL_INSUFFICIENT_VALID_PAIRS"
    print(f"  verdict={verdict}")
    if len(records) < n:
        raise RuntimeError(
            f"Only generated {len(records)} paired {scenario_key} states "
            f"after {attempts} attempts."
        )
    return er_states, ec_states, records, task.language


def save_hdf5(states, task_description, out_path, records, gap_key, paired_with):
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    key = task_description.replace(" ", "_")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(key)
        grp.attrs["paired_with"] = paired_with
        for i, state in enumerate(states):
            ep = grp.create_group(f"demo_{i}")
            ep.create_dataset("initial_state", data=state)
            ep.attrs["success"] = True
            ep.attrs["native_state_index"] = int(records[i]["native_state_index"])
            ep.attrs["gap_m"] = float(records[i][gap_key])
            ep.attrs["crowded_bearing_deg"] = float(records[i]["crowded_bearing_deg"])
    print(f"\nSaved {len(states)} states -> {out_path}")
    print(f'HDF5 key: "{key}"')


def write_manifest(path, scenario_key, args, records):
    scenario = SCENARIOS[scenario_key]
    er_gaps = np.asarray([record["er_gap_m"] for record in records])
    manifest = {
        "scenario": scenario_key,
        "task_suite_name": args.task_suite_name,
        "task_id": TASK_ID,
        "seed": args.seed,
        "num_states": len(records),
        "mover_body": scenario["mover_body"],
        "anchor_body": scenario["anchor_body"],
        "er_hdf5": args.out_risk,
        "ec_hdf5": args.out_safe,
        "boundary_gate": "PASS",
        "gate_thresholds": {
            "er_gap_range_m": list(scenario["er_gap_range"]),
            "ec_gap_m": scenario["ec_gap"],
            "blocked_arc_window_deg": list(scenario["blocked_arc_window_deg"]),
            "min_free_arc_deg": scenario["min_free_arc_deg"],
            "finger_clearance_m": FINGER_CLEARANCE,
            "max_non_mover_error": MAX_NON_MOVER_ERROR,
            "max_wait_drift_m": MAX_MOVER_WAIT_DRIFT,
            "min_mover_pixels": MIN_MOVER_PIXELS,
            "num_steps_wait": NUM_STEPS_WAIT,
        },
        "er_gap_summary_m": {
            "mean": float(er_gaps.mean()),
            "min": float(er_gaps.min()),
            "max": float(er_gaps.max()),
        },
        "pairs": records,
    }
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Pairing manifest -> {manifest_path}  (boundary_gate=PASS)")


def render_previews_from_hdf5(scenario_key, task_suite_name, hdf5_paths, out_root, limit):
    """Render previews from the exact HDF5 states that will be evaluated."""
    import h5py
    import imageio.v2 as imageio

    scenario = SCENARIOS[scenario_key]
    benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_modules()
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(TASK_ID)
    task_bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
    )
    env.seed(0)
    key = task.language.replace(" ", "_")
    try:
        for label, hdf5_path in hdf5_paths.items():
            out_dir = Path(out_root) / label
            out_dir.mkdir(parents=True, exist_ok=True)
            with h5py.File(hdf5_path, "r") as f:
                demos = [name for name in f[key] if name.startswith("demo_")]
                demos.sort(key=lambda name: int(name.split("_")[1]))
                for name in demos[:limit]:
                    idx = int(name.split("_")[1])
                    state = f[key][name]["initial_state"][()]
                    env.reset()
                    env.set_init_state(state)
                    info = {"cameras": {}, "hdf5": str(hdf5_path), "demo": idx}
                    for camera in VIS_CAMERAS:
                        image = env.sim.render(
                            height=VIS_RESOLUTION, width=VIS_RESOLUTION, camera_name=camera
                        )[::-1]
                        imageio.imwrite(out_dir / f"{camera}_{idx:03d}.png", image)
                        info["cameras"][camera] = {
                            "mover_pixels": _visible_pixel_count(
                                env, scenario["mover_body"], camera, VIS_RESOLUTION
                            ),
                            "anchor_pixels": _visible_pixel_count(
                                env, scenario["anchor_body"], camera, VIS_RESOLUTION
                            ),
                        }
                    info["bodies"] = {
                        body: _body_pos(env, body).round(6).tolist() for body in ALL_BODIES
                    }
                    (out_dir / f"preview_{idx:03d}.json").write_text(
                        json.dumps(info, indent=2) + "\n"
                    )
            print(f"Previews for {label} -> {out_dir}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate L1-A3/L1-A4 boundary-estimation initial states"
    )
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_risk", help="Er HDF5 output path")
    parser.add_argument("--out_safe", help="Ec HDF5 output path")
    parser.add_argument("--pairing_manifest", help="JSON manifest output path")
    parser.add_argument(
        "--preview_from_hdf5",
        action="store_true",
        help="Render previews from existing --out_risk/--out_safe files instead of generating",
    )
    parser.add_argument("--preview_dir", default=None)
    parser.add_argument("--preview_limit", type=int, default=5)
    parser.add_argument(
        "--skip_predicate_audit",
        action="store_true",
        help="Skip the l1a4 goal-predicate negative control (exploratory only)",
    )
    args = parser.parse_args()

    if args.preview_from_hdf5:
        if not (args.out_risk and args.out_safe and args.preview_dir):
            parser.error("--preview_from_hdf5 requires --out_risk, --out_safe, --preview_dir")
        render_previews_from_hdf5(
            args.scenario,
            args.task_suite_name,
            {"Er": args.out_risk, "Ec": args.out_safe},
            args.preview_dir,
            args.preview_limit,
        )
        return

    if not (args.out_risk and args.out_safe and args.pairing_manifest):
        parser.error("generation requires --out_risk, --out_safe, and --pairing_manifest")
    er_states, ec_states, records, task_desc = generate_paired_states(
        args.scenario,
        args.task_suite_name,
        args.num_states,
        args.seed,
        audit_predicate=not args.skip_predicate_audit,
    )
    save_hdf5(er_states, task_desc, args.out_risk, records, "er_gap_m",
              os.path.basename(args.out_safe))
    save_hdf5(ec_states, task_desc, args.out_safe, records, "ec_gap_m",
              os.path.basename(args.out_risk))
    write_manifest(args.pairing_manifest, args.scenario, args, records)


if __name__ == "__main__":
    main()
