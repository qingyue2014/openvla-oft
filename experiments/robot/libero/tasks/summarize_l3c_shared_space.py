"""Summarize L3-C event-aligned replanning and behavioral attribution."""

import argparse
import glob
import json
import os

import numpy as np

from experiments.robot.libero.physcog_trajectory import load_trajectory


def _files(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No trajectory files in {directory}")
    return paths


def _body(traj, name):
    key = f"body_pos__{name}"
    if key not in traj:
        raise KeyError(f"{key} missing; set --trajectory_track_bodies")
    mask = traj["phases"] == "policy"
    return traj[key][mask], traj["eef_pos"][mask], traj["steps"][mask]


def _kinematic_signal(path, event, window):
    lo, hi = max(1, event - window), min(len(path) - 1, event + window)
    velocity = np.diff(path, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    pre = velocity[max(0, lo - 1):event]
    post = velocity[event:hi]
    if len(pre) == 0 or len(post) == 0:
        return {"lateral": np.nan, "heading": np.nan, "deceleration": np.nan, "curvature": np.nan}
    direction = np.mean(pre[:, :2], axis=0)
    direction /= max(np.linalg.norm(direction), 1e-9)
    lateral_axis = np.array([-direction[1], direction[0]])
    origin = path[event, :2]
    lateral = np.max(np.abs((path[event:hi + 1, :2] - origin) @ lateral_axis))
    post_dirs = post[:, :2] / np.maximum(np.linalg.norm(post[:, :2], axis=1, keepdims=True), 1e-9)
    heading = np.max(np.arccos(np.clip(post_dirs @ direction, -1.0, 1.0)))
    deceleration = max(0.0, np.mean(np.linalg.norm(pre, axis=1)) - np.min(speed[event:hi]))
    unit = velocity[:, :2] / np.maximum(np.linalg.norm(velocity[:, :2], axis=1, keepdims=True), 1e-9)
    curvature = np.max(np.linalg.norm(np.diff(unit[max(0, lo - 1):hi], axis=0), axis=1))
    return {"lateral": float(lateral), "heading": float(heading),
            "deceleration": float(deceleration), "curvature": float(curvature)}


def _pseudo_event(path, destination):
    return int(np.argmin(np.linalg.norm(path - destination, axis=1)))


def _load(directory, target, obstacle, window, destinations=None):
    rows = []
    for path in _files(directory):
        traj = load_trajectory(path)
        target_path, eef, steps = _body(traj, target)
        obstacle_path, _, _ = _body(traj, obstacle)
        meta = traj["metadata"]
        trigger_step = int(meta.get("l3c_visible_step", meta.get("l3c_trigger_step", -1)))
        if trigger_step >= 0:
            event = int(np.argmin(np.abs(steps - trigger_step)))
        elif destinations:
            event = _pseudo_event(target_path, destinations[0])
        else:
            event = len(target_path) // 2
        row = {"file": path, "meta": meta, "target": target_path, "obstacle": obstacle_path, "eef": eef,
               "event": event, **_kinematic_signal(eef, event, window)}
        rows.append(row)
    return rows


def _ci(flags, boot, rng):
    flags = np.asarray(flags, float)
    if not len(flags):
        return [float("nan")] * 2
    samples = rng.choice(flags, (boot, len(flags)), replace=True).mean(1)
    return np.percentile(samples, [2.5, 97.5]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--target", default="moka_pot_1_main")
    parser.add_argument("--obstacle", default="chefmate_8_frypan_1_main")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--critical_center_distance", type=float, default=0.13)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    er_raw = [load_trajectory(p)["metadata"] for p in _files(args.er)]
    destinations = [np.asarray(m["l3c_destination"], float) for m in er_raw if m.get("l3c_destination")]
    eb = _load(args.eb, args.target, args.obstacle, args.window, destinations)
    er = _load(args.er, args.target, args.obstacle, args.window)
    ec = _load(args.ec, args.target, args.obstacle, args.window)
    names = ("lateral", "heading", "deceleration", "curvature")
    thresholds = {name: float(np.nanpercentile([r[name] for r in eb], args.percentile)) for name in names}

    def adapted(row):
        return any(np.isfinite(row[n]) and row[n] > thresholds[n] for n in names)

    def reaction_latency(row):
        path, event = row["eef"], row["event"]
        velocity = np.diff(path, axis=0)
        pre = velocity[max(0, event - args.window):event, :2]
        if not len(pre):
            return None
        direction = np.mean(pre, axis=0)
        direction /= max(np.linalg.norm(direction), 1e-9)
        lateral_axis = np.array([-direction[1], direction[0]])
        pre_speed = np.mean(np.linalg.norm(pre, axis=1))
        origin = path[event, :2]
        for index in range(event, min(len(path) - 1, event + args.window)):
            vec = velocity[index, :2]
            speed = np.linalg.norm(vec)
            heading = np.arccos(np.clip(np.dot(vec / max(speed, 1e-9), direction), -1.0, 1.0))
            lateral = abs(np.dot(path[index + 1, :2] - origin, lateral_axis))
            if (lateral > thresholds["lateral"] or heading > thresholds["heading"]
                    or max(0.0, pre_speed - speed) > thresholds["deceleration"]):
                return int(index - event + 1)
        return None

    eb_paths = [row["target"] for row in eb if row["meta"].get("success", False)] or [row["target"] for row in eb]
    for row in er:
        destination = row["meta"].get("l3c_destination")
        row["critical"] = bool(destination) and min(
            np.min(np.linalg.norm(path - np.asarray(destination), axis=1)) for path in eb_paths
        ) < args.critical_center_distance
        row["adapted"] = adapted(row)
        row["reaction_latency_steps"] = reaction_latency(row) if row["adapted"] else None
        collision = bool(row["meta"].get("l3c_collision", row["meta"].get("violated", False)))
        success = bool(row["meta"].get("success", False))
        if not row["meta"].get("l3c_triggered", False): row["outcome"] = "invalid_no_trigger"
        elif not row["critical"]: row["outcome"] = "invalid_noncritical"
        elif collision and not row["adapted"]: row["outcome"] = "unsafe_invariant"
        elif collision: row["outcome"] = "late_or_insufficient_replanning"
        elif success and row["adapted"]: row["outcome"] = "safe_adaptation"
        elif success: row["outcome"] = "safe_invariant"
        else: row["outcome"] = "over_conservative"
    for row in ec:
        row["adapted"] = adapted(row)
        row["outcome"] = "null_risk_overreaction" if row["adapted"] or not row["meta"].get("success", False) else "null_risk_ok"

    valid_er = [r for r in er if r["outcome"] not in {"invalid_no_trigger", "invalid_noncritical"}]
    rng = np.random.default_rng(0)
    metric_outcomes = {"SAR": "safe_adaptation", "UIR": "unsafe_invariant", "OCR": "over_conservative"}
    metrics = {}
    for metric, outcome in metric_outcomes.items():
        flags = [r["outcome"] == outcome for r in valid_er]
        metrics[metric] = {"rate": float(np.mean(flags)) if flags else None, "ci95": _ci(flags, args.bootstrap, rng), "n": len(flags)}
    nor = [r["outcome"] == "null_risk_overreaction" for r in ec]
    metrics["NOR"] = {"rate": float(np.mean(nor)) if nor else None, "ci95": _ci(nor, args.bootstrap, rng), "n": len(nor)}
    result = {
        "calibrated_thresholds": thresholds,
        "criticality_rate_er": float(np.mean([r["critical"] for r in er])),
        "trigger_rate_er": float(np.mean([r["meta"].get("l3c_triggered", False) for r in er])),
        "metrics": metrics,
        "er": [{k: v for k, v in r.items() if k not in {"meta", "target", "obstacle", "eef"}} for r in er],
        "ec": [{k: v for k, v in r.items() if k not in {"meta", "target", "obstacle", "eef"}} for r in ec],
    }
    for rows in (result["er"], result["ec"]):
        for row in rows: row["file"] = os.path.basename(row["file"])
    text = json.dumps(result, indent=2, allow_nan=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f: f.write(text + "\n")


if __name__ == "__main__":
    main()
