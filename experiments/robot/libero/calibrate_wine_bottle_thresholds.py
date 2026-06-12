"""
Calibrate ContactForceOracle thresholds from LIBERO human demonstration data.

Replays all wine bottle demos in libero_spatial (or libero_goal) and records
per-episode peak approach speed and peak impact force via ContactForceOracle.
Outputs mean ± 1 std as the "glass-object safe operation limit".

Usage:
    python experiments/robot/libero/calibrate_wine_bottle_thresholds.py \
        --demo_dir /path/to/libero_spatial \
        [--task_suite libero_spatial] \
        [--wine_bottle_body wine_bottle_1_main] \
        [--approach_radius 0.15] \
        [--impact_window 30] \
        [--settle_steps 5] \
        [--out calibration_thresholds.json]

The demo_dir should contain HDF5 files named <task_name>_demo.hdf5, e.g.:
    put_the_wine_bottle_on_the_rack_demo.hdf5
    put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5

These are the original LIBERO demo files downloadable from the LIBERO project page.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# LIBERO import bootstrap (mirrors run_physcog_libero_l1_eval.py)
# ---------------------------------------------------------------------------

def _ensure_libero_importable():
    try:
        import libero  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (repo_root / "_deps" / "LIBERO", repo_root.parent / "LIBERO", repo_root.parent / "libero"):
        if (candidate / "libero").is_dir():
            sys.path.insert(0, str(candidate))
            print(f"[info] Added LIBERO path: {candidate}")
            return


_ensure_libero_importable()

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from libero.libero import benchmark, get_libero_path
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.physcog_oracles import ContactForceOracle


# ---------------------------------------------------------------------------
# Wine bottle task detection
# ---------------------------------------------------------------------------

_WINE_BOTTLE_KEYWORDS = ("wine_bottle",)

# LIBERO object → MuJoCo body name convention: <object_name>_main
_DEFAULT_WINE_BODY = "wine_bottle_1_main"

def _is_wine_task(task_name: str) -> bool:
    name = task_name.lower()
    return any(k in name for k in _WINE_BOTTLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Demo replay with ContactForceOracle
# ---------------------------------------------------------------------------

def replay_demo(env, oracle: ContactForceOracle, demo_data, settle_steps: int = 5):
    """Replay one demo episode and return per-episode peak metrics.

    Returns (peak_approach_speed, peak_impact_force) or None if replay fails.
    """
    orig_actions = demo_data["actions"][()]
    orig_states  = demo_data["states"][()]

    env.reset()
    env.set_init_state(orig_states[0])

    # Settle the physics; oracle reset after settling so initial state is clean
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(get_libero_dummy_action("llava"))

    oracle.reset(env, obs)

    for step, action in enumerate(orig_actions):
        obs, _, done, _ = env.step(action)
        oracle.check(env, obs, action, step)
        if done:
            break

    return oracle.peak_approach_speed, oracle.peak_impact_force


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo_dir",          required=True,
                   help="Directory containing <task>_demo.hdf5 files from LIBERO")
    p.add_argument("--task_suite",        default="libero_spatial",
                   choices=["libero_spatial", "libero_goal", "libero_90"],
                   help="LIBERO benchmark suite to iterate over")
    p.add_argument("--wine_bottle_body",  default=_DEFAULT_WINE_BODY,
                   help="MuJoCo body name for the wine bottle")
    p.add_argument("--approach_radius",   type=float, default=0.15,
                   help="Gripper distance (m) within which approach speed is sampled")
    p.add_argument("--impact_window",     type=int,   default=30,
                   help="Number of steps after landing to track impact force")
    p.add_argument("--settle_steps",      type=int,   default=5,
                   help="Physics settle steps before replaying each demo")
    p.add_argument("--out",               default="calibration_thresholds.json",
                   help="Output JSON file for computed thresholds")
    return p.parse_args()


def main():
    args = parse_args()
    demo_dir = Path(args.demo_dir)
    if not demo_dir.is_dir():
        sys.exit(f"[error] demo_dir not found: {demo_dir}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks   = task_suite.n_tasks

    approach_speeds: list[float] = []
    impact_forces:   list[float] = []
    episodes_per_task: dict[str, int] = {}

    print(f"\nScanning {num_tasks} tasks in {args.task_suite} for wine bottle demos …\n")

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name

        if not _is_wine_task(task_name):
            continue

        demo_path = demo_dir / f"{task_name}_demo.hdf5"
        if not demo_path.exists():
            print(f"  [skip] {task_name}: demo file not found at {demo_path}")
            continue

        print(f"  [task] {task_name}")
        print(f"         bottle_body={args.wine_bottle_body}")

        env, _ = get_libero_env(task, "llava", resolution=128)

        oracle = ContactForceOracle(
            target_body=args.wine_bottle_body,
            violation_metric="approach_speed",   # metric doesn't matter; thresholds=999 = no violations
            max_approach_speed=999.0,
            max_grasp_force=999.0,
            max_impact_force=999.0,
            approach_radius=args.approach_radius,
            impact_window=args.impact_window,
        )

        with h5py.File(demo_path, "r") as f:
            demo_group = f["data"]
            n_demos = len(demo_group.keys())
            print(f"         replaying {n_demos} demos …")
            ep_count = 0
            for i in range(n_demos):
                demo_key = f"demo_{i}"
                if demo_key not in demo_group:
                    continue
                try:
                    spd, imp = replay_demo(
                        env, oracle, demo_group[demo_key],
                        settle_steps=args.settle_steps,
                    )
                    approach_speeds.append(spd)
                    impact_forces.append(imp)
                    ep_count += 1
                except Exception as exc:
                    print(f"         [warn] demo_{i} failed: {exc}")

            episodes_per_task[task_name] = ep_count
            print(f"         collected {ep_count} episodes  "
                  f"| mean_speed={np.mean(approach_speeds[-ep_count:]):.4f}  "
                  f"| mean_impact={np.mean(impact_forces[-ep_count:]):.4f}")

        env.close()

    if not approach_speeds:
        sys.exit("\n[error] No demo episodes collected. "
                 "Check --demo_dir and that the wine bottle tasks exist.")

    approach_speeds = np.array(approach_speeds, dtype=np.float64)
    impact_forces   = np.array(impact_forces,   dtype=np.float64)

    # ---- Compute thresholds (mean + 1 std, i.e. ~84th percentile) ----------
    speed_mean,  speed_std  = float(np.mean(approach_speeds)), float(np.std(approach_speeds))
    impact_mean, impact_std = float(np.mean(impact_forces)),   float(np.std(impact_forces))

    speed_threshold  = speed_mean  + speed_std
    impact_threshold = impact_mean + impact_std

    # Additional percentiles for reference
    p50_speed,  p75_speed,  p90_speed  = (float(np.percentile(approach_speeds, p)) for p in (50, 75, 90))
    p50_impact, p75_impact, p90_impact = (float(np.percentile(impact_forces,   p)) for p in (50, 75, 90))

    result = {
        "source":          "libero_human_demo_wine_bottle",
        "task_suite":      args.task_suite,
        "wine_bottle_body": args.wine_bottle_body,
        "n_episodes":      int(len(approach_speeds)),
        "episodes_per_task": episodes_per_task,
        "approach_speed": {
            "mean":      speed_mean,
            "std":       speed_std,
            "threshold_mean_plus_1std": speed_threshold,
            "p50": p50_speed, "p75": p75_speed, "p90": p90_speed,
            "unit": "m/s",
        },
        "impact_force": {
            "mean":      impact_mean,
            "std":       impact_std,
            "threshold_mean_plus_1std": impact_threshold,
            "p50": p50_impact, "p75": p75_impact, "p90": p90_impact,
            "unit": "N",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Total episodes:  {result['n_episodes']}")
    print(f"\n  Approach speed  (m/s)")
    print(f"    mean={speed_mean:.4f}  std={speed_std:.4f}")
    print(f"    threshold (mean+1σ) = {speed_threshold:.4f} m/s")
    print(f"    p50={p50_speed:.4f}  p75={p75_speed:.4f}  p90={p90_speed:.4f}")
    print(f"\n  Impact force  (N)")
    print(f"    mean={impact_mean:.4f}  std={impact_std:.4f}")
    print(f"    threshold (mean+1σ) = {impact_threshold:.4f} N")
    print(f"    p50={p50_impact:.4f}  p75={p75_impact:.4f}  p90={p90_impact:.4f}")
    print(f"\n  Saved → {out_path}")
    print("=" * 60)
    print()
    print("Next step: pass thresholds to ContactForceOracle via run_physcog_libero_l1_eval.py:")
    print(f"  --contact_max_approach_speed {speed_threshold:.4f}")
    print(f"  --contact_max_impact_force   {impact_threshold:.4f}")


if __name__ == "__main__":
    main()
