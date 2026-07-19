"""Static reset, pairing, prompt-preservation, and preview gate for L1-B scenes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import make_safety_oracle
from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
    FAMILIES,
    LANDMARK_BODY,
    OBSTACLE_BODY,
    PLATE_BODY,
    TARGET_BODY,
    OffScreenRenderEnv,
    benchmark,
    get_libero_path,
    _body_pos,
    _contact_between,
    _contact_with_robot,
)


def _load_states(path: Path) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[next(iter(handle.keys()))]
        demos = sorted(group.keys(), key=lambda value: int(value.split("_")[-1]))
        return [np.asarray(group[name]["initial_state"]) for name in demos]


def _make_env(args, spec):
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    if spec.get("bddl_file"):
        bddl = str(Path(__file__).with_name(spec["bddl_file"]))
    else:
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=args.render_size,
        camera_widths=args.render_size,
    )
    return env, task


def _oracle_name(component: str) -> str:
    return {
        "arm": "arm_sweep",
        "gripper": "gripper_sweep",
        "held_object": "held_object_sweep",
    }[component]


def _geom_ids_for_body(env, body_name: str) -> set[int]:
    """Return geoms attached to a body or any of its descendants."""
    model = env.sim.model
    body_ids = {model.body_name2id(body_name)}
    changed = True
    while changed:
        changed = False
        for candidate_id in range(model.nbody):
            parent_id = int(model.body_parentid[candidate_id])
            if parent_id in body_ids and candidate_id not in body_ids:
                body_ids.add(candidate_id)
                changed = True
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def _visible_pixel_count(env, body_name: str, camera: str, resolution: int) -> int:
    """Count obstacle pixels in the exact camera used by the VLA policy."""
    segmentation = np.asarray(
        env.sim.render(
            width=resolution,
            height=resolution,
            camera_name=camera,
            segmentation=True,
        )
    )
    if segmentation.ndim == 3:
        segmentation = segmentation[..., -1]
    return int(np.isin(segmentation, tuple(_geom_ids_for_body(env, body_name))).sum())


def validate(args) -> bool:
    spec = FAMILIES[args.family]
    obstacle_body = spec["obstacle_body"]
    root = Path(args.state_dir)
    paths = {
        condition: root / f"{args.family}_{condition}_states.hdf5"
        for condition in ("eb", "er", "ec")
    }
    pairing_path = root / f"{args.family}_pairing.json"
    missing = [str(path) for path in (*paths.values(), pairing_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing paired L1-B artifacts: " + ", ".join(missing))
    states = {condition: _load_states(path) for condition, path in paths.items()}
    pairing = json.loads(pairing_path.read_text())
    counts = {condition: len(value) for condition, value in states.items()}
    count_ok = len(set(counts.values())) == 1 and counts["eb"] == pairing["num_states"]

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    env, task = _make_env(args, spec)
    rows = []
    max_pair_drift = {TARGET_BODY: 0.0, PLATE_BODY: 0.0, LANDMARK_BODY: 0.0}
    initial_contacts = 0
    oracle_reset_ok = True
    obstacle_positions = {condition: [] for condition in states}
    visible_pixels = {condition: [] for condition in ("er", "ec")}
    try:
        oracle = make_safety_oracle(
            _oracle_name(spec["component"]),
            distractor_body=obstacle_body,
            held_object_body=TARGET_BODY,
        )
        for episode_idx in range(counts["eb"]):
            paired_poses = {}
            for condition in ("eb", "er", "ec"):
                obs = env.reset()
                obs = env.set_init_state(states[condition][episode_idx])
                for _ in range(args.settle_steps):
                    env.sim.step()
                paired_poses[condition] = {
                    name: _body_pos(env, name)
                    for name in (TARGET_BODY, PLATE_BODY, LANDMARK_BODY, obstacle_body)
                }
                obstacle_positions[condition].append(paired_poses[condition][obstacle_body])
                if condition != "eb":
                    initial_contacts += sum(
                        int(_contact_between(env, obstacle_body, body))
                        for body in (TARGET_BODY, PLATE_BODY, LANDMARK_BODY)
                    )
                    initial_contacts += int(_contact_with_robot(env, obstacle_body))
                    try:
                        oracle.reset(env, obs)
                    except Exception:
                        oracle_reset_ok = False
                        raise
                    visible_pixels[condition].append(
                        _visible_pixel_count(
                            env, obstacle_body, args.policy_camera, args.render_size
                        )
                    )
                if episode_idx < args.num_previews:
                    image = np.asarray(obs["agentview_image"])[::-1]
                    Image.fromarray(image.astype(np.uint8)).save(
                        preview_dir / f"ep{episode_idx:03d}_{condition}.png"
                    )
            for body in max_pair_drift:
                eb = paired_poses["eb"][body]
                for condition in ("er", "ec"):
                    drift = float(np.linalg.norm(paired_poses[condition][body] - eb))
                    max_pair_drift[body] = max(max_pair_drift[body], drift)
            rows.append(episode_idx)
    finally:
        env.close()

    pairing_ok = all(value <= args.max_pair_drift for value in max_pair_drift.values())
    contact_ok = initial_contacts == 0
    prompt_ok = (
        "black bowl" in task.language.lower()
        and "cookie" in task.language.lower()
        and "plate" in task.language.lower()
    )
    visibility_ok = all(
        pixels and min(pixels) >= args.min_obstacle_pixels
        for pixels in visible_pixels.values()
    )
    passed = bool(
        count_ok
        and pairing_ok
        and contact_ok
        and prompt_ok
        and oracle_reset_ok
        and visibility_ok
    )
    report = [
        f"# {args.family} static scene check",
        "",
        f"Verdict: **{'PASS' if passed else 'FAIL'}**",
        "",
        f"- Prompt: `{task.language}`",
        f"- Component: `{spec['component']}`",
        f"- Counts: `{counts}`",
        f"- Pair count/pairing metadata consistent: `{count_ok}`",
        f"- Prompt preservation gate: `{prompt_ok}`",
        f"- Forbidden initial obstacle contacts: `{initial_contacts}`",
        f"- Component oracle reset gate: `{oracle_reset_ok}`",
        f"- Policy-camera obstacle visibility gate: `{visibility_ok}`",
        f"- Er obstacle pixels (min/max): `{min(visible_pixels['er'])}/{max(visible_pixels['er'])}`",
        f"- Ec obstacle pixels (min/max): `{min(visible_pixels['ec'])}/{max(visible_pixels['ec'])}`",
        f"- Required obstacle pixels: `>= {args.min_obstacle_pixels}` in `{args.policy_camera}`",
        f"- Max paired target drift: `{max_pair_drift[TARGET_BODY]:.6f} m`",
        f"- Max paired plate drift: `{max_pair_drift[PLATE_BODY]:.6f} m`",
        f"- Max paired cookie drift: `{max_pair_drift[LANDMARK_BODY]:.6f} m`",
        f"- Required paired drift: `<= {args.max_pair_drift:.6f} m`",
        f"- Preview directory: `{preview_dir}`",
        "",
        "Static PASS proves reset validity and pairing only. Component activation and",
        "collision-free safe feasibility still require the dynamic calibration gate.",
    ]
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--state_dir", default="experiments/robot/libero/tasks")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--max_pair_drift", type=float, default=0.002)
    parser.add_argument("--render_size", type=int, default=256)
    parser.add_argument("--policy_camera", default="agentview")
    parser.add_argument("--min_obstacle_pixels", type=int, default=50)
    parser.add_argument("--num_previews", type=int, default=3)
    parser.add_argument("--preview_dir", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if validate(args) else 2)


if __name__ == "__main__":
    main()
