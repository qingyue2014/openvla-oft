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
    _forbidden_initial_contact_pairs,
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
    segmentation = _center_policy_crop(segmentation)
    return int(np.isin(segmentation, tuple(_geom_ids_for_body(env, body_name))).sum())


def _center_policy_crop(image: np.ndarray, crop_area: float = 0.9) -> np.ndarray:
    """Match OpenVLA's configured 0.9-area center crop before model input."""
    height, width = image.shape[:2]
    scale = float(np.sqrt(crop_area))
    crop_height = max(1, int(round(height * scale)))
    crop_width = max(1, int(round(width * scale)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[top:top + crop_height, left:left + crop_width]


def _fresh_observation(env) -> dict:
    for owner in (env, getattr(env, "env", None)):
        if owner is None:
            continue
        for method_name in ("get_observation", "_get_observations", "_get_observation"):
            method = getattr(owner, method_name, None)
            if method is not None:
                return method()
    return {}


def _policy_camera_image(env, camera: str, resolution: int) -> np.ndarray:
    """Reproduce policy orientation/crop after final settle, never stale obs."""
    obs = _fresh_observation(env)
    key = f"{camera}_image"
    image = obs.get(key)
    if image is None:
        image = env.sim.render(
            width=resolution,
            height=resolution,
            camera_name=camera,
        )[::-1]
    # libero_utils.get_libero_image rotates the observation by 180 degrees.
    image = np.asarray(image)[::-1, ::-1]
    cropped = _center_policy_crop(image)
    return np.asarray(
        Image.fromarray(cropped.astype(np.uint8)).resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
    )


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
    source_indices = [
        pair.get("source_state_index") for pair in pairing.get("pairs", [])
    ]
    unique_native_sources_ok = bool(
        not spec.get("preserve_native_layout")
        or (
            len(source_indices) == counts["eb"]
            and len(set(source_indices)) == len(source_indices)
            and pairing.get("unique_source_state_indices") == len(source_indices)
        )
    )

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    env, task = _make_env(args, spec)
    model_body_names = [
        env.sim.model.body_id2name(body_id) or ""
        for body_id in range(env.sim.model.nbody)
    ]
    invariant_bodies = [
        body
        for body in (TARGET_BODY, PLATE_BODY, LANDMARK_BODY)
        if body != obstacle_body and body in model_body_names
    ]
    max_pair_drift = {body: 0.0 for body in invariant_bodies}
    initial_contacts = 0
    initial_contact_pairs = []
    oracle_reset_ok = True
    obstacle_positions = {condition: [] for condition in states}
    visible_pixels = {condition: [] for condition in ("eb", "er", "ec")}
    prompt_relation_distances = {condition: [] for condition in states}
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
                contact_first_seen = {}
                for pair in _forbidden_initial_contact_pairs(env, obstacle_body):
                    contact_first_seen.setdefault(pair, "restore")
                for settle_step in range(args.settle_steps):
                    env.sim.step()
                    for pair in _forbidden_initial_contact_pairs(env, obstacle_body):
                        contact_first_seen.setdefault(
                            pair, f"settle_step_{settle_step + 1}"
                        )
                obs = _fresh_observation(env)
                tracked_bodies = set(invariant_bodies) | {obstacle_body}
                relation_body = spec.get("prompt_relation_body")
                if relation_body:
                    tracked_bodies.add(relation_body)
                paired_poses[condition] = {
                    name: _body_pos(env, name)
                    for name in tracked_bodies
                }
                obstacle_positions[condition].append(paired_poses[condition][obstacle_body])
                if relation_body:
                    prompt_relation_distances[condition].append(
                        float(
                            np.linalg.norm(
                                paired_poses[condition][obstacle_body][:2]
                                - paired_poses[condition][relation_body][:2]
                            )
                        )
                    )
                for pair, first_seen in contact_first_seen.items():
                    initial_contacts += 1
                    initial_contact_pairs.append(
                        f"ep{episode_idx:03d}/{condition}: "
                        f"{pair} ({first_seen})"
                    )
                visible_pixels[condition].append(
                    _visible_pixel_count(
                        env, obstacle_body, args.policy_camera, args.render_size
                    )
                )
                if condition != "eb":
                    try:
                        oracle.reset(env, obs)
                    except Exception:
                        oracle_reset_ok = False
                        raise
                if episode_idx < args.num_previews:
                    image = _policy_camera_image(
                        env, args.policy_camera, args.render_size
                    )
                    Image.fromarray(image.astype(np.uint8)).save(
                        preview_dir / f"ep{episode_idx:03d}_{condition}.png"
                    )
            for body in max_pair_drift:
                eb = paired_poses["eb"][body]
                for condition in ("er", "ec"):
                    drift = float(np.linalg.norm(paired_poses[condition][body] - eb))
                    max_pair_drift[body] = max(max_pair_drift[body], drift)
    finally:
        env.close()

    pairing_ok = all(value <= args.max_pair_drift for value in max_pair_drift.values())
    only_obstacle_pose_ok = all(
        pair.get("only_obstacle_pose_changed", False)
        for pair in pairing.get("pairs", [])
    )
    contact_ok = initial_contacts == 0
    required_prompt_terms = spec.get(
        "required_prompt_terms", ("black bowl", "cookie", "plate")
    )
    prompt_ok = all(
        term.lower() in task.language.lower() for term in required_prompt_terms
    )
    visibility_ok = all(
        pixels and min(pixels) >= args.min_obstacle_pixels
        for pixels in visible_pixels.values()
    )
    native_asset_gate = bool(
        not spec.get("native_assets_only")
        or (
            spec.get("bddl_file") is None
            and not any(
                body_name.startswith("l1_b_")
                for body_name in model_body_names
            )
        )
    )
    relation_limit = spec.get("prompt_relation_max_distance")
    prompt_relation_ok = bool(
        relation_limit is None
        or all(
            distances and max(distances) <= relation_limit
            for distances in prompt_relation_distances.values()
        )
    )
    passed = bool(
        count_ok
        and unique_native_sources_ok
        and pairing_ok
        and contact_ok
        and prompt_ok
        and oracle_reset_ok
        and visibility_ok
        and native_asset_gate
        and only_obstacle_pose_ok
        and prompt_relation_ok
    )
    report = [
        f"# {args.family} static scene check",
        "",
        f"Verdict: **{'PASS' if passed else 'FAIL'}**",
        "",
        f"- Prompt: `{task.language}`",
        f"- Required prompt terms: `{list(required_prompt_terms)}`",
        f"- Component: `{spec['component']}`",
        f"- Counts: `{counts}`",
        f"- Pair count/pairing metadata consistent: `{count_ok}`",
        f"- Unique native source reset gate: `{unique_native_sources_ok}`",
        f"- Prompt preservation gate: `{prompt_ok}`",
        f"- Native task asset-set gate: `{native_asset_gate}`",
        f"- Only protected obstacle pose changed: `{only_obstacle_pose_ok}`",
        f"- Prompt landmark relation gate: `{prompt_relation_ok}`",
        *(
            f"- {condition} prompt-relation distance (min/max): "
            f"`{min(distances):.4f}/{max(distances):.4f} m`"
            for condition, distances in prompt_relation_distances.items()
            if distances
        ),
        f"- Forbidden initial obstacle contacts/interpenetrations: `{initial_contacts}`",
        *(
            f"  - `{pair}`"
            for pair in initial_contact_pairs
        ),
        f"- Component oracle reset gate: `{oracle_reset_ok}`",
        f"- Policy-camera obstacle visibility gate: `{visibility_ok}`",
        *(
            f"- {condition.upper()} obstacle pixels (min/max): "
            f"`{min(pixels)}/{max(pixels)}`"
            for condition, pixels in visible_pixels.items()
        ),
        f"- Required obstacle pixels: `>= {args.min_obstacle_pixels}` in `{args.policy_camera}`",
        *(
            f"- Max paired {body} drift: `{drift:.6f} m`"
            for body, drift in max_pair_drift.items()
        ),
        f"- Required paired drift: `<= {args.max_pair_drift:.6f} m`",
        f"- Preview directory: `{preview_dir}`",
        "",
        "This static gate covers reset validity and pairing only. Component activation",
        "and collision-free safe feasibility still require the dynamic calibration gate.",
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
