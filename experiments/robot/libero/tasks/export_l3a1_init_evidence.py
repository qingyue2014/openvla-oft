#!/usr/bin/env python3
"""Restore exact L3-A1 Eb/Er/Ec states and export policy-view evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASK_ID = 3
TASK_DESCRIPTION = "put the black bowl in the bottom drawer of the cabinet and close it"
DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L3A1_bowl_drawer_bottle.bddl"
DEFAULT_ER = "experiments/robot/libero/tasks/l3a1_drawer_bottle_risk_initial_states.hdf5"
DEFAULT_EC = "experiments/robot/libero/tasks/l3a1_drawer_bottle_stable_initial_states.hdf5"
DEFAULT_OUT = "experiments/logs/l3a1_init_evidence"
TRACKED_BODIES = (
    "gripper0_eef",
    "white_cabinet_1_cabinet_bottom",
    "wine_bottle_1_main",
    "akita_black_bowl_1_main",
)


def policy_agentview(obs: dict[str, Any]) -> np.ndarray:
    """Apply the exact orientation used by get_libero_image()."""
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 RGB agentview_image, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descendants(model: Any, root: int) -> set[int]:
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for candidate in range(int(model.nbody)):
            if candidate not in bodies and int(model.body_parentid[candidate]) in bodies:
                bodies.add(candidate)
                changed = True
    return bodies


def contact_report(env: Any, tracked: tuple[str, ...] = TRACKED_BODIES) -> list[dict[str, Any]]:
    model, data = env.sim.model, env.sim.data
    labels_by_body: dict[int, list[str]] = {}
    for label in tracked:
        try:
            root = int(model.body_name2id(label))
        except Exception:
            continue
        for body_id in _descendants(model, root):
            labels_by_body.setdefault(body_id, []).append(label)
    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        labels = sorted(set(labels_by_body.get(body1, []) + labels_by_body.get(body2, [])))
        if labels:
            rows.append({
                "contact_index": index,
                "distance_m": float(contact.dist),
                "geom1": model.geom_id2name(geom1) or f"geom_{geom1}",
                "geom2": model.geom_id2name(geom2) or f"geom_{geom2}",
                "body1": model.body_id2name(body1) or f"body_{body1}",
                "body2": model.body_id2name(body2) or f"body_{body2}",
                "tracked_bodies": labels,
            })
    return rows


def pose_report(env: Any) -> dict[str, Any]:
    model, data = env.sim.model, env.sim.data
    poses = {}
    for name in TRACKED_BODIES:
        try:
            body_id = int(model.body_name2id(name))
        except Exception:
            poses[name] = {"present": False}
            continue
        poses[name] = {
            "present": True,
            "xyz_m": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
            "quat_wxyz": np.asarray(data.body_xquat[body_id], dtype=float).tolist(),
        }
    return poses


def _delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"same_shape": True, "max_abs": float(delta.max(initial=0)),
            "nonzero_scalars": int(np.count_nonzero(delta))}


def _task_group(handle: h5py.File) -> h5py.Group:
    key = TASK_DESCRIPTION.replace(" ", "_")
    if key in handle:
        return handle[key]
    groups = [value for value in handle.values() if isinstance(value, h5py.Group)]
    if len(groups) == 1:
        return groups[0]
    raise KeyError(f"task group {key!r} not found in {handle.filename}")


def load_hdf_states(path: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    with h5py.File(path, "r") as handle:
        group = _task_group(handle)
        demos = sorted((key for key in group if key.startswith("demo_")),
                       key=lambda key: int(key.split("_")[-1]))
        if len(demos) < count:
            raise ValueError(f"{path} has {len(demos)} states; {count} required")
        for demo_name in demos[:count]:
            demo = group[demo_name]
            rows.append({
                "state": np.asarray(demo["initial_state"][:]),
                "source": f"{group.name}/{demo_name}/initial_state",
                "reset_attempt": _scalar(demo.attrs.get("reset_attempt")),
                "source_demo_index": _scalar(demo.attrs.get("source_demo_index")),
            })
    return rows


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def capture(env: Any, state: np.ndarray, condition: str, episode: int, out_dir: Path) -> dict[str, Any]:
    import imageio.v2 as imageio

    env.reset()
    env.set_init_state(state)
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    # Explicitly refresh the observation after the exact restore. Neither this
    # nor set_init_state advances MuJoCo with sim.step(). ControlEnv exposes
    # the refreshed observation through regenerate_obs_from_state rather than
    # forwarding robosuite's private _get_observations method.
    obs = env.regenerate_obs_from_state(restored)
    refreshed = np.asarray(env.sim.get_state().flatten()).copy()
    restore_delta = _delta(state, restored)
    refresh_delta = _delta(restored, refreshed)
    if not restore_delta.get("same_shape") or restore_delta.get("max_abs") != 0.0:
        raise RuntimeError(f"{condition}[{episode}] serialized state was not restored exactly: {restore_delta}")
    if not refresh_delta.get("same_shape") or refresh_delta.get("max_abs") != 0.0:
        raise RuntimeError(f"{condition}[{episode}] observable refresh changed simulator state: {refresh_delta}")
    image = policy_agentview(obs)
    image_path = out_dir / f"{condition.lower()}_ep{episode:03d}_policy_agentview.png"
    imageio.imwrite(image_path, image)
    return {
        "condition": condition,
        "episode": episode,
        "serialized_state_sha256": array_sha256(state),
        "restored_state_sha256": array_sha256(restored),
        "serialized_to_restored_delta": restore_delta,
        "restored_to_refreshed_delta": refresh_delta,
        "policy_image": str(image_path),
        "policy_image_sha256": file_sha256(image_path),
        "policy_image_shape": list(image.shape),
        "poses": pose_report(env),
        "contacts": contact_report(env),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# L3-A1 exact-state policy-view evidence", "",
        "- Verdict: **PASS_L3A1_INIT_EVIDENCE_GENERATED**",
        "- Manual visibility verdict: **PENDING_REVIEW**",
        f"- Episodes per condition: {report['episodes_per_condition']}",
        "- Policy transform: `agentview_image[::-1, ::-1]`", "- Steps after restore: **0**", "",
        "| Condition | Episode | State SHA256 | Restore max delta | Refresh max delta | Contacts | Image |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["captures"]:
        restore = row["serialized_to_restored_delta"].get("max_abs", "shape mismatch")
        refresh = row["restored_to_refreshed_delta"].get("max_abs", "shape mismatch")
        lines.append(f"| {row['condition']} | {row['episode']} | `{row['serialized_state_sha256']}` | "
                     f"{restore} | {refresh} | {len(row['contacts'])} | `{row['policy_image']}` |")
    lines += ["", "Pose/contact details and artifact hashes: `init_evidence.json`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--er", default=DEFAULT_ER)
    parser.add_argument("--ec", default=DEFAULT_EC)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--num_states", type=int, default=int(os.environ.get("PREVIEW_STATES", "5")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_gpu", type=int, default=int(os.environ.get("RENDER_GPU", "1")))
    args = parser.parse_args()
    if args.num_states <= 0:
        parser.error("--num_states must be positive")

    er_path, ec_path, bddl_path = Path(args.er), Path(args.ec), Path(args.bddl)
    for path in (er_path, ec_path, bddl_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(TASK_ID)
    native_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    native = suite.get_task_init_states(TASK_ID)
    if len(native) < args.num_states:
        raise ValueError(f"native suite has {len(native)} states; {args.num_states} required")
    state_rows = {
        "Eb": [{"state": np.asarray(state), "source": f"libero_10/task_{TASK_ID}/init_state_{i}"}
               for i, state in enumerate(native[:args.num_states])],
        "Er": load_hdf_states(er_path, args.num_states),
        "Ec": load_hdf_states(ec_path, args.num_states),
    }
    native_env = OffScreenRenderEnv(bddl_file_name=str(native_bddl), camera_heights=256,
                                    camera_widths=256, render_gpu_device_id=args.render_gpu)
    custom_env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256,
                                    camera_widths=256, render_gpu_device_id=args.render_gpu)
    envs = {"Eb": native_env, "Er": custom_env, "Ec": custom_env}
    native_env.seed(args.seed)
    custom_env.seed(args.seed)
    captures = []
    try:
        for condition in ("Eb", "Er", "Ec"):
            for episode, source in enumerate(state_rows[condition]):
                row = capture(envs[condition], source["state"], condition, episode, out_dir)
                row.update({key: value for key, value in source.items() if key != "state"})
                captures.append(row)
    finally:
        native_env.close()
        custom_env.close()

    report = {
        "verdict": "PASS_L3A1_INIT_EVIDENCE_GENERATED",
        "manual_visibility_verdict": "PENDING_REVIEW",
        "episodes_per_condition": args.num_states,
        "task_id": TASK_ID,
        "task_description": TASK_DESCRIPTION,
        "native_bddl": str(native_bddl),
        "custom_bddl": str(bddl_path),
        "seed": args.seed,
        "policy_image_transform": "agentview_image[::-1, ::-1]",
        "simulation_steps_after_restore": 0,
        "artifacts": {
            "Er": {"path": str(er_path), "sha256": file_sha256(er_path)},
            "Ec": {"path": str(ec_path), "sha256": file_sha256(ec_path)},
        },
        "captures": captures,
    }
    json_path = out_dir / "init_evidence.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(report, out_dir / "init_evidence.md")
    print(f"verdict=PASS_L3A1_INIT_EVIDENCE_GENERATED captures={len(captures)} report={json_path}")


if __name__ == "__main__":
    main()
