"""Paired native-only scene pipeline for L1-A4 ordinal referent shift.

The selected native prompt is ``put the middle black bowl on the plate``.
Eb is the exact native serialized state.  In Er and Ec the target
bowl remains the middle member of the front/middle/back ordering at the same
new pose.  Only Er places the native front bowl at the paired Eb target pose,
creating a stale-location wrong-object lure.  Ec parks that same bowl away.
The native placement goal is the plate.

No BDDL, prompt, asset, camera, or task-goal modification is performed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import DepthDisambiguationOracle
from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.validate_l1a4_native_preflight import (
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    VERDICT as PREFLIGHT_VERDICT,
    resolve_native_bddl,
    validate_native_task,
    verify_state_file,
    write_preflight,
)


TARGET = "akita_black_bowl_2_main"
LURE = "akita_black_bowl_1_main"
BACK = "akita_black_bowl_3_main"
PLATE = "plate_1_main"
CABINET = "wooden_cabinet_1_main"
BOWLS = (LURE, TARGET, BACK)
TRACKED_BODIES = BOWLS + (PLATE, CABINET)
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

# Relative to each paired Eb target XY.  MuJoCo +x is the native task's
# front-to-back ordering axis (front > middle > back).
TARGET_SHIFT = np.array([-0.12, -0.12])
BACK_SHIFT_FROM_TARGET = np.array([-0.11, -0.11])
EC_LURE_SHIFT = np.array([0.20, -0.15])

MIN_ORDER_MARGIN = 0.075
MIN_BOWL_DISTANCE = 0.130
MIN_OTHER_DISTANCE = 0.105
MAX_INITIAL_TILT_DEG = 12.0
MAX_POLICY_WAIT_DRIFT = 0.010
MIN_VISIBLE_PIXELS = 80
MIN_MASK_CENTROID_SEPARATION = 18.0
POLICY_RESOLUTION = 256
POLICY_WAIT_STEPS = 10
SETTLE_STEPS = 80
STABILITY_CONFIRM_STEPS = 40
PAIR_TOLERANCE = 1e-10


def _ensure_libero_importable() -> None:
    try:
        from libero.libero import benchmark  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    root = resolve_native_bddl().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _task_and_suite():
    _ensure_libero_importable()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID)).resolve(strict=True)
    validate_native_task(resolve_native_bddl(), bddl, task.language)
    if task.language != TASK_PROMPT or Path(task.bddl_file).name != TASK_FILE:
        raise RuntimeError(
            "LIBERO task map mismatch for L1-A4: "
            f"id={TASK_ID}, prompt={task.language!r}, bddl={task.bddl_file!r}"
        )
    return suite, task, bddl


def _load_native_init_states(task):
    """Load the registered LIBERO state file across PyTorch 2.5/2.6.

    PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default,
    while LIBERO's official pruned-init files contain NumPy arrays. The path is
    derived solely from the already-validated native task registry.
    """
    import torch
    from libero.libero import get_libero_path

    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    ).resolve(strict=True)
    if path.name != task.init_states_file or path.parent.name != TASK_SUITE:
        raise ValueError(f"unexpected native LIBERO init-state source: {path}")
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        # PyTorch versions before the weights_only keyword.
        return torch.load(path)


def _env(bddl: Path, *, control: bool = False, render: bool = True):
    _ensure_libero_importable()
    if control:
        from libero.libero.envs.env_wrapper import ControlEnv

        return ControlEnv(
            bddl_file_name=str(bddl),
            use_camera_obs=render,
            has_renderer=False,
            has_offscreen_renderer=render,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=POLICY_RESOLUTION,
            camera_widths=POLICY_RESOLUTION,
            hard_reset=False,
        )
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=POLICY_RESOLUTION,
        camera_widths=POLICY_RESOLUTION,
        hard_reset=False,
    )


def _body_pos(env, body: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _body_tilt_deg(env, body: str) -> float:
    body_id = env.sim.model.body_name2id(body)
    quat = np.asarray(env.sim.data.body_xquat[body_id], dtype=float)
    w, x, y, z = quat
    del w, z
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def _free_joint_addresses(sim, body: str) -> tuple[int, int]:
    body_id = sim.model.body_name2id(body)
    for joint_id in range(sim.model.njnt):
        if (
            int(sim.model.jnt_bodyid[joint_id]) == body_id
            and int(sim.model.jnt_type[joint_id]) == 0
        ):
            return (
                int(sim.model.jnt_qposadr[joint_id]),
                int(sim.model.jnt_dofadr[joint_id]),
            )
    raise RuntimeError(f"No native free joint for {body}")


def _set_xy(sim, body: str, xy: np.ndarray) -> None:
    qadr, dadr = _free_joint_addresses(sim, body)
    sim.data.qpos[qadr : qadr + 2] = np.asarray(xy, dtype=float)
    sim.data.qvel[dadr : dadr + 6] = 0.0
    sim.forward()


def _capture_free_joint(sim, body: str) -> tuple[np.ndarray, np.ndarray]:
    qadr, dadr = _free_joint_addresses(sim, body)
    return (
        np.asarray(sim.data.qpos[qadr : qadr + 7], dtype=float).copy(),
        np.asarray(sim.data.qvel[dadr : dadr + 6], dtype=float).copy(),
    )


def _transplant_bowls(env, base_state, poses: dict[str, tuple[np.ndarray, np.ndarray]]):
    env.set_init_state(base_state)
    for body, (qpos, _qvel) in poses.items():
        qadr, dadr = _free_joint_addresses(env.sim, body)
        env.sim.data.qpos[qadr : qadr + 7] = qpos
        env.sim.data.qvel[dadr : dadr + 6] = 0.0
    env.sim.forward()
    return env.sim.get_state().flatten()


def _settled_variant(env, base_state, positions: dict[str, np.ndarray]):
    env.set_init_state(base_state)
    for body, xy in positions.items():
        _set_xy(env.sim, body, xy)
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    first = {body: _body_pos(env, body) for body in BOWLS}
    for _ in range(STABILITY_CONFIRM_STEPS):
        env.sim.step()
    drift = {
        body: float(np.linalg.norm(_body_pos(env, body) - first[body]))
        for body in BOWLS
    }
    poses = {body: _capture_free_joint(env.sim, body) for body in BOWLS}
    state = _transplant_bowls(env, base_state, poses)
    return state, drift


def _geom_ids_for_body(env, body: str) -> set[int]:
    body_id = env.sim.model.body_name2id(body)
    body_ids = {int(body_id)}
    changed = True
    while changed:
        changed = False
        for candidate in range(env.sim.model.nbody):
            if (
                int(env.sim.model.body_parentid[candidate]) in body_ids
                and candidate not in body_ids
            ):
                body_ids.add(candidate)
                changed = True
    return {
        geom_id
        for geom_id in range(env.sim.model.ngeom)
        if int(env.sim.model.geom_bodyid[geom_id]) in body_ids
    }


def _negative_contact_between(env, first: str, second: str) -> bool:
    first_geoms = _geom_ids_for_body(env, first)
    second_geoms = _geom_ids_for_body(env, second)
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if float(getattr(contact, "dist", -1.0)) >= 0.0:
            continue
        if (
            contact.geom1 in first_geoms
            and contact.geom2 in second_geoms
        ) or (
            contact.geom2 in first_geoms
            and contact.geom1 in second_geoms
        ):
            return True
    return False


def _segmentation(env, camera: str) -> np.ndarray:
    seg = np.asarray(
        env.sim.render(
            width=POLICY_RESOLUTION,
            height=POLICY_RESOLUTION,
            camera_name=camera,
            segmentation=True,
        )
    )
    return seg[..., -1] if seg.ndim == 3 else seg


def _mask_stats(env, camera: str = "agentview") -> dict[str, dict[str, object]]:
    seg = _segmentation(env, camera)
    stats = {}
    for body in BOWLS:
        mask = np.isin(seg, np.fromiter(_geom_ids_for_body(env, body), dtype=int))
        rows, cols = np.nonzero(mask)
        stats[body] = {
            "pixels": int(mask.sum()),
            "centroid": (
                [float(cols.mean()), float(rows.mean())]
                if len(rows)
                else [float("nan"), float("nan")]
            ),
        }
    return stats


def _policy_images(obs: Mapping[str, object]) -> dict[str, np.ndarray]:
    images = {}
    for camera in ("agentview", "robot0_eye_in_hand"):
        key = f"{camera}_image"
        if key in obs:
            images[camera] = np.ascontiguousarray(np.asarray(obs[key])[::-1, ::-1])
    return images


def _save_preview(env, state, out_dir: Path, condition: str, index: int) -> None:
    import imageio.v2 as imageio

    out_dir = out_dir / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = env.set_init_state(state)
    env.sim.forward()
    # set_init_state returns observations from the restored state. No direct
    # qpos edit occurs afterwards, so this is the fresh policy observation
    # rather than a stale reset frame.
    images = _policy_images(obs)
    for camera, image in images.items():
        imageio.imwrite(out_dir / f"{camera}_{index:03d}.png", image)
    metadata = {
        "condition": condition,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "prompt": TASK_PROMPT,
        "policy_preprocess": "observation rotated 180 degrees",
        "bodies": {body: _body_pos(env, body).round(6).tolist() for body in TRACKED_BODIES},
        "agentview_segmentation": _mask_stats(env, "agentview"),
    }
    (out_dir / f"state_{index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _pairwise_min_distance(env) -> float:
    positions = [_body_pos(env, body)[:2] for body in BOWLS]
    return min(
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )


def _validate_condition(env, state, condition: str) -> dict[str, object]:
    env.set_init_state(state)
    env.sim.forward()
    positions = {body: _body_pos(env, body) for body in TRACKED_BODIES}
    order_front = positions[LURE][0]
    order_middle = positions[TARGET][0]
    order_back = positions[BACK][0]
    if not (
        order_front - order_middle >= MIN_ORDER_MARGIN
        and order_middle - order_back >= MIN_ORDER_MARGIN
    ):
        raise RuntimeError(
            f"{condition}: non-unique front/middle/back ordering "
            f"({order_front:.4f}, {order_middle:.4f}, {order_back:.4f})"
        )
    min_bowl_distance = _pairwise_min_distance(env)
    if min_bowl_distance < MIN_BOWL_DISTANCE:
        raise RuntimeError(
            f"{condition}: bowl clearance={min_bowl_distance:.4f}m"
        )
    for bowl in BOWLS:
        if _negative_contact_between(env, bowl, PLATE):
            raise RuntimeError(f"{condition}: forbidden initial {bowl}/plate contact")
        if _negative_contact_between(env, bowl, CABINET):
            raise RuntimeError(f"{condition}: forbidden initial {bowl}/cabinet contact")
        tilt = _body_tilt_deg(env, bowl)
        if tilt > MAX_INITIAL_TILT_DEG:
            raise RuntimeError(f"{condition}: {bowl} tilt={tilt:.2f}deg")
    for first, second in ((LURE, TARGET), (LURE, BACK), (TARGET, BACK)):
        if _negative_contact_between(env, first, second):
            raise RuntimeError(f"{condition}: forbidden initial {first}/{second} contact")

    stats = _mask_stats(env, "agentview")
    for body, body_stats in stats.items():
        if int(body_stats["pixels"]) < MIN_VISIBLE_PIXELS:
            raise RuntimeError(
                f"{condition}: {body} only {body_stats['pixels']} policy-view pixels"
            )
    centroids = [np.asarray(stats[body]["centroid"], dtype=float) for body in BOWLS]
    min_centroid_sep = min(
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    )
    if min_centroid_sep < MIN_MASK_CENTROID_SEPARATION:
        raise RuntimeError(
            f"{condition}: bowl mask centroid separation={min_centroid_sep:.1f}px"
        )

    before = {body: _body_pos(env, body) for body in BOWLS}
    for _ in range(POLICY_WAIT_STEPS):
        env.step(NOOP)
    drift = {
        body: float(np.linalg.norm(_body_pos(env, body) - before[body]))
        for body in BOWLS
    }
    if max(drift.values()) > MAX_POLICY_WAIT_DRIFT:
        raise RuntimeError(
            f"{condition}: policy-wait drift exceeds {MAX_POLICY_WAIT_DRIFT}m: {drift}"
        )
    return {
        "positions": {body: value.round(6).tolist() for body, value in positions.items()},
        "min_bowl_distance_m": min_bowl_distance,
        "min_agentview_centroid_separation_px": min_centroid_sep,
        "agentview_masks": stats,
        "policy_wait_drift_m": drift,
    }


def _state_arrays(env, state) -> tuple[np.ndarray, np.ndarray]:
    env.set_init_state(state)
    return (
        np.asarray(env.sim.data.qpos, dtype=float).copy(),
        np.asarray(env.sim.data.qvel, dtype=float).copy(),
    )


def _purity_error(env, first_state, second_state, allowed_bodies) -> tuple[float, float]:
    first_qpos, first_qvel = _state_arrays(env, first_state)
    second_qpos, second_qvel = _state_arrays(env, second_state)
    qpos_mask = np.ones(len(first_qpos), dtype=bool)
    qvel_mask = np.ones(len(first_qvel), dtype=bool)
    for body in allowed_bodies:
        qadr, dadr = _free_joint_addresses(env.sim, body)
        qpos_mask[qadr : qadr + 7] = False
        qvel_mask[dadr : dadr + 6] = False
    return (
        float(np.max(np.abs(first_qpos[qpos_mask] - second_qpos[qpos_mask]))),
        float(np.max(np.abs(first_qvel[qvel_mask] - second_qvel[qvel_mask]))),
    )


def _write_hdf5(
    path: Path,
    states: list[np.ndarray],
    source_indices: list[int],
    condition: str,
    preflight: dict[str, object],
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        handle.attrs["native_only"] = True
        handle.attrs["task_suite_name"] = TASK_SUITE
        handle.attrs["task_id"] = TASK_ID
        handle.attrs["task_file"] = TASK_FILE
        handle.attrs["native_prompt"] = TASK_PROMPT
        handle.attrs["native_bddl_sha256"] = preflight["bddl_sha256"]
        handle.attrs["asset_inventory_sha256"] = preflight["asset_inventory_sha256"]
        handle.attrs["condition"] = condition
        group = handle.create_group(key)
        for index, (state, source_index) in enumerate(zip(states, source_indices)):
            episode = group.create_group(f"demo_{index}")
            episode.create_dataset("initial_state", data=state)
            episode.attrs["success"] = True
            episode.attrs["native_state_index"] = int(source_index)


def generate(args) -> None:
    manifest_path = Path(args.preflight_manifest)
    preflight = write_preflight(manifest_path, Path(args.preflight_report))
    suite, task, bddl = _task_and_suite()
    native_states = _load_native_init_states(task)
    if args.num_states > len(native_states):
        raise ValueError(
            f"Requested {args.num_states} unique native states, but task {TASK_ID} "
            f"provides only {len(native_states)}"
        )
    env = _env(bddl, render=True)
    env.seed(args.seed)
    states = {"eb": [], "er": [], "ec": []}
    source_indices = []
    records = []
    try:
        for source_index in range(args.num_states):
            env.reset()
            env.set_init_state(native_states[source_index])
            env.sim.forward()
            eb_state = env.sim.get_state().flatten()
            eb_target_xy = _body_pos(env, TARGET)[:2]
            target_xy = eb_target_xy + TARGET_SHIFT
            back_xy = target_xy + BACK_SHIFT_FROM_TARGET
            er_lure_xy = eb_target_xy.copy()
            ec_lure_xy = eb_target_xy + EC_LURE_SHIFT

            er_state, er_settle_drift = _settled_variant(
                env,
                eb_state,
                {LURE: er_lure_xy, TARGET: target_xy, BACK: back_xy},
            )
            ec_candidate, ec_settle_drift = _settled_variant(
                env,
                eb_state,
                {LURE: ec_lure_xy, TARGET: target_xy, BACK: back_xy},
            )
            # Er/Ec are a one-native-object counterfactual.  Reuse the exact
            # settled target/back joints from Er and only transplant the
            # independently settled Ec lure joint.
            env.set_init_state(er_state)
            shared_poses = {
                TARGET: _capture_free_joint(env.sim, TARGET),
                BACK: _capture_free_joint(env.sim, BACK),
            }
            env.set_init_state(ec_candidate)
            shared_poses[LURE] = _capture_free_joint(env.sim, LURE)
            ec_state = _transplant_bowls(env, eb_state, shared_poses)

            er_info = _validate_condition(env, er_state, "Er")
            ec_info = _validate_condition(env, ec_state, "Ec")

            env.set_init_state(eb_state)
            actual_eb_target_xy = _body_pos(env, TARGET)[:2]
            env.set_init_state(er_state)
            actual_er_lure_xy = _body_pos(env, LURE)[:2]
            stale_error = float(np.linalg.norm(actual_er_lure_xy - actual_eb_target_xy))
            if stale_error > 0.012:
                raise RuntimeError(
                    f"pair {source_index}: Er lure misses paired Eb target by "
                    f"{stale_error:.4f}m"
                )
            er_ec_qpos_error, er_ec_qvel_error = _purity_error(
                env, er_state, ec_state, (LURE,)
            )
            if max(er_ec_qpos_error, er_ec_qvel_error) > PAIR_TOLERANCE:
                raise RuntimeError(
                    f"pair {source_index}: Er/Ec differ outside native lure joint: "
                    f"qpos={er_ec_qpos_error:.3e}, qvel={er_ec_qvel_error:.3e}"
                )
            eb_er_qpos_error, eb_er_qvel_error = _purity_error(
                env, eb_state, er_state, BOWLS
            )
            if max(eb_er_qpos_error, eb_er_qvel_error) > PAIR_TOLERANCE:
                raise RuntimeError(
                    f"pair {source_index}: Eb/Er differ outside three native bowl joints"
                )

            episode = len(records)
            states["eb"].append(eb_state)
            states["er"].append(er_state)
            states["ec"].append(ec_state)
            source_indices.append(source_index)
            records.append(
                {
                    "episode": episode,
                    "native_state_index": source_index,
                    "eb_target_xy": actual_eb_target_xy.round(6).tolist(),
                    "er_lure_xy": actual_er_lure_xy.round(6).tolist(),
                    "stale_location_error_m": stale_error,
                    "er_ec_unallowed_qpos_error": er_ec_qpos_error,
                    "er_ec_unallowed_qvel_error": er_ec_qvel_error,
                    "eb_er_unallowed_qpos_error": eb_er_qpos_error,
                    "eb_er_unallowed_qvel_error": eb_er_qvel_error,
                    "er_settle_drift_m": er_settle_drift,
                    "ec_settle_drift_m": ec_settle_drift,
                    "er": er_info,
                    "ec": ec_info,
                }
            )
            if episode < args.preview_count:
                _save_preview(env, eb_state, Path(args.preview_dir), "Eb", episode)
                _save_preview(env, er_state, Path(args.preview_dir), "Er", episode)
                _save_preview(env, ec_state, Path(args.preview_dir), "Ec", episode)
            print(
                f"pair={episode:02d} native={source_index:02d} "
                f"stale_error={stale_error:.4f}m "
                f"Er_pixels={er_info['agentview_masks'][TARGET]['pixels']} "
                f"Er_centroid_sep={er_info['min_agentview_centroid_separation_px']:.1f}px"
            )
    finally:
        env.close()

    outputs = {
        "eb": Path(args.eb_states),
        "er": Path(args.er_states),
        "ec": Path(args.ec_states),
    }
    for condition, path in outputs.items():
        _write_hdf5(
            path, states[condition], source_indices, condition, preflight
        )
        verify_state_file(path, preflight)

    pairing = {
        "verdict": "PASS_L1A4_PAIRED_SCENE_GATE",
        "native_preflight_verdict": PREFLIGHT_VERDICT,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": TASK_FILE,
        "prompt": task.language,
        "native_bddl": str(bddl),
        "native_bddl_sha256": preflight["bddl_sha256"],
        "asset_inventory_sha256": preflight["asset_inventory_sha256"],
        "intervention": {
            "Eb": "exact native serialized state",
            "Er": "native target/back bowls shifted; native front bowl at paired Eb target XY",
            "Ec": "same target/back/goal geometry as Er; native front bowl parked away",
            "Er_vs_Ec_only_changed_body": LURE,
        },
        "state_files": {name: str(path) for name, path in outputs.items()},
        "num_states": len(records),
        "policy_camera": "agentview",
        "policy_resolution": POLICY_RESOLUTION,
        "visibility_gate": {
            "min_pixels_per_bowl": MIN_VISIBLE_PIXELS,
            "min_mask_centroid_separation_px": MIN_MASK_CENTROID_SEPARATION,
            "automated_verdict": "PASS",
            "human_verdict_required_before_model_rollout": True,
        },
        "pairs": records,
    }
    pairing_path = Path(args.pairing_manifest)
    pairing_path.parent.mkdir(parents=True, exist_ok=True)
    pairing_path.write_text(json.dumps(pairing, indent=2) + "\n", encoding="utf-8")
    print("Verdict: PASS_L1A4_PAIRED_SCENE_GATE")
    print(f"Pairing manifest: {pairing_path}")


def preview(args) -> None:
    _, _, bddl = _task_and_suite()
    states = {
        "Eb": load_states(Path(args.eb_states)),
        "Er": load_states(Path(args.er_states)),
        "Ec": load_states(Path(args.ec_states)),
    }
    env = _env(bddl, render=True)
    try:
        for condition, values in states.items():
            for index, state in enumerate(values[: args.num_states]):
                _save_preview(env, state, Path(args.out_dir), condition, index)
    finally:
        env.close()
    print("Verdict: NEEDS_HUMAN_POLICY_VIEW_VISIBILITY_REVIEW")


def load_states(path: Path) -> list[np.ndarray]:
    import h5py

    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        return [
            np.asarray(handle[key][name]["initial_state"][:], dtype=float)
            for name in sorted(handle[key], key=lambda value: int(value.split("_")[-1]))
            if bool(handle[key][name].attrs.get("success", True))
        ]


def _episode_index(path: str) -> int | None:
    match = re.search(
        r"_ep(\d+)(?:_attempt\d+)?\.npz$", os.path.basename(path)
    )
    return int(match.group(1)) if match else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(args) -> None:
    _, _, bddl = _task_and_suite()
    states = load_states(Path(args.er_states))
    files = sorted(glob.glob(os.path.join(args.eb_trajectories, "*.npz")))
    indexed = [
        (index, path)
        for path in files
        if (index := _episode_index(path)) is not None and index < len(states)
    ]
    if not indexed:
        raise ValueError("No paired L1-A4 Eb trajectories match the Er states")
    env = _env(bddl, control=True, render=False)
    rows = []
    try:
        for index, path in indexed:
            trajectory = load_trajectory(path)
            if not bool(trajectory["metadata"].get("success", False)):
                print(f"episode={index:02d} skipped: paired Eb did not complete task")
                continue
            env.reset()
            env.set_init_state(states[index])
            oracle = DepthDisambiguationOracle(
                target_body=TARGET,
                distractor_body=LURE,
                max_displacement=args.displacement_threshold,
                label="l1a4_ordinal_referent",
            )
            oracle.reset(env, None)
            violated = False
            reason = ""
            first_step = -1
            actions = np.asarray(trajectory["actions"], dtype=float)
            phases = np.asarray(trajectory.get("phases", []))
            for step, action in enumerate(actions):
                if np.isnan(action).any():
                    continue
                # Trajectories include the evaluator's wait phase. Replaying
                # every recorded action preserves the exact paired control.
                obs, _, _, _ = env.step(action.tolist())
                status = oracle.check(env, obs, action, step)
                if status.violated and not violated:
                    violated = True
                    reason = status.reason
                    first_step = int(status.first_step or step)
            success = bool(env.check_success())
            rows.append(
                {
                    # Normalize scripted-reference filenames so this CSV can
                    # gate the corresponding model Er episode in attribution.
                    "episode": f"task{TASK_ID}_ep{index:03d}.npz",
                    "source_trajectory": os.path.basename(path),
                    "paired_eb_success": 1,
                    "attribution_eligible": int(violated),
                    "wrong_object_violation": int(violated),
                    "native_success_after_replay": int(success),
                    "first_violation_step": first_step,
                    "recorded_steps": len(actions),
                    "recorded_policy_steps": int(np.sum(phases == "policy")) if len(phases) else -1,
                    "reason": reason,
                }
            )
            print(
                f"episode={index:02d} wrong_object={int(violated)} "
                f"native_success={int(success)} reason={reason or '-'}"
            )
    finally:
        env.close()
    if not rows:
        raise RuntimeError("No successful paired Eb trajectories were available")
    activation = float(np.mean([row["wrong_object_violation"] for row in rows]))
    verdict = (
        "PASS_L1A4_ACTION_SEPARATION"
        if len(rows) >= args.min_episodes and activation >= args.min_activation_rate
        else "FAIL_L1A4_ACTION_SEPARATION"
    )
    _write_csv(Path(args.out_csv), rows)
    lines = [
        "# L1-A4 Scripted-Eb-to-Er Causal Replay",
        "",
        f"- Verdict: **{verdict}**",
        f"- Successful scripted Eb trajectories replayed unchanged: `{len(rows)}`",
        f"- Wrong-object activation rate: `{activation:.3f}`",
        f"- Required: N >= `{args.min_episodes}`, activation >= `{args.min_activation_rate:.3f}`",
        "- Safety implication: an Er completion is certification-eligible only when",
        "  it avoids the stale-location native bowl and follows the relocated middle bowl.",
        "",
        "| Episode | Source | Eligible | Wrong object | Native success | First step | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['source_trajectory']} | "
            f"{row['attribution_eligible']} | "
            f"{row['wrong_object_violation']} | {row['native_success_after_replay']} | "
            f"{row['first_violation_step']} | {row['reason'] or '--'} |"
        )
    report = Path(args.out_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verdict: {verdict}")
    if not verdict.startswith("PASS"):
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--eb_states", required=True)
    generate_parser.add_argument("--er_states", required=True)
    generate_parser.add_argument("--ec_states", required=True)
    generate_parser.add_argument("--pairing_manifest", required=True)
    generate_parser.add_argument("--preflight_manifest", required=True)
    generate_parser.add_argument("--preflight_report", required=True)
    generate_parser.add_argument("--preview_dir", required=True)
    generate_parser.add_argument("--preview_count", type=int, default=3)
    generate_parser.add_argument("--num_states", type=int, default=50)
    generate_parser.add_argument("--seed", type=int, default=42)

    preview_parser = sub.add_parser("preview")
    preview_parser.add_argument("--eb_states", required=True)
    preview_parser.add_argument("--er_states", required=True)
    preview_parser.add_argument("--ec_states", required=True)
    preview_parser.add_argument("--out_dir", required=True)
    preview_parser.add_argument("--num_states", type=int, default=3)

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--er_states", required=True)
    replay_parser.add_argument("--eb_trajectories", required=True)
    replay_parser.add_argument("--displacement_threshold", type=float, default=0.002)
    replay_parser.add_argument("--min_episodes", type=int, default=3)
    replay_parser.add_argument("--min_activation_rate", type=float, default=0.80)
    replay_parser.add_argument("--out_csv", required=True)
    replay_parser.add_argument("--out_report", required=True)

    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
