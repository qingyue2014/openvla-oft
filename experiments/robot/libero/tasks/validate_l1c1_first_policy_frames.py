#!/usr/bin/env python3
"""Validate frozen L1-C1 states at the formal evaluator's first policy frame.

Simulator execution is Superpod-only. Every Eb/Er/Ec episode is retained; a
single invalid state fails the frozen bundle rather than being filtered out.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[4]
SCENARIO = "L1-C1"
TASK_SUITE = "libero_spatial"
TASK_ID = 2
TASK_PROMPT = "pick up the black bowl from table center and place it on the plate"
FORMAL_WAIT_STEPS = 10
CONFIRM_STEPS = 5
MODEL_IMAGE_SIZE = 224
CENTER_CROP_AREA = 0.9

TRACKED_BODIES = (
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
)
RECEPTACLES = {
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
}
VISIBILITY_BODIES = (
    "akita_black_bowl_1_main",
    "akita_black_bowl_2_main",
    "plate_1_main",
)
EXPECTED_SUPPORT = {
    "eb": {body: "table" for body in TRACKED_BODIES},
    "er": {
        **{body: "table" for body in TRACKED_BODIES},
        "akita_black_bowl_2_main": "plate_1_main",
    },
    "ec": {body: "table" for body in TRACKED_BODIES},
}
EXPECTED_STATE_SHA256 = {
    "eb": "aa69d062b5f64807d58e67ef4497582b37f170f37015f6e3782b83343b1acb07",
    "er": "66aa396f565c9d80187e756b34a1370f52d1b3e43c87d751dab3e13ce65a326e",
    "ec": "423a229287c8b11bd6461d8f2536ed96d9d092d3647d191bb810af8843b1cd15",
}
STATE_FILENAMES = {
    "eb": "l1c1_task2_bowl_stack_eb_states.hdf5",
    "er": "l1c1_task2_bowl_stack_candidate_states.hdf5",
    "ec": "l1c1_task2_bowl_stack_ec_states.hdf5",
}

MAX_TRANSLATION_DRIFT_M = 0.005
MAX_RECEPTACLE_TILT_DEG = 1.0
MAX_NON_RECEPTACLE_TILT_DEG = 2.0
MAX_WINDOW_LINEAR_SPEED_MPS = 0.025
MAX_WINDOW_ANGULAR_SPEED_RADPS = 0.25
MAX_FIRST_POLICY_LINEAR_SPEED_MPS = 0.01
MAX_FIRST_POLICY_ANGULAR_SPEED_RADPS = 0.05
MIN_VISIBLE_PIXELS = 50
STATE_DIFF_ATOL = 1e-9


def _assert_superpod(allow_local_simulator: bool) -> dict[str, object]:
    hostname = socket.gethostname()
    scheduler = bool(os.environ.get("SLURM_JOB_ID"))
    declared = os.environ.get("PHYSCG_SUPERPOD") == "1"
    host_marker = any(
        token in hostname.lower()
        for token in ("superpod", "dgx", "slogin", "compute", "gpu")
    )
    verified = scheduler or declared or host_marker
    if not verified and not allow_local_simulator:
        raise RuntimeError(
            "L1-C1 first-policy-frame validation is Superpod-only; submit the "
            "registered Superpod phase instead of initializing MuJoCo locally"
        )
    return {
        "hostname": hostname,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "physcog_superpod_declared": declared,
        "superpod_verified": verified,
        "local_override": bool(allow_local_simulator and not verified),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_states(path: Path) -> tuple[list[np.ndarray], dict[str, str]]:
    import h5py

    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        if len(keys) != 1:
            raise ValueError(f"expected one prompt group in {path}, found {keys}")
        group = handle[keys[0]]
        expected_key = TASK_PROMPT.replace(" ", "_")
        if keys[0] != expected_key:
            raise ValueError(f"prompt group mismatch in {path}: {keys[0]!r}")
        demos = sorted(group.keys(), key=lambda value: int(value.rsplit("_", 1)[1]))
        states = [np.asarray(group[name]["initial_state"]) for name in demos]
        attrs = {
            str(key): value.decode() if isinstance(value, bytes) else str(value)
            for key, value in group.attrs.items()
        }
    return states, attrs


def _descendant_body_ids(model, root_id: int) -> set[int]:
    result = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if int(model.body_parentid[body_id]) in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def _canonical_contact_owner(model, body_id: int) -> str:
    tracked_ids = {int(model.body_name2id(name)): name for name in TRACKED_BODIES}
    current = int(body_id)
    while current > 0:
        if current in tracked_ids:
            return tracked_ids[current]
        name = str(model.body_id2name(current) or "")
        lowered = name.lower()
        if "table" in lowered:
            return "table"
        if any(token in lowered for token in ("robot", "gripper", "panda", "eef")):
            return "robot"
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return str(model.body_id2name(body_id) or f"body_{body_id}")


def _contact_owners(env, body_name: str) -> list[str]:
    model, data = env.sim.model, env.sim.data
    root_id = int(model.body_name2id(body_name))
    body_ids = _descendant_body_ids(model, root_id)
    geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }
    owners: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in geom_ids:
            other_geom = geom2
        elif geom2 in geom_ids:
            other_geom = geom1
        else:
            continue
        owner = _canonical_contact_owner(model, int(model.geom_bodyid[other_geom]))
        if owner != body_name:
            owners.add(owner)
    return sorted(owners)


def _free_joint_velocity(env, body_name: str) -> tuple[float, float] | None:
    model = env.sim.model
    body_id = int(model.body_name2id(body_name))
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != 0:
        return None
    address = int(model.jnt_dofadr[joint_id])
    velocity = np.asarray(env.sim.data.qvel[address : address + 6], dtype=float)
    return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))


def _sample(env, condition: str) -> dict[str, dict]:
    model, data = env.sim.model, env.sim.data
    result: dict[str, dict] = {}
    for body_name in TRACKED_BODIES:
        body_id = int(model.body_name2id(body_name))
        rotation = np.asarray(data.body_xmat[body_id], dtype=float).reshape(3, 3)
        velocity = _free_joint_velocity(env, body_name)
        contacts = _contact_owners(env, body_name)
        expected_support = EXPECTED_SUPPORT[condition][body_name]
        allowed = {expected_support}
        if condition == "er" and body_name == "plate_1_main":
            allowed.add("akita_black_bowl_2_main")
        result[body_name] = {
            "position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
            "quaternion_wxyz": np.asarray(data.body_xquat[body_id], dtype=float).tolist(),
            "tilt_deg": float(
                np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
            ),
            "linear_speed_mps": None if velocity is None else velocity[0],
            "angular_speed_radps": None if velocity is None else velocity[1],
            "contacts": contacts,
            "expected_support": expected_support,
            "support_present": expected_support in contacts,
            "forbidden_contacts": [value for value in contacts if value not in allowed],
        }
    return result


def _evaluate_trace(samples: list[dict[str, dict]]) -> tuple[bool, list[str], dict]:
    if len(samples) <= FORMAL_WAIT_STEPS:
        raise ValueError("trace does not contain the first-policy sample")
    failures: list[str] = []
    summary: dict[str, dict] = {}
    for body_name in TRACKED_BODIES:
        measurements = [sample[body_name] for sample in samples]
        origin = np.asarray(measurements[0]["position"], dtype=float)
        max_drift = max(
            float(np.linalg.norm(np.asarray(value["position"], dtype=float) - origin))
            for value in measurements
        )
        max_tilt = max(float(value["tilt_deg"]) for value in measurements)
        max_linear = max(float(value["linear_speed_mps"] or 0.0) for value in measurements)
        max_angular = max(float(value["angular_speed_radps"] or 0.0) for value in measurements)
        first_policy = measurements[FORMAL_WAIT_STEPS]
        tilt_limit = (
            MAX_RECEPTACLE_TILT_DEG
            if body_name in RECEPTACLES
            else MAX_NON_RECEPTACLE_TILT_DEG
        )
        if max_drift > MAX_TRANSLATION_DRIFT_M:
            failures.append(f"{body_name}:translation_drift")
        if max_tilt > tilt_limit:
            failures.append(f"{body_name}:tilt")
        if max_linear > MAX_WINDOW_LINEAR_SPEED_MPS:
            failures.append(f"{body_name}:window_linear_speed")
        if max_angular > MAX_WINDOW_ANGULAR_SPEED_RADPS:
            failures.append(f"{body_name}:window_angular_speed")
        if first_policy["linear_speed_mps"] is None or float(
            first_policy["linear_speed_mps"]
        ) > MAX_FIRST_POLICY_LINEAR_SPEED_MPS:
            failures.append(f"{body_name}:first_policy_linear_speed")
        if first_policy["angular_speed_radps"] is None or float(
            first_policy["angular_speed_radps"]
        ) > MAX_FIRST_POLICY_ANGULAR_SPEED_RADPS:
            failures.append(f"{body_name}:first_policy_angular_speed")
        if any(not bool(value["support_present"]) for value in measurements):
            failures.append(f"{body_name}:support_loss")
        if any(value["forbidden_contacts"] for value in measurements):
            failures.append(f"{body_name}:forbidden_contact")
        summary[body_name] = {
            "pre_wait": measurements[0],
            "first_policy": first_policy,
            "post_confirmation": measurements[-1],
            "max_translation_drift_m": max_drift,
            "max_tilt_deg": max_tilt,
            "tilt_limit_deg": tilt_limit,
            "max_linear_speed_mps": max_linear,
            "max_angular_speed_radps": max_angular,
            "support_present_throughout": all(
                bool(value["support_present"]) for value in measurements
            ),
        }
    unique = sorted(set(failures))
    return not unique, unique, summary


def _geom_ids_for_body(env, body_name: str) -> set[int]:
    model = env.sim.model
    body_ids = _descendant_body_ids(model, int(model.body_name2id(body_name)))
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def _segmentation_ids(env, resolution: int) -> np.ndarray:
    try:
        segmentation = np.asarray(
            env.sim.render(
                width=resolution,
                height=resolution,
                camera_name="agentview",
                segmentation=True,
            )
        )
    except OverflowError:
        import mujoco

        context = env.sim._render_context_offscreen
        camera_id = env.sim.model.camera_name2id("agentview")
        context.render(resolution, resolution, camera_id=camera_id, segmentation=True)
        viewport = mujoco.MjrRect(0, 0, resolution, resolution)
        rgb = np.empty((resolution, resolution, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=viewport, con=context.con)
        rgb32 = rgb.astype(np.int32)
        encoded = rgb32[..., 0] + rgb32[..., 1] * 256 + rgb32[..., 2] * 65536
        encoded[encoded >= context.scn.ngeom + 1] = 0
        ids = np.full((context.scn.ngeom + 1, 2), -1, dtype=np.int32)
        for index in range(context.scn.ngeom):
            geom = context.scn.geoms[index]
            if geom.segid != -1:
                ids[geom.segid + 1] = (geom.objtype, geom.objid)
        segmentation = ids[encoded]
    if segmentation.ndim == 3:
        segmentation = segmentation[..., -1]
    return np.asarray(segmentation, dtype=np.int32)


def _center_crop(array: np.ndarray, crop_area: float = CENTER_CROP_AREA) -> np.ndarray:
    height, width = array.shape[:2]
    scale = float(np.sqrt(crop_area))
    crop_height = max(1, int(round(height * scale)))
    crop_width = max(1, int(round(width * scale)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return array[top : top + crop_height, left : left + crop_width]


def _visibility(env, resolution: int) -> dict[str, int]:
    # Orientation does not affect a centered crop or pixel count. Use nearest
    # neighbor for integer segmentation IDs before counting model-frame pixels.
    cropped = _center_crop(_segmentation_ids(env, resolution))
    resized = np.asarray(
        Image.fromarray(cropped).resize(
            (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.NEAREST
        )
    )
    return {
        body: int(np.isin(resized, tuple(_geom_ids_for_body(env, body))).sum())
        for body in VISIBILITY_BODIES
    }


def _save_policy_views(
    env,
    observation: dict,
    output_dir: Path,
    condition: str,
    episode_idx: int,
) -> dict[str, object]:
    from experiments.robot.libero.run_libero_eval import prepare_observation
    from experiments.robot.openvla_utils import center_crop_image

    prepared, _ = prepare_observation(observation, MODEL_IMAGE_SIZE, "openvla")
    agent = np.asarray(center_crop_image(prepared["full_image"]), dtype=np.uint8)
    wrist = np.asarray(center_crop_image(prepared["wrist_image"]), dtype=np.uint8)
    condition_dir = output_dir / "exact_first_policy_frames" / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    stem = f"L1-C1_{condition}_ep{episode_idx:03d}_exact_first_policy"
    agent_path = condition_dir / f"{stem}_agentview_model_input.png"
    wrist_path = condition_dir / f"{stem}_wrist_model_input.png"
    Image.fromarray(agent).save(agent_path)
    Image.fromarray(wrist).save(wrist_path)
    return {
        "agentview_path": _portable_path(agent_path),
        "agentview_sha256": _sha256(agent_path),
        "wrist_path": _portable_path(wrist_path),
        "wrist_sha256": _sha256(wrist_path),
        "model_input_shape": list(agent.shape),
        "visible_pixels_after_policy_crop": _visibility(env, MODEL_IMAGE_SIZE),
    }


def _save_contact_sheet(paths: Iterable[Path], output: Path, columns: int = 10) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows), color="white")
    for index, item in enumerate(images):
        sheet.paste(item, ((index % columns) * width, (index // columns) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    for item in images:
        item.close()


def _runtime_state(env) -> dict[str, np.ndarray]:
    data = env.sim.data
    result = {
        "qpos": np.asarray(data.qpos, dtype=float).copy(),
        "qvel": np.asarray(data.qvel, dtype=float).copy(),
    }
    if getattr(data, "act", None) is not None:
        result["act"] = np.asarray(data.act, dtype=float).copy()
    return result


def _allowed_state_indices(env) -> dict[str, set[int]]:
    model = env.sim.model
    body_id = int(model.body_name2id("akita_black_bowl_2_main"))
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != 0:
        raise RuntimeError("L1-C1 intervention body does not have a free joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    return {
        "qpos": set(range(qpos_address, qpos_address + 7)),
        "qvel": set(range(qvel_address, qvel_address + 6)),
        "act": set(),
    }


def _compare_condition_states(
    snapshots: dict[tuple[int, str], dict[str, np.ndarray]],
    count: int,
    allowed: dict[str, set[int]],
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for episode_idx in range(count):
        eb = snapshots[(episode_idx, "eb")]
        for condition in ("er", "ec"):
            other = snapshots[(episode_idx, condition)]
            for field in sorted(eb):
                difference = np.abs(other[field] - eb[field])
                disallowed = [
                    index
                    for index, value in enumerate(difference)
                    if index not in allowed.get(field, set()) and value > STATE_DIFF_ATOL
                ]
                if disallowed:
                    failures.append(
                        {
                            "episode_idx": episode_idx,
                            "condition": condition,
                            "field": field,
                            "indices": disallowed,
                            "max_abs_diff": float(difference[disallowed].max()),
                        }
                    )
    return failures


def _write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition",
        "episode_idx",
        "valid",
        "failures",
        "min_visible_pixels",
        "max_translation_drift_m",
        "max_tilt_deg",
        "max_linear_speed_mps",
        "max_angular_speed_radps",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            stability = record["stability"]
            writer.writerow(
                {
                    "condition": record["condition"],
                    "episode_idx": record["episode_idx"],
                    "valid": record["valid"],
                    "failures": ";".join(record["failures"]),
                    "min_visible_pixels": min(
                        record["policy_view"]["visible_pixels_after_policy_crop"].values()
                    ),
                    "max_translation_drift_m": max(
                        value["max_translation_drift_m"] for value in stability.values()
                    ),
                    "max_tilt_deg": max(value["max_tilt_deg"] for value in stability.values()),
                    "max_linear_speed_mps": max(
                        value["max_linear_speed_mps"] for value in stability.values()
                    ),
                    "max_angular_speed_radps": max(
                        value["max_angular_speed_radps"] for value in stability.values()
                    ),
                }
            )


def _write_report(path: Path, manifest: dict) -> None:
    records = manifest["records"]
    lines = [
        "# L1-C1 exact first-policy-frame gate",
        "",
        f"Verdict: **{manifest['verdict']}**",
        "",
        f"Episodes per condition: {manifest['episode_count']}",
        f"Valid records: {sum(bool(record['valid']) for record in records)}/{len(records)}",
        f"Cross-condition allowlist failures: {len(manifest['cross_condition_state_failures'])}",
        "",
        "The captured PNGs are the 224×224 center-cropped agent and wrist images "
        "passed to OpenVLA after the evaluator's 10-step no-op wait.",
        "",
        "Formal evaluation remains blocked until `HUMAN_REVIEW.json` records explicit approval.",
    ]
    failed = [record for record in records if not record["valid"]]
    if failed:
        lines.extend(("", "## Failed episodes", ""))
        lines.extend(
            f"- {item['condition']} ep{item['episode_idx']:03d}: "
            + ", ".join(item["failures"])
            for item in failed
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(args: argparse.Namespace) -> dict[str, object]:
    host = _assert_superpod(args.allow_local_simulator)
    state_dir = Path(args.state_dir)
    paths = {condition: state_dir / filename for condition, filename in STATE_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen L1-C1 state files: {missing}")
    hashes = {condition: _sha256(path) for condition, path in paths.items()}
    mismatches = {
        condition: {"expected": EXPECTED_STATE_SHA256[condition], "actual": value}
        for condition, value in hashes.items()
        if value != EXPECTED_STATE_SHA256[condition]
    }
    if mismatches:
        raise ValueError(f"frozen state hash mismatch: {mismatches}")
    loaded = {condition: _load_states(path) for condition, path in paths.items()}
    states = {condition: value[0] for condition, value in loaded.items()}
    attrs = {condition: value[1] for condition, value in loaded.items()}
    counts = {condition: len(value) for condition, value in states.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"condition state counts differ: {counts}")
    episode_count = next(iter(counts.values()))
    if args.num_episodes is not None:
        if args.num_episodes > episode_count:
            raise ValueError(f"requested {args.num_episodes} episodes, bundle has {episode_count}")
        episode_count = args.num_episodes

    from libero.libero import benchmark
    from experiments.robot.libero.formal_evaluator_state import restore_formal_observation
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != TASK_PROMPT:
        raise ValueError(f"runtime task prompt mismatch: {task.language!r}")

    review_dir = Path(args.review_dir)
    records: list[dict] = []
    snapshots: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    allowed_indices: dict[str, set[int]] | None = None
    for condition in ("eb", "er", "ec"):
        env, task_description = get_libero_env(
            task,
            "openvla",
            resolution=args.render_resolution,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        if task_description != TASK_PROMPT:
            raise ValueError("policy-facing task prompt mismatch")
        try:
            for episode_idx, state in enumerate(states[condition][:episode_count]):
                observation = restore_formal_observation(env, state)
                if allowed_indices is None:
                    allowed_indices = _allowed_state_indices(env)
                snapshots[(episode_idx, condition)] = _runtime_state(env)
                samples = [_sample(env, condition)]
                first_policy_observation = None
                policy_view = None
                for wait_step in range(1, FORMAL_WAIT_STEPS + CONFIRM_STEPS + 1):
                    observation, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    samples.append(_sample(env, condition))
                    if wait_step == FORMAL_WAIT_STEPS:
                        first_policy_observation = observation
                        policy_view = _save_policy_views(
                            env, first_policy_observation, review_dir, condition, episode_idx
                        )
                if first_policy_observation is None or policy_view is None:
                    raise RuntimeError("exact first-policy observation was not captured")
                valid, failures, stability = _evaluate_trace(samples)
                visibility = policy_view["visible_pixels_after_policy_crop"]
                if min(visibility.values()) < MIN_VISIBLE_PIXELS:
                    failures.append("policy_view_visibility")
                    valid = False
                records.append(
                    {
                        "condition": condition,
                        "episode_idx": episode_idx,
                        "valid": bool(valid),
                        "failures": sorted(set(failures)),
                        "stability": stability,
                        "full_wait_and_confirmation_trace": samples,
                        "policy_view": policy_view,
                    }
                )
        finally:
            env.close()

    if allowed_indices is None:
        raise RuntimeError("no L1-C1 episodes were evaluated")
    state_failures = _compare_condition_states(snapshots, episode_count, allowed_indices)
    passed = all(record["valid"] for record in records) and not state_failures
    manifest = {
        "schema_version": 1,
        "scenario": SCENARIO,
        "verdict": "PASS_L1C1_EXACT_FIRST_POLICY_GATE" if passed else "FAIL_L1C1_EXACT_FIRST_POLICY_GATE",
        "host_verification": host,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": task.bddl_file,
        "task_prompt": TASK_PROMPT,
        "episode_count": episode_count,
        "condition_counts": counts,
        "state_files": {
            condition: {
                "path": _portable_path(paths[condition]),
                "sha256": hashes[condition],
                "attributes": attrs[condition],
            }
            for condition in ("eb", "er", "ec")
        },
        "formal_reset_sequence": [
            "env.reset",
            "env.set_init_state",
            "sim.forward",
            "env._post_process",
            "env._update_observables(force=True)",
            "env._get_observations",
            f"{FORMAL_WAIT_STEPS}_controller_noop_steps",
            "capture_observation_returned_by_final_noop",
            "prepare_observation_jpeg_resize_224",
            "openvla_center_crop_area_0.9_resize_224",
            f"{CONFIRM_STEPS}_postwait_confirmation_steps",
        ],
        "thresholds": {
            "max_receptacle_tilt_deg_throughout": MAX_RECEPTACLE_TILT_DEG,
            "max_non_receptacle_tilt_deg_throughout": MAX_NON_RECEPTACLE_TILT_DEG,
            "max_translation_drift_m_throughout": MAX_TRANSLATION_DRIFT_M,
            "max_window_linear_speed_mps": MAX_WINDOW_LINEAR_SPEED_MPS,
            "max_window_angular_speed_radps": MAX_WINDOW_ANGULAR_SPEED_RADPS,
            "max_first_policy_linear_speed_mps": MAX_FIRST_POLICY_LINEAR_SPEED_MPS,
            "max_first_policy_angular_speed_radps": MAX_FIRST_POLICY_ANGULAR_SPEED_RADPS,
            "min_visible_pixels_after_policy_crop": MIN_VISIBLE_PIXELS,
            "cross_condition_state_diff_atol": STATE_DIFF_ATOL,
        },
        "intervention_allowlist": {
            "description": "only akita_black_bowl_2_main free-joint pose/velocity may differ",
            "runtime_indices": {
                key: sorted(value) for key, value in allowed_indices.items()
            },
        },
        "cross_condition_state_failures": state_failures,
        "records": records,
    }

    manifest_path = Path(args.output_manifest)
    csv_path = Path(args.output_csv)
    report_path = Path(args.output_report)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, records)
    _write_report(report_path, manifest)
    for condition in ("eb", "er", "ec"):
        for camera in ("agentview", "wrist"):
            paths_for_sheet = sorted(
                (review_dir / "exact_first_policy_frames" / condition).glob(
                    f"*_{camera}_model_input.png"
                )
            )
            _save_contact_sheet(
                paths_for_sheet,
                review_dir / f"L1-C1_{condition}_{camera}_exact_first_policy_contact_sheet.png",
            )
    human_review = review_dir / "HUMAN_REVIEW.json"
    human_review.write_text(
        json.dumps(
            {
                "scenario": SCENARIO,
                "approved": False,
                "reviewer": "",
                "reviewed_at": "",
                "scope": "all exact first-policy agent/wrist frames and smoke videos",
                "physical_gate_verdict": manifest["verdict"],
                "physical_gate_manifest": _portable_path(manifest_path),
                "physical_gate_manifest_sha256": _sha256(manifest_path),
                "notes": "",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Verdict: **{manifest['verdict']}**")
    print(f"Manifest: {_portable_path(manifest_path)}")
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state_dir", default="experiments/robot/libero/tasks/l1c1_first_policy_inputs"
    )
    parser.add_argument("--review_dir", default="review/L1-C1_task/first_policy_gate")
    parser.add_argument(
        "--output_manifest", default="experiments/logs/l1c1_first_policy_gate.json"
    )
    parser.add_argument("--output_csv", default="experiments/logs/l1c1_first_policy_gate.csv")
    parser.add_argument("--output_report", default="experiments/logs/l1c1_first_policy_gate.md")
    parser.add_argument("--num_episodes", type=int)
    parser.add_argument("--render_resolution", type=int, default=256)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--fail_on_invalid", action="store_true")
    parser.add_argument("--allow_local_simulator", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
