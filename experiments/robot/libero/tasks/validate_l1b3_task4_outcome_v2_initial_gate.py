"""Validate L1-B3 Task-4 Outcome V2 at the exact first policy frame.

This command initializes and steps LIBERO, so it is Superpod-only.  Every
EB/ER/EC episode is checked before the formal wait, throughout the evaluator's
ten no-op wait steps, at the refreshed first policy observation, and through a
short post-wait confirmation window.  One invalid record fails the frozen
scene version; episodes are never silently filtered here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import (
    FAMILIES,
    _allowed_obstacle_state_indices,
)
from experiments.robot.libero.tasks.validate_l1b_swept_states import (
    _center_policy_crop,
    _fresh_observation,
    _load_states,
    _visible_pixel_count,
)
from experiments.robot.libero.tasks.validate_l1b3_task4_outcome_v2_preflight import (
    FAMILY,
    SCENE_ID,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    VERDICT as PREFLIGHT_VERDICT,
)


PROTECTED_BODY = "wine_bottle_1_main"
FORMAL_WAIT_STEPS = 10
CONFIRM_STEPS = 5
MAX_TRANSLATION_DRIFT_M = 0.005
MAX_RECEPTACLE_TILT_DEG = 1.0
MAX_OTHER_OBJECT_TILT_DEG = 2.0
# The serialized native state is sampled before the evaluator has taken its
# first controller no-op. Bound that restore-entry transient separately from
# the stabilization window; samples 1..N must satisfy the stricter window
# limits below, and the exact first-policy sample has its own tighter limits.
MAX_RESTORE_LINEAR_SPEED_MPS = 0.100
MAX_RESTORE_ANGULAR_SPEED_RADPS = 0.500
MAX_WINDOW_LINEAR_SPEED_MPS = 0.025
MAX_WINDOW_ANGULAR_SPEED_RADPS = 0.25
MAX_FIRST_POLICY_LINEAR_SPEED_MPS = 0.010
MAX_FIRST_POLICY_ANGULAR_SPEED_RADPS = 0.050
MAX_SUPPORT_PENETRATION_M = 0.002
MIN_VISIBLE_PIXELS = 50
MAX_CROSS_CONDITION_INVARIANT_DRIFT_M = 0.001
SUPPORT_PREFIXES = ("table", "main_table")
RECEPTACLE_TOKENS = ("bowl", "plate", "ramekin", "cup", "mug")
PASS_VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_INITIAL_GATE"
FAIL_VERDICT = "FAIL_L1B3_TASK4_OUTCOME_V2_INITIAL_GATE"


def _assert_superpod() -> dict[str, object]:
    hostname = socket.gethostname()
    host_marker = any(
        marker in hostname.lower()
        for marker in ("superpod", "dgx", "slogin", "compute", "gpu")
    )
    scheduler = bool(os.environ.get("SLURM_JOB_ID"))
    trusted_marker = os.environ.get("PHYSCG_SUPERPOD") == "1"
    declared = os.environ.get("PHYSCG_EXECUTION_HOST") == "superpod"
    verified = bool(declared and (host_marker or scheduler or trusted_marker))
    if not verified:
        raise RuntimeError(
            "L1-B3 Task-4 Outcome V2 simulator validation is Superpod-only. "
            "PHYSCG_EXECUTION_HOST=superpod plus a Superpod hostname, SLURM "
            "job, or trusted PHYSCG_SUPERPOD=1 marker is required."
        )
    return {
        "hostname": hostname,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "execution_host_declared": declared,
        "trusted_superpod_marker": trusted_marker,
        "superpod_verified": verified,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _verify_frozen_preflight_hashes(
    preflight: dict[str, object],
) -> dict[str, object]:
    checked = []
    mismatches = []
    for relative, expected in preflight.get("project_file_hashes", {}).items():
        path = REPO_ROOT / str(relative)
        observed = _sha256(path) if path.is_file() else None
        checked.append(str(relative))
        if observed != expected:
            mismatches.append(
                {"path": str(relative), "expected": expected, "observed": observed}
            )
    native_bddl = Path(str(preflight["native_bddl"]))
    bddl_observed = _sha256(native_bddl) if native_bddl.is_file() else None
    checked.append(str(native_bddl))
    if bddl_observed != preflight.get("native_bddl_sha256"):
        mismatches.append(
            {
                "path": str(native_bddl),
                "expected": preflight.get("native_bddl_sha256"),
                "observed": bddl_observed,
            }
        )
    libero_root = None
    for parent in native_bddl.parents:
        if parent.joinpath("libero/libero/assets").is_dir():
            libero_root = parent
            break
    if libero_root is None:
        mismatches.append(
            {
                "path": str(native_bddl),
                "expected": "resolvable LIBERO repository root",
                "observed": None,
            }
        )
    else:
        for relative, metadata in preflight.get("native_asset_files", {}).items():
            path = libero_root / str(relative)
            observed = _sha256(path) if path.is_file() else None
            expected = metadata.get("sha256")
            checked.append(str(path))
            if observed != expected:
                mismatches.append(
                    {
                        "path": str(path),
                        "expected": expected,
                        "observed": observed,
                    }
                )
    if mismatches:
        raise ValueError(f"frozen preflight hash mismatch: {mismatches}")
    return {"verified": True, "checked_file_count": len(checked), "mismatches": []}


def _free_joint_addresses(model, body_name: str) -> tuple[int, int]:
    body_id = int(model.body_name2id(body_name))
    for joint_id in range(int(model.njnt)):
        if (
            int(model.jnt_bodyid[joint_id]) == body_id
            and int(model.jnt_type[joint_id]) == 0
        ):
            return (
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
    raise ValueError(f"free joint not found for {body_name}")


def _movable_object_bodies(model) -> tuple[str, ...]:
    names = []
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) != 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        name = model.body_id2name(body_id) or ""
        if name and not name.startswith(("robot0_", "gripper0_")):
            names.append(str(name))
    result = tuple(sorted(set(names)))
    if PROTECTED_BODY not in result:
        raise ValueError(f"protected body {PROTECTED_BODY!r} is not a free object")
    return result


def _body_subtree_ids(model, body_name: str) -> set[int]:
    result = {int(model.body_name2id(body_name))}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if (
                body_id not in result
                and int(model.body_parentid[body_id]) in result
            ):
                result.add(body_id)
                changed = True
    return result


def _contact_measurement(env, body_name: str) -> dict[str, object]:
    model, data = env.sim.model, env.sim.data
    subtree = _body_subtree_ids(model, body_name)
    geom_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in subtree
    }
    support: set[str] = set()
    forbidden: set[str] = set()
    max_support_penetration = 0.0
    max_forbidden_penetration = 0.0
    contacts = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in geom_ids and second not in geom_ids:
            other_geom = second
        elif second in geom_ids and first not in geom_ids:
            other_geom = first
        else:
            continue
        other_body_id = int(model.geom_bodyid[other_geom])
        other_name = model.body_id2name(other_body_id) or (
            f"body_id_{other_body_id}"
        )
        distance = float(contact.dist)
        penetration = max(0.0, -distance)
        is_support = str(other_name).startswith(SUPPORT_PREFIXES)
        contacts.append({"body": str(other_name), "distance_m": distance})
        if is_support:
            support.add(str(other_name))
            max_support_penetration = max(
                max_support_penetration, penetration
            )
            if penetration > MAX_SUPPORT_PENETRATION_M:
                forbidden.add(f"{other_name}:support_penetration")
        elif distance <= 0.0:
            forbidden.add(str(other_name))
            max_forbidden_penetration = max(
                max_forbidden_penetration, penetration
            )
    return {
        "contacts": contacts,
        "support_contacts": sorted(support),
        "forbidden_contacts": sorted(forbidden),
        "max_support_penetration_m": max_support_penetration,
        "max_forbidden_penetration_m": max_forbidden_penetration,
    }


def _body_measurement(env, body_name: str) -> dict[str, object]:
    model, data = env.sim.model, env.sim.data
    body_id = int(model.body_name2id(body_name))
    rotation = np.asarray(data.body_xmat[body_id], dtype=float).reshape(3, 3)
    _, velocity_address = _free_joint_addresses(model, body_name)
    velocity = np.asarray(
        data.qvel[velocity_address : velocity_address + 6], dtype=float
    )
    result = {
        "position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(
            data.body_xquat[body_id], dtype=float
        ).tolist(),
        "tilt_deg": float(
            np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        ),
        "linear_speed_mps": float(np.linalg.norm(velocity[:3])),
        "angular_speed_radps": float(np.linalg.norm(velocity[3:])),
    }
    result.update(_contact_measurement(env, body_name))
    return result


def _sample(env, bodies: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {body: _body_measurement(env, body) for body in bodies}


def _tilt_limit(body_name: str) -> float:
    if body_name == PROTECTED_BODY or any(
        token in body_name.lower() for token in RECEPTACLE_TOKENS
    ):
        return MAX_RECEPTACLE_TILT_DEG
    return MAX_OTHER_OBJECT_TILT_DEG


def _evaluate_trace(
    samples: list[dict[str, dict[str, object]]],
    bodies: tuple[str, ...],
) -> tuple[bool, list[str], dict[str, object]]:
    failures: list[str] = []
    summary: dict[str, object] = {}
    for body_name in bodies:
        positions = [
            np.asarray(sample[body_name]["position"], dtype=float)
            for sample in samples
        ]
        max_drift = max(
            float(np.linalg.norm(position - positions[0]))
            for position in positions
        )
        max_tilt = max(float(sample[body_name]["tilt_deg"]) for sample in samples)
        pre_wait_linear = float(samples[0][body_name]["linear_speed_mps"])
        pre_wait_angular = float(samples[0][body_name]["angular_speed_radps"])
        max_linear = max(
            float(sample[body_name]["linear_speed_mps"])
            for sample in samples[1:]
        )
        max_angular = max(
            float(sample[body_name]["angular_speed_radps"])
            for sample in samples[1:]
        )
        first_policy = samples[FORMAL_WAIT_STEPS][body_name]
        tilt_limit = _tilt_limit(body_name)
        if max_drift > MAX_TRANSLATION_DRIFT_M:
            failures.append(f"{body_name}:translation_drift")
        if max_tilt > tilt_limit:
            failures.append(f"{body_name}:tilt")
        if pre_wait_linear > MAX_RESTORE_LINEAR_SPEED_MPS:
            failures.append(f"{body_name}:restore_linear_speed")
        if pre_wait_angular > MAX_RESTORE_ANGULAR_SPEED_RADPS:
            failures.append(f"{body_name}:restore_angular_speed")
        if max_linear > MAX_WINDOW_LINEAR_SPEED_MPS:
            failures.append(f"{body_name}:window_linear_speed")
        if max_angular > MAX_WINDOW_ANGULAR_SPEED_RADPS:
            failures.append(f"{body_name}:window_angular_speed")
        if (
            float(first_policy["linear_speed_mps"])
            > MAX_FIRST_POLICY_LINEAR_SPEED_MPS
        ):
            failures.append(f"{body_name}:first_policy_linear_speed")
        if (
            float(first_policy["angular_speed_radps"])
            > MAX_FIRST_POLICY_ANGULAR_SPEED_RADPS
        ):
            failures.append(f"{body_name}:first_policy_angular_speed")
        if any(not sample[body_name]["support_contacts"] for sample in samples):
            failures.append(f"{body_name}:support_loss")
        if any(sample[body_name]["forbidden_contacts"] for sample in samples):
            failures.append(f"{body_name}:forbidden_contact")
        summary[body_name] = {
            "pre_wait": samples[0][body_name],
            "first_policy": first_policy,
            "post_confirmation": samples[-1][body_name],
            "max_translation_drift_m": max_drift,
            "max_tilt_deg": max_tilt,
            "tilt_limit_deg": tilt_limit,
            "pre_wait_linear_speed_mps": pre_wait_linear,
            "pre_wait_angular_speed_radps": pre_wait_angular,
            "max_stabilization_linear_speed_mps": max_linear,
            "max_stabilization_angular_speed_radps": max_angular,
            "support_signatures_throughout": [
                sample[body_name]["support_contacts"] for sample in samples
            ],
        }
    return not failures, sorted(set(failures)), summary


def _save_policy_view(
    env,
    observation: dict,
    review_dir: Path,
    condition: str,
    episode_idx: int,
    resolution: int,
) -> dict[str, object]:
    image = np.asarray(get_libero_image(observation), dtype=np.uint8)
    raw = review_dir / "initial_frames" / condition / (
        f"L1-B3-task4-outcome-v2_{condition}_ep{episode_idx:03d}_"
        "exact_first_policy.png"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(raw)
    cropped = _center_policy_crop(image)
    cropped = np.asarray(
        Image.fromarray(cropped).resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
    )
    crop = raw.with_name(raw.stem + "_center_crop09.png")
    Image.fromarray(cropped.astype(np.uint8)).save(crop)
    visible = _visible_pixel_count(
        env, PROTECTED_BODY, "agentview", resolution
    )
    return {
        "raw_image_path": _portable(raw),
        "raw_image_sha256": _sha256(raw),
        "center_crop_path": _portable(crop),
        "center_crop_sha256": _sha256(crop),
        "wine_bottle_visible_pixels_after_policy_crop": visible,
    }


def _fixture_root_bodies(model, preflight: dict[str, object]) -> tuple[str, ...]:
    model_names = {
        model.body_id2name(body_id) or ""
        for body_id in range(int(model.nbody))
    }
    result = []
    fixtures = preflight.get("fixtures", {})
    for instance in fixtures:
        for candidate in (f"{instance}_main", str(instance)):
            if candidate in model_names:
                result.append(candidate)
                break
    return tuple(sorted(set(result)))


def _state_diff_audit(env, states: dict[str, list[np.ndarray]]) -> list[dict]:
    allowed = _allowed_obstacle_state_indices(
        env.sim, PROTECTED_BODY, FAMILIES[FAMILY]
    )
    records = []
    for episode_idx in range(len(states["eb"])):
        base = np.asarray(states["eb"][episode_idx])
        for condition in ("er", "ec"):
            other = np.asarray(states[condition][episode_idx])
            if base.shape != other.shape:
                changed = []
                outside = ["shape_mismatch"]
            else:
                changed = np.flatnonzero(base != other).astype(int).tolist()
                outside = sorted(set(changed) - allowed)
            records.append(
                {
                    "episode_idx": episode_idx,
                    "condition": condition,
                    "changed_flat_state_indices": changed,
                    "allowed_flat_state_indices": sorted(allowed),
                    "outside_allowlist": outside,
                    "all_other_state_fields_byte_identical": not outside,
                    "intervention_nonempty": bool(changed),
                    "valid": bool(changed and not outside),
                }
            )
    return records


def _source_to_project_state_diff_audit(
    env,
    native_source_states: list[np.ndarray],
    eb_states: list[np.ndarray],
    pairing: dict[str, object],
) -> list[dict[str, object]]:
    """Verify the declared native-source -> project-Eb layout delta exactly."""
    allowed = _allowed_obstacle_state_indices(
        env.sim, PROTECTED_BODY, FAMILIES[FAMILY]
    )
    expected_offset_xy = np.asarray(
        FAMILIES[FAMILY]["eb_obstacle_offset_xy"], dtype=float
    )
    pairs = pairing.get("pairs", [])
    records = []
    for episode_idx, (native, eb) in enumerate(
        zip(native_source_states, eb_states)
    ):
        native = np.asarray(native)
        eb = np.asarray(eb)
        pair = pairs[episode_idx] if episode_idx < len(pairs) else {}
        if native.shape != eb.shape:
            changed = []
            outside = ["shape_mismatch"]
        else:
            changed = np.flatnonzero(native != eb).astype(int).tolist()
            outside = sorted(set(changed) - allowed)
        reported_changed = pair.get("eb_layout_changed_state_indices", [])
        observed_native_hash = hashlib.sha256(
            np.ascontiguousarray(native).tobytes()
        ).hexdigest()
        observed_eb_hash = hashlib.sha256(
            np.ascontiguousarray(eb).tobytes()
        ).hexdigest()
        native_xy = np.asarray(
            pair.get("native_source_obstacle_xyz", [])[:2], dtype=float
        )
        expected_xy = native_xy + expected_offset_xy
        reported_xy = np.asarray(
            pair.get("eb_project_placement", []), dtype=float
        )
        anchor_matches = bool(
            native_xy.shape == (2,)
            and reported_xy.shape == (2,)
            and np.allclose(reported_xy, expected_xy, rtol=0.0, atol=1e-12)
        )
        valid = bool(
            changed
            and not outside
            and changed == reported_changed
            and pair.get("eb_layout_only_obstacle_pose_changed") is True
            and pair.get("native_source_state_sha256")
            == observed_native_hash
            and pair.get("source_state_sha256") == observed_eb_hash
            and anchor_matches
        )
        records.append(
            {
                "episode_idx": episode_idx,
                "changed_flat_state_indices": changed,
                "reported_changed_flat_state_indices": reported_changed,
                "allowed_flat_state_indices": sorted(allowed),
                "outside_allowlist": outside,
                "native_source_state_sha256": observed_native_hash,
                "project_eb_state_sha256": observed_eb_hash,
                "expected_eb_obstacle_xy": expected_xy.tolist(),
                "expected_eb_obstacle_offset_xy": expected_offset_xy.tolist(),
                "reported_eb_obstacle_xy": reported_xy.tolist(),
                "all_other_native_state_fields_byte_identical": not outside,
                "valid": valid,
            }
        )
    return records


def _write_human_review_template(
    review_dir: Path, manifest_path: Path
) -> Path:
    path = review_dir / "HUMAN_REVIEW.json"
    manifest_hash = _sha256(manifest_path)
    keep_existing = False
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            keep_existing = (
                existing.get("physical_gate_manifest_sha256") == manifest_hash
            )
        except (OSError, ValueError):
            keep_existing = False
    if not keep_existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "scenario": SCENE_ID,
                    "approved": False,
                    "reviewer": "",
                    "reviewed_at": "",
                    "scope": "all_initial_frames_and_smoke_videos",
                    "physical_gate_manifest": _portable(manifest_path),
                    "physical_gate_manifest_sha256": manifest_hash,
                    "notes": "",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def validate(args) -> dict[str, object]:
    host = _assert_superpod()
    state_dir = Path(args.state_dir)
    paths = {
        condition: state_dir / f"{FAMILY}_{condition}_states.hdf5"
        for condition in ("eb", "er", "ec")
    }
    native_source_path = state_dir / f"{FAMILY}_native_source_states.hdf5"
    pairing_path = state_dir / f"{FAMILY}_pairing.json"
    preflight_path = Path(args.preflight_manifest)
    prereg_path = Path(args.preregistration)
    required = [
        native_source_path,
        *paths.values(),
        pairing_path,
        preflight_path,
        prereg_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen v2 artifacts: {missing}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if preflight.get("verdict") != PREFLIGHT_VERDICT:
        raise ValueError("native preflight has not passed")
    frozen_hash_verification = _verify_frozen_preflight_hashes(preflight)
    if pairing.get("family") != FAMILY or prereg.get("family") != FAMILY:
        raise ValueError("family/version mismatch in frozen artifacts")
    states = {condition: _load_states(path) for condition, path in paths.items()}
    native_source_states = _load_states(native_source_path)
    counts = {condition: len(value) for condition, value in states.items()}
    counts["native_source"] = len(native_source_states)
    if len(set(counts.values())) != 1 or counts["eb"] <= 0:
        raise ValueError(f"condition state counts differ or are empty: {counts}")
    if pairing.get("num_states") != counts["eb"]:
        raise ValueError("pairing count does not match serialized state count")

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != TASK_PROMPT:
        raise ValueError("runtime native task prompt changed")
    review_dir = Path(args.review_dir)
    records: list[dict[str, object]] = []
    first_policy_poses: dict[tuple[int, str], dict[str, list[float]]] = {}
    cross_condition_bodies: tuple[str, ...] | None = None
    state_diff_records: list[dict] | None = None
    source_to_project_diff_records: list[dict] | None = None
    movable_bodies: tuple[str, ...] | None = None

    for condition in ("eb", "er", "ec"):
        env, task_description = get_libero_env(
            task,
            "openvla",
            resolution=args.resolution,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        if task_description != TASK_PROMPT:
            raise ValueError("runtime policy prompt mismatch")
        try:
            current_movable = _movable_object_bodies(env.sim.model)
            if movable_bodies is None:
                movable_bodies = current_movable
                fixture_bodies = _fixture_root_bodies(env.sim.model, preflight)
                cross_condition_bodies = tuple(
                    sorted(
                        set(current_movable) - {PROTECTED_BODY}
                        | set(fixture_bodies)
                    )
                )
                state_diff_records = _state_diff_audit(env, states)
                source_to_project_diff_records = (
                    _source_to_project_state_diff_audit(
                        env, native_source_states, states["eb"], pairing
                    )
                )
            elif current_movable != movable_bodies:
                raise ValueError("runtime movable inventory changed by condition")
            for episode_idx, state in enumerate(states[condition]):
                env.reset()
                env.set_init_state(state)
                env.sim.forward()
                samples = [_sample(env, current_movable)]
                first_observation = None
                first_policy_view = None
                first_positions = None
                for wait_step in range(1, FORMAL_WAIT_STEPS + CONFIRM_STEPS + 1):
                    env.step(get_libero_dummy_action("openvla"))
                    if wait_step == FORMAL_WAIT_STEPS:
                        # Refresh from the exact stabilized simulator state;
                        # never reuse the object returned by set_init_state.
                        first_observation = _fresh_observation(env)
                    samples.append(_sample(env, current_movable))
                    if wait_step == FORMAL_WAIT_STEPS:
                        first_policy_view = _save_policy_view(
                            env,
                            first_observation,
                            review_dir,
                            condition,
                            episode_idx,
                            args.resolution,
                        )
                        first_positions = {
                            body: np.asarray(
                                env.sim.data.body_xpos[
                                    env.sim.model.body_name2id(body)
                                ],
                                dtype=float,
                            ).tolist()
                            for body in cross_condition_bodies or ()
                        }
                if first_policy_view is None or first_positions is None:
                    raise RuntimeError("exact first policy frame was not captured")
                valid, failures, stability = _evaluate_trace(
                    samples, current_movable
                )
                visibility_ok = (
                    first_policy_view[
                        "wine_bottle_visible_pixels_after_policy_crop"
                    ]
                    >= MIN_VISIBLE_PIXELS
                )
                if not visibility_ok:
                    failures.append("wine_bottle:policy_view_visibility")
                first_policy_poses[(episode_idx, condition)] = first_positions
                records.append(
                    {
                        "episode_idx": episode_idx,
                        "condition": condition,
                        "valid": bool(valid and visibility_ok),
                        "failures": sorted(set(failures)),
                        "stability": stability,
                        "full_wait_trace": samples,
                        "policy_view": first_policy_view,
                    }
                )
        finally:
            env.close()

    if movable_bodies is None or cross_condition_bodies is None:
        raise RuntimeError("runtime body inventory was not captured")
    if state_diff_records is None:
        raise RuntimeError("serialized intervention audit was not captured")
    if source_to_project_diff_records is None:
        raise RuntimeError("source-to-project state audit was not captured")
    cross_condition_records = []
    for episode_idx in range(counts["eb"]):
        for body in cross_condition_bodies:
            eb = np.asarray(
                first_policy_poses[(episode_idx, "eb")][body], dtype=float
            )
            for condition in ("er", "ec"):
                other = np.asarray(
                    first_policy_poses[(episode_idx, condition)][body],
                    dtype=float,
                )
                drift = float(np.linalg.norm(other - eb))
                cross_condition_records.append(
                    {
                        "episode_idx": episode_idx,
                        "condition": condition,
                        "body": body,
                        "first_policy_position_drift_m": drift,
                        "valid": (
                            drift
                            <= MAX_CROSS_CONDITION_INVARIANT_DRIFT_M
                        ),
                    }
                )
    physical_ok = all(record["valid"] for record in records)
    state_diff_ok = all(record["valid"] for record in state_diff_records)
    source_to_project_diff_ok = all(
        record["valid"] for record in source_to_project_diff_records
    )
    invariant_ok = all(record["valid"] for record in cross_condition_records)
    passed = bool(
        physical_ok
        and source_to_project_diff_ok
        and state_diff_ok
        and invariant_ok
    )
    artifact_hashes = {
        _portable(path): _sha256(path)
        for path in required
    }
    manifest = {
        "schema_version": 1,
        "scenario": SCENE_ID,
        "family": FAMILY,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "host_verification": host,
        "task_suite_name": TASK_SUITE,
        "task_id": TASK_ID,
        "task_file": preflight.get("task_file"),
        "task_prompt": TASK_PROMPT,
        "episode_count": counts["eb"],
        "condition_counts": counts,
        "movable_object_bodies": list(movable_bodies),
        "cross_condition_invariant_bodies": list(cross_condition_bodies),
        "formal_reset_sequence": [
            "env.reset",
            "env.set_init_state",
            "sim.forward",
            f"{FORMAL_WAIT_STEPS}_controller_noop_steps",
            "forced_observation_refresh",
            "get_libero_image_policy_preprocessing",
            "first_policy_frame",
            f"{CONFIRM_STEPS}_postwait_confirmation_steps",
        ],
        "frozen_preflight_hash_verification": frozen_hash_verification,
        "thresholds": {
            "max_receptacle_and_wine_tilt_deg_throughout": (
                MAX_RECEPTACLE_TILT_DEG
            ),
            "max_other_object_tilt_deg_throughout": MAX_OTHER_OBJECT_TILT_DEG,
            "max_translation_drift_m_throughout": MAX_TRANSLATION_DRIFT_M,
            "max_restore_entry_linear_speed_mps": (
                MAX_RESTORE_LINEAR_SPEED_MPS
            ),
            "max_restore_entry_angular_speed_radps": (
                MAX_RESTORE_ANGULAR_SPEED_RADPS
            ),
            "stabilization_window_starts_after_controller_noop_step": 1,
            "max_stabilization_linear_speed_mps": (
                MAX_WINDOW_LINEAR_SPEED_MPS
            ),
            "max_stabilization_angular_speed_radps": (
                MAX_WINDOW_ANGULAR_SPEED_RADPS
            ),
            "max_first_policy_linear_speed_mps": (
                MAX_FIRST_POLICY_LINEAR_SPEED_MPS
            ),
            "max_first_policy_angular_speed_radps": (
                MAX_FIRST_POLICY_ANGULAR_SPEED_RADPS
            ),
            "max_support_penetration_m": MAX_SUPPORT_PENETRATION_M,
            "minimum_wine_bottle_policy_pixels": MIN_VISIBLE_PIXELS,
            "max_cross_condition_invariant_drift_m": (
                MAX_CROSS_CONDITION_INVARIANT_DRIFT_M
            ),
        },
        "artifact_sha256": artifact_hashes,
        "serialized_intervention_audit": state_diff_records,
        "native_source_to_project_eb_audit": source_to_project_diff_records,
        "cross_condition_first_policy_audit": cross_condition_records,
        "records": records,
    }
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report_path = Path(args.output_report)
    failed_records = [record for record in records if not record["valid"]]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            (
                "# L1-B3 Task-4 Outcome V2 exact initial-state gate",
                "",
                f"Verdict: **{manifest['verdict']}**",
                "",
                f"- Episodes per condition: `{counts['eb']}`",
                f"- Movable objects checked: `{list(movable_bodies)}`",
                f"- Invalid physical/visibility records: `{len(failed_records)}`",
                f"- Serialized intervention allowlist gate: `{state_diff_ok}`",
                f"- Native-source to project-Eb delta gate: "
                f"`{source_to_project_diff_ok}`",
                f"- Cross-condition invariant gate: `{invariant_ok}`",
                f"- Formal wait / confirmation steps: "
                f"`{FORMAL_WAIT_STEPS}` / `{CONFIRM_STEPS}`",
                f"- Wine-bottle visibility threshold: `>= {MIN_VISIBLE_PIXELS}` pixels",
                f"- Manifest: `{output_path}`",
                "",
                "Human approval remains pending until every paired initial frame and",
                "the corresponding smoke videos have been reviewed.",
                "",
            )
        ),
        encoding="utf-8",
    )
    review_path = _write_human_review_template(review_dir, output_path)
    print(f"Verdict: {manifest['verdict']}")
    print(f"Manifest: {output_path}")
    print(f"Human review record: {review_path}")
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state_dir", default="experiments/robot/libero/tasks"
    )
    parser.add_argument(
        "--preflight_manifest",
        default=(
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_native_preflight.json"
        ),
    )
    parser.add_argument(
        "--preregistration",
        default=(
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_design_prereg.json"
        ),
    )
    parser.add_argument(
        "--review_dir", default="review/L1-B3_task/task4-outcome-v2"
    )
    parser.add_argument(
        "--output_manifest",
        default=(
            "review/L1-B3_task/task4-outcome-v2/"
            "L1-B3-task4-outcome-v2_initial_gate_manifest.json"
        ),
    )
    parser.add_argument(
        "--output_report",
        default=(
            "experiments/logs/"
            "l1b3_task4_outcome_v2_initial_gate.md"
        ),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--fail_on_invalid", action="store_true")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
