"""Independent fail-closed artifact and runtime replay gate for L3-B3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.generate_l3b3_microwave_precondition_states import (
    POLICY_IMAGE_SPECS,
    _camera_observables_disabled,
    _gate_failures,
    _policy_images,
    _refresh_observation,
)
from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    CONDITIONS,
    CONDITION_INTERVENTION_BODY,
    CONDITION_INTERVENTION_KIND,
    CONDITION_LABEL,
    DESIGN_VERSION,
    DISTRACTOR_BODY,
    DOOR_BODY,
    DOOR_CLOSED_QPOS,
    DOOR_FULLY_OPEN_QPOS,
    DOOR_JOINT,
    DUMMY_ACTION,
    EXPECTED_FIXTURE_ROOTS,
    EXPECTED_INITIAL_PREDICATES,
    FORMAL_WAIT_STEPS,
    INITIAL_GATE_VERDICT,
    INTERVENTION_ALLOWLIST,
    MAX_DISTRACTOR_MUG_TILT_DEG,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_POST_WAIT_ANGULAR_SPEED_RADPS,
    MAX_POST_WAIT_LINEAR_SPEED_MPS,
    MAX_POST_WAIT_TRANSLATION_M,
    MAX_TARGET_MUG_TILT_DEG,
    PAIRING_METHOD,
    PAIRING_VERDICT,
    POST_WAIT_HOLD_STEPS,
    PROJECT_TARGET_LAYOUT_FIELDS,
    PROJECT_TARGET_WORLD_XY,
    RUNTIME_REPLAY_VERDICT,
    SCENE_ID,
    SUITE,
    TABLE_BODY,
    TARGET_BODY,
    TASK_FILE,
    TASK_ID,
    TASK_GOAL,
    TASK_KEY,
    TASK_PROMPT,
    flat_free_joint_slices,
    flat_scalar_joint_indices,
    inventory_sha256,
    native_bddl_path,
    native_asset_manifest_sha256,
    scene_measurement,
    sha256_path,
    state_sha256,
    validate_native_bddl,
    validate_runtime_inventory,
    validate_serialized_intervention,
    verify_native_asset_provenance,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)


NATIVE_PREFLIGHT_VERDICT = "PASS_L3B3_MICROWAVE_PRECONDITION_NATIVE_ONLY_PREFLIGHT"
EXPECTED_COUNT = 20


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _json_attr(attrs, name: str):
    if name not in attrs:
        raise ValueError(f"missing required attribute {name!r}")
    value = _decode(attrs[name])
    if not isinstance(value, str):
        raise ValueError(f"attribute {name!r} is not encoded JSON")
    return json.loads(value)


def _record(demo) -> dict[str, object]:
    record = {"initial_state": np.asarray(demo["initial_state"][:], dtype=float)}
    for name, value in demo.attrs.items():
        record[name] = _decode(value)
    return record


def _validate_images(paths: dict, context: str) -> dict[str, str]:
    if set(paths) != set(POLICY_IMAGE_SPECS):
        raise ValueError(f"{context} policy image inventory mismatch")
    hashes = {}
    for label, expected_hw in POLICY_IMAGE_SPECS.items():
        path = Path(paths[label]).resolve(strict=True)
        array = np.asarray(imageio.imread(path))
        if tuple(array.shape) != (*expected_hw, 3):
            raise ValueError(f"{context} {label} shape mismatch: {array.shape}")
        if float(np.std(array)) < 1.0:
            raise ValueError(f"{context} {label} is blank or near-constant")
        hashes[label] = sha256_path(path)
    return hashes


def _require_table_support(measure: dict, body: str, context: str) -> None:
    contacts = [str(value) for value in measure[body].get("contacts", [])]
    if TABLE_BODY not in contacts:
        raise ValueError(f"{context} {body} lacks table support")
    forbidden_prefixes = ("robot0_", "microwave_1")
    other = DISTRACTOR_BODY if body == TARGET_BODY else TARGET_BODY
    if any(
        contact.startswith(forbidden_prefixes)
        or contact.startswith(other.removesuffix("_main"))
        for contact in contacts
    ):
        raise ValueError(f"{context} {body} has forbidden first-frame contact")


def _validate_physical_metadata(attrs, condition: str, context: str) -> dict:
    if not bool(attrs.get("physical_gate_pass", False)):
        raise ValueError(f"{context} is missing physical_gate_pass")
    pre = _json_attr(attrs, "formal_pre_wait_json")
    first = _json_attr(attrs, "formal_first_policy_json")
    formal = _json_attr(attrs, "formal_window_stats_json")
    hold = _json_attr(attrs, "post_wait_hold_stats_json")
    required = {TARGET_BODY, DISTRACTOR_BODY, DOOR_BODY, "door_joint"}
    for label, record in (("pre", pre), ("first", first), ("formal", formal), ("hold", hold)):
        if not required.issubset(record):
            raise ValueError(f"{context} {label} physical inventory mismatch")
    for body, limit in (
        (TARGET_BODY, MAX_TARGET_MUG_TILT_DEG),
        (DISTRACTOR_BODY, MAX_DISTRACTOR_MUG_TILT_DEG),
    ):
        if float(formal[body]["max_tilt_deg"]) > limit:
            raise ValueError(f"{context} {body} formal tilt failed")
        if float(hold[body]["max_tilt_deg"]) > limit:
            raise ValueError(f"{context} {body} hold tilt failed")
        for label, stats, limits in (
            (
                "formal",
                formal[body],
                (
                    MAX_NATIVE_WINDOW_TRANSLATION_M,
                    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
                    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
                ),
            ),
            (
                "hold",
                hold[body],
                (
                    MAX_POST_WAIT_TRANSLATION_M,
                    MAX_POST_WAIT_LINEAR_SPEED_MPS,
                    MAX_POST_WAIT_ANGULAR_SPEED_RADPS,
                ),
            ),
        ):
            translation, linear, angular = limits
            if float(stats["max_translation_drift_m"]) > translation:
                raise ValueError(f"{context} {body} {label} translation failed")
            if float(stats["max_linear_speed_mps"]) > linear:
                raise ValueError(f"{context} {body} {label} linear speed failed")
            if float(stats["max_angular_speed_radps"]) > angular:
                raise ValueError(f"{context} {body} {label} angular speed failed")
        _require_table_support(pre, body, context)
        _require_table_support(first, body, context)
    if first.get("predicates") != EXPECTED_INITIAL_PREDICATES[condition]:
        raise ValueError(f"{context} first-policy predicate mismatch")
    qpos = float(first["door_joint"]["qpos"])
    if condition == "closed_microwave" and abs(qpos - DOOR_CLOSED_QPOS) > 1e-12:
        raise ValueError(f"{context} Er door pose mismatch")
    if condition == "open_control" and abs(qpos - DOOR_FULLY_OPEN_QPOS) > 1e-12:
        raise ValueError(f"{context} Ec door pose mismatch")
    return {"pre": pre, "first": first, "formal": formal, "hold": hold}


def _load_bundle(path: Path, condition: str) -> list[dict[str, object]]:
    path = path.resolve(strict=True)
    with h5py.File(path, "r") as handle:
        if set(handle) != {TASK_KEY}:
            raise ValueError(f"{path} task key mismatch")
        group = handle[TASK_KEY]
        expected = {
            "scenario": SCENE_ID,
            "design_version": DESIGN_VERSION,
            "condition": condition,
            "condition_label": CONDITION_LABEL[condition],
            "task_suite_name": SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "task_file": TASK_FILE,
            "count": EXPECTED_COUNT,
            "pairing_method": PAIRING_METHOD,
            "custom_bddl": False,
            "custom_assets": False,
            "prompt_override": False,
            "native_bddl_sha256": sha256_path(native_bddl_path()),
            "native_goal_signature": TASK_GOAL,
            "native_asset_manifest_sha256": native_asset_manifest_sha256(),
            "intervention_allowlist": json.dumps(
                INTERVENTION_ALLOWLIST[CONDITION_LABEL[condition]],
                sort_keys=True,
            ),
        }
        for name, wanted in expected.items():
            actual = _decode(group.attrs.get(name))
            if actual != wanted:
                raise ValueError(f"{path} {name} mismatch: {actual!r} != {wanted!r}")
        if len(group) != EXPECTED_COUNT:
            raise ValueError(f"{path} does not contain exactly 20 episodes")

        records = []
        for index in range(EXPECTED_COUNT):
            name = f"demo_{index}"
            if name not in group:
                raise ValueError(f"{path} missing contiguous {name}")
            demo = group[name]
            context = f"{path}:{name}"
            if set(demo) != {"initial_state", "base_reset_state"}:
                raise ValueError(f"{context} state dataset inventory mismatch")
            initial = np.asarray(demo["initial_state"][:], dtype=float)
            base = np.asarray(demo["base_reset_state"][:], dtype=float)
            if initial.ndim != 1 or initial.shape != base.shape:
                raise ValueError(f"{context} invalid state shape")
            if _decode(demo.attrs.get("condition")) != condition:
                raise ValueError(f"{context} condition mismatch")
            if _decode(demo.attrs.get("base_state_sha256")) != state_sha256(base):
                raise ValueError(f"{context} base hash mismatch")
            if _decode(demo.attrs.get("initial_state_sha256")) != state_sha256(initial):
                raise ValueError(f"{context} initial hash mismatch")
            layout = _json_attr(demo.attrs, "common_layout_delta_json")
            if layout.get("fields") != list(PROJECT_TARGET_LAYOUT_FIELDS):
                raise ValueError(f"{context} common layout fields mismatch")
            if layout.get("project_target_world_xy") != list(PROJECT_TARGET_WORLD_XY):
                raise ValueError(f"{context} common target position mismatch")
            if layout.get("identical_across_conditions") is not True:
                raise ValueError(f"{context} common layout is not paired")
            if layout.get("asset_modified") is not False:
                raise ValueError(f"{context} common layout modified an asset")
            qpos_slice, _ = flat_free_joint_slices(_MODEL_PROXY.model, TARGET_BODY)
            distractor_qpos_slice, _ = flat_free_joint_slices(
                _MODEL_PROXY.model, DISTRACTOR_BODY
            )
            if not np.array_equal(
                base[qpos_slice.start : qpos_slice.start + 2],
                np.asarray(PROJECT_TARGET_WORLD_XY, dtype=float),
            ):
                raise ValueError(f"{context} project-base target x/y mismatch")
            expected_layout_indices = sorted(
                (
                    qpos_slice.start,
                    qpos_slice.start + 1,
                    qpos_slice.start + 2,
                    distractor_qpos_slice.start + 2,
                )
            )
            if layout.get("changed_flat_state_indices") != expected_layout_indices:
                raise ValueError(f"{context} source-to-project index mismatch")
            if not _decode(demo.attrs.get("official_source_state_sha256", "")):
                raise ValueError(f"{context} official source-state hash missing")
            for flag in ("asset_inventory_changed", "prompt_changed", "bddl_changed"):
                if bool(demo.attrs.get(flag, True)):
                    raise ValueError(f"{context} prohibited {flag}=True")
            if _decode(demo.attrs.get("intervention_body", "")) != CONDITION_INTERVENTION_BODY[condition]:
                raise ValueError(f"{context} intervention body mismatch")
            if _decode(demo.attrs.get("intervention_kind", "")) != CONDITION_INTERVENTION_KIND[condition]:
                raise ValueError(f"{context} intervention kind mismatch")
            diff = validate_serialized_intervention(
                _MODEL_PROXY.model,
                base,
                initial,
                condition,
            )
            intervention = _json_attr(demo.attrs, "intervention_json")
            if intervention.get("state_diff") != diff:
                raise ValueError(f"{context} serialized state-diff metadata mismatch")
            fixture_names = _json_attr(demo.attrs, "fixture_replay_bodies_json")
            fixture_positions = np.asarray(demo.attrs["fixture_replay_positions"], dtype=float)
            fixture_quaternions = np.asarray(demo.attrs["fixture_replay_quaternions"], dtype=float)
            if set(fixture_names) != EXPECTED_FIXTURE_ROOTS:
                raise ValueError(f"{context} fixture inventory mismatch")
            if fixture_positions.shape != (len(fixture_names), 3):
                raise ValueError(f"{context} fixture position shape mismatch")
            if fixture_quaternions.shape != (len(fixture_names), 4):
                raise ValueError(f"{context} fixture quaternion shape mismatch")
            physical = _validate_physical_metadata(demo.attrs, condition, context)
            image_paths = _json_attr(demo.attrs, "policy_images_json")
            records.append(
                {
                    "initial": initial,
                    "base": base,
                    "native_index": int(demo.attrs["native_init_state_index"]),
                    "fixture_names": fixture_names,
                    "fixture_positions": fixture_positions,
                    "fixture_quaternions": fixture_quaternions,
                    "physical": physical,
                    "image_paths": image_paths,
                    "image_hashes": _validate_images(image_paths, context),
                    "hdf_record": _record(demo),
                }
            )
    return records


class _ModelProxy:
    model = None


_MODEL_PROXY = _ModelProxy()


def _validate_manifest(path: Path, bundles: dict[str, Path]) -> dict:
    path = path.resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "count": EXPECTED_COUNT,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "openvla_exact_224_preprocessing_complete": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": INITIAL_GATE_VERDICT,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(f"initial manifest {name} mismatch")
    if record.get("intervention_allowlist") != INTERVENTION_ALLOWLIST:
        raise ValueError("initial manifest intervention allowlist mismatch")
    if record.get("native_asset_manifest_sha256") != native_asset_manifest_sha256():
        raise ValueError("initial manifest native asset hash mismatch")
    prereg = Path(__file__).with_name(
        "l3b3_microwave_v6_design_prereg.json"
    ).resolve(strict=True)
    prereg_binding = record.get("design_preregistration_artifact", {})
    if Path(prereg_binding.get("path", "")).resolve() != prereg:
        raise ValueError("initial manifest design preregistration path mismatch")
    if prereg_binding.get("sha256") != sha256_path(prereg):
        raise ValueError("initial manifest design preregistration hash mismatch")
    current_prereg = json.loads(prereg.read_text(encoding="utf-8"))
    if record.get("design_preregistration") != current_prereg:
        raise ValueError("initial manifest design preregistration snapshot mismatch")
    for condition, bundle in bundles.items():
        binding = record.get("state_bundles", {}).get(condition, {})
        if Path(binding.get("path", "")).resolve() != bundle:
            raise ValueError(f"initial manifest {condition} path mismatch")
        if binding.get("sha256") != sha256_path(bundle):
            raise ValueError(f"initial manifest {condition} hash mismatch")
    return record


def _compare_images(actual: dict[str, np.ndarray], paths: dict, context: str) -> dict:
    result = {}
    for label, array in actual.items():
        expected = np.asarray(imageio.imread(Path(paths[label]).resolve(strict=True)), dtype=np.int16)
        observed = np.asarray(array, dtype=np.int16)
        difference = np.abs(observed - expected)
        mean = float(np.mean(difference))
        percentile = float(np.percentile(difference, 99))
        if mean > 1.0 or percentile > 3.0:
            raise ValueError(f"{context} {label} runtime RGB mismatch: mean={mean}, p99={percentile}")
        result[label] = {"mean_absolute_error": mean, "p99_absolute_error": percentile}
    return result


def _validate_openvla_view_gate(
    path: str | Path | None,
    *,
    initial_manifest: Path,
    checkpoint: str | None,
) -> dict[str, object] | None:
    if path is None:
        return None
    from experiments.robot.libero.tasks.certify_l3b3_openvla_policy_views import (
        VERDICT as OPENVLA_VIEW_VERDICT,
    )

    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_prompt": TASK_PROMPT,
        "model_family": "openvla",
        "center_crop": True,
        "image_size": [224, 224],
        "count": 60,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": OPENVLA_VIEW_VERDICT,
    }
    for name, wanted in expected.items():
        if record.get(name) != wanted:
            raise ValueError(f"OpenVLA policy-view gate {name} mismatch")
    if checkpoint is None or record.get("checkpoint") != checkpoint:
        raise ValueError("OpenVLA policy-view checkpoint mismatch")
    binding = record.get("initial_manifest", {})
    if Path(binding.get("path", "")).resolve() != initial_manifest:
        raise ValueError("OpenVLA policy-view initial manifest path mismatch")
    if binding.get("sha256") != sha256_path(initial_manifest):
        raise ValueError("OpenVLA policy-view initial manifest hash mismatch")
    records = record.get("records", [])
    if len(records) != 60:
        raise ValueError("OpenVLA policy-view record count mismatch")
    identities = set()
    for item in records:
        identity = (int(item.get("episode_index", -1)), item.get("condition"))
        if identity in identities:
            raise ValueError(f"duplicate OpenVLA policy-view identity: {identity}")
        identities.add(identity)
        for camera in ("agentview", "wrist"):
            image = item.get("images", {}).get(camera, {})
            raw = Path(image.get("raw_path", "")).resolve(strict=True)
            processed = Path(image.get("path", "")).resolve(strict=True)
            if image.get("raw_sha256") != sha256_path(raw):
                raise ValueError("OpenVLA raw policy-view hash mismatch")
            if image.get("sha256") != sha256_path(processed):
                raise ValueError("OpenVLA processed policy-view hash mismatch")
            array = np.asarray(imageio.imread(processed))
            if array.shape != (224, 224, 3) or float(np.std(array)) < 1.0:
                raise ValueError("invalid OpenVLA processed policy view")
    expected_identities = {
        (index, condition) for index in range(20) for condition in CONDITIONS
    }
    if identities != expected_identities:
        raise ValueError("OpenVLA policy-view episode inventory mismatch")
    visibility = record.get("visibility", [])
    if len(visibility) != 40 or not all(item.get("passed") is True for item in visibility):
        raise ValueError("OpenVLA door-visibility gate incomplete")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "checkpoint": checkpoint,
        "verdict": OPENVLA_VIEW_VERDICT,
    }


def validate_all(
    *,
    manifest_path: str | Path,
    bundle_paths: dict[str, str | Path],
    render_gpu_device_id: int,
    seed: int,
    openvla_view_gate_path: str | Path | None = None,
    openvla_checkpoint: str | None = None,
) -> tuple[dict, dict, dict]:
    from libero.libero.envs import OffScreenRenderEnv

    native = validate_native_bddl(native_bddl_path())
    asset_provenance = verify_native_asset_provenance()
    bundles = {name: Path(path).resolve(strict=True) for name, path in bundle_paths.items()}
    if set(bundles) != set(CONDITIONS):
        raise ValueError("L3-B3 requires native, closed_microwave, and open_control bundles")
    manifest = _validate_manifest(Path(manifest_path), bundles)
    openvla_view_gate = _validate_openvla_view_gate(
        openvla_view_gate_path,
        initial_manifest=Path(manifest_path).resolve(strict=True),
        checkpoint=openvla_checkpoint,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(native_bddl_path()),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(seed)
    replay_episodes = []
    try:
        env.reset()
        runtime_inventory = validate_runtime_inventory(env.sim.model)
        _MODEL_PROXY.model = env.sim.model
        loaded = {condition: _load_bundle(path, condition) for condition, path in bundles.items()}
        for index in range(EXPECTED_COUNT):
            triplet = [loaded[condition][index] for condition in CONDITIONS]
            base = triplet[0]["base"]
            if any(not np.array_equal(item["base"], base) for item in triplet[1:]):
                raise ValueError(f"paired base mismatch at demo_{index}")
            if [item["native_index"] for item in triplet] != [index] * 3:
                raise ValueError(f"paired native-state index mismatch at demo_{index}")
            for field in ("fixture_names", "fixture_positions", "fixture_quaternions"):
                first = triplet[0][field]
                for item in triplet[1:]:
                    if isinstance(first, list):
                        equal = item[field] == first
                    else:
                        equal = np.array_equal(item[field], first)
                    if not equal:
                        raise ValueError(f"paired {field} mismatch at demo_{index}")

        for condition in CONDITIONS:
            for index, item in enumerate(loaded[condition]):
                env.reset()
                state = materialize_native_scene_state(env, item["hdf_record"])
                env.set_init_state(state)
                samples = [scene_measurement(env)]
                for _ in range(FORMAL_WAIT_STEPS):
                    env.step(DUMMY_ACTION)
                    samples.append(scene_measurement(env))
                observation = _refresh_observation(env)
                failures, formal_stats = _gate_failures(samples, condition)
                hold_samples = [samples[-1]]
                with _camera_observables_disabled(env):
                    for _ in range(POST_WAIT_HOLD_STEPS):
                        env.step(DUMMY_ACTION)
                        hold_samples.append(scene_measurement(env))
                hold_failures, hold_stats = _gate_failures(
                    hold_samples, condition, post_wait_hold=True
                )
                failures.extend(f"post_wait:{value}" for value in hold_failures)
                if bool(env.check_success()):
                    failures.append("partial_state_unexpectedly_satisfies_full_goal")
                if failures:
                    raise ValueError(f"runtime replay {condition} demo_{index} failed: {sorted(set(failures))}")
                image_comparison = _compare_images(
                    _policy_images(observation), item["image_paths"], f"{condition}:demo_{index}"
                )
                replay_episodes.append(
                    {
                        "condition": condition,
                        "episode_index": index,
                        "state_artifact": str(bundles[condition]),
                        "state_artifact_sha256": sha256_path(bundles[condition]),
                        "physical_gate_pass": True,
                        "formal_window_stats": formal_stats,
                        "post_wait_hold_stats": hold_stats,
                        "policy_image_comparison": image_comparison,
                    }
                )
    finally:
        _MODEL_PROXY.model = None
        env.close()

    pairing = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "count": EXPECTED_COUNT,
        "bindings": {
            condition: {"path": str(path), "sha256": sha256_path(path)}
            for condition, path in bundles.items()
        },
        "initial_manifest": {
            "path": str(Path(manifest_path).resolve()),
            "sha256": sha256_path(manifest_path),
            "verdict": manifest["verdict"],
        },
        "verdict": PAIRING_VERDICT,
    }
    replay = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "count": len(replay_episodes),
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "post_wait_hold_steps": POST_WAIT_HOLD_STEPS,
        "episodes": replay_episodes,
        "verdict": RUNTIME_REPLAY_VERDICT,
    }
    preflight = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "native_bddl": native,
        "evaluated_bddl": native,
        "native_asset_provenance": asset_provenance,
        "native_asset_manifest_sha256": native_asset_manifest_sha256(),
        "native_asset_inventory_sha256": inventory_sha256(),
        "runtime_inventory": runtime_inventory,
        "intervention_allowlist": INTERVENTION_ALLOWLIST,
        "source_to_project_delta": manifest["source_to_project_delta"],
        "state_bundles": pairing["bindings"],
        "initial_manifest": pairing["initial_manifest"],
        "pairing_gate_verdict": pairing["verdict"],
        "runtime_replay_verdict": replay["verdict"],
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "openvla_exact_224_preprocessing_complete": openvla_view_gate is not None,
        "openvla_policy_view_gate": openvla_view_gate,
        # The state generator saves and validates the exact resize-with-pad
        # 224x224 agent and wrist images consumed by pi0.5 for every episode.
        # This authorizes only the bounded Eb capability diagnostic below;
        # it does not change the OpenVLA-first formal release sequence.
        "pi05_exact_224_preprocessing_complete": True,
        "pi05_eb_diagnostic_only": True,
        "pi05_formal_authorized": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": NATIVE_PREFLIGHT_VERDICT,
    }
    return pairing, replay, preflight


def verify_evaluation_request(
    manifest_path: str | Path,
    *,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    task_bddl: str | Path,
    policy_prompt: str,
    initial_states_path: str | Path,
) -> dict[str, object]:
    """Bind a smoke/formal evaluator request to an exact qualified artifact."""

    path = Path(manifest_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "verdict": NATIVE_PREFLIGHT_VERDICT,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "openvla_exact_224_preprocessing_complete": True,
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise ValueError(f"L3-B3 preflight {key} mismatch")
    if not isinstance(record.get("openvla_policy_view_gate"), dict):
        raise ValueError("L3-B3 evaluator requires a bound OpenVLA policy-view gate")
    if (task_suite_name, int(task_id)) != (SUITE, TASK_ID):
        raise ValueError("L3-B3 evaluator suite/task mismatch")
    if task_language != TASK_PROMPT or policy_prompt != TASK_PROMPT:
        raise ValueError("L3-B3 evaluator prompt mismatch")
    native = validate_native_bddl(task_bddl)
    if native["bddl_sha256"] != record["native_bddl"]["bddl_sha256"]:
        raise ValueError("L3-B3 evaluator BDDL hash mismatch")
    if native["goal_signature_sha256"] != record["native_bddl"]["goal_signature_sha256"]:
        raise ValueError("L3-B3 evaluator goal signature mismatch")
    states = Path(initial_states_path).resolve(strict=True)
    matches = [
        binding
        for binding in record.get("state_bundles", {}).values()
        if Path(binding.get("path", "")).resolve() == states
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_path(states):
        raise ValueError("L3-B3 evaluator state bundle is not hash-bound")
    provenance = verify_native_asset_provenance()
    if provenance["asset_manifest_sha256"] != record.get("native_asset_manifest_sha256"):
        raise ValueError("L3-B3 evaluator native asset manifest changed")
    return record


def verify_pi05_eb_diagnostic_evaluation_request(
    manifest_path: str | Path,
    *,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    task_bddl: str | Path,
    policy_prompt: str,
    initial_states_path: str | Path,
) -> dict[str, object]:
    """Bind a non-formal pi0.5 Eb diagnostic to the moved-mug artifact.

    The diagnostic answers only whether the learned policy retains native-task
    capability after the common target relocation.  It cannot authorize Er,
    paired smoke, formal evaluation, or the post-formal cascade.
    """

    path = Path(manifest_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "verdict": NATIVE_PREFLIGHT_VERDICT,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "pi05_exact_224_preprocessing_complete": True,
        "pi05_eb_diagnostic_only": True,
        "pi05_formal_authorized": False,
        "human_review_approved": False,
        "formal_authorized": False,
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise ValueError(f"L3-B3 pi0.5 diagnostic preflight {key} mismatch")
    if (task_suite_name, int(task_id)) != (SUITE, TASK_ID):
        raise ValueError("L3-B3 pi0.5 diagnostic suite/task mismatch")
    if task_language != TASK_PROMPT or policy_prompt != TASK_PROMPT:
        raise ValueError("L3-B3 pi0.5 diagnostic prompt mismatch")
    native = validate_native_bddl(task_bddl)
    if native["bddl_sha256"] != record["native_bddl"]["bddl_sha256"]:
        raise ValueError("L3-B3 pi0.5 diagnostic BDDL hash mismatch")
    if (
        native["goal_signature_sha256"]
        != record["native_bddl"]["goal_signature_sha256"]
    ):
        raise ValueError("L3-B3 pi0.5 diagnostic goal signature mismatch")
    states = Path(initial_states_path).resolve(strict=True)
    binding = record.get("state_bundles", {}).get("native", {})
    if (
        Path(binding.get("path", "")).resolve() != states
        or binding.get("sha256") != sha256_path(states)
    ):
        raise ValueError("L3-B3 pi0.5 diagnostic Eb bundle is not hash-bound")
    with h5py.File(states, "r") as handle:
        group = handle[TASK_KEY]
        if _decode(group.attrs.get("condition")) != "native":
            raise ValueError("L3-B3 pi0.5 diagnostic is restricted to Eb/native")
    provenance = verify_native_asset_provenance()
    if (
        provenance["asset_manifest_sha256"]
        != record.get("native_asset_manifest_sha256")
    ):
        raise ValueError("L3-B3 pi0.5 diagnostic native assets changed")
    return record


def verify_runtime_asset_inventory(
    manifest_path: str | Path, model
) -> dict[str, object]:
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    runtime = validate_runtime_inventory(model)
    if runtime != record.get("runtime_inventory"):
        raise ValueError("L3-B3 compiled runtime inventory mismatch")
    return runtime


def _write_json(path: str | Path, record: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--pairing-output", required=True)
    parser.add_argument("--runtime-output", required=True)
    parser.add_argument("--preflight-output", required=True)
    parser.add_argument("--openvla-view-gate")
    parser.add_argument("--openvla-checkpoint")
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pairing, replay, preflight = validate_all(
        manifest_path=args.manifest,
        bundle_paths={"native": args.native, "closed_microwave": args.er, "open_control": args.ec},
        render_gpu_device_id=args.render_gpu_device_id,
        seed=args.seed,
        openvla_view_gate_path=args.openvla_view_gate,
        openvla_checkpoint=args.openvla_checkpoint,
    )
    _write_json(args.pairing_output, pairing)
    _write_json(args.runtime_output, replay)
    _write_json(args.preflight_output, preflight)
    print(
        f"{pairing['verdict']} {replay['verdict']} {preflight['verdict']} "
        "formal_authorized=false"
    )


if __name__ == "__main__":
    main()
