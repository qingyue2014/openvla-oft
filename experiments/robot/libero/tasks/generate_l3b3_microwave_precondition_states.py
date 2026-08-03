"""Generate and gate native-only Eb/Er/Ec states for L3-B3.

This generator stops at the static scene gate.  It writes the exact first
policy observations used after the evaluator-style wait, but it does not
authorize policy smoke or formal evaluation.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    CONDITIONS,
    CONDITION_INTERVENTION_BODY,
    CONDITION_INTERVENTION_KIND,
    CONDITION_LABEL,
    CONSTRUCTION_SETTLE_STEPS,
    DESIGN_VERSION,
    DISTRACTOR_BODY,
    DOOR_BODY,
    DOOR_CLOSED_QPOS,
    DOOR_FULLY_OPEN_QPOS,
    DOOR_JOINT,
    DUMMY_ACTION,
    EXPECTED_INITIAL_PREDICATES,
    FORMAL_WAIT_STEPS,
    INITIAL_GATE_VERDICT,
    INTERVENTION_ALLOWLIST,
    MAX_DISTRACTOR_MUG_TILT_DEG,
    MAX_DOOR_FINAL_SPEED,
    MAX_DOOR_WINDOW_QPOS_DRIFT,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_POST_WAIT_ANGULAR_SPEED_RADPS,
    MAX_POST_WAIT_LINEAR_SPEED_MPS,
    MAX_POST_WAIT_TRANSLATION_M,
    MAX_TARGET_MUG_TILT_DEG,
    MICROWAVE_BODY,
    POST_WAIT_HOLD_STEPS,
    PROJECT_TARGET_LAYOUT_FIELDS,
    PROJECT_TARGET_WORLD_XY,
    SCENE_ID,
    SUITE,
    TABLE_BODY,
    TARGET_BODY,
    TASK_FILE,
    TASK_ID,
    TASK_GOAL,
    TASK_PROMPT,
    body_window_stats,
    door_window_stats,
    flat_free_joint_slices,
    flat_scalar_joint_indices,
    native_bddl_path,
    native_asset_manifest_sha256,
    predicate_state,
    save_state_bundle,
    scalar_joint_addresses,
    scene_measurement,
    sha256_path,
    state_sha256,
    validate_native_bddl,
    validate_runtime_inventory,
    validate_serialized_intervention,
    verify_native_asset_provenance,
)
from experiments.robot.libero.tasks.validate_l3b3_microwave_design import (
    validate_spec as validate_design_preregistration,
)
from experiments.robot.pi05_utils import PI05_IMAGE_SIZE, resize_with_pad


FIXTURE_REPLAY_BODIES = (TABLE_BODY, MICROWAVE_BODY)
POLICY_IMAGE_SPECS = {
    "agentview_raw_256": (256, 256),
    "wrist_raw_256": (256, 256),
    "agentview_pi05_224": (224, 224),
    "wrist_pi05_224": (224, 224),
}
MIN_VISIBLE_CHANGED_PIXELS = 100
MIN_VISIBLE_MEAN_ABS_ERROR = 0.5


def _apply_common_project_layout(model, official_state: np.ndarray):
    """Relocate the native target identically in all three conditions."""
    official = np.asarray(official_state, dtype=float)
    project = official.copy()
    qpos_slice, _ = flat_free_joint_slices(model, TARGET_BODY)
    before = project[qpos_slice].copy()
    project[qpos_slice.start] = PROJECT_TARGET_WORLD_XY[0]
    project[qpos_slice.start + 1] = PROJECT_TARGET_WORLD_XY[1]
    changed = set(np.flatnonzero(project != official).tolist())
    allowed = {qpos_slice.start, qpos_slice.start + 1}
    if changed != allowed:
        raise ValueError(
            "common target layout delta must change exactly target qpos x/y: "
            f"changed={sorted(changed)} expected={sorted(allowed)}"
        )
    return project, {
        "kind": "common_native_target_free_joint_translation",
        "object": "white_yellow_mug_1",
        "body": TARGET_BODY,
        "fields": list(PROJECT_TARGET_LAYOUT_FIELDS),
        "changed_flat_state_indices": sorted(changed),
        "official_free_joint_qpos_xyz_wxyz": before.tolist(),
        "project_free_joint_qpos_xyz_wxyz": project[qpos_slice].tolist(),
        "project_target_world_xy": list(PROJECT_TARGET_WORLD_XY),
        "identical_across_conditions": True,
        "asset_modified": False,
    }


def _refresh_observation(env):
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def _is_camera_observable(name: str) -> bool:
    lowered = str(name).lower()
    return any(token in lowered for token in ("image", "depth", "segmentation"))


@contextmanager
def _camera_observables_disabled(env):
    observables = getattr(env.env, "_observables", {})
    changed = []
    for name, observable in observables.items():
        if _is_camera_observable(name) and observable.is_enabled():
            observable.set_enabled(False)
            changed.append(observable)
    try:
        yield
    finally:
        for observable in changed:
            observable.set_enabled(True)


def _fast_noop_steps(env, count: int) -> None:
    """Run real controller no-ops while suppressing unused camera renders."""
    with _camera_observables_disabled(env):
        for _ in range(int(count)):
            env.step(DUMMY_ACTION)


def _fixture_snapshot(env) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, positions, quaternions = [], [], []
    for body_name in FIXTURE_REPLAY_BODIES:
        body_id = int(env.sim.model.body_name2id(body_name))
        names.append(body_name)
        positions.append(
            np.asarray(env.sim.model.body_pos[body_id], dtype=float).copy()
        )
        quaternions.append(
            np.asarray(env.sim.model.body_quat[body_id], dtype=float).copy()
        )
    return names, np.asarray(positions), np.asarray(quaternions)


def _restore_fixtures(env, names, positions, quaternions) -> None:
    for index, body_name in enumerate(names):
        body_id = int(env.sim.model.body_name2id(body_name))
        env.sim.model.body_pos[body_id] = positions[index]
        env.sim.model.body_quat[body_id] = quaternions[index]
    env.sim.forward()


def _reset_to(env, state, names, positions, quaternions):
    env.reset()
    _restore_fixtures(env, names, positions, quaternions)
    return env.set_init_state(np.asarray(state, dtype=float))


def _policy_images(observation) -> dict[str, np.ndarray]:
    agent = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    return {
        "agentview_raw_256": agent,
        "wrist_raw_256": wrist,
        "agentview_pi05_224": resize_with_pad(agent, PI05_IMAGE_SIZE),
        "wrist_pi05_224": resize_with_pad(wrist, PI05_IMAGE_SIZE),
    }


def _write_images(
    review_dir: Path,
    condition: str,
    index: int,
    images: dict[str, np.ndarray],
) -> dict[str, str]:
    result = {}
    for label, image in images.items():
        path = (
            review_dir
            / "initial"
            / condition
            / f"{condition}_state{index:03d}_{label}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.asarray(image, dtype=np.uint8))
        result[label] = str(path.resolve())
    return result


def _trusted_native_states(task) -> tuple[np.ndarray, Path]:
    import torch
    from libero.libero import get_libero_path

    path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    ).resolve(strict=True)
    expected_name = f"{Path(task.bddl_file).stem}.pruned_init"
    if path.name != expected_name:
        raise ValueError(f"unexpected official init-state file: {path.name!r}")
    return np.asarray(torch.load(path, weights_only=False)), path


def _runtime_task():
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.bddl_file != TASK_FILE or task.language != TASK_PROMPT:
        raise ValueError("runtime task id does not resolve to locked native task")
    path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    ).resolve(strict=True)
    validate_native_bddl(path)
    return task, path


def _intervene_door_joint(
    env,
    base,
    fixture_names,
    fixture_positions,
    fixture_quaternions,
    *,
    condition: str,
    target_qpos: float,
):
    if condition not in ("closed_microwave", "open_control"):
        raise ValueError(condition)
    _reset_to(
        env, base, fixture_names, fixture_positions, fixture_quaternions
    )
    qpos_address, qvel_address = scalar_joint_addresses(
        env.sim.model, DOOR_JOINT
    )
    with _camera_observables_disabled(env):
        for _ in range(CONSTRUCTION_SETTLE_STEPS):
            env.sim.data.qpos[qpos_address] = target_qpos
            env.sim.data.qvel[qvel_address] = 0.0
            env.sim.forward()
            env.step(DUMMY_ACTION)
    env.sim.data.qpos[qpos_address] = target_qpos
    env.sim.data.qvel[qvel_address] = 0.0
    env.sim.forward()
    settled = np.asarray(env.sim.get_state().flatten(), dtype=float)
    candidate = np.asarray(base, dtype=float).copy()
    qpos_index, qvel_index = flat_scalar_joint_indices(
        env.sim.model, DOOR_JOINT
    )
    candidate[qpos_index] = settled[qpos_index]
    candidate[qvel_index] = settled[qvel_index]
    diff = validate_serialized_intervention(
        env.sim.model, base, candidate, condition
    )
    return candidate, {
        "intervention_body": DOOR_BODY,
        "intervention_kind": CONDITION_INTERVENTION_KIND[condition],
        "joint_name": DOOR_JOINT,
        "target_qpos": target_qpos,
        "settled_qpos": float(settled[qpos_index]),
        "qpos_flat_index": qpos_index,
        "qvel_flat_index": qvel_index,
        "construction_settle_steps": CONSTRUCTION_SETTLE_STEPS,
        "state_diff": diff,
    }


def _support_ok(measure: dict[str, object], expected: str) -> bool:
    contacts = [str(value) for value in measure.get("contacts", [])]
    if expected == "table":
        return TABLE_BODY in contacts
    if expected == "microwave":
        return any(value.startswith("microwave_1") for value in contacts)
    raise ValueError(expected)


def _forbidden_contacts(
    measure: dict[str, object], *, allow_microwave: bool
) -> list[str]:
    failures = []
    for contact in measure.get("contacts", []):
        name = str(contact)
        if name.startswith("robot0_") or name.startswith(
            DISTRACTOR_BODY.removesuffix("_main")
        ):
            failures.append(name)
        elif not allow_microwave and name.startswith("microwave_1"):
            failures.append(name)
    return sorted(set(failures))


def _gate_failures(
    samples: list[dict[str, object]],
    condition: str,
    *,
    post_wait_hold: bool = False,
) -> tuple[list[str], dict[str, object]]:
    stats = {
        TARGET_BODY: body_window_stats(samples, TARGET_BODY),
        DISTRACTOR_BODY: body_window_stats(samples, DISTRACTOR_BODY),
        DOOR_BODY: body_window_stats(samples, DOOR_BODY),
        "door_joint": door_window_stats(samples),
    }
    first = samples[-1]
    failures = []
    for body, tilt_limit in (
        (TARGET_BODY, MAX_TARGET_MUG_TILT_DEG),
        (DISTRACTOR_BODY, MAX_DISTRACTOR_MUG_TILT_DEG),
    ):
        if float(stats[body]["max_tilt_deg"]) > tilt_limit:
            failures.append(f"{body}:tilt")
    if post_wait_hold:
        target_translation = distractor_translation = MAX_POST_WAIT_TRANSLATION_M
        target_linear = distractor_linear = MAX_POST_WAIT_LINEAR_SPEED_MPS
        target_angular = distractor_angular = MAX_POST_WAIT_ANGULAR_SPEED_RADPS
    else:
        target_translation = MAX_NATIVE_WINDOW_TRANSLATION_M
        target_linear = MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
        target_angular = MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
        distractor_translation = MAX_NATIVE_WINDOW_TRANSLATION_M
        distractor_linear = MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
        distractor_angular = MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
    for body, limits in (
        (
            TARGET_BODY,
            (target_translation, target_linear, target_angular),
        ),
        (
            DISTRACTOR_BODY,
            (distractor_translation, distractor_linear, distractor_angular),
        ),
    ):
        translation, linear, angular = limits
        if float(stats[body]["max_translation_drift_m"]) > translation:
            failures.append(f"{body}:translation")
        if float(stats[body]["max_linear_speed_mps"]) > linear:
            failures.append(f"{body}:linear_speed")
        if float(stats[body]["max_angular_speed_radps"]) > angular:
            failures.append(f"{body}:angular_speed")
        if float(first[body]["linear_speed_mps"]) > MAX_FINAL_LINEAR_SPEED_MPS:
            failures.append(f"{body}:first_policy_linear_speed")
        if (
            float(first[body]["angular_speed_radps"])
            > MAX_FINAL_ANGULAR_SPEED_RADPS
        ):
            failures.append(f"{body}:first_policy_angular_speed")
    if not _support_ok(first[TARGET_BODY], "table"):
        failures.append("target:missing_expected_support")
    if not _support_ok(first[DISTRACTOR_BODY], "table"):
        failures.append("distractor:missing_table_support")
    if _forbidden_contacts(
        first[TARGET_BODY], allow_microwave=False
    ):
        failures.append("target:forbidden_contact")
    distractor_contacts = [
        str(value) for value in first[DISTRACTOR_BODY].get("contacts", [])
    ]
    if any(
        name.startswith(("robot0_", "white_yellow_mug_1", "microwave_1"))
        for name in distractor_contacts
    ):
        failures.append("distractor:forbidden_contact")
    if float(stats["door_joint"]["max_qpos_drift"]) > MAX_DOOR_WINDOW_QPOS_DRIFT:
        failures.append("door:qpos_drift")
    if float(first["door_joint"]["speed"]) > MAX_DOOR_FINAL_SPEED:
        failures.append("door:first_policy_speed")
    expected = EXPECTED_INITIAL_PREDICATES[condition]
    if any(sample["predicates"] != expected for sample in samples):
        failures.append("predicate:window_mismatch")
    return sorted(set(failures)), stats


def _formal_gate(
    env,
    state,
    *,
    condition,
    fixture_names,
    fixture_positions,
    fixture_quaternions,
    fast_observation_steps: bool = False,
):
    _reset_to(
        env,
        state,
        fixture_names,
        fixture_positions,
        fixture_quaternions,
    )
    samples = [scene_measurement(env)]
    camera_context = (
        _camera_observables_disabled(env)
        if fast_observation_steps
        else nullcontext()
    )
    with camera_context:
        for _ in range(FORMAL_WAIT_STEPS):
            env.step(DUMMY_ACTION)
            samples.append(scene_measurement(env))
    observation = _refresh_observation(env)
    images = _policy_images(observation)
    failures, stats = _gate_failures(samples, condition)
    hold_samples = [samples[-1]]
    with _camera_observables_disabled(env):
        for _ in range(POST_WAIT_HOLD_STEPS):
            env.step(DUMMY_ACTION)
            hold_samples.append(scene_measurement(env))
    hold_failures, hold_stats = _gate_failures(
        hold_samples, condition, post_wait_hold=True
    )
    failures.extend(f"post_wait:{item}" for item in hold_failures)
    if bool(env.check_success()):
        failures.append("partial_state_unexpectedly_satisfies_full_goal")
    return {
        "condition": condition,
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "pre_wait": samples[0],
        "first_policy": samples[-1],
        "formal_window_stats": stats,
        "post_wait_hold_steps": POST_WAIT_HOLD_STEPS,
        "post_wait_hold_stats": hold_stats,
        "physical_gate_pass": not failures,
        "failures": sorted(set(failures)),
    }, images


def _visibility_metrics(reference: np.ndarray, changed: np.ndarray) -> dict:
    delta = np.abs(
        np.asarray(changed, dtype=np.int16)
        - np.asarray(reference, dtype=np.int16)
    )
    pixel_delta = np.max(delta, axis=2)
    return {
        "changed_pixels_abs_gt_10": int(np.count_nonzero(pixel_delta > 10)),
        "mean_absolute_error": float(np.mean(delta)),
        "max_absolute_error": int(np.max(delta)),
    }


def generate(args) -> dict[str, object]:
    from libero.libero.envs import OffScreenRenderEnv

    task, runtime_bddl = _runtime_task()
    design = validate_design_preregistration(args.design_preregistration)
    native_states, native_init_path = _trusted_native_states(task)
    indices = design["official_state_indices"]
    if args.num_states != len(indices):
        raise ValueError("--num-states must equal preregistered native pool")
    if max(indices) >= len(native_states):
        raise ValueError("preregistered official index exceeds native pool")
    env = OffScreenRenderEnv(
        bddl_file_name=str(runtime_bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(args.seed)
    review_dir = Path(args.review_dir)
    bundles = {condition: [] for condition in CONDITIONS}
    manifest_episodes = []
    try:
        env.reset()
        runtime_inventory = validate_runtime_inventory(env.sim.model)
        for episode_index, native_index in enumerate(indices):
            official_base = np.asarray(
                native_states[native_index], dtype=float
            ).copy()
            base, common_layout_delta = _apply_common_project_layout(
                env.sim.model, official_base
            )
            env.reset()
            fixture_names, fixture_positions, fixture_quaternions = (
                _fixture_snapshot(env)
            )
            states = {"native": base.copy()}
            interventions = {
                "native": {
                    "intervention_body": "",
                    "intervention_kind": "none",
                    "construction_settle_steps": 0,
                    "state_diff": validate_serialized_intervention(
                        env.sim.model, base, base, "native"
                    ),
                }
            }
            states["closed_microwave"], interventions["closed_microwave"] = (
                _intervene_door_joint(
                    env,
                    base,
                    fixture_names,
                    fixture_positions,
                    fixture_quaternions,
                    condition="closed_microwave",
                    target_qpos=DOOR_CLOSED_QPOS,
                )
            )
            states["open_control"], interventions["open_control"] = (
                _intervene_door_joint(
                    env,
                    base,
                    fixture_names,
                    fixture_positions,
                    fixture_quaternions,
                    condition="open_control",
                    target_qpos=DOOR_FULLY_OPEN_QPOS,
                )
            )
            episode = {
                "episode_index": episode_index,
                "native_init_state_index": native_index,
                "official_source_state_sha256": state_sha256(official_base),
                "base_state_sha256": state_sha256(base),
                "common_layout_delta": common_layout_delta,
                "fixture_replay_bodies": fixture_names,
                "fixture_replay_positions": fixture_positions.tolist(),
                "fixture_replay_quaternions": fixture_quaternions.tolist(),
                "conditions": {},
            }
            episode_images = {}
            for condition in CONDITIONS:
                gate, images = _formal_gate(
                    env,
                    states[condition],
                    condition=condition,
                    fixture_names=fixture_names,
                    fixture_positions=fixture_positions,
                    fixture_quaternions=fixture_quaternions,
                )
                episode_images[condition] = images
                image_paths = _write_images(
                    review_dir, condition, episode_index, images
                )
                gate["policy_images"] = image_paths
                if not gate["physical_gate_pass"]:
                    raise ValueError(
                        f"{condition} episode {episode_index} failed: "
                        f"{gate['failures']}"
                    )
                attrs = {
                    "scenario": SCENE_ID,
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "design_version": DESIGN_VERSION,
                    "task_suite_name": SUITE,
                    "task_id": TASK_ID,
                    "task_prompt": TASK_PROMPT,
                    "native_goal_signature": TASK_GOAL,
                    "native_asset_manifest_sha256": native_asset_manifest_sha256(),
                    "intervention_allowlist": INTERVENTION_ALLOWLIST[
                        CONDITION_LABEL[condition]
                    ],
                    "episode_index": episode_index,
                    "native_init_state_index": native_index,
                    "official_source_state_sha256": state_sha256(official_base),
                    "base_state_sha256": state_sha256(base),
                    "common_layout_delta_json": common_layout_delta,
                    "initial_state_sha256": state_sha256(states[condition]),
                    "intervention_body": CONDITION_INTERVENTION_BODY[condition],
                    "intervention_kind": CONDITION_INTERVENTION_KIND[condition],
                    "intervention_json": interventions[condition],
                    "fixture_replay_bodies_json": fixture_names,
                    "fixture_replay_positions": fixture_positions,
                    "fixture_replay_quaternions": fixture_quaternions,
                    "formal_pre_wait_json": gate["pre_wait"],
                    "formal_first_policy_json": gate["first_policy"],
                    "formal_window_stats_json": gate["formal_window_stats"],
                    "post_wait_hold_stats_json": gate["post_wait_hold_stats"],
                    "policy_images_json": image_paths,
                    "physical_gate_pass": True,
                    "asset_inventory_changed": False,
                    "prompt_changed": False,
                    "bddl_changed": False,
                }
                bundles[condition].append(
                    {
                        "initial_state": states[condition],
                        "base_reset_state": base,
                        "attrs": attrs,
                    }
                )
                episode["conditions"][condition] = gate
            episode["visibility"] = {}
            visibility_references = {
                "closed_microwave": "native",
                "open_control": "closed_microwave",
            }
            for condition, reference_condition in visibility_references.items():
                metrics = _visibility_metrics(
                    episode_images[reference_condition]["agentview_pi05_224"],
                    episode_images[condition]["agentview_pi05_224"],
                )
                metrics["reference_condition"] = reference_condition
                metrics["passed"] = bool(
                    metrics["changed_pixels_abs_gt_10"]
                    >= MIN_VISIBLE_CHANGED_PIXELS
                    and metrics["mean_absolute_error"]
                    >= MIN_VISIBLE_MEAN_ABS_ERROR
                )
                if not metrics["passed"]:
                    raise ValueError(
                        f"{condition} episode {episode_index} is not visible "
                        f"in policy RGB: {metrics}"
                    )
                episode["visibility"][condition] = metrics
            manifest_episodes.append(episode)
            print(
                f"PASS episode={episode_index} native_index={native_index} "
                f"er_qpos={interventions['closed_microwave']['settled_qpos']:.6f} "
                f"ec_qpos={interventions['open_control']['settled_qpos']:.6f}",
                flush=True,
            )
    finally:
        env.close()

    outputs = {
        "native": Path(args.native_output),
        "closed_microwave": Path(args.er_output),
        "open_control": Path(args.ec_output),
    }
    for condition, path in outputs.items():
        save_state_bundle(path, condition=condition, records=bundles[condition])
    bddl_record = validate_native_bddl(runtime_bddl)
    asset_provenance = verify_native_asset_provenance()
    result = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene_labels": {
            "Eb": "native",
            "Er": "closed_microwave",
            "Ec": "open_control",
            "Safe": "real_action_open_insert_reclose_from_exact_Er",
        },
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "native_bddl": bddl_record,
        "evaluated_bddl": bddl_record,
        "native_asset_inventory": {
            "fixtures": bddl_record["fixtures"],
            "objects": bddl_record["objects"],
        },
        "runtime_inventory": runtime_inventory,
        "native_asset_provenance": asset_provenance,
        "native_asset_manifest_sha256": native_asset_manifest_sha256(),
        "source_to_project_delta": design["source_to_project_delta"],
        "intervention_allowlist": INTERVENTION_ALLOWLIST,
        "native_init_states_path": str(native_init_path),
        "native_init_states_sha256": sha256_path(native_init_path),
        "official_native_state_indices": indices,
        "design_preregistration": design,
        "design_preregistration_artifact": {
            "path": str(Path(args.design_preregistration).resolve(strict=True)),
            "sha256": sha256_path(args.design_preregistration),
        },
        "count": len(indices),
        "state_bundles": {
            condition: {
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
            }
            for condition, path in outputs.items()
        },
        "episodes": manifest_episodes,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "openvla_exact_224_preprocessing_complete": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": INITIAL_GATE_VERDICT,
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", default=str(native_bddl_path()))
    parser.add_argument("--native-output", required=True)
    parser.add_argument("--er-output", required=True)
    parser.add_argument("--ec-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review-dir", default="review/L3-B3_task")
    parser.add_argument("--num-states", type=int, default=20)
    parser.add_argument(
        "--design-preregistration",
        default=str(
            Path(__file__).with_name("l3b3_microwave_v2_design_prereg.json")
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    if (
        Path(args.bddl).resolve(strict=True)
        != native_bddl_path().resolve(strict=True)
    ):
        raise ValueError("--bddl must be the exact locked native BDDL")
    result = generate(args)
    print(
        f"{result['verdict']} count={result['count']} manifest={args.manifest}"
    )


if __name__ == "__main__":
    main()
