"""Generate and gate native-only Eb/Er/Ec states for L3-B bowl order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    BOTTLE_BODY,
    BOWL_BODY,
    BOWL_SPAWN_HEIGHT_M,
    CABINET_BODY,
    CONDITIONS,
    CONDITION_INTERVENTION_BODY,
    CONDITION_INTERVENTION_KIND,
    CONDITION_LABEL,
    CONSTRUCTION_SETTLE_STEPS,
    DESIGN_VERSION,
    DRAWER_BODY,
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT,
    DRAWER_SITE,
    DUMMY_ACTION,
    EXPECTED_INITIAL_PREDICATES,
    FORMAL_WAIT_STEPS,
    INITIAL_GATE_VERDICT,
    MAX_BOTTLE_TILT_DEG,
    MAX_BOWL_TILT_DEG,
    MAX_DRAWER_FINAL_SPEED,
    MAX_DRAWER_CABINET_PENETRATION_M,
    MAX_DRAWER_WINDOW_QPOS_DRIFT,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_PLACED_WINDOW_TRANSLATION_M,
    MAX_POST_WAIT_ANGULAR_SPEED_RADPS,
    MAX_POST_WAIT_LINEAR_SPEED_MPS,
    MAX_POST_WAIT_TRANSLATION_M,
    POST_WAIT_HOLD_STEPS,
    SCENE_ID,
    SUITE,
    TABLE_BODY,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    WINE_RACK_BODY,
    body_window_stats,
    drawer_window_stats,
    flat_free_joint_slices,
    flat_scalar_joint_indices,
    free_joint_addresses,
    native_bddl_path,
    predicate_state,
    save_state_bundle,
    scalar_joint_addresses,
    scene_measurement,
    sha256_path,
    state_sha256,
    validate_native_bddl,
)
from experiments.robot.libero.tasks.validate_l3b_bowl_design import (
    validate_spec as validate_design_preregistration,
)
from experiments.robot.pi05_utils import PI05_IMAGE_SIZE, resize_with_pad


FIXTURE_REPLAY_BODIES = (TABLE_BODY, CABINET_BODY, WINE_RACK_BODY)
POLICY_IMAGE_SPECS = {
    "agentview_raw_256": (256, 256),
    "wrist_raw_256": (256, 256),
    "agentview_pi05_224": (224, 224),
    "wrist_pi05_224": (224, 224),
}
MIN_VISIBLE_CHANGED_PIXELS = 100
MIN_VISIBLE_MEAN_ABS_ERROR = 0.5


def _refresh_observation(env):
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def _fixture_snapshot(env) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, positions, quaternions = [], [], []
    for body_name in FIXTURE_REPLAY_BODIES:
        body_id = int(env.sim.model.body_name2id(body_name))
        names.append(body_name)
        positions.append(np.asarray(env.sim.model.body_pos[body_id], dtype=float).copy())
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


def _write_images(review_dir: Path, condition: str, index: int, images: dict) -> dict:
    result = {}
    for label, image in images.items():
        path = review_dir / "initial" / condition / f"{condition}_state{index:03d}_{label}.png"
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
        raise ValueError("runtime task id does not resolve to the locked native task")
    path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    ).resolve(strict=True)
    validate_native_bddl(path)
    return task, path


def _intervene_closed_drawer(
    env,
    base,
    fixture_names,
    fixture_positions,
    fixture_quaternions,
    *,
    target_qpos,
):
    env.reset()
    _restore_fixtures(env, fixture_names, fixture_positions, fixture_quaternions)
    env.set_init_state(base)
    qpos_address, qvel_address = scalar_joint_addresses(env, DRAWER_JOINT)
    env.sim.data.qpos[qpos_address] = target_qpos
    env.sim.data.qvel[qvel_address] = 0.0
    env.sim.forward()
    for _ in range(CONSTRUCTION_SETTLE_STEPS):
        env.step(DUMMY_ACTION)
    settled = np.asarray(env.sim.get_state().flatten(), dtype=float)
    candidate = np.asarray(base, dtype=float).copy()
    qpos_index, qvel_index = flat_scalar_joint_indices(env, DRAWER_JOINT)
    candidate[qpos_index] = settled[qpos_index]
    candidate[qvel_index] = settled[qvel_index]
    allowed = np.zeros(candidate.shape, dtype=bool)
    allowed[[qpos_index, qvel_index]] = True
    if not np.array_equal(candidate[~allowed], np.asarray(base)[~allowed]):
        raise ValueError("Er changed state outside the drawer scalar joint")
    return candidate, {
        "intervention_body": DRAWER_BODY,
        "intervention_kind": "drawer_joint_only",
        "joint_name": DRAWER_JOINT,
        "target_qpos": target_qpos,
        "settled_qpos": float(settled[qpos_index]),
        "qpos_flat_index": qpos_index,
        "qvel_flat_index": qvel_index,
        "construction_settle_steps": CONSTRUCTION_SETTLE_STEPS,
    }


def _intervene_bowl_in_drawer(env, base, fixture_names, fixture_positions, fixture_quaternions):
    env.reset()
    _restore_fixtures(env, fixture_names, fixture_positions, fixture_quaternions)
    env.set_init_state(base)
    site_id = int(env.sim.model.site_name2id(DRAWER_SITE))
    site_center = np.asarray(env.sim.data.site_xpos[site_id], dtype=float).copy()
    spawn = site_center.copy()
    spawn[2] += BOWL_SPAWN_HEIGHT_M
    qpos_address, qvel_address = free_joint_addresses(env, BOWL_BODY)
    drawer_qpos_address, drawer_qvel_address = scalar_joint_addresses(
        env, DRAWER_JOINT
    )
    native_drawer_qpos = float(env.sim.data.qpos[drawer_qpos_address])
    env.sim.data.qpos[qpos_address : qpos_address + 3] = spawn
    env.sim.data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    env.sim.data.qvel[qvel_address : qvel_address + 6] = 0.0
    env.sim.forward()
    for _ in range(CONSTRUCTION_SETTLE_STEPS):
        # Keep the open native drawer at its exact paired position while the
        # bowl settles.  Only the bowl free joint is later copied into Ec.
        env.sim.data.qpos[drawer_qpos_address] = native_drawer_qpos
        env.sim.data.qvel[drawer_qvel_address] = 0.0
        env.sim.forward()
        env.step(DUMMY_ACTION)
    settled = np.asarray(env.sim.get_state().flatten(), dtype=float)
    candidate = np.asarray(base, dtype=float).copy()
    qpos_slice, qvel_slice = flat_free_joint_slices(env, BOWL_BODY)
    candidate[qpos_slice] = settled[qpos_slice]
    candidate[qvel_slice] = settled[qvel_slice]
    allowed = np.zeros(candidate.shape, dtype=bool)
    allowed[qpos_slice] = True
    allowed[qvel_slice] = True
    if not np.array_equal(candidate[~allowed], np.asarray(base)[~allowed]):
        raise ValueError("Ec changed state outside the bowl free joint")
    return candidate, {
        "intervention_body": BOWL_BODY,
        "intervention_kind": "bowl_free_joint_only",
        "target_site": DRAWER_SITE,
        "site_center_xyz": site_center.tolist(),
        "spawn_xyz": spawn.tolist(),
        "settled_xyz": settled[qpos_slice][:3].tolist(),
        "qpos_flat_start": qpos_slice.start,
        "qvel_flat_start": qvel_slice.start,
        "construction_settle_steps": CONSTRUCTION_SETTLE_STEPS,
        "drawer_joint_locked_during_construction": True,
        "native_drawer_qpos": native_drawer_qpos,
    }


def _support_ok(measure: dict, support: str) -> bool:
    contacts = [str(value) for value in measure.get("contacts", [])]
    if support == "table":
        return TABLE_BODY in contacts
    if support == "drawer":
        return any(value.startswith("white_cabinet_1_cabinet_bottom") for value in contacts)
    raise ValueError(support)


def _forbidden_contacts(measure: dict, *, allow_cabinet: bool) -> list[str]:
    forbidden = []
    for contact in measure.get("contacts", []):
        name = str(contact)
        if name.startswith("robot0_") or name.startswith("wine_bottle_1"):
            forbidden.append(name)
        elif not allow_cabinet and name.startswith("white_cabinet_1"):
            forbidden.append(name)
    return sorted(set(forbidden))


def _gate_failures(
    samples: list[dict], condition: str, *, post_wait_hold: bool = False
) -> tuple[list[str], dict]:
    stats = {
        BOWL_BODY: body_window_stats(samples, BOWL_BODY),
        BOTTLE_BODY: body_window_stats(samples, BOTTLE_BODY),
        DRAWER_BODY: body_window_stats(samples, DRAWER_BODY),
        "drawer_joint": drawer_window_stats(samples),
    }
    first = samples[-1]
    failures = []
    bowl_placed = condition == "prerequisite_done"
    bowl_stats = stats[BOWL_BODY]
    bottle_stats = stats[BOTTLE_BODY]
    if bowl_stats["max_tilt_deg"] > MAX_BOWL_TILT_DEG:
        failures.append("bowl:tilt")
    if bottle_stats["max_tilt_deg"] > MAX_BOTTLE_TILT_DEG:
        failures.append("bottle:tilt")
    if post_wait_hold:
        bowl_translation_limit = MAX_POST_WAIT_TRANSLATION_M
        bowl_linear_limit = MAX_POST_WAIT_LINEAR_SPEED_MPS
        bowl_angular_limit = MAX_POST_WAIT_ANGULAR_SPEED_RADPS
    else:
        bowl_translation_limit = (
            MAX_PLACED_WINDOW_TRANSLATION_M
            if bowl_placed
            else MAX_NATIVE_WINDOW_TRANSLATION_M
        )
        bowl_linear_limit = (
            MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS
            if bowl_placed
            else MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
        )
        bowl_angular_limit = (
            MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS
            if bowl_placed
            else MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
        )
    if bowl_stats["max_translation_drift_m"] > bowl_translation_limit:
        failures.append("bowl:translation")
    if bowl_stats["max_linear_speed_mps"] > bowl_linear_limit:
        failures.append("bowl:linear_speed")
    if bowl_stats["max_angular_speed_radps"] > bowl_angular_limit:
        failures.append("bowl:angular_speed")
    bottle_translation_limit = (
        MAX_POST_WAIT_TRANSLATION_M
        if post_wait_hold
        else MAX_NATIVE_WINDOW_TRANSLATION_M
    )
    bottle_linear_limit = (
        MAX_POST_WAIT_LINEAR_SPEED_MPS
        if post_wait_hold
        else MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
    )
    bottle_angular_limit = (
        MAX_POST_WAIT_ANGULAR_SPEED_RADPS
        if post_wait_hold
        else MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
    )
    if bottle_stats["max_translation_drift_m"] > bottle_translation_limit:
        failures.append("bottle:translation")
    if bottle_stats["max_linear_speed_mps"] > bottle_linear_limit:
        failures.append("bottle:linear_speed")
    if bottle_stats["max_angular_speed_radps"] > bottle_angular_limit:
        failures.append("bottle:angular_speed")
    for body in (BOWL_BODY, BOTTLE_BODY):
        if first[body]["linear_speed_mps"] > MAX_FINAL_LINEAR_SPEED_MPS:
            failures.append(f"{body}:first_policy_linear_speed")
        if first[body]["angular_speed_radps"] > MAX_FINAL_ANGULAR_SPEED_RADPS:
            failures.append(f"{body}:first_policy_angular_speed")
    if not _support_ok(first[BOWL_BODY], "drawer" if bowl_placed else "table"):
        failures.append("bowl:missing_expected_support")
    if not _support_ok(first[BOTTLE_BODY], "table"):
        failures.append("bottle:missing_table_support")
    if _forbidden_contacts(first[BOWL_BODY], allow_cabinet=bowl_placed):
        failures.append("bowl:forbidden_contact")
    if any(
        str(name).startswith(("robot0_", "akita_black_bowl_1"))
        for name in first[BOTTLE_BODY].get("contacts", [])
    ):
        failures.append("bottle:forbidden_contact")
    if stats["drawer_joint"]["max_qpos_drift"] > MAX_DRAWER_WINDOW_QPOS_DRIFT:
        failures.append("drawer:qpos_drift")
    if first["drawer_joint"]["speed"] > MAX_DRAWER_FINAL_SPEED:
        failures.append("drawer:first_policy_speed")
    if any(
        float(contact["distance_m"]) < -MAX_DRAWER_CABINET_PENETRATION_M
        for sample in samples
        for contact in sample["drawer_cabinet_self_contacts"]
    ):
        failures.append("drawer:cabinet_self_contact")
    expected = EXPECTED_INITIAL_PREDICATES[condition]
    if any(sample["predicates"] != expected for sample in samples):
        failures.append("predicate:formal_window_mismatch")
    return sorted(set(failures)), stats


def _formal_gate(env, state, *, condition, fixture_names, fixture_positions, fixture_quaternions):
    env.reset()
    _restore_fixtures(env, fixture_names, fixture_positions, fixture_quaternions)
    observation = env.set_init_state(state)
    samples = [scene_measurement(env)]
    for _ in range(FORMAL_WAIT_STEPS):
        observation, _, _, _ = env.step(DUMMY_ACTION)
        samples.append(scene_measurement(env))
    images = _policy_images(observation)
    failures, stats = _gate_failures(samples, condition)
    hold_samples = [samples[-1]]
    for _ in range(POST_WAIT_HOLD_STEPS):
        env.step(DUMMY_ACTION)
        hold_samples.append(scene_measurement(env))
    hold_failures, hold_stats = _gate_failures(
        hold_samples, condition, post_wait_hold=True
    )
    failures.extend(f"post_wait:{item}" for item in hold_failures)
    if bool(env.check_success()):
        failures.append("partial_state_unexpectedly_satisfies_full_goal")
    predicate_diagnostics = {}
    if any("predicate:formal_window_mismatch" in item for item in failures):
        expected = EXPECTED_INITIAL_PREDICATES[condition]

        def _trace(samples):
            return [
                {
                    "step": step,
                    "drawer_qpos": float(sample["drawer_joint"]["qpos"]),
                    "drawer_speed": float(sample["drawer_joint"]["speed"]),
                    "predicates": sample["predicates"],
                    "matches_expected": sample["predicates"] == expected,
                }
                for step, sample in enumerate(samples)
            ]

        predicate_diagnostics = {
            "expected": expected,
            "formal_window": _trace(samples),
            "post_wait_hold": _trace(hold_samples),
        }
    return {
        "condition": condition,
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "pre_wait": samples[0],
        "first_policy": samples[-1],
        "formal_window_stats": stats,
        "post_wait_hold_steps": POST_WAIT_HOLD_STEPS,
        "post_wait_hold_stats": hold_stats,
        "predicate_diagnostics": predicate_diagnostics,
        "physical_gate_pass": not failures,
        "failures": sorted(set(failures)),
    }, images


def _visibility_metrics(reference: np.ndarray, changed: np.ndarray) -> dict:
    delta = np.abs(np.asarray(changed, dtype=np.int16) - np.asarray(reference, dtype=np.int16))
    pixel_delta = np.max(delta, axis=2)
    return {
        "changed_pixels_abs_gt_10": int(np.count_nonzero(pixel_delta > 10)),
        "mean_absolute_error": float(np.mean(delta)),
        "max_absolute_error": int(np.max(delta)),
    }


def generate(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    task, runtime_bddl = _runtime_task()
    design = validate_design_preregistration(args.design_preregistration)
    closed_drawer_target_qpos = float(
        design["physical_thresholds"].get(
            "er_closed_drawer_target_qpos", DRAWER_CLOSED_QPOS
        )
    )
    native_states, native_init_path = _trusted_native_states(task)
    indices = design["official_state_indices"]
    if args.num_states != len(indices):
        raise ValueError("--num-states must equal the preregistered native pool")
    if max(indices) >= len(native_states):
        raise ValueError("preregistered official index exceeds native state count")
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
        for episode_index, native_index in enumerate(indices):
            base = np.asarray(native_states[native_index], dtype=float).copy()
            env.reset()
            fixture_names, fixture_positions, fixture_quaternions = _fixture_snapshot(env)
            states = {"native": base.copy()}
            interventions = {
                "native": {
                    "intervention_body": "",
                    "intervention_kind": "none",
                    "construction_settle_steps": 0,
                }
            }
            states["premature_close"], interventions["premature_close"] = (
                _intervene_closed_drawer(
                    env,
                    base,
                    fixture_names,
                    fixture_positions,
                    fixture_quaternions,
                    target_qpos=closed_drawer_target_qpos,
                )
            )
            states["prerequisite_done"], interventions["prerequisite_done"] = (
                _intervene_bowl_in_drawer(
                    env, base, fixture_names, fixture_positions, fixture_quaternions
                )
            )
            episode = {
                "episode_index": episode_index,
                "native_init_state_index": native_index,
                "base_state_sha256": state_sha256(base),
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
                image_paths = _write_images(review_dir, condition, episode_index, images)
                gate["policy_images"] = image_paths
                if not gate["physical_gate_pass"]:
                    raise ValueError(
                        f"{condition} episode {episode_index} failed: "
                        f"{gate['failures']}; intervention="
                        f"{json.dumps(interventions[condition], sort_keys=True)}; "
                        f"predicate_diagnostics="
                        f"{json.dumps(gate['predicate_diagnostics'], sort_keys=True)}"
                    )
                attrs = {
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "design_version": DESIGN_VERSION,
                    "episode_index": episode_index,
                    "native_init_state_index": native_index,
                    "base_state_sha256": state_sha256(base),
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
                    {"initial_state": states[condition], "base_reset_state": base, "attrs": attrs}
                )
                episode["conditions"][condition] = gate
            episode["visibility"] = {}
            for condition in ("premature_close", "prerequisite_done"):
                metrics = _visibility_metrics(
                    episode_images["native"]["agentview_pi05_224"],
                    episode_images[condition]["agentview_pi05_224"],
                )
                metrics["passed"] = (
                    metrics["changed_pixels_abs_gt_10"] >= MIN_VISIBLE_CHANGED_PIXELS
                    and metrics["mean_absolute_error"] >= MIN_VISIBLE_MEAN_ABS_ERROR
                )
                if not metrics["passed"]:
                    raise ValueError(
                        f"{condition} episode {episode_index} is not visible in policy RGB: {metrics}"
                    )
                episode["visibility"][condition] = metrics
            manifest_episodes.append(episode)
    finally:
        env.close()

    outputs = {
        "native": Path(args.native_output),
        "premature_close": Path(args.er_output),
        "prerequisite_done": Path(args.ec_output),
    }
    for condition, path in outputs.items():
        save_state_bundle(path, condition=condition, records=bundles[condition])
    bddl_record = validate_native_bddl(runtime_bddl)
    result = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene_labels": {
            "Eb": "native",
            "Er": "premature_close",
            "Ec": "prerequisite_done",
            "Safe": "scripted_real_action_reference_from_Er",
        },
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_bddl": bddl_record,
        "native_init_states_path": str(native_init_path),
        "native_init_states_sha256": sha256_path(native_init_path),
        "official_native_state_indices": indices,
        "design_preregistration": design,
        "count": len(indices),
        "state_bundles": {
            condition: {"path": str(path.resolve()), "sha256": sha256_path(path)}
            for condition, path in outputs.items()
        },
        "episodes": manifest_episodes,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "verdict": INITIAL_GATE_VERDICT,
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", default=str(native_bddl_path()))
    parser.add_argument("--native-output", required=True)
    parser.add_argument("--er-output", required=True)
    parser.add_argument("--ec-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review-dir", default="review/L3-B_bowl_order_task")
    parser.add_argument("--num-states", type=int, default=20)
    parser.add_argument(
        "--design-preregistration",
        default=str(Path(__file__).with_name("l3b_bowl_v1_design_prereg.json")),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    if Path(args.bddl).resolve(strict=True) != native_bddl_path().resolve(strict=True):
        raise ValueError("--bddl must be the exact locked native BDDL")
    result = generate(args)
    print(f"{result['verdict']} count={result['count']} manifest={args.manifest}")


if __name__ == "__main__":
    main()
