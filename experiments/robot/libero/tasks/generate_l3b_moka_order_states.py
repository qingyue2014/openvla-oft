"""Generate Eb, Er, and Ec states for the native moka sequence probe.

All three conditions use the exact native ``libero_10`` task and prompt.
``near_first`` (Er) and ``far_first`` (Ec) both change only moka pot 1's
free-joint pose/velocity, placing that same pot in the near or far stove slot.
Moka pot 2 remains on the table in both partial conditions. ``native`` (Eb) is
the bit-exact official state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITION_INTERVENTION_BODY,
    CONDITION_LABEL,
    CONDITION_SLOT,
    CONDITIONS,
    CONSTRUCTION_SETTLE_STEPS,
    COOK_SITE,
    DESIGN_VERSION,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_PLACED_WINDOW_TRANSLATION_M,
    MAX_RECEPTACLE_TILT_DEG,
    POST_WAIT_HOLD_STEPS,
    POT_1,
    POT_2,
    POT_BODIES,
    SCENE_ID,
    SLOT_SEPARATION_M,
    STOVE_BODY,
    SUITE,
    TABLE_BODY,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    flat_state_slices,
    free_joint_addresses,
    measurement,
    native_bddl_path,
    save_state_bundle,
    sha256_path,
    validate_native_bddl,
    window_stats,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v5_design import (
    validate_spec as validate_design_preregistration,
)
from experiments.robot.pi05_utils import PI05_IMAGE_SIZE, resize_with_pad


TRACKED_BODIES = (*POT_BODIES, STOVE_BODY)
FIXTURE_REPLAY_BODIES = (TABLE_BODY, STOVE_BODY)


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
        positions.append(
            np.asarray(env.sim.model.body_pos[body_id], dtype=float).copy()
        )
        quaternions.append(
            np.asarray(env.sim.model.body_quat[body_id], dtype=float).copy()
        )
    return names, np.asarray(positions), np.asarray(quaternions)


def _restore_fixtures(
    env,
    names: list[str],
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    for index, body_name in enumerate(names):
        body_id = int(env.sim.model.body_name2id(body_name))
        env.sim.model.body_pos[body_id] = positions[index]
        env.sim.model.body_quat[body_id] = quaternions[index]
    env.sim.forward()


def _policy_images(observation) -> dict[str, np.ndarray]:
    agent = np.ascontiguousarray(
        observation["agentview_image"][::-1, ::-1]
    )
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
    state_index: int,
    images: dict[str, np.ndarray],
) -> dict[str, str]:
    paths = {}
    for label, image in images.items():
        path = (
            review_dir
            / "initial"
            / condition
            / f"{condition}_state{state_index:03d}_{label}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, np.asarray(image, dtype=np.uint8))
        paths[label] = str(path.resolve())
    return paths


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
        raise ValueError(
            f"unexpected official init-state file: {path.name!r}"
        )
    return np.asarray(torch.load(path, weights_only=False)), path


def _runtime_task():
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.bddl_file != TASK_FILE or task.language != TASK_PROMPT:
        raise ValueError(
            "runtime task id does not resolve to the locked native moka task"
        )
    path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    ).resolve(strict=True)
    validate_native_bddl(path)
    return suite, task, path


def _slot_targets(
    env,
    native_state: np.ndarray,
    fixture_names: list[str],
    fixture_positions: np.ndarray,
    fixture_quaternions: np.ndarray,
) -> dict[str, np.ndarray]:
    env.reset()
    _restore_fixtures(
        env, fixture_names, fixture_positions, fixture_quaternions
    )
    observation = env.set_init_state(native_state)
    for _ in range(FORMAL_WAIT_STEPS):
        observation, _, _, _ = env.step(DUMMY_ACTION)
    del observation
    site_id = int(env.sim.model.site_name2id(COOK_SITE))
    center = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
    site_matrix = np.asarray(
        env.sim.data.site_xmat[site_id], dtype=float
    ).reshape(3, 3)
    site_size = np.asarray(env.sim.model.site_size[site_id], dtype=float)
    landing_axis = (
        np.asarray(site_matrix[:2, 0], dtype=float)
        + np.asarray(site_matrix[:2, 1], dtype=float)
    )
    landing_axis /= max(float(np.linalg.norm(landing_axis)), 1e-9)
    targets = {
        "far": np.asarray(
            [
                *(center[:2] - landing_axis * SLOT_SEPARATION_M / 2.0),
                center[2],
            ],
            dtype=float,
        ),
        "near": np.asarray(
            [
                *(center[:2] + landing_axis * SLOT_SEPARATION_M / 2.0),
                center[2],
            ],
            dtype=float,
        ),
    }
    for label, target in targets.items():
        local = site_matrix.T @ (target - center)
        if np.any(np.abs(local[:2]) >= site_size[:2]):
            raise ValueError(
                f"{label} target is outside the native cook region: {local}"
            )
    return {
        **targets,
        "center": center,
        "site_size": site_size,
        "landing_axis_xy": landing_axis,
    }


def _intervene_one_pot(
    env,
    base_state: np.ndarray,
    *,
    body_name: str,
    target_label: str,
    target: np.ndarray,
    fixture_names: list[str],
    fixture_positions: np.ndarray,
    fixture_quaternions: np.ndarray,
) -> tuple[np.ndarray, dict]:
    env.reset()
    _restore_fixtures(
        env, fixture_names, fixture_positions, fixture_quaternions
    )
    env.set_init_state(base_state)
    qpos_address, velocity_address = free_joint_addresses(env, body_name)
    spawn_z = float(target[2] + 0.13)
    env.sim.data.qpos[qpos_address : qpos_address + 3] = np.asarray(
        [target[0], target[1], spawn_z], dtype=float
    )
    # Absolute zero yaw, upright, in MuJoCo wxyz convention.
    env.sim.data.qpos[qpos_address + 3 : qpos_address + 7] = np.asarray(
        [1.0, 0.0, 0.0, 0.0], dtype=float
    )
    env.sim.data.qvel[velocity_address : velocity_address + 6] = 0.0
    env.sim.forward()
    for _ in range(CONSTRUCTION_SETTLE_STEPS):
        env.step(DUMMY_ACTION)
    settled = np.asarray(env.sim.get_state().flatten(), dtype=float)
    candidate = np.asarray(base_state, dtype=float).copy()
    qpos_slice, velocity_slice = flat_state_slices(env, body_name)
    candidate[qpos_slice] = settled[qpos_slice]
    candidate[velocity_slice] = settled[velocity_slice]
    allowed = np.zeros(candidate.shape, dtype=bool)
    allowed[qpos_slice] = True
    allowed[velocity_slice] = True
    if not np.array_equal(candidate[~allowed], base_state[~allowed]):
        raise ValueError("intervention changed non-target serialized state")
    return candidate, {
        "intervention_body": body_name,
        "target_slot": target_label,
        "target_xy": np.asarray(target[:2], dtype=float).tolist(),
        "spawn_z": spawn_z,
        "qpos_flat_start": qpos_slice.start,
        "qvel_flat_start": velocity_slice.start,
        "construction_settle_steps": CONSTRUCTION_SETTLE_STEPS,
    }


def _support_ok(measure: dict, expected: str) -> bool:
    contacts = [str(value) for value in measure["contacts"]]
    if expected == "stove":
        return any(value.startswith("flat_stove_1_") for value in contacts)
    if expected == "table":
        return TABLE_BODY in contacts
    raise ValueError(expected)


def _formal_gate(
    env,
    state: np.ndarray,
    *,
    condition: str,
    fixture_names: list[str],
    fixture_positions: np.ndarray,
    fixture_quaternions: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    env.reset()
    _restore_fixtures(
        env, fixture_names, fixture_positions, fixture_quaternions
    )
    observation = env.set_init_state(state)
    samples = [
        {body: measurement(env, body) for body in TRACKED_BODIES}
    ]
    for _ in range(FORMAL_WAIT_STEPS):
        observation, _, _, _ = env.step(DUMMY_ACTION)
        samples.append(
            {body: measurement(env, body) for body in TRACKED_BODIES}
        )
    first_policy_images = _policy_images(observation)
    post = samples[-1]
    stats = {
        body: window_stats(samples, body) for body in TRACKED_BODIES
    }
    failures = []
    placed_body = CONDITION_INTERVENTION_BODY[condition]
    for body in POT_BODIES:
        body_stats = stats[body]
        placed = body == placed_body
        max_translation = (
            MAX_PLACED_WINDOW_TRANSLATION_M
            if placed
            else MAX_NATIVE_WINDOW_TRANSLATION_M
        )
        max_linear = (
            MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS
            if placed
            else MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
        )
        max_angular = (
            MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS
            if placed
            else MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
        )
        if body_stats["max_translation_drift_m"] > max_translation:
            failures.append(f"{body}:formal_translation")
        if body_stats["max_tilt_deg"] > MAX_RECEPTACLE_TILT_DEG:
            failures.append(f"{body}:formal_tilt")
        if body_stats["max_linear_speed_mps"] > max_linear:
            failures.append(f"{body}:formal_linear_speed")
        if body_stats["max_angular_speed_radps"] > max_angular:
            failures.append(f"{body}:formal_angular_speed")
        final = post[body]
        if final["linear_speed_mps"] > MAX_FINAL_LINEAR_SPEED_MPS:
            failures.append(f"{body}:final_linear_speed")
        if final["angular_speed_radps"] > MAX_FINAL_ANGULAR_SPEED_RADPS:
            failures.append(f"{body}:final_angular_speed")
        expected_support = "stove" if placed else "table"
        if not _support_ok(final, expected_support):
            failures.append(f"{body}:missing_{expected_support}_support")
        other = POT_2 if body == POT_1 else POT_1
        other_prefix = other.removesuffix("_main")
        if any(
            str(contact).startswith(other_prefix)
            for contact in final["contacts"]
        ):
            failures.append(f"{body}:initial_pot_contact")
    if stats[STOVE_BODY]["max_translation_drift_m"] > 1e-6:
        failures.append(f"{STOVE_BODY}:formal_translation")
    if stats[STOVE_BODY]["max_tilt_deg"] > MAX_RECEPTACLE_TILT_DEG:
        failures.append(f"{STOVE_BODY}:formal_tilt")

    # The exact first-policy frame is the acceptance state. Continue with a
    # controller-backed hold to reject latent rocking or delayed tipping.
    hold_samples = [samples[-1]]
    for _ in range(POST_WAIT_HOLD_STEPS):
        env.step(DUMMY_ACTION)
        hold_samples.append(
            {body: measurement(env, body) for body in TRACKED_BODIES}
        )
    hold_stats = {
        body: window_stats(hold_samples, body) for body in TRACKED_BODIES
    }
    for body in POT_BODIES:
        if hold_stats[body]["max_tilt_deg"] > MAX_RECEPTACLE_TILT_DEG:
            failures.append(f"{body}:post_wait_tilt")
        if (
            hold_stats[body]["max_translation_drift_m"]
            > MAX_PLACED_WINDOW_TRANSLATION_M
        ):
            failures.append(f"{body}:post_wait_drift")
    if hold_stats[STOVE_BODY]["max_translation_drift_m"] > 1e-6:
        failures.append(f"{STOVE_BODY}:post_wait_drift")
    if condition == "native" and bool(env.check_success()):
        failures.append("native_state_unexpectedly_satisfies_goal")
    if condition != "native" and bool(env.check_success()):
        failures.append("partial_state_unexpectedly_satisfies_full_goal")

    return (
        {
            "condition": condition,
            "formal_wait_steps": FORMAL_WAIT_STEPS,
            "pre_wait": samples[0],
            "first_policy": post,
            "formal_window_stats": stats,
            "post_wait_hold_steps": POST_WAIT_HOLD_STEPS,
            "post_wait_hold_stats": hold_stats,
            "physical_gate_pass": not failures,
            "failures": sorted(set(failures)),
        },
        first_policy_images,
    )


def _state_sha256(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=float).tobytes()).hexdigest()


def generate(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    _, task, runtime_bddl = _runtime_task()
    native_states, native_init_path = _trusted_native_states(task)
    pool = validate_design_preregistration(args.design_preregistration)
    native_state_indices = pool["official_state_indices"]
    if args.num_states != len(native_state_indices):
        raise ValueError(
            "--num-states must equal the locked capability-conditioned pool "
            f"count {len(native_state_indices)}"
        )
    if max(native_state_indices) >= len(native_states):
        raise ValueError(
            "locked official state index exceeds the native init-state file"
        )
    count = len(native_state_indices)
    env = OffScreenRenderEnv(
        bddl_file_name=str(runtime_bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(args.seed)
    bundles = {condition: [] for condition in CONDITIONS}
    manifest_episodes = []
    review_dir = Path(args.review_dir)
    try:
        for episode_index, native_state_index in enumerate(
            native_state_indices
        ):
            base = np.asarray(
                native_states[native_state_index], dtype=float
            ).copy()
            env.reset()
            (
                fixture_names,
                fixture_positions,
                fixture_quaternions,
            ) = _fixture_snapshot(env)
            slots = _slot_targets(
                env,
                base,
                fixture_names,
                fixture_positions,
                fixture_quaternions,
            )
            states = {"native": base.copy()}
            intervention_meta = {
                "native": {
                    "intervention_body": "",
                    "target_slot": "",
                    "target_xy": [],
                    "construction_settle_steps": 0,
                }
            }
            states["near_first"], intervention_meta["near_first"] = (
                _intervene_one_pot(
                    env,
                    base,
                    body_name=POT_1,
                    target_label="near",
                    target=slots["near"],
                    fixture_names=fixture_names,
                    fixture_positions=fixture_positions,
                    fixture_quaternions=fixture_quaternions,
                )
            )
            states["far_first"], intervention_meta["far_first"] = (
                _intervene_one_pot(
                    env,
                    base,
                    body_name=POT_1,
                    target_label="far",
                    target=slots["far"],
                    fixture_names=fixture_names,
                    fixture_positions=fixture_positions,
                    fixture_quaternions=fixture_quaternions,
                )
            )

            episode_record = {
                "episode_index": episode_index,
                "native_init_state_index": native_state_index,
                "base_state_sha256": _state_sha256(base),
                "fixture_replay_bodies": fixture_names,
                "fixture_replay_positions": fixture_positions.tolist(),
                "fixture_replay_quaternions": fixture_quaternions.tolist(),
                "slots": {
                    "far_xyz": slots["far"].tolist(),
                    "near_xyz": slots["near"].tolist(),
                    "cook_center_xyz": slots["center"].tolist(),
                    "cook_half_size_xyz": slots["site_size"].tolist(),
                    "landing_axis_xy": slots["landing_axis_xy"].tolist(),
                    "separation_m": SLOT_SEPARATION_M,
                },
                "conditions": {},
            }
            for condition in CONDITIONS:
                gate, images = _formal_gate(
                    env,
                    states[condition],
                    condition=condition,
                    fixture_names=fixture_names,
                    fixture_positions=fixture_positions,
                    fixture_quaternions=fixture_quaternions,
                )
                image_paths = _write_images(
                    review_dir,
                    condition,
                    episode_index,
                    images,
                )
                gate["policy_images"] = image_paths
                if not gate["physical_gate_pass"]:
                    raise ValueError(
                        f"{condition} episode {episode_index} "
                        f"(official state {native_state_index}) failed "
                        "physical gate: "
                        f"{gate['failures']}"
                    )
                attrs = {
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "target_slot": CONDITION_SLOT[condition] or "",
                    "design_version": DESIGN_VERSION,
                    "episode_index": episode_index,
                    "native_init_state_index": native_state_index,
                    "base_state_sha256": _state_sha256(base),
                    "initial_state_sha256": _state_sha256(states[condition]),
                    "intervention_body": intervention_meta[condition][
                        "intervention_body"
                    ],
                    "intervention_json": intervention_meta[condition],
                    "slot_targets_json": episode_record["slots"],
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
                episode_record["conditions"][condition] = gate
            manifest_episodes.append(episode_record)
    finally:
        env.close()

    output_paths = {
        "native": Path(args.native_output),
        "near_first": Path(args.near_first_output),
        "far_first": Path(args.far_first_output),
    }
    for condition, path in output_paths.items():
        save_state_bundle(
            path,
            condition=condition,
            records=bundles[condition],
            pool_preregistration=pool,
        )
    bddl_record = validate_native_bddl(runtime_bddl)
    result = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene_labels": {
            "Eb": "native",
            "Er": "near_first",
            "Ec": "far_first",
            "Safe": "real_action_reference_from_Er",
        },
        "diagnostic_only": True,
        "formal_authorized": False,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_bddl": bddl_record,
        "native_init_states_path": str(native_init_path),
        "native_init_states_sha256": sha256_path(native_init_path),
        "asset_inventory_changed": False,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "count": count,
        "official_native_state_indices": native_state_indices,
        "pool_preregistration": pool,
        "state_bundles": {
            condition: {
                "path": str(path.resolve()),
                "sha256": sha256_path(path),
            }
            for condition, path in output_paths.items()
        },
        "episodes": manifest_episodes,
        "verdict": "PASS_L3B_MOKA_INITIAL_PHYSICAL_GATES",
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate native-only provisional L3-B moka order states"
    )
    parser.add_argument(
        "--bddl",
        default=str(native_bddl_path()),
        help="Must resolve to the locked native BDDL",
    )
    parser.add_argument(
        "--native-output",
        default="experiments/robot/libero/tasks/l3b_moka_native_states.hdf5",
    )
    parser.add_argument(
        "--near-first-output",
        default=(
            "experiments/robot/libero/tasks/"
            "l3b_moka_near_first_states.hdf5"
        ),
    )
    parser.add_argument(
        "--far-first-output",
        default=(
            "experiments/robot/libero/tasks/"
            "l3b_moka_far_first_states.hdf5"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "review/L3-B_moka_order_task/"
            "L3-B_moka_initial_gate_manifest.json"
        ),
    )
    parser.add_argument(
        "--review-dir",
        default="review/L3-B_moka_order_task",
    )
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument(
        "--design-preregistration",
        default=(
            "experiments/robot/libero/tasks/"
            "l3b_moka_v5_design_prereg.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    if Path(args.bddl).resolve(strict=True) != native_bddl_path().resolve(
        strict=True
    ):
        raise ValueError("--bddl must be the exact locked native BDDL")
    result = generate(args)
    print(
        f"{result['verdict']} count={result['count']} "
        f"manifest={args.manifest}"
    )


if __name__ == "__main__":
    main()
