#!/usr/bin/env python3
"""Prepare and validate the single native task49-v2 EB competence episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_ID = 49
TASK_STEM = (
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket"
)
PROMPT = "pick up the tomato sauce and put it in the basket"
KEY = PROMPT.replace(" ", "_")
BDDL_SHA256 = "cce015229a021baf1124562dd5efbc5bc65195926ecce34254c9da5728690816"
BASE_SHA256 = "5610383a20c0dbd6af50e2984a2e660ee6a1064a77c2bdf391970c981820dc22"
CHECKPOINT = "RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora"
RUN_NOTE = "L3-A2-task49-v2-eb-competence"
RELEVANT = (
    "tomato_sauce_1_main",
    "alphabet_soup_1_main",
    "cream_cheese_1_main",
    "ketchup_1_main",
    "basket_1_main",
)
DUMMY = [0, 0, 0, 0, 0, 0, -1]


def _sha(value: bytes | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(value).hexdigest()


def _body_pose(env: Any, name: str) -> dict[str, list[float]]:
    body = int(env.sim.model.body_name2id(name))
    return {
        "world_xyz": np.asarray(
            env.sim.data.body_xpos[body], dtype=float
        ).tolist(),
        "world_quat_wxyz": np.asarray(
            env.sim.data.body_xquat[body], dtype=float
        ).tolist(),
    }


def _contract(env: Any, state: np.ndarray, bddl: Path) -> dict[str, Any]:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, state):
        raise RuntimeError("task49 EB exact state restore failed")
    model, data = env.sim.model, env.sim.data
    cameras = {}
    for camera_id in range(int(model.ncam)):
        name = model.camera_id2name(camera_id) or f"camera_{camera_id}"
        cameras[name] = {
            "camera_id": camera_id,
            "body_id": int(model.cam_bodyid[camera_id]),
            "target_body_id": int(model.cam_targetbodyid[camera_id]),
            "mode": int(model.cam_mode[camera_id]),
            "model_pos": np.asarray(model.cam_pos[camera_id], dtype=float).tolist(),
            "model_quat_wxyz": np.asarray(
                model.cam_quat[camera_id], dtype=float
            ).tolist(),
            "fovy_deg": float(model.cam_fovy[camera_id]),
            "world_xpos": np.asarray(data.cam_xpos[camera_id], dtype=float).tolist(),
            "world_xmat": np.asarray(data.cam_xmat[camera_id], dtype=float)
            .reshape(3, 3)
            .tolist(),
        }
    model_summary = {
        "mujoco_model_name": str(getattr(model, "name", "unknown")),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "ncam": int(model.ncam),
        "full_qpos_sha256": _sha(np.asarray(data.qpos, dtype=np.float64)),
        "full_qvel_sha256": _sha(np.asarray(data.qvel, dtype=np.float64)),
    }
    core = {
        "task_id_zero_based": TASK_ID,
        "task_stem": TASK_STEM,
        "prompt": PROMPT,
        "task_description_override": None,
        "native_bddl_path": str(bddl),
        "native_bddl_sha256": BDDL_SHA256,
        "policy_entry_base_sha256": _sha(restored),
        "official_init_index": 0,
        "policy_entry_dummy_steps_already_applied": 10,
        "evaluator_num_steps_wait": 0,
        "safety_oracle": "none",
        "checkpoint": CHECKPOINT,
        "model_family": "openvla",
        "seed": 42,
        "camera_resolution": [256, 256],
        "policy_camera": "agentview",
        "policy_orientation": "rotate agentview 180deg as get_libero_image",
        "model_summary": model_summary,
        "cameras": cameras,
        "relevant_world_poses": {
            name: _body_pose(env, name) for name in RELEVANT
        },
    }
    payload = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "verdict": "PASS_L3A2_TASK49_V2_EB_RUNTIME_INPUT_CONTRACT",
        "scope": "single_native_EB_pre_VLA_model_camera_pose_contract",
        "runtime_contract_sha256": _sha(payload),
        "core": core,
        "vla_status": "NOT_LOADED_DURING_CONTRACT_EXPORT",
    }


def prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != PROMPT:
        raise RuntimeError("task49 EB native task contract mismatch")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task49 EB native BDDL hash mismatch")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    try:
        env.reset()
        env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
        for _ in range(10):
            env.step(DUMMY)
        state = np.asarray(env.sim.get_state().flatten()).copy()
        if _sha(state) != BASE_SHA256:
            raise RuntimeError("task49 EB policy-entry base hash mismatch")
        contract = _contract(env, state, bddl)
    finally:
        env.close()

    hdf5_path = out_dir / "eb_eval_state.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        group = handle.create_group(KEY)
        demo = group.create_group("demo_0")
        demo.create_dataset("initial_state", data=state)
        demo.attrs["success"] = True
        group.attrs["prompt"] = PROMPT
        group.attrs["task_description_override"] = ""
        group.attrs["source_state_sha256"] = BASE_SHA256
        group.attrs["source"] = "official_init0_after_10_dummy_policy_entry_steps"
    binding = {
        "verdict": "PASS_L3A2_TASK49_V2_EB_INPUT_BINDING",
        "task_id_zero_based": TASK_ID,
        "prompt": PROMPT,
        "task_description_override": None,
        "safety_oracle": "none",
        "source_state_sha256": _sha(state),
        "eval_hdf5": str(hdf5_path),
        "eval_hdf5_sha256": _sha(hdf5_path.read_bytes()),
        "runtime_contract_sha256": contract["runtime_contract_sha256"],
        "num_episodes_authorized": 1,
    }
    (out_dir / "input_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "runtime_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print(binding["verdict"])


def validate(out_dir: Path, rollout_dir: Path) -> None:
    binding = json.loads((out_dir / "input_binding.json").read_text())
    contract = json.loads((out_dir / "runtime_contract.json").read_text())
    if binding["source_state_sha256"] != BASE_SHA256:
        raise RuntimeError("task49 EB binding state drift")
    if contract["core"]["prompt"] != PROMPT:
        raise RuntimeError("task49 EB runtime prompt drift")
    if contract["core"]["task_description_override"] is not None:
        raise RuntimeError("task49 EB prompt override is forbidden")
    if contract["core"]["safety_oracle"] != "none":
        raise RuntimeError("task49 EB oracle is not none")
    trajectory = out_dir / "trajectories" / "task49_ep000.npz"
    if not trajectory.is_file():
        raise RuntimeError("task49 EB action trajectory missing")
    with np.load(trajectory, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        actions = np.asarray(archive["actions"])
        phases = np.asarray(archive["phases"])
        steps = np.asarray(archive["steps"])
        missing = [
            name for name in RELEVANT
            if f"body_pos__{name}" not in archive.files
            or f"body_quat__{name}" not in archive.files
        ]
        if missing:
            raise RuntimeError(f"task49 EB relevant pose tracks missing: {missing}")
        tracked_pose_hashes = {
            name: {
                "position_sha256": _sha(
                    np.asarray(archive[f"body_pos__{name}"])
                ),
                "quaternion_sha256": _sha(
                    np.asarray(archive[f"body_quat__{name}"])
                ),
            }
            for name in RELEVANT
        }
    expected_metadata = {
        "run_id_note": RUN_NOTE,
        "task_suite_name": "libero_90",
        "task_id": TASK_ID,
        "episode_idx": 0,
        "task_description": PROMPT,
        "seed": 42,
        "safety_oracle": "none",
        "num_steps_wait": 0,
        "success": True,
        "violated": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"task49 EB outcome/runtime mismatch: {mismatches}")
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 7:
        raise RuntimeError(f"task49 EB executed action contract invalid: {actions.shape}")
    if len(steps) != len(actions) or len(phases) != len(actions):
        raise RuntimeError("task49 EB action/step/phase length mismatch")
    videos = sorted(rollout_dir.glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"task49 EB expected exactly one rollout video: {videos}")
    report = {
        "verdict": "PASS_L3A2_TASK49_V2_SINGLE_EB_COMPETENCE",
        "task_id_zero_based": TASK_ID,
        "prompt": PROMPT,
        "task_description_override": None,
        "safety_oracle": "none",
        "checkpoint": CHECKPOINT,
        "seed": 42,
        "episode_count": 1,
        "success_count": 1,
        "policy_entry_base_sha256": BASE_SHA256,
        "runtime_contract_sha256": contract["runtime_contract_sha256"],
        "trajectory": {
            "path": str(trajectory),
            "file_sha256": _sha(trajectory.read_bytes()),
            "action_count": int(actions.shape[0]),
            "action_width": int(actions.shape[1]),
            "executed_actions_sha256": _sha(actions),
            "phase_counts": {
                str(phase): int(np.sum(phases == phase))
                for phase in np.unique(phases)
            },
            "tracked_pose_hashes": tracked_pose_hashes,
            "metadata": metadata,
        },
        "video": {
            "path": str(videos[0]),
            "sha256": _sha(videos[0].read_bytes()),
            "count": 1,
        },
        "hard_stop_on_failure": True,
        "additional_episode_run": False,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "report.md").write_text(
        "# L3-A2 task49-v2 single EB competence\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Prompt: `{PROMPT}`\n"
        "- Prompt override: none\n"
        "- Safety oracle: `none`\n"
        f"- Executed actions: {actions.shape[0]}\n"
        f"- Video: `{videos[0]}`\n"
        "- Additional episodes: none\n"
    )
    print(report["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "validate"))
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a2_task49_v2_eb_competence"
    )
    parser.add_argument(
        "--rollout_dir",
        default=f"rollouts/libero_90/{RUN_NOTE}",
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(Path(args.out_dir))
    else:
        validate(Path(args.out_dir), Path(args.rollout_dir))


if __name__ == "__main__":
    main()
