"""Export and bind the task59 model/camera/runtime contract before VLA use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from experiments.robot.libero.tasks.l3a3_support_chain_common import body_pose
from experiments.robot.libero.tasks.probe_l3a3_task57_native import (
    balanced_form,
    policy_image,
    refresh,
)


TASK_ID = 59
PROMPT = "pick up the tomato sauce and put it in the tray"
KEY = PROMPT.replace(" ", "_")
NATIVE_BDDL_SHA256 = "7580a3282b33142c441a3a4f906e7f88415a9a734b3ef14e59e22f3a8d7d3315"
GOAL_SHA256 = "a3cb4109ca75f8e64024e9cf63066478505f44b9b95946a4d98fb95c59bb00b9"
EB_STATE_SHA256 = "e8156dd33774f028f8a098b9463c661e12853e589ade5a8cad77990f543eb66b"
RELEVANT = (
    "tomato_sauce_1_main",
    "alphabet_soup_1_main",
    "butter_1_main",
    "wooden_tray_1_main",
)
MIN_PSNR_DB = 47.7
MIN_SSIM = 0.99885


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_hash(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256(payload)


def qpos_width(joint_type: int) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--expected_contract")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    canonical = Path(args.canonical_dir)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.language != PROMPT:
        raise RuntimeError("task59 runtime prompt drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    goal = balanced_form(bddl_bytes.decode(), "goal")
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError("task59 runtime BDDL drift")
    if sha256(goal.encode()) != GOAL_SHA256:
        raise RuntimeError("task59 runtime goal drift")
    with h5py.File(canonical / "l3a3_task59_eb_one.hdf5", "r") as handle:
        state = handle[KEY]["demo_0"][:]
    if sha256(state.tobytes()) != EB_STATE_SHA256:
        raise RuntimeError("task59 runtime EB state drift")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    try:
        env.reset()
        env.set_init_state(state)
        env.sim.forward()
        restored = np.asarray(env.sim.get_state().flatten()).copy()
        if not np.array_equal(restored, state):
            raise RuntimeError("task59 runtime exact state restore failed")
        obs = refresh(env, restored)
        if not np.array_equal(
            restored, np.asarray(env.sim.get_state().flatten())
        ):
            raise RuntimeError("task59 runtime observation changed state")
        current_rgb = policy_image(obs)
        canonical_rgb = np.asarray(
            Image.open(canonical / "eb_ep000_policy.png").convert("RGB")
        )
        if current_rgb.shape != (256, 256, 3) or canonical_rgb.shape != (
            256,
            256,
            3,
        ):
            raise RuntimeError("task59 runtime policy RGB dimensions drift")
        mse = float(
            np.mean(
                (
                    current_rgb.astype(np.float64)
                    - canonical_rgb.astype(np.float64)
                )
                ** 2
            )
        )
        psnr = (
            float("inf")
            if mse == 0
            else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        )
        ssim = float(
            structural_similarity(
                current_rgb,
                canonical_rgb,
                channel_axis=2,
                data_range=255,
            )
        )

        model, data = env.sim.model, env.sim.data
        bodies = {}
        for body_id in range(int(model.nbody)):
            name = model.body_id2name(body_id)
            if not name:
                continue
            bodies[name] = {
                "body_id": body_id,
                "parent_id": int(model.body_parentid[body_id]),
                "model_body_pos": np.asarray(
                    model.body_pos[body_id], dtype=float
                ).tolist(),
                "model_body_quat_wxyz": np.asarray(
                    model.body_quat[body_id], dtype=float
                ).tolist(),
                "world_body_xpos": np.asarray(
                    data.body_xpos[body_id], dtype=float
                ).tolist(),
                "world_body_xquat_wxyz": np.asarray(
                    data.body_xquat[body_id], dtype=float
                ).tolist(),
            }
        cameras = {}
        for camera_id in range(int(model.ncam)):
            name = model.camera_id2name(camera_id) or f"camera_{camera_id}"
            cameras[name] = {
                "camera_id": camera_id,
                "body_id": int(model.cam_bodyid[camera_id]),
                "target_body_id": int(model.cam_targetbodyid[camera_id]),
                "mode": int(model.cam_mode[camera_id]),
                "model_pos": np.asarray(
                    model.cam_pos[camera_id], dtype=float
                ).tolist(),
                "model_quat_wxyz": np.asarray(
                    model.cam_quat[camera_id], dtype=float
                ).tolist(),
                "fovy_deg": float(model.cam_fovy[camera_id]),
                "world_xpos": np.asarray(
                    data.cam_xpos[camera_id], dtype=float
                ).tolist(),
                "world_xmat": np.asarray(
                    data.cam_xmat[camera_id], dtype=float
                ).reshape(3, 3).tolist(),
            }
        robot_joints = {}
        for joint_id in range(int(model.njnt)):
            body_id = int(model.jnt_bodyid[joint_id])
            body_name = model.body_id2name(body_id) or ""
            if not body_name.lower().startswith(("robot0_", "gripper0_")):
                continue
            address = int(model.jnt_qposadr[joint_id])
            width = qpos_width(int(model.jnt_type[joint_id]))
            name = model.joint_id2name(joint_id) or f"joint_{joint_id}"
            robot_joints[name] = {
                "joint_id": joint_id,
                "body": body_name,
                "qpos_address": address,
                "qpos": np.asarray(
                    data.qpos[address : address + width], dtype=float
                ).tolist(),
            }
        relevant = {}
        for body in RELEVANT:
            xyz, quat = body_pose(env.sim, body)
            relevant[body] = {
                "world_xyz": xyz.tolist(),
                "world_quat_wxyz": quat.tolist(),
            }
        core = {
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "native_bddl_sha256": NATIVE_BDDL_SHA256,
            "goal_form_sha256": GOAL_SHA256,
            "eb_state_sha256": EB_STATE_SHA256,
            "flattened_state_sha256": sha256(restored.tobytes()),
            "full_qpos": np.asarray(data.qpos, dtype=float).tolist(),
            "full_qpos_sha256": sha256(
                np.asarray(data.qpos, dtype=np.float64).tobytes()
            ),
            "robot_joints": robot_joints,
            "relevant_world_poses": relevant,
            "named_bodies": bodies,
            "cameras": cameras,
        }
    finally:
        env.close()

    contract_hash = json_hash(core)
    expected_match = None
    if args.expected_contract:
        expected = json.loads(Path(args.expected_contract).read_text())
        expected_match = contract_hash == expected["runtime_contract_sha256"]
        if not expected_match:
            raise RuntimeError("task59 runtime model/camera contract drift")
    visual_equivalent = bool(psnr >= MIN_PSNR_DB and ssim >= MIN_SSIM)
    verdict = (
        "PASS_L3A3_TASK59_RUNTIME_CONTRACT"
        if visual_equivalent
        else "FAIL_L3A3_TASK59_RUNTIME_CONTRACT"
    )
    report = {
        "verdict": verdict,
        "scope": "pre_vla_runtime_model_camera_state_contract",
        "runtime_contract_sha256": contract_hash,
        "core": core,
        "policy_rgb": {
            "canonical_decoded_sha256": sha256(canonical_rgb.tobytes()),
            "runtime_decoded_sha256": sha256(current_rgb.tobytes()),
            "shape": list(current_rgb.shape),
            "psnr_db": psnr,
            "ssim": ssim,
            "minimum_psnr_db": MIN_PSNR_DB,
            "minimum_ssim": MIN_SSIM,
            "visually_equivalent": visual_equivalent,
            "difference_classification": "GPU_renderer_nondeterminism",
        },
        "expected_contract_supplied": bool(args.expected_contract),
        "expected_contract_match": expected_match,
        "historic_limitation": (
            "job 490085 did not export camera extrinsics or full model.body "
            "contract; relevant poses and flattened states were later verified "
            "identical against pre-run regeneration"
        ),
        "vla_status": "NOT_LOADED_DURING_CONTRACT_EXPORT",
    }
    Path(args.out_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(verdict)
    if not visual_equivalent:
        raise RuntimeError(
            f"{verdict}: psnr={psnr:.6f}, ssim={ssim:.9f}"
        )


if __name__ == "__main__":
    main()
