#!/usr/bin/env python3
"""Read-only native contract, hinge, asset, and policy-view audit for task33."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_ID = 33
TASK_STEM = "KITCHEN_SCENE6_close_the_microwave"
TASK_PROMPT = "close the microwave"
BDDL_SHA256 = "97df87deffb264990bcb06b877deaace02d296c01de7cc5c312c7bfc30da3b00"
ASSET_SHA256 = {
    "articulated_objects/microwave.xml":
        "d06c1e44acb830c529a843e566deedfc5693a6e11f68a3d1d59d49021890fbdf",
    "turbosquid_objects/porcelain_mug/porcelain_mug.xml":
        "cd72828fc40d20ccd8d3fb08ae37fbc87cb9957309eb4e3d04244554491f753e",
    "turbosquid_objects/white_yellow_mug/white_yellow_mug.xml":
        "134bfdb2e605b60609b63c4bae60d833ad027902973dfb6b912768e0870b435e",
}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def _sha(data: bytes | np.ndarray) -> str:
    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data).tobytes()
    return hashlib.sha256(data).hexdigest()


def _one_name(names: list[str], suffix: str) -> str:
    matched = [name for name in names if name and name.endswith(suffix)]
    if len(matched) != 1:
        raise RuntimeError(f"expected one compiled name ending {suffix!r}: {matched}")
    return matched[0]


def _descendant_bodies(model: Any, root: int) -> set[int]:
    result = {root}
    changed = True
    while changed:
        changed = False
        for body in range(int(model.nbody)):
            if body not in result and int(model.body_parentid[body]) in result:
                result.add(body)
                changed = True
    return result


def _geom_groups(model: Any, bodies: set[int]) -> dict[str, int]:
    groups = np.asarray(model.geom_group, dtype=int)
    body_ids = np.asarray(model.geom_bodyid, dtype=int)
    return {
        "group0_physical": int(sum(
            body_ids[index] in bodies and groups[index] == 0
            for index in range(int(model.ngeom))
        )),
        "group1_visible": int(sum(
            body_ids[index] in bodies and groups[index] == 1
            for index in range(int(model.ngeom))
        )),
    }


def _body_xyz(env: Any, body_name: str) -> list[float]:
    body = int(env.sim.model.body_name2id(body_name))
    return np.asarray(env.sim.data.body_xpos[body], dtype=float).tolist()


def _door_pose(env: Any, joint_id: int) -> dict[str, Any]:
    model, data = env.sim.model, env.sim.data
    door_body = int(model.jnt_bodyid[joint_id])
    door_bodies = _descendant_bodies(model, door_body)
    physical = [
        geom for geom in range(int(model.ngeom))
        if int(model.geom_bodyid[geom]) in door_bodies
        and int(model.geom_group[geom]) == 0
    ]
    if not physical:
        raise RuntimeError("compiled microwave door has no group-0 geometry")
    centers = np.asarray([data.geom_xpos[geom] for geom in physical], dtype=float)
    hinge = np.asarray(data.body_xpos[door_body], dtype=float)
    farthest = physical[int(np.argmax(np.linalg.norm(centers - hinge, axis=1)))]
    return {
        "hinge_world_xyz_m": hinge.tolist(),
        "door_collision_centroid_world_xyz_m": centers.mean(axis=0).tolist(),
        "farthest_collision_geom_world_xyz_m":
            np.asarray(data.geom_xpos[farthest], dtype=float).tolist(),
        "door_body_xmat_row_major":
            np.asarray(data.body_xmat[door_body], dtype=float).tolist(),
    }


def main() -> None:
    out = Path("experiments/logs/l3a2_task33_native_audit")
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError(
            f"task33 contract mismatch: name={task.name!r} language={task.language!r}"
        )
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task33 BDDL hash mismatch")
    assets = Path(get_libero_path("assets"))
    asset_hashes = {}
    for relative, expected in ASSET_SHA256.items():
        actual = _sha((assets / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(f"native asset hash mismatch for {relative}: {actual}")
        asset_hashes[relative] = actual

    states = suite.get_task_init_states(TASK_ID)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        image = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise RuntimeError(f"unexpected policy image: {image.shape} {image.dtype}")

        model = env.sim.model
        body_names = [model.body_id2name(i) for i in range(int(model.nbody))]
        joint_names = [model.joint_id2name(i) for i in range(int(model.njnt))]
        microwave = _one_name(body_names, "microwave_1_main")
        porcelain = _one_name(body_names, "porcelain_mug_1_main")
        yellow = _one_name(body_names, "white_yellow_mug_1_main")
        joint_name = _one_name(joint_names, "microwave_1_microjoint")
        joint_id = int(model.joint_name2id(joint_name))
        qadr = int(model.jnt_qposadr[joint_id])
        joint_range = np.asarray(model.jnt_range[joint_id], dtype=float).copy()
        entry_qpos = float(env.sim.data.qpos[qadr])

        fixture_bodies = _descendant_bodies(model, int(model.body_name2id(microwave)))
        mug_groups = {
            porcelain: _geom_groups(
                model, _descendant_bodies(model, int(model.body_name2id(porcelain)))
            ),
            yellow: _geom_groups(
                model, _descendant_bodies(model, int(model.body_name2id(yellow)))
            ),
        }
        door_body = int(model.jnt_bodyid[joint_id])
        door_groups = _geom_groups(model, _descendant_bodies(model, door_body))
        fixture_groups = _geom_groups(model, fixture_bodies)

        kinematics = {}
        for label, value in (
            ("open_limit", float(joint_range[0])),
            ("midpoint", float(joint_range.mean())),
            ("closed_limit", float(joint_range[1])),
        ):
            env.sim.set_state_from_flattened(base)
            env.sim.data.qpos[qadr] = value
            env.sim.data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0
            env.sim.forward()
            kinematics[label] = {"joint_qpos_rad": value, **_door_pose(env, joint_id)}
        env.sim.set_state_from_flattened(base)
        env.sim.forward()

        imageio.imwrite(out / "task33_native_policy_entry_agentview.png", image)
        report = {
            "verdict": "PASS_L3A2_TASK33_NATIVE_READ_ONLY_AUDIT_EXPORTED",
            "scope": "official_native_init_state_0_after_10_dummy_actions",
            "task_id_zero_based": TASK_ID,
            "task_stem": task.name,
            "prompt": task.language,
            "goal": "( :goal ( And ( Close microwave_1 ) ) )",
            "bddl_sha256": BDDL_SHA256,
            "asset_sha256": asset_hashes,
            "settled_state_sha256": _sha(base),
            "policy_image_sha256": _sha(image),
            "policy_image_shape": list(image.shape),
            "policy_image_orientation": "agentview rotated 180deg as get_libero_image",
            "body_xyz_m": {
                microwave: _body_xyz(env, microwave),
                porcelain: _body_xyz(env, porcelain),
                yellow: _body_xyz(env, yellow),
            },
            "compiled_geometry": {
                "microwave_fixture": fixture_groups,
                "microwave_door": door_groups,
                "mugs": mug_groups,
            },
            "door_joint": {
                "name": joint_name,
                "type": "hinge",
                "axis_local": np.asarray(model.jnt_axis[joint_id], dtype=float).tolist(),
                "range_rad": joint_range.tolist(),
                "policy_entry_qpos_rad": entry_qpos,
                "closing_direction": "increasing qpos toward 0 rad",
                "kinematic_samples": kinematics,
            },
            "topology_note":
                "vertical side-hinged door; possible lateral leaning support only, "
                "not a horizontal gravity-support shelf",
            "scene_or_asset_modified": False,
            "vla_run": False,
            "manual_policy_visibility_review": "PENDING",
        }
        (out / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(report["verdict"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
