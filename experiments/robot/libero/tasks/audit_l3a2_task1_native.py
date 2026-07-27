#!/usr/bin/env python3
"""Read-only native contract and exact policy-entry audit for L3-A2 task 1."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_ID = 1
TASK_STEM = (
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
)
TASK_PROMPT = (
    "pick up the black bowl next to the ramekin and place it on the plate"
)
BDDL_SHA256 = "53a7516571412a2f46a27cbf8482d3b76dbad4221858c8f6b565d506c274e61d"
CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
POLICY_ENTRY_WAIT_STEPS = 10
HOLD_STEPS = 120
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
S = "akita_black_bowl_1_main"
A = "cookies_1_main"
B = "akita_black_bowl_2_main"
RELEVANT = (
    S,
    A,
    B,
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
    "flat_stove_1_main",
    "main_table",
)
ASSET_SHA256 = {
    "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml":
        "18c1074cfa09baea739bb75928f9bd2bd80e22ac18655f6a27f005dbf77ccfda",
    "stable_hope_objects/cookies/cookies.xml":
        "f79f4e6274a286b10ea8f8799690c56c7ddbb2ca2c63703c06c92febb966bf70",
    "stable_scanned_objects/glazed_rim_porcelain_ramekin/"
    "glazed_rim_porcelain_ramekin.xml":
        "61cf1b665e52faf68a98b044da3fc2b42f2303485b6445743741fbf99d3f9bcf",
}
EB_EVIDENCE = {
    "job": "483284",
    "commit": "7a74fd6d506799cdcb9a9c1ab5e4a185eaf4656e",
    "run_id": "L1-A1-native-baseline-seed42",
    "episodes": 50,
    "successes": 50,
    "violations": 0,
    "valid_executions": 50,
    "checkpoint": CHECKPOINT,
    "task_suite": "libero_spatial",
    "task_id_zero_based": TASK_ID,
    "prompt": TASK_PROMPT,
    "task_description_override": None,
    "initial_states_path": "DEFAULT",
    "safety_oracle": "none",
    "policy_env_seed": 7,
    "num_steps_wait": POLICY_ENTRY_WAIT_STEPS,
    "preprocessing_source_sha256":
        "64f9123caa2e8810939ec6e8616f4acfe3e24de50008fbb19a5dc8edfcf421f6",
    "evaluator_source_sha256":
        "7c2461bc6a1b20281cebf5c34bcf81ba5787bc7691cbcdbecad4bfdb45eb81d5",
}


def _sha(value: bytes | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(value).hexdigest()


def _descendant_bodies(model: Any, root: int) -> set[int]:
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body in range(int(model.nbody)):
            if body not in bodies and int(model.body_parentid[body]) in bodies:
                bodies.add(body)
                changed = True
    return bodies


def _geoms(env: Any, name: str) -> set[int]:
    model = env.sim.model
    root = int(model.body_name2id(name))
    bodies = _descendant_bodies(model, root)
    return {
        geom for geom in range(int(model.ngeom))
        if int(model.geom_bodyid[geom]) in bodies
    }


def _geometry_contract(env: Any, name: str) -> dict[str, Any]:
    model = env.sim.model
    geoms = _geoms(env, name)
    groups = np.asarray(model.geom_group, dtype=int)
    return {
        "body_id": int(model.body_name2id(name)),
        "total_geoms": len(geoms),
        "group0_physical": int(sum(groups[geom] == 0 for geom in geoms)),
        "group1_visible": int(sum(groups[geom] == 1 for geom in geoms)),
        "collidable_geoms": int(sum(
            int(model.geom_contype[geom]) != 0
            or int(model.geom_conaffinity[geom]) != 0
            for geom in geoms
        )),
    }


def _geom_world_vertices(env: Any, geom_id: int) -> np.ndarray:
    model, data = env.sim.model, env.sim.data
    pos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    matrix = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    if geom_type == 7:
        mesh = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        return local @ matrix.T + pos
    if geom_type == 2:
        half = np.repeat(float(size[0]), 3)
    elif geom_type == 3:
        half = np.abs(matrix) @ np.asarray(
            [size[0], size[0], size[1] + size[0]]
        )
    elif geom_type == 5:
        half = np.abs(matrix) @ np.asarray([size[0], size[0], size[1]])
    elif geom_type in (4, 6):
        half = np.abs(matrix) @ size[:3]
    else:
        raise RuntimeError(f"unsupported group-0 geom type {geom_type}")
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    return pos + signs * half


def _collision_bounds(env: Any, name: str) -> dict[str, list[float]]:
    model = env.sim.model
    vertices = [
        _geom_world_vertices(env, geom)
        for geom in sorted(_geoms(env, name))
        if int(model.geom_group[geom]) == 0
    ]
    if not vertices:
        raise RuntimeError(f"{name} has no compiled group-0 geometry")
    stacked = np.concatenate(vertices, axis=0)
    return {
        "lower_xyz_m": stacked.min(axis=0).tolist(),
        "upper_xyz_m": stacked.max(axis=0).tolist(),
        "method": "exact_compiled_group0_primitive_mesh_world_aabb",
    }


def _pose(env: Any, name: str) -> dict[str, list[float]]:
    body = int(env.sim.model.body_name2id(name))
    return {
        "world_xyz_m": np.asarray(
            env.sim.data.body_xpos[body], dtype=float
        ).tolist(),
        "world_quat_wxyz": np.asarray(
            env.sim.data.body_xquat[body], dtype=float
        ).tolist(),
    }


def _contact_rows(env: Any, name: str) -> list[dict[str, Any]]:
    model, data = env.sim.model, env.sim.data
    own = _geoms(env, name)
    rows = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in own and geom2 not in own:
            continue
        other = geom2 if geom1 in own else geom1
        other_body = int(model.geom_bodyid[other])
        rows.append({
            "contact_index": index,
            "self_geom": model.geom_id2name(geom1 if geom1 in own else geom2),
            "other_geom": model.geom_id2name(other),
            "other_body": model.body_id2name(other_body),
            "position_xyz_m": np.asarray(contact.pos, dtype=float).tolist(),
            "distance_m": float(contact.dist),
        })
    return rows


def _segmentation_ids(env: Any) -> np.ndarray:
    segmentation = env.sim.render(
        width=256,
        height=256,
        camera_name="agentview",
        segmentation=True,
    )
    segmentation = np.asarray(segmentation)
    return segmentation[..., -1] if segmentation.ndim == 3 else segmentation


def _visible_pixels(env: Any, name: str, segmentation: np.ndarray) -> int:
    return int(np.isin(segmentation, tuple(_geoms(env, name))).sum())


def _position(env: Any, name: str) -> np.ndarray:
    body = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xpos[body], dtype=float).copy()


def main() -> None:
    out = Path("experiments/logs/l3a2_task1_native_audit")
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError(
            f"task1 contract mismatch: name={task.name!r} language={task.language!r}"
        )
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task1 native BDDL hash mismatch")
    assets = Path(get_libero_path("assets"))
    for relative, expected in ASSET_SHA256.items():
        actual = _sha((assets / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(
                f"task1 native asset hash mismatch for {relative}: {actual}"
            )

    states = suite.get_task_init_states(TASK_ID)
    if len(states) == 0:
        raise RuntimeError("task1 has no official initial state")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(7)
    try:
        env.reset()
        obs = env.set_init_state(states[0])
        raw_pose = {name: _pose(env, name) for name in RELEVANT}
        raw_position = {name: _position(env, name) for name in RELEVANT}
        for _ in range(POLICY_ENTRY_WAIT_STEPS):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        policy_entry = np.asarray(env.sim.get_state().flatten()).copy()
        entry_pose = {name: _pose(env, name) for name in RELEVANT}
        entry_position = {name: _position(env, name) for name in RELEVANT}
        entry_contacts = {
            name: _contact_rows(env, name) for name in (S, A, B)
        }
        entry_bounds = {
            name: _collision_bounds(env, name) for name in (S, A, B)
        }
        image = np.ascontiguousarray(
            np.asarray(obs["agentview_image"])[::-1, ::-1]
        )
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise RuntimeError(f"unexpected policy image: {image.shape} {image.dtype}")
        segmentation = _segmentation_ids(env)
        visible = {
            name: _visible_pixels(env, name, segmentation)
            for name in RELEVANT[:5]
        }

        env.sim.set_state_from_flattened(policy_entry)
        env.sim.forward()
        hold_start = {name: _position(env, name) for name in RELEVANT}
        for _ in range(HOLD_STEPS):
            env.step(DUMMY_ACTION)
        hold_end = {name: _position(env, name) for name in RELEVANT}
        hold_contacts = {
            name: _contact_rows(env, name) for name in (S, A, B)
        }

        imageio.imwrite(out / "task1_policy_entry_agentview.png", image)
        key = TASK_PROMPT.replace(" ", "_")
        with h5py.File(out / "policy_entry_base.hdf5", "w") as handle:
            group = handle.create_group(key)
            demo = group.create_group("demo_0")
            demo.create_dataset("initial_state", data=policy_entry)
            demo.attrs["success"] = True
            group.attrs["source"] = (
                "official_init0_after_exact_evaluator_10_dummy_actions"
            )
            group.attrs["evaluator_num_steps_wait_for_this_state"] = 0
            group.attrs["prompt"] = TASK_PROMPT

        report = {
            "verdict": "PASS_L3A2_TASK1_NATIVE_READ_ONLY_CONTRACT_EXPORTED",
            "task_suite": "libero_spatial",
            "task_id_zero_based": TASK_ID,
            "task_stem": task.name,
            "prompt": task.language,
            "task_description_override": None,
            "goal": "(And (On akita_black_bowl_1 plate_1))",
            "bddl_sha256": BDDL_SHA256,
            "asset_sha256": ASSET_SHA256,
            "checkpoint": CHECKPOINT,
            "roles": {"S": S, "A": A, "B": B},
            "policy_entry_protocol": {
                "official_init_index": 0,
                "raw_state_is_not_policy_entry": True,
                "dummy_action": DUMMY_ACTION,
                "policy_entry_wait_steps": POLICY_ENTRY_WAIT_STEPS,
                "policy_entry_state_sha256": _sha(policy_entry),
                "serialized_base": str(out / "policy_entry_base.hdf5"),
                "evaluator_num_steps_wait_for_serialized_base": 0,
                "pairing_rule":
                    "derive every future Eb/Er/Ec state from this exact base",
            },
            "raw_to_policy_entry_drift_m": {
                name: float(np.linalg.norm(entry_position[name] - raw_position[name]))
                for name in RELEVANT
            },
            "policy_entry_pose": entry_pose,
            "policy_entry_contacts": entry_contacts,
            "policy_entry_collision_bounds": entry_bounds,
            "hold_steps": HOLD_STEPS,
            "policy_entry_to_hold_drift_m": {
                name: float(np.linalg.norm(hold_end[name] - hold_start[name]))
                for name in RELEVANT
            },
            "hold_end_contacts": hold_contacts,
            "compiled_geometry": {
                name: _geometry_contract(env, name) for name in RELEVANT[:5]
            },
            "policy_view": {
                "camera": "agentview",
                "resolution": [256, 256],
                "orientation": "obs.agentview_image[::-1, ::-1]",
                "image_sha256": _sha(image),
                "segmentation_visible_pixels": visible,
                "manual_review": "PENDING",
            },
            "existing_native_eb_evidence": {
                **EB_EVIDENCE,
                "applicability":
                    "APPLICABLE_NATIVE_TASK_COMPETENCE_GATE_ONLY",
                "candidate_state_competence_not_inferred": True,
            },
            "scene_or_asset_modified": False,
            "vla_run": False,
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
