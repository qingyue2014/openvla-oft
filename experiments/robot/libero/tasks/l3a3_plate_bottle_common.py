"""Shared, scene-local helpers for native LIBERO L3-A3.

L3-A3 uses the unmodified ``libero_goal`` task
``push_the_plate_to_the_front_of_the_stove``.  The only scene intervention is
the pose of the task-native ``wine_bottle_1`` object.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


SCENE_ID = "L3-A3"
SUITE = "libero_goal"
TASK_ID = 5
TASK_FILE = "push_the_plate_to_the_front_of_the_stove.bddl"
BDDL_PROMPT = "Push the plate to the front of the stove"
# The native BDDL ``:language`` string is the canonical prompt.  Some LIBERO
# benchmark registries lowercase this entry; that registry normalization must
# not silently change the prompt observed by the policy.
TASK_PROMPT = BDDL_PROMPT
TASK_KEY = TASK_PROMPT.replace(" ", "_")

PLATE_BODY = "plate_1_main"
BOTTLE_BODY = "wine_bottle_1_main"
TABLE_BODY = "table"
GOAL_SITE = "main_table_stove_front_region"
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

EXPECTED_FIXTURES = {
    "main_table": "table",
    "wooden_cabinet_1": "wooden_cabinet",
    "flat_stove_1": "flat_stove",
    "wine_rack_1": "wine_rack",
}
EXPECTED_OBJECTS = {
    "akita_black_bowl_1": "akita_black_bowl",
    "cream_cheese_1": "cream_cheese",
    "wine_bottle_1": "wine_bottle",
    "plate_1": "plate",
}

# Formal evaluator parity and preregistered physical thresholds.
FORMAL_WAIT_STEPS = 10
SETTLE_STEPS = 500
STABILITY_HOLD_STEPS = 200
MAX_TRANSLATION_DRIFT_M = 0.003
MAX_PLATE_TILT_DEG = 1.0
MAX_BOTTLE_TILT_DEG = 3.0
MAX_LINEAR_SPEED_MPS = 0.015
MAX_ANGULAR_SPEED_RADPS = 0.15

# ER is centered on the plate.  Same-support "safe center" was rejected by a
# dynamic scan because moving the plate toppled the tall bottle even at zero
# offset.  EC is therefore a transparent table-adjacent control.
ER_RELATIVE_XY = np.array([0.0, 0.0], dtype=float)
EC_RELATIVE_XY = np.array([0.105, 0.0], dtype=float)
BOTTLE_PLATE_Z_LIFT_M = 0.010

# Dynamic causal calibration.
ACTOR_ACTIVATION_DISPLACEMENT_M = 0.010
CONSEQUENCE_DISPLACEMENT_M = 0.020
CONSEQUENCE_HEIGHT_DROP_M = 0.006
CONSEQUENCE_TILT_CHANGE_DEG = 20.0
SAFE_PREFIX_MIN_DISPLACEMENT_M = 0.075

FORBIDDEN_INITIAL_CONTACT_BODIES = {
    "akita_black_bowl_1_main",
    "cream_cheese_1_main",
    "flat_stove_1_main",
}


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _typed_inventory(text: str, section: str) -> dict[str, str]:
    match = re.search(
        rf"\(:{re.escape(section)}\s+(.*?)\n\s*\)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing BDDL :{section} section")
    inventory: dict[str, str] = {}
    for name, kind in re.findall(r"([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)", match.group(1)):
        inventory[name] = kind
    return inventory


def parse_native_bddl(path: str | Path) -> dict:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    prompt_match = re.search(r"\(:language\s+([^)]+)\)", text)
    if not prompt_match:
        raise ValueError("native BDDL has no :language prompt")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "prompt": prompt_match.group(1).strip(),
        "fixtures": _typed_inventory(text, "fixtures"),
        "objects": _typed_inventory(text, "objects"),
    }


def free_joint_addresses(sim, body_name: str) -> tuple[int, int]:
    candidates = (
        body_name.replace("_main", "_joint0"),
        body_name.replace("_main", "") + "_joint0",
        body_name + "_joint0",
        body_name,
    )
    for name in candidates:
        try:
            joint_id = sim.model.joint_name2id(name)
        except Exception:
            continue
        if int(sim.model.jnt_type[joint_id]) != 0:
            continue
        return (
            int(sim.model.jnt_qposadr[joint_id]),
            int(sim.model.jnt_dofadr[joint_id]),
        )
    raise ValueError(f"free joint not found for {body_name!r}")


def body_pose(env, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = env.sim.model.body_name2id(body_name)
    return (
        np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy(),
        np.asarray(env.sim.data.body_xquat[body_id], dtype=float).copy(),
    )


def body_tilt_deg(env, body_name: str) -> float:
    body_id = env.sim.model.body_name2id(body_name)
    rotation = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))


def body_velocity(env, body_name: str) -> tuple[float, float]:
    _, vadr = free_joint_addresses(env.sim, body_name)
    velocity = np.asarray(env.sim.data.qvel[vadr:vadr + 6], dtype=float)
    return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))


def contact_body_names(env, body_name: str) -> set[str]:
    model, data = env.sim.model, env.sim.data
    root_id = model.body_name2id(body_name)
    descendant_ids = {root_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if int(model.body_parentid[body_id]) in descendant_ids and body_id not in descendant_ids:
                descendant_ids.add(body_id)
                changed = True
    geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in descendant_ids
    }
    names: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        if int(contact.geom1) in geom_ids:
            other = int(contact.geom2)
        elif int(contact.geom2) in geom_ids:
            other = int(contact.geom1)
        else:
            continue
        other_name = model.body_id2name(int(model.geom_bodyid[other]))
        if other_name:
            names.add(other_name)
    return names


def bodies_contact(env, first: str, second: str) -> bool:
    return second in contact_body_names(env, first)


def measurement(env, body_name: str) -> dict:
    position, quaternion = body_pose(env, body_name)
    linear_speed, angular_speed = body_velocity(env, body_name)
    return {
        "position": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "tilt_deg": body_tilt_deg(env, body_name),
        "linear_speed_mps": linear_speed,
        "angular_speed_radps": angular_speed,
        "contacts": sorted(contact_body_names(env, body_name)),
    }


def maximum_pose_drift(samples: Iterable[dict], body_name: str) -> dict:
    values = [sample[body_name] for sample in samples]
    start = np.asarray(values[0]["position"], dtype=float)
    return {
        "max_translation_drift_m": max(
            float(np.linalg.norm(np.asarray(value["position"], dtype=float) - start))
            for value in values
        ),
        "max_tilt_deg": max(float(value["tilt_deg"]) for value in values),
        "max_linear_speed_mps": max(float(value["linear_speed_mps"]) for value in values),
        "max_angular_speed_radps": max(float(value["angular_speed_radps"]) for value in values),
    }


def encode_attr(value):
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def save_state_bundle(
    path: str | Path,
    condition: str,
    states: list[np.ndarray],
    records: list[dict],
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        for key, value in metadata.items():
            group.attrs[key] = encode_attr(value)
        group.attrs["condition"] = condition
        group.attrs["scenario"] = SCENE_ID
        group.attrs["task_suite_name"] = SUITE
        group.attrs["task_id"] = TASK_ID
        group.attrs["task_prompt"] = TASK_PROMPT
        for index, (state, record) in enumerate(zip(states, records)):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset("initial_state", data=np.asarray(state))
            base_reset_state = record.get("base_reset_state")
            if base_reset_state is None:
                raise ValueError(f"demo_{index} missing paired base_reset_state")
            demo.create_dataset("base_reset_state", data=np.asarray(base_reset_state))
            demo.attrs["success"] = True
            for key, value in record.items():
                if key == "base_reset_state":
                    continue
                demo.attrs[key] = encode_attr(value)


def load_state_bundle(path: str | Path) -> tuple[dict, list[np.ndarray], list[dict]]:
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        metadata = {key: group.attrs[key] for key in group.attrs}
        states, records = [], []
        for name in sorted(group, key=lambda value: int(value.split("_")[-1])):
            demo = group[name]
            states.append(np.asarray(demo["initial_state"][:]))
            records.append({key: demo.attrs[key] for key in demo.attrs})
    return metadata, states, records
