"""Shared native-only contract for the L3-B moka sequence scene.

The external four-scene interpretation is:

* Eb = ``native``: the bit-exact official state;
* Er = ``near_first``: moka pot 1 occupies the near stove slot;
* Ec = ``far_first``: the same moka pot 1 occupies the far stove slot;
* Safe = a real-action reference starting from Er.

Both partial conditions therefore leave moka pot 2 as the only unfinished
object.  This removes object-instance identity from the Er/Ec contrast.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np


SCENE_ID = "L3-B-MOKA-ORDER"
DESIGN_VERSION = 3
SUITE = "libero_10"
TASK_ID = 8
TASK_FILE = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl"
TASK_PROMPT = "put both moka pots on the stove"
TASK_KEY = TASK_PROMPT.replace(" ", "_")

POT_1 = "moka_pot_1_main"
POT_2 = "moka_pot_2_main"
POT_BODIES = (POT_1, POT_2)
STOVE_BODY = "flat_stove_1_main"
TABLE_BODY = "table"
COOK_SITE = "flat_stove_1_cook_region"

EXPECTED_FIXTURES = {
    "kitchen_table": "kitchen_table",
    "flat_stove_1": "flat_stove",
}
EXPECTED_OBJECTS = {
    "moka_pot_1": "moka_pot",
    "moka_pot_2": "moka_pot",
}
EXPECTED_MOVABLE_ROOTS = set(POT_BODIES)
EXPECTED_FIXTURE_ROOTS = {TABLE_BODY, STOVE_BODY}

CONDITIONS = ("native", "near_first", "far_first")
CONDITION_LABEL = {
    "native": "Eb",
    "near_first": "Er",
    "far_first": "Ec",
}
CONDITION_SLOT = {
    "native": None,
    "near_first": "near",
    "far_first": "far",
}
CONDITION_INTERVENTION_BODY = {
    "native": None,
    "near_first": POT_1,
    "far_first": POT_1,
}
CONDITION_REMAINING_BODY = {
    "native": None,
    "near_first": POT_2,
    "far_first": POT_2,
}

DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
FORMAL_WAIT_STEPS = 10
CONSTRUCTION_SETTLE_STEPS = 80
POST_WAIT_HOLD_STEPS = 100
# The native cook site has 0.075 m half-extents.  A 0.105 m diagonal
# separation keeps both centers inside that site while leaving clearance
# between the native moka collision bodies across the Safe grasp orientation.
SLOT_SEPARATION_M = 0.105

MAX_RECEPTACLE_TILT_DEG = 1.0
MAX_NATIVE_WINDOW_TRANSLATION_M = 0.006
MAX_PLACED_WINDOW_TRANSLATION_M = 0.003
MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS = 0.03
MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS = 0.10
MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS = 0.015
MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS = 0.05
MAX_FINAL_LINEAR_SPEED_MPS = 0.01
MAX_FINAL_ANGULAR_SPEED_RADPS = 0.05


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_bddl_path() -> Path:
    configured_root = os.environ.get("LIBERO_ROOT")
    libero_root = (
        Path(configured_root)
        if configured_root
        else repository_root() / "_deps" / "LIBERO"
    )
    return libero_root / "libero" / "libero" / "bddl_files" / SUITE / TASK_FILE


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
        raise ValueError(f"native BDDL has no :{section} section")
    inventory = {}
    for line in match.group(1).splitlines():
        declaration = line.strip()
        if not declaration or "-" not in declaration:
            continue
        names, kind = declaration.split("-", maxsplit=1)
        for name in names.split():
            inventory[name] = kind.strip()
    return inventory


def parse_native_bddl(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    text = path.read_text(encoding="utf-8")
    prompt_match = re.search(r"\(:language\s+([^)]+)\)", text)
    if not prompt_match:
        raise ValueError("native BDDL has no :language prompt")
    return {
        "path": str(path),
        "bddl_sha256": sha256_path(path),
        "prompt": prompt_match.group(1).strip(),
        "fixtures": _typed_inventory(text, "fixtures"),
        "objects": _typed_inventory(text, "objects"),
    }


def validate_native_bddl(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    expected = native_bddl_path().resolve(strict=True)
    if path != expected:
        raise ValueError(
            "evaluated BDDL is not the selected native libero_10 task"
        )
    record = parse_native_bddl(path)
    if path.parent.name != SUITE or path.name != TASK_FILE:
        raise ValueError(f"wrong native task: {path}")
    if record["prompt"] != TASK_PROMPT:
        raise ValueError(f"native prompt mismatch: {record['prompt']!r}")
    if record["fixtures"] != EXPECTED_FIXTURES:
        raise ValueError(f"native fixture inventory mismatch: {record['fixtures']}")
    if record["objects"] != EXPECTED_OBJECTS:
        raise ValueError(f"native object inventory mismatch: {record['objects']}")
    return record


def inventory_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            {"fixtures": EXPECTED_FIXTURES, "objects": EXPECTED_OBJECTS},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def free_joint_addresses(env, body_name: str) -> tuple[int, int]:
    model = env.sim.model
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


def flat_state_slices(env, body_name: str) -> tuple[slice, slice]:
    qpos_address, velocity_address = free_joint_addresses(env, body_name)
    return (
        slice(1 + qpos_address, 1 + qpos_address + 7),
        slice(
            1 + int(env.sim.model.nq) + velocity_address,
            1 + int(env.sim.model.nq) + velocity_address + 6,
        ),
    )


def _descendant_body_ids(model, root_id: int) -> set[int]:
    result = {int(root_id)}
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


def contact_body_names(env, body_name: str) -> list[str]:
    model, data = env.sim.model, env.sim.data
    root_id = int(model.body_name2id(body_name))
    descendants = _descendant_body_ids(model, root_id)
    geoms = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendants
    }
    contacts = set()
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in geoms:
            other = second
        elif second in geoms:
            other = first
        else:
            continue
        other_name = model.body_id2name(int(model.geom_bodyid[other]))
        if other_name:
            contacts.add(str(other_name))
    return sorted(contacts)


def measurement(env, body_name: str) -> dict:
    model, data = env.sim.model, env.sim.data
    body_id = int(model.body_name2id(body_name))
    rotation = np.asarray(data.body_xmat[body_id], dtype=float).reshape(3, 3)
    try:
        _, velocity_address = free_joint_addresses(env, body_name)
    except ValueError:
        linear_speed = angular_speed = None
    else:
        velocity = np.asarray(
            data.qvel[velocity_address : velocity_address + 6], dtype=float
        )
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
    return {
        "position": np.asarray(data.body_xpos[body_id], dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(
            data.body_xquat[body_id], dtype=float
        ).tolist(),
        "tilt_deg": float(
            np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        ),
        "linear_speed_mps": linear_speed,
        "angular_speed_radps": angular_speed,
        "contacts": contact_body_names(env, body_name),
    }


def window_stats(samples: list[dict[str, dict]], body_name: str) -> dict:
    origin = np.asarray(samples[0][body_name]["position"], dtype=float)
    linear = [
        float(sample[body_name]["linear_speed_mps"])
        for sample in samples
        if sample[body_name]["linear_speed_mps"] is not None
    ]
    angular = [
        float(sample[body_name]["angular_speed_radps"])
        for sample in samples
        if sample[body_name]["angular_speed_radps"] is not None
    ]
    return {
        "max_translation_drift_m": max(
            float(
                np.linalg.norm(
                    np.asarray(sample[body_name]["position"], dtype=float)
                    - origin
                )
            )
            for sample in samples
        ),
        "max_tilt_deg": max(
            float(sample[body_name]["tilt_deg"]) for sample in samples
        ),
        "max_linear_speed_mps": max(linear, default=None),
        "max_angular_speed_radps": max(angular, default=None),
    }


def save_state_bundle(
    path: str | Path,
    *,
    condition: str,
    records: list[dict],
    pool_preregistration: dict | None = None,
) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        group.attrs.update(
            {
                "scenario": SCENE_ID,
                "design_version": DESIGN_VERSION,
                "condition": condition,
                "condition_label": CONDITION_LABEL[condition],
                "task_suite_name": SUITE,
                "task_id": TASK_ID,
                "task_prompt": TASK_PROMPT,
                "task_file": TASK_FILE,
                "count": len(records),
                "pairing_method": (
                    "same_official_native_state_same_moka_pot_alternate_slot"
                ),
                "custom_bddl": False,
                "custom_assets": False,
                "prompt_override": False,
            }
        )
        if pool_preregistration is not None:
            group.attrs["pool_preregistration_id"] = pool_preregistration[
                "preregistration_id"
            ]
            group.attrs["pool_preregistration_sha256"] = (
                pool_preregistration["sha256"]
            )
            group.attrs["official_native_state_indices_json"] = json.dumps(
                pool_preregistration["official_state_indices"]
            )
        for index, record in enumerate(records):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset(
                "initial_state",
                data=np.asarray(record["initial_state"], dtype=float),
            )
            demo.create_dataset(
                "base_reset_state",
                data=np.asarray(record["base_reset_state"], dtype=float),
            )
            demo.attrs["success"] = True
            for key, value in record["attrs"].items():
                if isinstance(value, (dict, list, tuple)):
                    demo.attrs[key] = json.dumps(value, sort_keys=True)
                else:
                    demo.attrs[key] = value
    return path
