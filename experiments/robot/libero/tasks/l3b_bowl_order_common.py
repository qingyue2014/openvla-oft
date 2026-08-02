"""Native-only contract for the L3-B bowl / drawer order probe.

The selected task is the unmodified ``libero_10`` task whose native goal is
``In(bowl, bottom_drawer) AND Close(bottom_drawer)``.  The intervention changes
only serialized native state:

* Eb / ``native``: open drawer, bowl on the table;
* Er / ``premature_close``: empty drawer is already closed;
* Ec / ``prerequisite_done``: bowl is already in the open drawer.

Er is diagnostic because success requires a non-monotonic predicate trace:
``Close: True -> False -> True`` while ``In: False -> True``.  No collision or
safety oracle defines the result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np


SCENE_ID = "L3-B-BOWL-ORDER"
DESIGN_VERSION = 2
SUITE = "libero_10"
TASK_ID = 3
TASK_FILE = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_"
    "cabinet_and_close_it.bddl"
)
TASK_PROMPT = (
    "put the black bowl in the bottom drawer of the cabinet and close it"
)
TASK_KEY = TASK_PROMPT.replace(" ", "_")

BOWL_BODY = "akita_black_bowl_1_main"
BOTTLE_BODY = "wine_bottle_1_main"
TABLE_BODY = "table"
CABINET_BODY = "white_cabinet_1_main"
DRAWER_BODY = "white_cabinet_1_cabinet_bottom"
WINE_RACK_BODY = "wine_rack_1_main"
DRAWER_JOINT = "white_cabinet_1_bottom_level"
DRAWER_SITE = "white_cabinet_1_bottom_region"

EXPECTED_FIXTURES = {
    "kitchen_table": "kitchen_table",
    "white_cabinet_1": "white_cabinet",
    "wine_rack_1": "wine_rack",
}
EXPECTED_OBJECTS = {
    "akita_black_bowl_1": "akita_black_bowl",
    "wine_bottle_1": "wine_bottle",
}
EXPECTED_MOVABLE_ROOTS = {BOWL_BODY, BOTTLE_BODY}
EXPECTED_FIXTURE_ROOTS = {TABLE_BODY, CABINET_BODY, WINE_RACK_BODY}

CONDITIONS = ("native", "premature_close", "prerequisite_done")
CONDITION_LABEL = {
    "native": "Eb",
    "premature_close": "Er",
    "prerequisite_done": "Ec",
}
CONDITION_INTERVENTION_BODY = {
    "native": "",
    "premature_close": DRAWER_BODY,
    "prerequisite_done": BOWL_BODY,
}
CONDITION_INTERVENTION_KIND = {
    "native": "none",
    "premature_close": "drawer_joint_only",
    "prerequisite_done": "bowl_free_joint_only",
}
EXPECTED_INITIAL_PREDICATES = {
    "native": {"close": False, "in": False},
    "premature_close": {"close": True, "in": False},
    "prerequisite_done": {"close": False, "in": True},
}

DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
FORMAL_WAIT_STEPS = 10
CONSTRUCTION_SETTLE_STEPS = 100
POST_WAIT_HOLD_STEPS = 100
# The native Close predicate accepts qpos in [0.0, 0.005], but the upper end
# of that interval interpenetrates the cabinet base and mechanically locks the
# drawer.  Starting from 0.002 lets the passive construction settle to about
# 0.00072: still Close, collision-free, and executable through OSC actions.
DRAWER_CLOSED_QPOS = 0.002
DRAWER_OPEN_PREDICATE_MAX_QPOS = -0.14
DRAWER_CLOSE_PREDICATE_MIN_QPOS = 0.0
BOWL_SPAWN_HEIGHT_M = 0.0

MAX_BOWL_TILT_DEG = 1.0
MAX_BOTTLE_TILT_DEG = 5.0
# Official LIBERO init states restore the bowl and bottle at z=0.97 m and the
# evaluator's native ten-step wait drops them about 0.074 m onto the table.
# This limit covers that native spawn-settling motion; uprightness is still
# enforced throughout, and the post-wait hold uses the strict limits below.
MAX_NATIVE_WINDOW_TRANSLATION_M = 0.080
MAX_PLACED_WINDOW_TRANSLATION_M = 0.003
MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS = 1.10
MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS = 0.10
MAX_PLACED_TRANSIENT_LINEAR_SPEED_MPS = 0.015
MAX_PLACED_TRANSIENT_ANGULAR_SPEED_RADPS = 0.05
MAX_POST_WAIT_TRANSLATION_M = 0.003
MAX_POST_WAIT_LINEAR_SPEED_MPS = 0.015
MAX_POST_WAIT_ANGULAR_SPEED_RADPS = 0.05
MAX_FINAL_LINEAR_SPEED_MPS = 0.01
MAX_FINAL_ANGULAR_SPEED_RADPS = 0.05
MAX_DRAWER_WINDOW_QPOS_DRIFT = 0.003
MAX_DRAWER_FINAL_SPEED = 0.01
# Closed drawers may rest on the cabinet stop with floating-point contact at
# roughly 1e-10 m.  Reject material interpenetration, not a valid zero-distance
# supporting contact.
MAX_DRAWER_CABINET_PENETRATION_M = 1e-5

PAIRING_METHOD = "same_official_native_state_predicate_progress_intervention"
INITIAL_GATE_VERDICT = "PASS_L3B_BOWL_INITIAL_PHYSICAL_GATES"
PAIRING_VERDICT = "PASS_L3B_BOWL_EXACT_SERIALIZED_PAIRING"
RUNTIME_REPLAY_VERDICT = "PASS_L3B_BOWL_RUNTIME_REPLAY"
NATIVE_PREFLIGHT_VERDICT = "PASS_L3B_BOWL_NATIVE_ONLY_PREFLIGHT"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def native_bddl_path() -> Path:
    configured = os.environ.get("LIBERO_ROOT")
    root = Path(configured) if configured else repository_root() / "_deps" / "LIBERO"
    return root / "libero" / "libero" / "bddl_files" / SUITE / TASK_FILE


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state, dtype=float).tobytes()).hexdigest()


def inventory_sha256() -> str:
    payload = {"fixtures": EXPECTED_FIXTURES, "objects": EXPECTED_OBJECTS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _typed_inventory(text: str, section: str) -> dict[str, str]:
    match = re.search(
        rf"\(:{re.escape(section)}\s+(.*?)\n\s*\)", text, flags=re.DOTALL
    )
    if not match:
        raise ValueError(f"native BDDL has no :{section} section")
    result: dict[str, str] = {}
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line or "-" not in line:
            continue
        names, kind = line.split("-", maxsplit=1)
        for name in names.split():
            result[name] = kind.strip()
    return result


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
        "init_has_open_bottom_drawer": (
            "(Open white_cabinet_1_bottom_region)" in text
        ),
        "goal_has_close_bottom_drawer": (
            "(Close white_cabinet_1_bottom_region)" in text
        ),
        "goal_has_bowl_in_bottom_drawer": (
            "(In akita_black_bowl_1 white_cabinet_1_bottom_region)" in text
        ),
    }


def validate_native_bddl(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    expected = native_bddl_path().resolve(strict=True)
    if path != expected:
        raise ValueError("evaluated BDDL is not the selected native task")
    record = parse_native_bddl(path)
    if path.parent.name != SUITE or path.name != TASK_FILE:
        raise ValueError(f"wrong native task: {path}")
    if record["prompt"] != TASK_PROMPT:
        raise ValueError(f"native prompt mismatch: {record['prompt']!r}")
    if record["fixtures"] != EXPECTED_FIXTURES:
        raise ValueError(f"native fixture inventory mismatch: {record['fixtures']}")
    if record["objects"] != EXPECTED_OBJECTS:
        raise ValueError(f"native object inventory mismatch: {record['objects']}")
    for key in (
        "init_has_open_bottom_drawer",
        "goal_has_close_bottom_drawer",
        "goal_has_bowl_in_bottom_drawer",
    ):
        if record[key] is not True:
            raise ValueError(f"native predicate contract missing: {key}")
    return record


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


def scalar_joint_addresses(env, joint_name: str) -> tuple[int, int]:
    joint_id = int(env.sim.model.joint_name2id(joint_name))
    if int(env.sim.model.jnt_type[joint_id]) not in (2, 3):
        raise ValueError(f"{joint_name} is not a scalar hinge/slide joint")
    return (
        int(env.sim.model.jnt_qposadr[joint_id]),
        int(env.sim.model.jnt_dofadr[joint_id]),
    )


def flat_free_joint_slices(env, body_name: str) -> tuple[slice, slice]:
    qpos, qvel = free_joint_addresses(env, body_name)
    return (
        slice(1 + qpos, 1 + qpos + 7),
        slice(1 + int(env.sim.model.nq) + qvel, 1 + int(env.sim.model.nq) + qvel + 6),
    )


def flat_scalar_joint_indices(env, joint_name: str) -> tuple[int, int]:
    qpos, qvel = scalar_joint_addresses(env, joint_name)
    return (1 + qpos, 1 + int(env.sim.model.nq) + qvel)


def _descendant_body_ids(model, root_id: int) -> set[int]:
    result = {int(root_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def contact_body_names(env, body_name: str) -> list[str]:
    model, data = env.sim.model, env.sim.data
    root = int(model.body_name2id(body_name))
    descendants = _descendant_body_ids(model, root)
    geoms = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendants
    }
    contacts: set[str] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first in geoms:
            other = second
        elif second in geoms:
            other = first
        else:
            continue
        name = model.body_id2name(int(model.geom_bodyid[other]))
        if name:
            contacts.add(str(name))
    return sorted(contacts)


def drawer_cabinet_self_contacts(env) -> list[dict]:
    """Return contacts between the bottom drawer and the rest of its cabinet."""
    model, data = env.sim.model, env.sim.data
    drawer_ids = _descendant_body_ids(model, int(model.body_name2id(DRAWER_BODY)))
    cabinet_ids = _descendant_body_ids(model, int(model.body_name2id(CABINET_BODY)))
    other_cabinet_ids = cabinet_ids - drawer_ids
    records = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first_geom, second_geom = int(contact.geom1), int(contact.geom2)
        first_body = int(model.geom_bodyid[first_geom])
        second_body = int(model.geom_bodyid[second_geom])
        if not (
            (first_body in drawer_ids and second_body in other_cabinet_ids)
            or (second_body in drawer_ids and first_body in other_cabinet_ids)
        ):
            continue
        records.append(
            {
                "geom1": str(model.geom_id2name(first_geom) or ""),
                "geom2": str(model.geom_id2name(second_geom) or ""),
                "body1": str(model.body_id2name(first_body) or ""),
                "body2": str(model.body_id2name(second_body) or ""),
                "distance_m": float(contact.dist),
            }
        )
    return records


def body_measurement(env, body_name: str) -> dict:
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
        "quaternion_wxyz": np.asarray(data.body_xquat[body_id], dtype=float).tolist(),
        "tilt_deg": float(
            np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        ),
        "linear_speed_mps": linear_speed,
        "angular_speed_radps": angular_speed,
        "contacts": contact_body_names(env, body_name),
    }


def predicate_state(env) -> dict[str, bool]:
    inner = env.env if hasattr(env, "env") else env
    return {
        "close": bool(
            inner._eval_predicate(
                ["close", "white_cabinet_1_bottom_region"]
            )
        ),
        "in": bool(
            inner._eval_predicate(
                [
                    "in",
                    "akita_black_bowl_1",
                    "white_cabinet_1_bottom_region",
                ]
            )
        ),
    }


def scene_measurement(env) -> dict:
    qpos_address, qvel_address = scalar_joint_addresses(env, DRAWER_JOINT)
    return {
        BOWL_BODY: body_measurement(env, BOWL_BODY),
        BOTTLE_BODY: body_measurement(env, BOTTLE_BODY),
        DRAWER_BODY: body_measurement(env, DRAWER_BODY),
        "drawer_joint": {
            "qpos": float(env.sim.data.qpos[qpos_address]),
            "speed": float(abs(env.sim.data.qvel[qvel_address])),
        },
        "drawer_cabinet_self_contacts": drawer_cabinet_self_contacts(env),
        "predicates": predicate_state(env),
    }


def body_window_stats(samples: list[dict], body_name: str) -> dict:
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
                    np.asarray(sample[body_name]["position"], dtype=float) - origin
                )
            )
            for sample in samples
        ),
        "max_tilt_deg": max(float(sample[body_name]["tilt_deg"]) for sample in samples),
        "max_linear_speed_mps": max(linear, default=None),
        "max_angular_speed_radps": max(angular, default=None),
    }


def drawer_window_stats(samples: list[dict]) -> dict:
    initial = float(samples[0]["drawer_joint"]["qpos"])
    return {
        "max_qpos_drift": max(
            abs(float(sample["drawer_joint"]["qpos"]) - initial)
            for sample in samples
        ),
        "max_speed": max(float(sample["drawer_joint"]["speed"]) for sample in samples),
        "min_qpos": min(float(sample["drawer_joint"]["qpos"]) for sample in samples),
        "max_qpos": max(float(sample["drawer_joint"]["qpos"]) for sample in samples),
    }


def save_state_bundle(path: str | Path, *, condition: str, records: list[dict]) -> Path:
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
                "pairing_method": PAIRING_METHOD,
                "custom_bddl": False,
                "custom_assets": False,
                "prompt_override": False,
            }
        )
        for index, record in enumerate(records):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset(
                "initial_state", data=np.asarray(record["initial_state"], dtype=float)
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
