"""Shared, fail-closed helpers for the L3-A3 support-chain experiment.

Imports of LIBERO are intentionally lazy so schema and unit tests run on CPU
development machines without robosuite installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

SCHEMA = "physcog_l3a3_support_chain_v3_task87_pad_block"
PROMPT = "pick up the book on the left and place it on top of the shelf"
HDF5_KEY = PROMPT.replace(" ", "_")

SUPPORT_BODY = "yellow_book_2_main"             # S, goal-relevant left book
TARGET_BODY = SUPPORT_BODY                       # the original task target
MIDDLE_BODY = "l3_a3_support_pad_1_main"         # A, low-friction flat pad
TOP_BODY = "l3_a3_top_block_1_main"              # B, compact cylindrical load
TRAY_BODY = "wooden_two_layer_shelf_1_main"

ALL_CHAIN_BODIES = (SUPPORT_BODY, MIDDLE_BODY, TOP_BODY)


@dataclass(frozen=True)
class GateThresholds:
    settle_steps: int = 500
    hold_steps: int = 200
    max_hold_xy_m: float = 0.006
    max_hold_drop_m: float = 0.006
    remove_dx_m: float = 0.14
    chain_event_xy_m: float = 0.020
    chain_event_drop_m: float = 0.012
    min_event_lag_steps: int = 0
    max_event_lag_steps: int = 180


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _body_id(model, name: str) -> int:
    try:
        return int(model.body_name2id(name))
    except Exception as exc:
        raise ValueError(f"required L3-A3 body is missing: {name}") from exc


def find_free_joint(sim, body_name: str) -> tuple[int, int]:
    """Return qpos/dof addresses for a body's free joint."""
    body_id = _body_id(sim.model, body_name)
    for joint_id in range(int(sim.model.njnt)):
        if int(sim.model.jnt_bodyid[joint_id]) != body_id:
            continue
        if int(sim.model.jnt_type[joint_id]) == 0:
            return (
                int(sim.model.jnt_qposadr[joint_id]),
                int(sim.model.jnt_dofadr[joint_id]),
            )
    raise ValueError(f"required free joint is missing for {body_name}")


def body_pose(sim, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = _body_id(sim.model, body_name)
    return (
        np.asarray(sim.data.body_xpos[body_id], dtype=float).copy(),
        np.asarray(sim.data.body_xquat[body_id], dtype=float).copy(),
    )


def set_free_pose(
    sim,
    body_name: str,
    xyz: Iterable[float],
    quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    qadr, vadr = find_free_joint(sim, body_name)
    sim.data.qpos[qadr : qadr + 3] = np.asarray(xyz, dtype=float)
    quat = np.asarray(quat_wxyz, dtype=float)
    quat /= np.linalg.norm(quat)
    sim.data.qpos[qadr + 3 : qadr + 7] = quat
    sim.data.qvel[vadr : vadr + 6] = 0.0
    sim.forward()


def zero_body_velocity(sim, body_name: str) -> None:
    _, vadr = find_free_joint(sim, body_name)
    sim.data.qvel[vadr : vadr + 6] = 0.0


def _geom_body_ancestors(model, geom_id: int) -> set[int]:
    body_id = int(model.geom_bodyid[geom_id])
    result = set()
    while body_id >= 0 and body_id not in result:
        result.add(body_id)
        if body_id == 0:
            break
        body_id = int(model.body_parentid[body_id])
    return result


def bodies_in_contact(sim, first: str, second: str) -> bool:
    first_id = _body_id(sim.model, first)
    second_id = _body_id(sim.model, second)
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        a = _geom_body_ancestors(sim.model, int(contact.geom1))
        b = _geom_body_ancestors(sim.model, int(contact.geom2))
        if (first_id in a and second_id in b) or (second_id in a and first_id in b):
            return True
    return False


def chain_contacts(sim) -> dict[str, bool]:
    return {
        "s_a": bodies_in_contact(sim, SUPPORT_BODY, MIDDLE_BODY),
        "a_b": bodies_in_contact(sim, MIDDLE_BODY, TOP_BODY),
        "s_b_forbidden": bodies_in_contact(sim, SUPPORT_BODY, TOP_BODY),
    }


def pose_delta(
    before: tuple[np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    p0, _ = before
    p1, _ = after
    return {
        "xy_m": float(np.linalg.norm((p1 - p0)[:2])),
        "drop_m": float(p0[2] - p1[2]),
        "distance_m": float(np.linalg.norm(p1 - p0)),
    }


def event_triggered(delta: dict[str, float], thresholds: GateThresholds) -> bool:
    return (
        delta["xy_m"] >= thresholds.chain_event_xy_m
        or delta["drop_m"] >= thresholds.chain_event_drop_m
    )


def load_states(path: str | Path) -> tuple[list[np.ndarray], dict]:
    with h5py.File(path, "r") as handle:
        if HDF5_KEY not in handle:
            raise ValueError(f"missing HDF5 key {HDF5_KEY!r}")
        group = handle[HDF5_KEY]
        metadata = {key: value for key, value in group.attrs.items()}
        if metadata.get("schema", "") != SCHEMA:
            raise ValueError(f"stale/mismatched L3-A3 schema in {path}")
        demos = sorted(group, key=lambda name: int(name.split("_")[-1]))
        states = [np.asarray(group[name]["initial_state"]) for name in demos]
    if not states:
        raise ValueError(f"no serialized states in {path}")
    return states, metadata


def save_states(
    path: str | Path,
    states: list[np.ndarray],
    condition: str,
    seed: int,
    episode_metadata: list[dict],
) -> None:
    if condition not in {"eb", "er", "ec"}:
        raise ValueError(condition)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        group = handle.create_group(HDF5_KEY)
        group.attrs["schema"] = SCHEMA
        group.attrs["condition"] = condition
        group.attrs["prompt"] = PROMPT
        group.attrs["seed"] = int(seed)
        group.attrs["support_body"] = SUPPORT_BODY
        group.attrs["middle_body"] = MIDDLE_BODY
        group.attrs["top_body"] = TOP_BODY
        for index, (state, evidence) in enumerate(zip(states, episode_metadata)):
            demo = group.create_group(f"demo_{index}")
            demo.create_dataset("initial_state", data=np.asarray(state))
            demo.attrs["success"] = True
            demo.attrs["pair_id"] = int(index)
            demo.attrs["evidence_json"] = json.dumps(evidence, sort_keys=True)


def validate_triplet_metadata(eb: str | Path, er: str | Path, ec: str | Path) -> int:
    triplet = [load_states(path) for path in (eb, er, ec)]
    counts = {len(states) for states, _ in triplet}
    if len(counts) != 1:
        raise ValueError(f"Eb/Er/Ec episode counts differ: {sorted(counts)}")
    conditions = [str(meta.get("condition", "")) for _, meta in triplet]
    if conditions != ["eb", "er", "ec"]:
        raise ValueError(f"expected eb/er/ec metadata, got {conditions}")
    prompts = {str(meta.get("prompt", "")) for _, meta in triplet}
    if prompts != {PROMPT}:
        raise ValueError("prompt differs across paired conditions")
    seeds = {int(meta.get("seed", -1)) for _, meta in triplet}
    if len(seeds) != 1:
        raise ValueError("seed differs across paired conditions")
    return counts.pop()
