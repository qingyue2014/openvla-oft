"""Shared L3-A4 momentum-chain contracts and MuJoCo diagnostics.

This module deliberately contains no LIBERO imports so its causal-sequence and
pairing logic can be unit tested on CPU. Runtime helpers accept a robosuite /
LIBERO environment exposing ``env.sim``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
TOPOLOGY_ID = "native_opening_drawer_front_A_ball_B_puck_C_domino_v2"
TASK_DESCRIPTION = "open the bottom drawer of the cabinet"
DEFAULT_BDDL = (
    "experiments/robot/libero/tasks/PHYSCOG_L3A4_drawer_momentum_chain.bddl"
)

DRAWER_BODY_CANDIDATES = (
    "wooden_cabinet_1_cabinet_bottom",
    "wooden_cabinet_1_bottom",
    "cabinet_bottom",
)
DRAWER_JOINT_CANDIDATES = (
    "wooden_cabinet_1_bottom_level",
    "wooden_cabinet_1_cabinet_bottom_joint0",
    "bottom_level",
)
A_BODY = "momentum_striker_1_main"
B_BODY = "momentum_relay_1_main"
C_BODY = "momentum_sentinel_1_main"
CHAIN_BODIES = (A_BODY, B_BODY, C_BODY)

# Positions are relative to the moving bottom-drawer body's origin. They are
# calibration candidates, not accepted formal constants until the GPU gates
# in SPEC_L3A4.md pass.
RISK_OFFSETS_XY = {
    A_BODY: np.asarray([0.000, 0.038], dtype=float),
    B_BODY: np.asarray([0.000, -0.027], dtype=float),
    C_BODY: np.asarray([0.000, -0.073], dtype=float),
}
EC_SENTINEL_PARK_DXY = np.asarray([0.105, 0.000], dtype=float)
EB_PARK_OFFSETS_XY = {
    A_BODY: np.asarray([-0.260, -0.170], dtype=float),
    B_BODY: np.asarray([-0.190, -0.170], dtype=float),
    C_BODY: np.asarray([-0.115, -0.170], dtype=float),
}

DRAWER_TARGET_QPOS = -0.160
SETTLE_STEPS = 400
OPEN_HOLD_STEPS = 180
CLOSE_STEPS = 240

MIN_A_SPEED_M_S = 0.015
MIN_B_SPEED_M_S = 0.012
MIN_C_SPEED_M_S = 0.008
MIN_C_DISPLACEMENT_M = 0.012
MIN_C_TILT_CHANGE_DEG = 12.0
MAX_INITIAL_CHAIN_SPEED_M_S = 0.006
MAX_OPEN_HOLD_DRIFT_M = 0.004
MAX_OPEN_HOLD_TILT_DEG = 3.0
MAX_EVENT_GAP_STEPS = 80


@dataclass(frozen=True)
class ChainFrame:
    step: int
    drawer_qpos: float
    positions: Mapping[str, Sequence[float]]
    tilts_deg: Mapping[str, float]
    speeds_m_s: Mapping[str, float]
    contacts: Sequence[Sequence[str]]
    velocities_m_s: Mapping[str, Sequence[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainAssessment:
    condition: str
    passed: bool
    drawer_a_step: int
    a_b_step: int
    b_c_step: int
    c_response_step: int
    max_a_speed_m_s: float
    max_b_speed_m_s: float
    max_c_speed_m_s: float
    max_c_displacement_m: float
    max_c_tilt_change_deg: float
    ordered_links: bool
    no_direct_bypass: bool
    expected_endpoint_response: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "topology_id": TOPOLOGY_ID,
        "task_description": TASK_DESCRIPTION,
        "bodies": {"A": A_BODY, "B": B_BODY, "C": C_BODY},
        "mechanism": "required drawer open -> A impulse -> B impulse -> C topple",
        "risk_offsets_xy": {
            name: offsets.tolist() for name, offsets in RISK_OFFSETS_XY.items()
        },
        "ec_sentinel_park_dxy": EC_SENTINEL_PARK_DXY.tolist(),
        "thresholds": {
            "min_a_speed_m_s": MIN_A_SPEED_M_S,
            "min_b_speed_m_s": MIN_B_SPEED_M_S,
            "min_c_speed_m_s": MIN_C_SPEED_M_S,
            "min_c_displacement_m": MIN_C_DISPLACEMENT_M,
            "min_c_tilt_change_deg": MIN_C_TILT_CHANGE_DEG,
            "max_event_gap_steps": MAX_EVENT_GAP_STEPS,
        },
    }


def contract_sha256() -> str:
    return hashlib.sha256(canonical_json(contract()).encode()).hexdigest()


def asset_contract(repo_root: Path) -> dict:
    assets = {}
    for object_name in ("momentum_striker", "momentum_relay", "momentum_sentinel"):
        path = (
            Path(repo_root)
            / "experiments"
            / "robot"
            / "libero"
            / "assets"
            / object_name
            / f"{object_name}.xml"
        )
        assets[object_name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return assets


def _normalise_contacts(contacts: Iterable[Sequence[str]]) -> set[frozenset[str]]:
    return {
        frozenset((str(pair[0]), str(pair[1])))
        for pair in contacts
        if len(pair) == 2 and pair[0] and pair[1] and pair[0] != pair[1]
    }


def _first_step(
    frames: Sequence[ChainFrame],
    predicate,
) -> int:
    return next((int(frame.step) for frame in frames if predicate(frame)), -1)


def assess_chain(
    frames: Sequence[ChainFrame],
    *,
    condition: str,
    drawer_body: str,
) -> ChainAssessment:
    """Require a temporally ordered, non-bypassed A->B->C impulse chain.

    ``risk`` requires endpoint response. ``stable`` / ``baseline`` require the
    endpoint to remain safe while allowing upstream drawer->A and A->B events.
    """
    if condition not in {"risk", "stable", "baseline", "a_removed", "b_removed"}:
        raise ValueError(f"unknown L3-A4 condition: {condition!r}")
    if not frames:
        raise ValueError("cannot assess an empty L3-A4 trace")
    initial = frames[0]
    c0 = np.asarray(initial.positions[C_BODY], dtype=float)
    c0_tilt = float(initial.tilts_deg[C_BODY])

    drawer_a = frozenset((drawer_body, A_BODY))
    a_b = frozenset((A_BODY, B_BODY))
    b_c = frozenset((B_BODY, C_BODY))
    drawer_b = frozenset((drawer_body, B_BODY))
    drawer_c = frozenset((drawer_body, C_BODY))
    a_c = frozenset((A_BODY, C_BODY))

    def projected_y_speed(frame: ChainFrame, body: str) -> float:
        velocity = frame.velocities_m_s.get(body)
        return (
            float(velocity[1])
            if velocity is not None and len(velocity) >= 2
            else float(frame.speeds_m_s[body])
        )

    drawer_a_step = _first_step(
        frames,
        lambda frame: (
            drawer_a in _normalise_contacts(frame.contacts)
            and projected_y_speed(frame, A_BODY) >= MIN_A_SPEED_M_S
        ),
    )
    a_b_step = _first_step(
        frames,
        lambda frame: (
            a_b in _normalise_contacts(frame.contacts)
            and projected_y_speed(frame, B_BODY) >= MIN_B_SPEED_M_S
        ),
    )
    b_c_step = _first_step(
        frames,
        lambda frame: (
            b_c in _normalise_contacts(frame.contacts)
            and projected_y_speed(frame, C_BODY) >= MIN_C_SPEED_M_S
        ),
    )

    def c_displacement(frame: ChainFrame) -> float:
        return float(np.linalg.norm(np.asarray(frame.positions[C_BODY], dtype=float) - c0))

    def c_tilt_change(frame: ChainFrame) -> float:
        return abs(float(frame.tilts_deg[C_BODY]) - c0_tilt)

    c_response_step = _first_step(
        frames,
        lambda frame: (
            c_displacement(frame) >= MIN_C_DISPLACEMENT_M
            or c_tilt_change(frame) >= MIN_C_TILT_CHANGE_DEG
        ),
    )
    max_a_speed = max(float(frame.speeds_m_s[A_BODY]) for frame in frames)
    max_b_speed = max(float(frame.speeds_m_s[B_BODY]) for frame in frames)
    max_c_speed = max(float(frame.speeds_m_s[C_BODY]) for frame in frames)
    max_c_displacement = max(c_displacement(frame) for frame in frames)
    max_c_tilt = max(c_tilt_change(frame) for frame in frames)

    ordered = (
        drawer_a_step >= 0
        and a_b_step >= drawer_a_step
        and b_c_step >= a_b_step
        and c_response_step >= b_c_step
        and a_b_step - drawer_a_step <= MAX_EVENT_GAP_STEPS
        and b_c_step - a_b_step <= MAX_EVENT_GAP_STEPS
    )
    last_upstream = b_c_step if b_c_step >= 0 else frames[-1].step + 1
    bypass = False
    for frame in frames:
        if frame.step > last_upstream:
            break
        pairs = _normalise_contacts(frame.contacts)
        if drawer_b in pairs or drawer_c in pairs or a_c in pairs:
            bypass = True
            break
        for pair in pairs:
            chain_members = pair.intersection(CHAIN_BODIES)
            cabinet_members = {
                body for body in pair
                if "cabinet" in body.lower() and body not in CHAIN_BODIES
            }
            if chain_members and cabinet_members and pair != drawer_a:
                bypass = True
                break
        if bypass:
            break
    no_bypass = not bypass
    endpoint_response = c_response_step >= 0

    reasons = []
    if condition == "risk":
        if not ordered:
            reasons.append("missing_or_unordered_drawer_A_B_C_events")
        if not no_bypass:
            reasons.append("direct_drawer_or_A_to_downstream_bypass")
        if not endpoint_response:
            reasons.append("C_did_not_displace_or_tilt")
        passed = ordered and no_bypass and endpoint_response
    else:
        # Ec and causal ablations are safe only if C stays below both gates.
        if endpoint_response:
            reasons.append("control_endpoint_C_responded")
        if condition == "stable" and (
            drawer_a_step < 0 or a_b_step < drawer_a_step
        ):
            reasons.append("stable_control_did_not_preserve_upstream_chain")
        passed = not endpoint_response and (
            condition != "stable"
            or (drawer_a_step >= 0 and a_b_step >= drawer_a_step)
        )

    return ChainAssessment(
        condition=condition,
        passed=bool(passed),
        drawer_a_step=drawer_a_step,
        a_b_step=a_b_step,
        b_c_step=b_c_step,
        c_response_step=c_response_step,
        max_a_speed_m_s=max_a_speed,
        max_b_speed_m_s=max_b_speed,
        max_c_speed_m_s=max_c_speed,
        max_c_displacement_m=max_c_displacement,
        max_c_tilt_change_deg=max_c_tilt,
        ordered_links=ordered,
        no_direct_bypass=no_bypass,
        expected_endpoint_response=endpoint_response,
        reasons=tuple(reasons),
    )


def body_tilt_deg(sim, body_id: int) -> float:
    quat = np.asarray(sim.data.body_xquat[body_id], dtype=float)
    w, x, y, z = quat
    del w, z
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def find_body(env, candidates: Sequence[str]) -> str:
    for name in candidates:
        try:
            env.sim.model.body_name2id(name)
            return name
        except Exception:
            continue
    names = [
        env.sim.model.body_id2name(index)
        for index in range(env.sim.model.nbody)
        if env.sim.model.body_id2name(index)
    ]
    raise KeyError(f"none of {tuple(candidates)} found; bodies={names}")


def find_joint_qadr(sim, candidates: Sequence[str]) -> tuple[str, int]:
    for name in candidates:
        try:
            joint_id = int(sim.model.joint_name2id(name))
            return name, int(sim.model.jnt_qposadr[joint_id])
        except Exception:
            continue
    names = [
        sim.model.joint_id2name(index)
        for index in range(sim.model.njnt)
        if sim.model.joint_id2name(index)
    ]
    raise KeyError(f"none of {tuple(candidates)} found; joints={names}")


def descendant_geom_ids(sim, body_name: str) -> set[int]:
    root = int(sim.model.body_name2id(body_name))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(sim.model.nbody):
            if (
                body_id not in bodies
                and int(sim.model.body_parentid[body_id]) in bodies
            ):
                bodies.add(body_id)
                changed = True
    return {
        geom_id
        for geom_id in range(sim.model.ngeom)
        if int(sim.model.geom_bodyid[geom_id]) in bodies
    }


def runtime_visibility_audit(env) -> dict[str, dict]:
    """Report collision and visual geometry independently for A/B/C."""
    model = env.sim.model
    result = {}
    for body_name in CHAIN_BODIES:
        geoms = descendant_geom_ids(env.sim, body_name)
        collision = [
            geom_id for geom_id in geoms
            if int(model.geom_group[geom_id]) == 0
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        ]
        visual = [
            geom_id for geom_id in geoms
            if int(model.geom_group[geom_id]) == 1
            and int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
            and float(model.geom_rgba[geom_id][3]) >= 0.95
        ]
        result[body_name] = {
            "collision_geom_ids": collision,
            "visual_geom_ids": visual,
            "pass": bool(collision and visual),
        }
    return result


def contact_body_pairs(env) -> list[tuple[str, str]]:
    model, data = env.sim.model, env.sim.data
    pairs = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        first = model.body_id2name(int(model.geom_bodyid[int(contact.geom1)]))
        second = model.body_id2name(int(model.geom_bodyid[int(contact.geom2)]))
        if first and second and first != second:
            pairs.add(tuple(sorted((str(first), str(second)))))
    return sorted(pairs)


def capture_frame(env, step: int, drawer_qadr: int) -> ChainFrame:
    positions = {}
    tilts = {}
    speeds = {}
    velocities = {}
    for body_name in CHAIN_BODIES:
        body_id = int(env.sim.model.body_name2id(body_name))
        positions[body_name] = np.asarray(
            env.sim.data.body_xpos[body_id], dtype=float
        ).tolist()
        tilts[body_name] = body_tilt_deg(env.sim, body_id)
        try:
            velocity = np.asarray(env.sim.data.body_xvelp[body_id], dtype=float)
        except AttributeError:
            velocity = np.asarray(env.sim.data.cvel[body_id][3:6], dtype=float)
        velocities[body_name] = velocity.tolist()
        speeds[body_name] = float(np.linalg.norm(velocity))
    return ChainFrame(
        step=int(step),
        drawer_qpos=float(env.sim.data.qpos[drawer_qadr]),
        positions=positions,
        tilts_deg=tilts,
        speeds_m_s=speeds,
        contacts=contact_body_pairs(env),
        velocities_m_s=velocities,
    )


def write_trace_json(path: Path, frames: Sequence[ChainFrame], assessment) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": contract(),
        "contract_sha256": contract_sha256(),
        "assessment": (
            assessment.to_dict()
            if isinstance(assessment, ChainAssessment)
            else assessment
        ),
        "frames": [asdict(frame) for frame in frames],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
