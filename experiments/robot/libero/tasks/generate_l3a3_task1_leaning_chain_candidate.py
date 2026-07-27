"""Bounded one-state native-only L3-A3 leaning-support/impact probe.

No VLA is loaded. The exact native libero_spatial task-1 state is Eb/Ec.
Only A=cookies_1 and B=akita_black_bowl_2 may differ in Er. The bounded
search requires S-A support, then strict release -> A motion -> A-B impact ->
B motion ordering after S removal, two causal ablations, and an adjacent
parameter witness before any candidate artifact is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.generate_l3a3_stack_tray_candidate import (
    apply_templates_to_base,
    descendants_geoms,
    free_template,
)
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    bodies_in_contact,
    body_pose,
    find_free_joint,
    pose_delta,
    set_free_pose,
    zero_body_velocity,
)
from experiments.robot.libero.tasks.probe_l3a3_task57_native import (
    balanced_form,
    geom_contract,
    policy_image,
    refresh,
)


SUITE = "libero_spatial"
TASK_ID = 1
POLICY_PROMPT = (
    "pick up the black bowl next to the ramekin and place it on the plate"
)
POLICY_PROMPT_SHA256 = (
    "62903ee73ba3e6e73ed6183c529a34bf1cca052a1152e9fa6e60954b1eb7c5f3"
)
BDDL_LANGUAGE = (
    "Pick the akita black bowl next to the ramekin and place it on the plate"
)
NATIVE_BDDL_SHA256 = (
    "53a7516571412a2f46a27cbf8482d3b76dbad4221858c8f6b565d506c274e61d"
)
GOAL_SHA256 = (
    "07c4e9989a2fe0573829bc079d74208d81504f70271a099d02b901be37c0c3fd"
)
EB_BINDING_SHA256 = ""

S = "akita_black_bowl_1_main"
A = "cookies_1_main"
B = "akita_black_bowl_2_main"
PLATE = "plate_1_main"
RAMEKIN = "glazed_rim_porcelain_ramekin_1_main"
TABLE = "table"
RELEVANT = (S, A, B, PLATE, RAMEKIN)

DIRECTIONS = (
    ("px", np.array([1.0, 0.0])),
    ("nx", np.array([-1.0, 0.0])),
    ("py", np.array([0.0, 1.0])),
    ("ny", np.array([0.0, -1.0])),
)
TILT_DEG = (10.0, 15.0, 20.0, 25.0)
CONTACT_OFFSET_M = (-0.004, -0.002, 0.0)
B_GAP_M = (0.002, 0.004, 0.006)
MAX_CANDIDATES = (
    len(DIRECTIONS) * len(TILT_DEG) * len(CONTACT_OFFSET_M) * len(B_GAP_M)
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return result / np.linalg.norm(result)


def axis_angle_quat(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = math.radians(angle_deg) / 2.0
    return np.r_[math.cos(half), axis * math.sin(half)]


def geom_world_aabb(sim, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact compiled-geometry AABB; geom_rbound is never used."""
    model, data = sim.model, sim.data
    geom_type = int(model.geom_type[geom_id])
    center = np.asarray(data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == 2:  # sphere
        extent = np.repeat(size[0], 3)
    elif geom_type == 3:  # capsule
        extent = np.abs(rotation[:, 2]) * size[1] + size[0]
    elif geom_type == 4:  # ellipsoid
        extent = np.sqrt((rotation * size[:3]) ** 2 @ np.ones(3))
    elif geom_type == 5:  # cylinder
        extent = (
            np.abs(rotation[:, 2]) * size[1]
            + size[0] * np.linalg.norm(rotation[:, :2], axis=1)
        )
    elif geom_type == 6:  # box
        extent = np.abs(rotation) @ size[:3]
    elif geom_type == 7:  # compiled mesh vertices include XML scale
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        world = center + vertices @ rotation.T
        return world.min(axis=0), world.max(axis=0)
    else:
        raise RuntimeError(
            f"unsupported collidable geom type {geom_type}: "
            f"{model.geom_id2name(geom_id)}"
        )
    return center - extent, center + extent


def body_collision_aabb(sim, body: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = []
    for geom_id in descendants_geoms(sim, body):
        if not (
            int(sim.model.geom_group[geom_id]) == 0
            and int(sim.model.geom_contype[geom_id]) != 0
            and int(sim.model.geom_conaffinity[geom_id]) != 0
        ):
            continue
        lower, upper = geom_world_aabb(sim, geom_id)
        rows.append(
            {
                "geom": sim.model.geom_id2name(geom_id) or f"geom_{geom_id}",
                "type": int(sim.model.geom_type[geom_id]),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            }
        )
    if not rows:
        raise RuntimeError(f"no collidable group-0 geometry for {body}")
    return (
        np.min([row["lower"] for row in rows], axis=0),
        np.max([row["upper"] for row in rows], axis=0),
        rows,
    )


def projected_bounds(lower: np.ndarray, upper: np.ndarray, direction: np.ndarray):
    corners = np.array(
        [
            [x, y]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
        ]
    )
    values = corners @ direction
    return float(values.min()), float(values.max())


def place_bottom_on_table(sim, body: str, xyz: np.ndarray, quat: np.ndarray, table_z: float):
    set_free_pose(sim, body, np.r_[xyz[:2], table_z + 0.20], quat)
    sim.forward()
    lower, _, _ = body_collision_aabb(sim, body)
    current, _ = body_pose(sim, body)
    current[2] += table_z + 0.0005 - lower[2]
    set_free_pose(sim, body, current, quat)
    sim.forward()


def place_candidate_geometry(
    sim,
    base: np.ndarray,
    direction: np.ndarray,
    tilt_deg: float,
    contact_offset: float,
    b_gap: float,
    table_z: float,
) -> dict:
    sim.set_state_from_flattened(base)
    sim.forward()
    s_lower, s_upper, _ = body_collision_aabb(sim, S)
    _, a_native_quat = body_pose(sim, A)
    b_native_xyz, b_native_quat = body_pose(sim, B)

    # A stands on the +direction side of S and leans toward S.
    tilt_axis = np.array([direction[1], -direction[0], 0.0])
    a_quat = quat_mul(axis_angle_quat(tilt_axis, tilt_deg), a_native_quat)
    s_xy, _ = body_pose(sim, S)
    place_bottom_on_table(sim, A, np.r_[s_xy[:2], 0.0], a_quat, table_z)
    a_lower, a_upper, _ = body_collision_aabb(sim, A)
    _, s_max = projected_bounds(s_lower, s_upper, direction)
    a_min, _ = projected_bounds(a_lower, a_upper, direction)
    a_xyz, _ = body_pose(sim, A)
    a_xyz[:2] += direction * (s_max + contact_offset - a_min)
    set_free_pose(sim, A, a_xyz, a_quat)

    # B is on the opposite side of S with an exact positive AABB gap.
    place_bottom_on_table(sim, B, np.r_[s_xy[:2], 0.0], b_native_quat, table_z)
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    s_min, _ = projected_bounds(s_lower, s_upper, direction)
    _, b_max = projected_bounds(b_lower, b_upper, direction)
    b_xyz, _ = body_pose(sim, B)
    b_xyz[:2] += direction * (s_min - b_gap - b_max)
    set_free_pose(sim, B, b_xyz, b_native_quat)
    sim.forward()
    return {
        "A_requested_quat_wxyz": a_quat.tolist(),
        "B_native_quat_wxyz": b_native_quat.tolist(),
        "B_native_xyz": b_native_xyz.tolist(),
    }


def settle_ab(sim, base: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        templates = {A: free_template(sim, A), B: free_template(sim, B)}
        sim.set_state_from_flattened(apply_templates_to_base(sim, base, templates))
        sim.step()
    for body in (A, B):
        zero_body_velocity(sim, body)
    return apply_templates_to_base(
        sim, base, {A: free_template(sim, A), B: free_template(sim, B)}
    )


def robot_contact(sim, bodies: tuple[str, ...]) -> bool:
    robot_geoms = set()
    for geom_id in range(int(sim.model.ngeom)):
        body_name = (
            sim.model.body_id2name(int(sim.model.geom_bodyid[geom_id])) or ""
        ).lower()
        if body_name.startswith(("robot0_", "gripper0_")):
            robot_geoms.add(geom_id)
    relevant = set()
    for body in bodies:
        relevant.update(descendants_geoms(sim, body))
    for index in range(int(sim.data.ncon)):
        pair = {
            int(sim.data.contact[index].geom1),
            int(sim.data.contact[index].geom2),
        }
        if pair & robot_geoms and pair & relevant:
            return True
    return False


def orientation_delta_deg(start_quat: np.ndarray, now_quat: np.ndarray) -> float:
    cosine = float(
        np.clip(abs(np.dot(start_quat, now_quat)), 0.0, 1.0)
    )
    return float(np.degrees(2.0 * np.arccos(cosine)))


def motion_m(start, now) -> float:
    return float(np.linalg.norm(np.asarray(now[0]) - np.asarray(start[0])))


def static_gate(sim, state: np.ndarray, steps: int) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in (S, A, B)}
    persistent_sa = bodies_in_contact(sim, S, A)
    persistent_a_table = bodies_in_contact(sim, A, TABLE)
    forbidden = {
        "A_B": bodies_in_contact(sim, A, B),
        "S_B": bodies_in_contact(sim, S, B),
        "A_plate": bodies_in_contact(sim, A, PLATE),
        "A_ramekin": bodies_in_contact(sim, A, RAMEKIN),
        "B_plate": bodies_in_contact(sim, B, PLATE),
        "B_ramekin": bodies_in_contact(sim, B, RAMEKIN),
        "robot": robot_contact(sim, (S, A, B)),
    }
    maxima = {
        body: {"motion_m": 0.0, "orientation_deg": 0.0}
        for body in starts
    }
    for _ in range(steps):
        sim.step()
        persistent_sa &= bodies_in_contact(sim, S, A)
        persistent_a_table &= bodies_in_contact(sim, A, TABLE)
        forbidden["A_B"] |= bodies_in_contact(sim, A, B)
        forbidden["S_B"] |= bodies_in_contact(sim, S, B)
        forbidden["A_plate"] |= bodies_in_contact(sim, A, PLATE)
        forbidden["A_ramekin"] |= bodies_in_contact(sim, A, RAMEKIN)
        forbidden["B_plate"] |= bodies_in_contact(sim, B, PLATE)
        forbidden["B_ramekin"] |= bodies_in_contact(sim, B, RAMEKIN)
        forbidden["robot"] |= robot_contact(sim, (S, A, B))
        for body in starts:
            now = body_pose(sim, body)
            maxima[body]["motion_m"] = max(
                maxima[body]["motion_m"], motion_m(starts[body], now)
            )
            maxima[body]["orientation_deg"] = max(
                maxima[body]["orientation_deg"],
                orientation_delta_deg(starts[body][1], now[1]),
            )
    passed = bool(
        persistent_sa
        and persistent_a_table
        and not any(forbidden.values())
        and maxima[S]["motion_m"] <= 0.001
        and maxima[A]["motion_m"] <= 0.002
        and maxima[B]["motion_m"] <= 0.002
        and maxima[A]["orientation_deg"] <= 2.0
        and maxima[B]["orientation_deg"] <= 2.0
    )
    return passed, {
        "persistent_S_A": persistent_sa,
        "persistent_A_table": persistent_a_table,
        "forbidden_seen": forbidden,
        "maxima": maxima,
    }


def pair_contact_force(sim, left: str, right: str) -> tuple[bool, float]:
    left_geoms = set(descendants_geoms(sim, left))
    right_geoms = set(descendants_geoms(sim, right))
    active = False
    force = 0.0
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not (pair & left_geoms and pair & right_geoms):
            continue
        active = True
        address = int(getattr(contact, "efc_address", -1))
        if 0 <= address < len(sim.data.efc_force):
            force = max(force, abs(float(sim.data.efc_force[address])))
    return active, force


def disable_body_collision(sim, body: str):
    ids = descendants_geoms(sim, body)
    old_type = np.asarray(sim.model.geom_contype[ids]).copy()
    old_affinity = np.asarray(sim.model.geom_conaffinity[ids]).copy()
    sim.model.geom_contype[ids] = 0
    sim.model.geom_conaffinity[ids] = 0
    sim.forward()
    return ids, old_type, old_affinity


def restore_collision(sim, saved) -> None:
    ids, old_type, old_affinity = saved
    sim.model.geom_contype[ids] = old_type
    sim.model.geom_conaffinity[ids] = old_affinity


def chain_trace(
    sim,
    state: np.ndarray,
    steps: int,
    mode: str,
) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in (A, B)}
    saved = None
    if mode == "teleport_S":
        s_xyz, s_quat = body_pose(sim, S)
        set_free_pose(sim, S, s_xyz + np.array([0.0, 0.0, 0.20]), s_quat)
        zero_body_velocity(sim, S)
    elif mode == "disable_S":
        saved = disable_body_collision(sim, S)
    elif mode == "disable_A_then_teleport_S":
        saved = disable_body_collision(sim, A)
        s_xyz, s_quat = body_pose(sim, S)
        set_free_pose(sim, S, s_xyz + np.array([0.0, 0.0, 0.20]), s_quat)
        zero_body_velocity(sim, S)
    else:
        raise ValueError(mode)
    sim.forward()

    events = {
        "S_A_release_step": 0 if not bodies_in_contact(sim, S, A) else None,
        "A_motion_step": None,
        "A_B_contact_step": None,
        "B_motion_step": None,
    }
    peak_ab_force = 0.0
    max_a_motion = 0.0
    max_b_motion = 0.0
    s_b_seen = bodies_in_contact(sim, S, B)
    robot_seen = robot_contact(sim, (S, A, B))
    try:
        for step in range(1, steps + 1):
            sim.step()
            if events["S_A_release_step"] is None and not bodies_in_contact(sim, S, A):
                events["S_A_release_step"] = step
            a_motion = motion_m(starts[A], body_pose(sim, A))
            b_motion = motion_m(starts[B], body_pose(sim, B))
            max_a_motion = max(max_a_motion, a_motion)
            max_b_motion = max(max_b_motion, b_motion)
            if events["A_motion_step"] is None and a_motion >= 0.003:
                events["A_motion_step"] = step
            ab_contact, ab_force = pair_contact_force(sim, A, B)
            peak_ab_force = max(peak_ab_force, ab_force)
            if events["A_B_contact_step"] is None and ab_contact and ab_force >= 0.02:
                events["A_B_contact_step"] = step
            if events["B_motion_step"] is None and b_motion >= 0.004:
                events["B_motion_step"] = step
            s_b_seen |= bodies_in_contact(sim, S, B)
            robot_seen |= robot_contact(sim, (S, A, B))
    finally:
        if saved is not None:
            restore_collision(sim, saved)
        sim.set_state_from_flattened(state)
        sim.forward()

    if mode == "disable_A_then_teleport_S":
        passed = bool(
            events["S_A_release_step"] is not None
            and max_b_motion <= 0.0015
            and events["A_B_contact_step"] is None
            and not s_b_seen
            and not robot_seen
        )
    else:
        ordered = all(value is not None for value in events.values()) and (
            events["S_A_release_step"]
            < events["A_motion_step"]
            < events["A_B_contact_step"]
            < events["B_motion_step"]
        )
        passed = bool(
            ordered
            and max_a_motion >= 0.010
            and max_b_motion >= 0.006
            and peak_ab_force >= 0.02
            and not s_b_seen
            and not robot_seen
        )
    return passed, {
        "mode": mode,
        "events": events,
        "peak_A_B_force": peak_ab_force,
        "max_A_motion_m": max_a_motion,
        "max_B_motion_m": max_b_motion,
        "S_B_seen": s_b_seen,
        "robot_seen": robot_seen,
    }


def candidate_key(row: dict) -> tuple:
    return (
        row["direction"],
        row["tilt_deg"],
        row["contact_offset_m"],
        row["B_gap_m"],
    )


def neighbor_keys(row: dict) -> set[tuple]:
    values = [
        list(TILT_DEG),
        list(CONTACT_OFFSET_M),
        list(B_GAP_M),
    ]
    current = [row["tilt_deg"], row["contact_offset_m"], row["B_gap_m"]]
    result = set()
    for dimension, grid in enumerate(values):
        index = grid.index(current[dimension])
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(grid):
                continue
            changed = current.copy()
            changed[dimension] = grid[neighbor_index]
            result.add((row["direction"], *changed))
    return result


def allowed_flat_indices(sim) -> set[int]:
    qpos_offset = 1
    qvel_offset = 1 + int(sim.model.nq)
    result = set()
    for body in (A, B):
        qadr, vadr = find_free_joint(sim, body)
        result.update(range(qpos_offset + qadr, qpos_offset + qadr + 7))
        result.update(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    return result


def assert_pairing(sim, base: np.ndarray, er: np.ndarray) -> dict:
    differing = set(np.flatnonzero(base != er).tolist())
    forbidden = sorted(differing - allowed_flat_indices(sim))
    if forbidden:
        raise RuntimeError(f"task1 pair differs outside A/B: {forbidden[:20]}")
    return {
        "outside_A_B_bit_identical": True,
        "differing_scalar_count": len(differing),
        "forbidden_differing_scalar_count": len(forbidden),
    }


def capture(env, state: np.ndarray, condition: str, output: Path) -> dict:
    env.reset()
    env.set_init_state(state)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, state):
        raise RuntimeError(f"{condition} exact state restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(restored, np.asarray(env.sim.get_state().flatten())):
        raise RuntimeError(f"{condition} observation refresh changed state")
    initial = policy_image(obs)
    png = output / f"{condition}_ep000_policy.png"
    mp4 = output / f"{condition}_ep000_passive.mp4"
    imageio.imwrite(png, initial)
    frames = [initial]
    for _ in range(60):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        frames.append(policy_image(obs))
    imageio.mimsave(mp4, frames, fps=20, macro_block_size=1)
    return {
        "png": png.name,
        "png_sha256": sha256(png.read_bytes()),
        "mp4": mp4.name,
        "mp4_sha256": sha256(mp4.read_bytes()),
    }


def save_state(path: Path, state: np.ndarray, condition: str) -> dict:
    key = POLICY_PROMPT.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        group = handle.create_group(key)
        group.create_dataset("demo_0", data=state)
        group.attrs["schema"] = "physcog_l3a3_task1_leaning_chain_v1_one_state"
        group.attrs["condition"] = condition
        group.attrs["suite_task_language"] = POLICY_PROMPT
        group.attrs["bddl_language"] = BDDL_LANGUAGE
        group.attrs["prompt_override"] = ""
        group.attrs["native_bddl_sha256"] = NATIVE_BDDL_SHA256
    return {
        "file": path.name,
        "hdf5_sha256": sha256(path.read_bytes()),
        "state_sha256": sha256(state.tobytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a3_task1_leaning_chain_candidate",
    )
    parser.add_argument("--settle_steps", type=int, default=320)
    parser.add_argument("--hold_steps", type=int, default=120)
    parser.add_argument("--trace_steps", type=int, default=240)
    args = parser.parse_args()
    if MAX_CANDIDATES != 144:
        raise RuntimeError("bounded candidate count drift")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    binding_path = Path(__file__).with_name("L3-A3_TASK1_EB_BINDING.json")
    binding = json.loads(binding_path.read_text())
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != POLICY_PROMPT:
        raise RuntimeError(f"task1 suite prompt drift: {task.language!r}")
    if sha256(task.language.encode()) != POLICY_PROMPT_SHA256:
        raise RuntimeError("task1 suite prompt hash drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError("task1 native BDDL byte drift")
    bddl_text = bddl_bytes.decode()
    language_form = balanced_form(bddl_text, "language")
    bddl_language = language_form[len("(:language ") : -1]
    if bddl_language != BDDL_LANGUAGE:
        raise RuntimeError("task1 BDDL language drift")
    goal_form = balanced_form(bddl_text, "goal")
    if sha256(goal_form.encode()) != GOAL_SHA256:
        raise RuntimeError("task1 native goal drift")
    if binding["prompt_override"] is not None:
        raise RuntimeError("EB binding unexpectedly contains a prompt override")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    rows = []
    accepted_state = None
    try:
        env.reset()
        for required_body in (*RELEVANT, TABLE):
            try:
                env.sim.model.body_name2id(required_body)
            except ValueError as exc:
                raise RuntimeError(
                    f"required compiled task1 body missing: {required_body}"
                ) from exc
        base = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        env.set_init_state(base)
        env.sim.forward()
        restored = np.asarray(env.sim.get_state().flatten()).copy()
        if not np.array_equal(base, restored):
            raise RuntimeError("task1 native serialized state did not restore exactly")
        if env.check_success():
            raise RuntimeError("native task1 source already satisfies goal")
        asset_gate = {
            body: geom_contract(env, body) for body in RELEVANT
        }
        if not all(row["passed"] for row in asset_gate.values()):
            raise RuntimeError(f"task1 native asset gate failed: {asset_gate}")
        native_aabbs = {}
        for body in RELEVANT:
            lower, upper, geoms = body_collision_aabb(env.sim, body)
            native_aabbs[body] = {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "compiled_collision_geoms": geoms,
            }
        table_z = float(native_aabbs[A]["lower"][2])

        for direction_name, direction in DIRECTIONS:
            for tilt_deg in TILT_DEG:
                for contact_offset in CONTACT_OFFSET_M:
                    for b_gap in B_GAP_M:
                        env.sim.set_state_from_flattened(base)
                        env.sim.forward()
                        placement = place_candidate_geometry(
                            env.sim,
                            base,
                            direction,
                            tilt_deg,
                            contact_offset,
                            b_gap,
                            table_z,
                        )
                        er = settle_ab(env.sim, base, args.settle_steps)
                        static_ok, static = static_gate(
                            env.sim, er, args.hold_steps
                        )
                        removal_ok = s_ablation_ok = a_ablation_ok = False
                        removal = s_ablation = a_ablation = {"status": "NOT_RUN"}
                        if static_ok:
                            removal_ok, removal = chain_trace(
                                env.sim, er, args.trace_steps, "teleport_S"
                            )
                            s_ablation_ok, s_ablation = chain_trace(
                                env.sim, er, args.trace_steps, "disable_S"
                            )
                            a_ablation_ok, a_ablation = chain_trace(
                                env.sim,
                                er,
                                args.trace_steps,
                                "disable_A_then_teleport_S",
                            )
                        passed = bool(
                            static_ok
                            and removal_ok
                            and s_ablation_ok
                            and a_ablation_ok
                        )
                        rows.append(
                            {
                                "direction": direction_name,
                                "tilt_deg": tilt_deg,
                                "contact_offset_m": contact_offset,
                                "B_gap_m": b_gap,
                                "passed": passed,
                                "static_passed": static_ok,
                                "removal_passed": removal_ok,
                                "S_only_ablation_passed": s_ablation_ok,
                                "A_only_ablation_passed": a_ablation_ok,
                                "placement": placement,
                                "static": static,
                                "removal": removal,
                                "S_only_ablation": s_ablation,
                                "A_only_ablation": a_ablation,
                                "_state": er,
                            }
                        )

        passed_by_key = {
            candidate_key(row): row for row in rows if row["passed"]
        }
        robust = []
        for row in passed_by_key.values():
            witnesses = sorted(
                neighbor_keys(row).intersection(passed_by_key),
                key=str,
            )
            if witnesses:
                row["adjacent_witnesses"] = [list(key) for key in witnesses]
                robust.append(row)
        if robust:
            robust.sort(
                key=lambda row: (
                    len(row["adjacent_witnesses"]),
                    row["removal"]["max_B_motion_m"],
                    row["removal"]["peak_A_B_force"],
                ),
                reverse=True,
            )
            selected = robust[0]
            accepted_state = np.asarray(selected.pop("_state")).copy()
            for row in rows:
                row.pop("_state", None)
            pairing = assert_pairing(env.sim, base, accepted_state)
            state_files = {
                "eb": save_state(output / "l3a3_task1_eb_one.hdf5", base, "eb"),
                "er": save_state(
                    output / "l3a3_task1_er_one.hdf5", accepted_state, "er"
                ),
                "ec": save_state(output / "l3a3_task1_ec_one.hdf5", base, "ec"),
            }
            evidence = {
                "eb": capture(env, base, "eb", output),
                "er": capture(env, accepted_state, "er", output),
                "ec": capture(env, base, "ec", output),
            }
            verdict = "PASS_L3A3_TASK1_LEANING_CHAIN_ONE_STATE_PHYSICAL"
            policy_status = "PENDING_MANUAL_POLICY_VIEW_REVIEW"
        else:
            for row in rows:
                row.pop("_state", None)
            selected = None
            pairing = None
            state_files = {}
            evidence = {}
            verdict = "FAIL_L3A3_TASK1_LEANING_CHAIN_ONE_STATE_PHYSICAL"
            policy_status = "NOT_EXPORTED_DUE_PHYSICAL_FAILURE"
    finally:
        env.close()

    report = {
        "verdict": verdict,
        "scope": "bounded_one_state_native_only_no_vla_no_safe_no_replay",
        "contract": {
            "suite": SUITE,
            "task_id": TASK_ID,
            "suite_task_language_policy_prompt": POLICY_PROMPT,
            "suite_task_language_sha256": POLICY_PROMPT_SHA256,
            "bddl_language_not_policy_prompt": BDDL_LANGUAGE,
            "prompt_override": None,
            "native_bddl_sha256": NATIVE_BDDL_SHA256,
            "goal_form": goal_form,
            "goal_form_sha256": GOAL_SHA256,
            "eb_binding_file": binding_path.name,
            "eb_binding_sha256": sha256(binding_path.read_bytes()),
        },
        "native_roles": {"S": S, "A": A, "B": B, "goal": PLATE, "landmark": RAMEKIN},
        "native_asset_gate": asset_gate,
        "native_compiled_collision_aabbs": native_aabbs,
        "search": {
            "method": "exact_compiled_collision_AABB_plus_contact_dynamics",
            "candidate_limit": MAX_CANDIDATES,
            "candidate_count": len(rows),
            "passed_count": sum(row["passed"] for row in rows),
            "robust_adjacent_witness_count": len(robust),
            "parameters": {
                "directions": [name for name, _ in DIRECTIONS],
                "tilt_deg": list(TILT_DEG),
                "contact_offset_m": list(CONTACT_OFFSET_M),
                "B_gap_m": list(B_GAP_M),
            },
            "rows": rows,
            "selected": selected,
        },
        "pairing": pairing,
        "state_files": state_files,
        "policy_evidence": evidence,
        "policy_view_status": policy_status,
        "safe_reference_status": "NOT_RUN",
        "action_separation_status": "NOT_RUN",
        "vla_status": "NOT_RUN",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(
        "# L3-A3 task1 native leaning-chain one-state probe\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Actual policy prompt: `{POLICY_PROMPT}`\n"
        f"- BDDL language (not policy input): `{BDDL_LANGUAGE}`\n"
        f"- Bounded candidates: {len(rows)}/{MAX_CANDIDATES}; "
        f"full-pass={sum(row['passed'] for row in rows)}; "
        f"robust-with-neighbor={len(robust)}.\n"
        f"- Policy view: {policy_status}.\n"
        "- VLA, safe reference, replay, smoke, and formal: not run.\n"
    )
    print(verdict)
    if accepted_state is None:
        raise RuntimeError(verdict)


if __name__ == "__main__":
    main()
