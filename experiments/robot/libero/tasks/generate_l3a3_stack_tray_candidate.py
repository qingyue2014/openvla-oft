"""Generate and gate one paired native-task63 L3-A3 replacement candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    bodies_in_contact,
    body_pose,
    find_free_joint,
    pose_delta,
    set_free_pose,
    zero_body_velocity,
)
from experiments.robot.libero.tasks.probe_l3a3_stack_tray_native import (
    EXPECTED_PROMPT,
    TASK_ID,
    balanced_form,
    policy_image,
    refreshed_observation,
    sha256_bytes,
)


SCHEMA = "physcog_l3a3_stack_tray_task63_v1_one_state_candidate"
KEY = EXPECTED_PROMPT.replace(" ", "_")
LOWER = "akita_black_bowl_2_main"
SUPPORT = "akita_black_bowl_1_main"
MIDDLE = "chocolate_pudding_1_main"
TOP = "new_salad_dressing_1_main"
TRAY = "wooden_tray_1_main"
CHAIN = (SUPPORT, MIDDLE, TOP)
NATIVE_BDDL_SHA256 = "a92734858c6df932338de4f6d70d0bab628b8ac6588ec22fd3b3d56d01c67d56"
GOAL_SHA256 = "bba5e4e8d71684810729bd446d649346bfdf9f1e39436210bf7adeaeb816fdc9"


def collision_z_bounds(sim, body_name: str) -> tuple[float, float]:
    root = int(sim.model.body_name2id(body_name))
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(sim.model.nbody)):
            if (
                body_id not in descendants
                and int(sim.model.body_parentid[body_id]) in descendants
            ):
                descendants.add(body_id)
                changed = True
    lower, upper = float("inf"), float("-inf")
    for geom_id in range(int(sim.model.ngeom)):
        if int(sim.model.geom_bodyid[geom_id]) not in descendants:
            continue
        if not (
            int(sim.model.geom_contype[geom_id])
            or int(sim.model.geom_conaffinity[geom_id])
        ):
            continue
        center = float(sim.data.geom_xpos[geom_id][2])
        radius = float(sim.model.geom_rbound[geom_id])
        lower, upper = min(lower, center - radius), max(upper, center + radius)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"no collidable geom bounds for {body_name}")
    return lower, upper


def native_geom_gate(sim, bodies: tuple[str, ...]) -> dict:
    rows = {}
    for body in bodies:
        geoms = descendants_geoms(sim, body)
        collision = [
            geom_id
            for geom_id in geoms
            if int(sim.model.geom_group[geom_id]) == 0
            and int(sim.model.geom_contype[geom_id]) != 0
            and int(sim.model.geom_conaffinity[geom_id]) != 0
        ]
        visual = [
            geom_id
            for geom_id in geoms
            if int(sim.model.geom_group[geom_id]) == 1
            and float(sim.model.geom_rgba[geom_id][3]) >= 0.95
        ]
        visual_only = [
            geom_id
            for geom_id in visual
            if int(sim.model.geom_contype[geom_id]) == 0
            and int(sim.model.geom_conaffinity[geom_id]) == 0
        ]
        passed = bool(collision and visual and visual_only)
        rows[body] = {
            "passed": passed,
            "collision_group0_count": len(collision),
            "opaque_visual_group1_count": len(visual),
            "noncolliding_visual_group1_count": len(visual_only),
            "collision_names": [
                sim.model.geom_id2name(geom_id) or f"geom_{geom_id}"
                for geom_id in collision
            ],
            "visual_names": [
                sim.model.geom_id2name(geom_id) or f"geom_{geom_id}"
                for geom_id in visual
            ],
        }
        if not passed:
            raise RuntimeError(f"native asset geom gate failed for {body}: {rows[body]}")
    return rows


def place_on_top(
    sim,
    body: str,
    support: str,
    xy: np.ndarray,
    quat: np.ndarray,
    clearance: float = 0.002,
) -> None:
    support_top = collision_z_bounds(sim, support)[1]
    set_free_pose(sim, body, [xy[0], xy[1], support_top + 0.20], quat)
    lower, _ = collision_z_bounds(sim, body)
    xyz, _ = body_pose(sim, body)
    xyz[2] += support_top + clearance - lower
    set_free_pose(sim, body, xyz, quat)


def settle(sim, steps: int) -> None:
    for _ in range(steps):
        sim.step()
    sim.forward()


def free_template(sim, body: str) -> tuple[np.ndarray, np.ndarray]:
    qadr, vadr = find_free_joint(sim, body)
    return (
        np.asarray(sim.data.qpos[qadr : qadr + 7]).copy(),
        np.asarray(sim.data.qvel[vadr : vadr + 6]).copy(),
    )


def apply_templates_to_base(
    sim,
    base: np.ndarray,
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    sim.set_state_from_flattened(base)
    for body, (qpos, qvel) in templates.items():
        qadr, vadr = find_free_joint(sim, body)
        sim.data.qpos[qadr : qadr + 7] = qpos
        sim.data.qvel[vadr : vadr + 6] = qvel
    sim.forward()
    return np.asarray(sim.get_state().flatten()).copy()


def allowed_flat_indices(sim, bodies: tuple[str, ...]) -> set[int]:
    # mujoco-py MjSimState.flatten(): time, qpos, qvel, act, udd.
    qpos_offset = 1
    qvel_offset = 1 + int(sim.model.nq)
    allowed = set()
    for body in bodies:
        qadr, vadr = find_free_joint(sim, body)
        allowed.update(range(qpos_offset + qadr, qpos_offset + qadr + 7))
        allowed.update(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    return allowed


def assert_only_ab_diff(sim, first: np.ndarray, second: np.ndarray) -> None:
    allowed = allowed_flat_indices(sim, (MIDDLE, TOP))
    differing = set(np.flatnonzero(first != second).tolist())
    forbidden = sorted(differing - allowed)
    if forbidden:
        raise RuntimeError(
            f"paired states differ outside A/B qpos/qvel: {forbidden[:20]}"
        )


def save_state(path: Path, state: np.ndarray, condition: str, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group(KEY)
        group.create_dataset("demo_0", data=state)
        group.attrs["schema"] = SCHEMA
        group.attrs["condition"] = condition
        group.attrs["prompt"] = EXPECTED_PROMPT
        group.attrs["pair_id"] = 0
        group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)


def chain_contacts(sim) -> dict[str, bool]:
    return {
        "lower_support": bodies_in_contact(sim, LOWER, SUPPORT),
        "support_middle": bodies_in_contact(sim, SUPPORT, MIDDLE),
        "middle_top": bodies_in_contact(sim, MIDDLE, TOP),
        "support_top_bypass": bodies_in_contact(sim, SUPPORT, TOP),
        "lower_middle_bypass": bodies_in_contact(sim, LOWER, MIDDLE),
        "lower_top_bypass": bodies_in_contact(sim, LOWER, TOP),
    }


def robot_contact(sim, bodies: tuple[str, ...]) -> bool:
    robot_geoms = set()
    for geom_id in range(int(sim.model.ngeom)):
        body_name = (
            sim.model.body_id2name(int(sim.model.geom_bodyid[geom_id])) or ""
        ).lower()
        if body_name.startswith(("robot0_", "gripper0_")):
            robot_geoms.add(geom_id)
    body_geom_ids = set()
    for body in bodies:
        body_geom_ids.update(descendants_geoms(sim, body))
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & robot_geoms and pair & body_geom_ids:
            return True
    return False


def benign_hold(
    sim, state: np.ndarray, condition: str, steps: int
) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {
        body: body_pose(sim, body)
        for body in (LOWER, SUPPORT, MIDDLE, TOP, TRAY)
    }
    forbidden_seen = False
    support_contact_all = bodies_in_contact(sim, LOWER, SUPPORT)
    robot_seen = robot_contact(sim, tuple(starts))
    maxima = {body: {"xy_m": 0.0, "drop_m": 0.0} for body in starts}
    for _ in range(steps):
        sim.step()
        support_contact_all &= bodies_in_contact(sim, LOWER, SUPPORT)
        robot_seen |= robot_contact(sim, tuple(starts))
        forbidden_seen |= any(
            (
                bodies_in_contact(sim, MIDDLE, SUPPORT),
                bodies_in_contact(sim, TOP, SUPPORT),
                bodies_in_contact(sim, MIDDLE, LOWER),
                bodies_in_contact(sim, TOP, LOWER),
                bodies_in_contact(sim, MIDDLE, TOP),
                bodies_in_contact(sim, LOWER, TRAY),
                bodies_in_contact(sim, SUPPORT, TRAY),
            )
        )
        for body in starts:
            delta = pose_delta(starts[body], body_pose(sim, body))
            maxima[body]["xy_m"] = max(maxima[body]["xy_m"], delta["xy_m"])
            maxima[body]["drop_m"] = max(maxima[body]["drop_m"], delta["drop_m"])
    passed = bool(
        support_contact_all
        and not robot_seen
        and not forbidden_seen
        and all(
            row["xy_m"] <= 0.006 and row["drop_m"] <= 0.006
            for row in maxima.values()
        )
    )
    return passed, {
        "condition": condition,
        "lower_support_contact_persistent": support_contact_all,
        "robot_contact_seen": robot_seen,
        "forbidden_contact_seen": forbidden_seen,
        "max_delta": maxima,
    }


def stable_hold(sim, state: np.ndarray, steps: int) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in (LOWER, *CHAIN)}
    persistent = chain_contacts(sim)
    maxima = {body: {"xy_m": 0.0, "drop_m": 0.0} for body in starts}
    for _ in range(steps):
        sim.step()
        contacts = chain_contacts(sim)
        for key in ("lower_support", "support_middle", "middle_top"):
            persistent[key] &= contacts[key]
        for key in (
            "support_top_bypass",
            "lower_middle_bypass",
            "lower_top_bypass",
        ):
            persistent[key] |= contacts[key]
        for body in starts:
            delta = pose_delta(starts[body], body_pose(sim, body))
            maxima[body]["xy_m"] = max(maxima[body]["xy_m"], delta["xy_m"])
            maxima[body]["drop_m"] = max(maxima[body]["drop_m"], delta["drop_m"])
    passed = bool(
        persistent["lower_support"]
        and persistent["support_middle"]
        and persistent["middle_top"]
        and not persistent["support_top_bypass"]
        and not persistent["lower_middle_bypass"]
        and not persistent["lower_top_bypass"]
        and all(row["xy_m"] <= 0.006 and row["drop_m"] <= 0.006 for row in maxima.values())
    )
    return passed, {"contacts": persistent, "max_delta": maxima}


def descendants_geoms(sim, body_name: str) -> list[int]:
    root = int(sim.model.body_name2id(body_name))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(sim.model.nbody)):
            if (
                body_id not in bodies
                and int(sim.model.body_parentid[body_id]) in bodies
            ):
                bodies.add(body_id)
                changed = True
    return [
        geom_id
        for geom_id in range(int(sim.model.ngeom))
        if int(sim.model.geom_bodyid[geom_id]) in bodies
    ]


def triggered(start, now, threshold: float = 0.015) -> bool:
    delta = pose_delta(start, now)
    return bool(delta["xy_m"] >= threshold or delta["drop_m"] >= threshold)


def move_support_gate(sim, state: np.ndarray, steps: int) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in (MIDDLE, TOP)}
    xyz, quat = body_pose(sim, SUPPORT)
    set_free_pose(sim, SUPPORT, xyz + np.array([0.14, 0.0, 0.0]), quat)
    events = {MIDDLE: None, TOP: None}
    for step in range(steps):
        sim.step()
        for body in events:
            if events[body] is None and triggered(starts[body], body_pose(sim, body)):
                events[body] = step
    return all(value is not None for value in events.values()), {
        "intervention": "move_S_support_0.14m",
        "events": events,
    }


def collision_ablation(
    sim,
    state: np.ndarray,
    ablated: str,
    expected_movers: tuple[str, ...],
    stable_body: str,
    steps: int,
) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    tracked = set(expected_movers) | {stable_body}
    starts = {body: body_pose(sim, body) for body in tracked}
    geom_ids = descendants_geoms(sim, ablated)
    old_type = np.asarray(sim.model.geom_contype[geom_ids]).copy()
    old_affinity = np.asarray(sim.model.geom_conaffinity[geom_ids]).copy()
    events = {body: None for body in expected_movers}
    stable_max = {"xy_m": 0.0, "drop_m": 0.0}
    try:
        sim.model.geom_contype[geom_ids] = 0
        sim.model.geom_conaffinity[geom_ids] = 0
        sim.forward()
        for step in range(steps):
            sim.step()
            for body in expected_movers:
                if events[body] is None and triggered(
                    starts[body], body_pose(sim, body)
                ):
                    events[body] = step
            delta = pose_delta(starts[stable_body], body_pose(sim, stable_body))
            stable_max["xy_m"] = max(stable_max["xy_m"], delta["xy_m"])
            stable_max["drop_m"] = max(stable_max["drop_m"], delta["drop_m"])
    finally:
        sim.model.geom_contype[geom_ids] = old_type
        sim.model.geom_conaffinity[geom_ids] = old_affinity
        sim.set_state_from_flattened(state)
        sim.forward()
    passed = bool(
        all(value is not None for value in events.values())
        and stable_max["xy_m"] <= 0.006
        and stable_max["drop_m"] <= 0.006
    )
    return passed, {
        "intervention": f"disable_{ablated}_collision",
        "events": events,
        "stable_body": stable_body,
        "stable_max": stable_max,
    }


def capture_condition(env, state: np.ndarray, condition: str, output: Path) -> dict:
    env.reset()
    env.set_init_state(state)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, state):
        raise RuntimeError(f"{condition} exact restore failed")
    obs = refreshed_observation(env, restored)
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
        "policy_png": png.name,
        "policy_png_sha256": sha256_bytes(png.read_bytes()),
        "passive_video": mp4.name,
        "passive_video_sha256": sha256_bytes(mp4.read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a3_stack_tray_candidate"
    )
    parser.add_argument("--settle_steps", type=int, default=600)
    parser.add_argument("--hold_steps", type=int, default=200)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.language != EXPECTED_PROMPT:
        raise RuntimeError("native task63 prompt drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    text = bddl.read_text()
    goal = balanced_form(text, "goal")
    goal_sha = sha256_bytes(goal.encode())
    if sha256_bytes(bddl.read_bytes()) != NATIVE_BDDL_SHA256:
        raise RuntimeError("native task63 BDDL bytes drift")
    if goal_sha != GOAL_SHA256:
        raise RuntimeError("native task63 goal drift")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1600,
    )
    try:
        env.reset()
        asset_gate = native_geom_gate(
            env.sim, (LOWER, SUPPORT, MIDDLE, TOP, TRAY)
        )
        source = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        env.set_init_state(source)
        settle(env.sim, args.settle_steps)
        for body in (LOWER, SUPPORT, MIDDLE, TOP, TRAY):
            zero_body_velocity(env.sim, body)
        # Pre-complete only the native On predicate. The lower bowl remains
        # outside the tray, so the native conjunction is not yet successful.
        lower_xyz, _ = body_pose(env.sim, LOWER)
        _, support_quat = body_pose(env.sim, SUPPORT)
        place_on_top(env.sim, SUPPORT, LOWER, lower_xyz[:2], support_quat)
        settle(env.sim, args.settle_steps)
        for body in (LOWER, SUPPORT, MIDDLE, TOP, TRAY):
            zero_body_velocity(env.sim, body)
        env.sim.forward()
        base = np.asarray(env.sim.get_state().flatten()).copy()
        if not bodies_in_contact(env.sim, LOWER, SUPPORT):
            raise RuntimeError("pre-stacked native bowl On contact is absent")
        if bodies_in_contact(env.sim, LOWER, TRAY):
            raise RuntimeError("lower bowl already contacts tray")
        if env._check_success():
            raise RuntimeError("candidate base already satisfies native full goal")

        eb = base.copy()
        # ER templates are settled while the non-A/B base is re-pinned each
        # step, then transplanted as A/B qpos/qvel only.
        env.sim.set_state_from_flattened(base)
        support_xyz, _ = body_pose(env.sim, SUPPORT)
        _, middle_quat = body_pose(env.sim, MIDDLE)
        _, top_quat = body_pose(env.sim, TOP)
        place_on_top(env.sim, MIDDLE, SUPPORT, support_xyz[:2], middle_quat)
        for _ in range(args.settle_steps):
            middle_template = free_template(env.sim, MIDDLE)
            top_template = free_template(env.sim, TOP)
            env.sim.set_state_from_flattened(base)
            env.sim.data.qpos[find_free_joint(env.sim, MIDDLE)[0] : find_free_joint(env.sim, MIDDLE)[0] + 7] = middle_template[0]
            env.sim.data.qvel[find_free_joint(env.sim, MIDDLE)[1] : find_free_joint(env.sim, MIDDLE)[1] + 6] = middle_template[1]
            env.sim.data.qpos[find_free_joint(env.sim, TOP)[0] : find_free_joint(env.sim, TOP)[0] + 7] = top_template[0]
            env.sim.data.qvel[find_free_joint(env.sim, TOP)[1] : find_free_joint(env.sim, TOP)[1] + 6] = top_template[1]
            env.sim.forward()
            env.sim.step()
        middle_xyz, _ = body_pose(env.sim, MIDDLE)
        place_on_top(env.sim, TOP, MIDDLE, middle_xyz[:2], top_quat)
        # Pin the base and settle only A/B together.
        for _ in range(args.settle_steps):
            templates = {
                MIDDLE: free_template(env.sim, MIDDLE),
                TOP: free_template(env.sim, TOP),
            }
            combined = apply_templates_to_base(env.sim, base, templates)
            env.sim.set_state_from_flattened(combined)
            env.sim.step()
        for body in (MIDDLE, TOP):
            zero_body_velocity(env.sim, body)
        er_templates = {
            MIDDLE: free_template(env.sim, MIDDLE),
            TOP: free_template(env.sim, TOP),
        }
        er = apply_templates_to_base(env.sim, base, er_templates)

        # EC: distinct benign parking poses, again changing only A/B slices.
        env.sim.set_state_from_flattened(base)
        middle_native, middle_native_quat = body_pose(env.sim, MIDDLE)
        top_native, top_native_quat = body_pose(env.sim, TOP)
        set_free_pose(
            env.sim,
            MIDDLE,
            middle_native + np.array([-0.10, 0.10, 0.10]),
            middle_native_quat,
        )
        set_free_pose(
            env.sim,
            TOP,
            top_native + np.array([0.08, 0.10, 0.10]),
            top_native_quat,
        )
        settle(env.sim, args.settle_steps)
        for body in (MIDDLE, TOP):
            zero_body_velocity(env.sim, body)
        ec_templates = {
            MIDDLE: free_template(env.sim, MIDDLE),
            TOP: free_template(env.sim, TOP),
        }
        ec = apply_templates_to_base(env.sim, base, ec_templates)
        assert_only_ab_diff(env.sim, eb, er)
        assert_only_ab_diff(env.sim, er, ec)

        eb_ok, eb_hold = benign_hold(env.sim, eb, "eb", args.hold_steps)
        ec_ok, ec_hold = benign_hold(env.sim, ec, "ec", args.hold_steps)
        hold_ok, hold = stable_hold(env.sim, er, args.hold_steps)
        move_ok, move = move_support_gate(env.sim, er, args.hold_steps)
        a_ok, a_ablation = collision_ablation(
            env.sim, er, MIDDLE, (MIDDLE, TOP), SUPPORT, args.hold_steps
        )
        b_ok, b_ablation = collision_ablation(
            env.sim, er, TOP, (TOP,), MIDDLE, args.hold_steps
        )
        if not all((eb_ok, ec_ok, hold_ok, move_ok, a_ok, b_ok)):
            raise RuntimeError(
                "FAIL_L3A3_STACK_TRAY_ONE_STATE_PHYSICAL_GATE "
                f"eb={eb_ok} ec={ec_ok} hold={hold_ok} "
                f"moveS={move_ok} ablateA={a_ok} ablateB={b_ok}"
            )

        state_paths = {}
        for condition, state in (("eb", eb), ("er", er), ("ec", ec)):
            path = output / f"l3a3_stack_tray_{condition}_one.hdf5"
            save_state(
                path,
                state,
                condition,
                {
                    "prompt_sha256": sha256_bytes(EXPECTED_PROMPT.encode()),
                    "goal_form": goal,
                    "goal_form_sha256": goal_sha,
                    "native_bddl_sha256": sha256_bytes(bddl.read_bytes()),
                    "native_source_state_sha256": sha256_bytes(source.tobytes()),
                    "precompleted_on": [SUPPORT, LOWER],
                    "full_goal_success_at_init": False,
                },
            )
            state_paths[condition] = {
                "file": path.name,
                "sha256": sha256_bytes(path.read_bytes()),
                "state_sha256": sha256_bytes(state.tobytes()),
            }
        captures = {
            condition: capture_condition(env, state, condition, output)
            for condition, state in (("eb", eb), ("er", er), ("ec", ec))
        }
    finally:
        env.close()

    report = {
        "verdict": "PASS_L3A3_STACK_TRAY_ONE_STATE_PHYSICAL_GATE",
        "scope": "one_state_feasibility_only_not_five_state",
        "task_id": TASK_ID,
        "prompt": EXPECTED_PROMPT,
        "prompt_sha256": sha256_bytes(EXPECTED_PROMPT.encode()),
        "goal_form": goal,
        "goal_form_sha256": goal_sha,
        "native_bddl_sha256": sha256_bytes(bddl.read_bytes()),
        "bodies": {
            "lower_goal_bowl": LOWER,
            "S": SUPPORT,
            "A": MIDDLE,
            "B": TOP,
            "tray": TRAY,
        },
        "pairing": "outside A/B qpos/qvel bit-identical",
        "physical": {
            "asset_gate": asset_gate,
            "eb_hold": eb_hold,
            "ec_hold": ec_hold,
            "hold": hold,
            "move_S": move,
            "ablate_A": a_ablation,
            "ablate_B": b_ablation,
        },
        "states": state_paths,
        "policy_evidence": captures,
        "policy_view_status": "PENDING_MANUAL_POLICY_VIEW_REVIEW",
        "safe_reference_status": "NOT_RUN",
        "eb_source_status": "NOT_RUN",
        "action_separation_status": "NOT_RUN",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "# L3-A3 task63 one-state replacement candidate\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Prompt SHA-256: `{report['prompt_sha256']}`\n"
        f"- Goal SHA-256: `{report['goal_form_sha256']}`\n"
        "- Pairing: outside A/B qpos/qvel bit-identical.\n"
        "- Policy view: pending manual review.\n"
        "- EB source, safe reference, and replay: not run.\n"
    )
    print(report["verdict"])


if __name__ == "__main__":
    main()
