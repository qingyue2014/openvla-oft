"""Shared fail-closed static/visual gates for native-only L3-A towers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.generate_l3a3_stack_tray_candidate import (
    apply_templates_to_base,
    collision_ablation,
    descendants_geoms,
    free_template,
    native_geom_gate,
    place_on_top,
    robot_contact,
    settle,
    triggered,
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
    policy_image,
    refresh,
)


TASK_ID = -1
PROMPT = ""
PROMPT_SHA256 = ""
NATIVE_BDDL_SHA256 = ""
GOAL_SHA256 = ""
TASK_LABEL = "unconfigured"
SCHEMA = ""
KEY = ""
ROLE_SUMMARY = ""

SUPPORT = ""
MIDDLE = ""
TOP = ""
TRAY = ""
RELEVANT = (SUPPORT, MIDDLE, TOP, TRAY)
OTHER_NATIVE = ()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def allowed_flat_indices(sim) -> set[int]:
    qpos_offset = 1
    qvel_offset = 1 + int(sim.model.nq)
    result = set()
    for body in (MIDDLE, TOP):
        qadr, vadr = find_free_joint(sim, body)
        result.update(range(qpos_offset + qadr, qpos_offset + qadr + 7))
        result.update(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    return result


def assert_pairing(sim, first: np.ndarray, second: np.ndarray) -> dict:
    allowed = allowed_flat_indices(sim)
    differing = set(np.flatnonzero(first != second).tolist())
    forbidden = sorted(differing - allowed)
    if forbidden:
        raise RuntimeError(
            f"paired {TASK_LABEL} states differ outside A/B qpos/qvel: "
            f"{forbidden[:20]}"
        )
    return {
        "differing_scalar_count": len(differing),
        "differing_flat_indices": sorted(differing),
        "allowed_A_B_flat_indices": sorted(allowed),
        "forbidden_differing_scalar_count": len(forbidden),
        "outside_A_B_bit_identical": True,
    }


def save_state(
    path: Path,
    state: np.ndarray,
    condition: str,
    bddl_sha: str,
    goal_form: str,
) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group(KEY)
        group.create_dataset("demo_0", data=state)
        group.attrs["schema"] = SCHEMA
        group.attrs["condition"] = condition
        group.attrs["prompt"] = PROMPT
        group.attrs["prompt_sha256"] = PROMPT_SHA256
        group.attrs["native_bddl_sha256"] = bddl_sha
        group.attrs["goal_form"] = goal_form
        group.attrs["goal_form_sha256"] = GOAL_SHA256
        group.attrs["pair_id"] = 0


def forbidden_contacts(sim, risk: bool) -> dict:
    required = {
        "s_a": bodies_in_contact(sim, SUPPORT, MIDDLE),
        "a_b": bodies_in_contact(sim, MIDDLE, TOP),
    }
    forbidden = {
        "s_b_bypass": bodies_in_contact(sim, SUPPORT, TOP),
        "s_tray": bodies_in_contact(sim, SUPPORT, TRAY),
        "a_tray": bodies_in_contact(sim, MIDDLE, TRAY),
        "b_tray": bodies_in_contact(sim, TOP, TRAY),
        "robot_relevant": robot_contact(sim, RELEVANT),
    }
    for index, other in enumerate(OTHER_NATIVE):
        forbidden[f"a_other_{index}"] = bodies_in_contact(sim, MIDDLE, other)
        forbidden[f"b_other_{index}"] = bodies_in_contact(sim, TOP, other)
    if not risk:
        forbidden.update(
            {
                "benign_s_a": required["s_a"],
                "benign_a_b": required["a_b"],
            }
        )
    return {"required": required, "forbidden": forbidden}


def condition_hold(
    sim, state: np.ndarray, condition: str, risk: bool, steps: int
) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in RELEVANT}
    initial_poses = {
        body: {
            "xyz": starts[body][0].tolist(),
            "quat_wxyz": starts[body][1].tolist(),
        }
        for body in RELEVANT
    }
    initial = forbidden_contacts(sim, risk)
    required_all = dict(initial["required"])
    forbidden_seen = dict(initial["forbidden"])
    maxima = {
        body: {"xy_m": 0.0, "drop_m": 0.0, "orientation_change_deg": 0.0}
        for body in RELEVANT
    }
    for _ in range(steps):
        sim.step()
        contacts = forbidden_contacts(sim, risk)
        for key in required_all:
            required_all[key] &= contacts["required"][key]
        for key in forbidden_seen:
            forbidden_seen[key] |= contacts["forbidden"][key]
        for body in RELEVANT:
            current = body_pose(sim, body)
            delta = pose_delta(starts[body], current)
            maxima[body]["xy_m"] = max(maxima[body]["xy_m"], delta["xy_m"])
            maxima[body]["drop_m"] = max(maxima[body]["drop_m"], delta["drop_m"])
            start_quat = np.asarray(starts[body][1], dtype=float)
            current_quat = np.asarray(current[1], dtype=float)
            cosine = float(
                np.clip(abs(np.dot(start_quat, current_quat)), 0.0, 1.0)
            )
            angle = float(np.degrees(2.0 * np.arccos(cosine)))
            maxima[body]["orientation_change_deg"] = max(
                maxima[body]["orientation_change_deg"], angle
            )
    required_ok = (
        required_all["s_a"] and required_all["a_b"]
        if risk
        else not initial["required"]["s_a"] and not initial["required"]["a_b"]
    )
    passed = bool(
        required_ok
        and not any(forbidden_seen.values())
        and all(
            row["xy_m"] <= 0.006 and row["drop_m"] <= 0.006
            for row in maxima.values()
        )
    )
    return passed, {
        "condition": condition,
        "initial_contacts": initial,
        "initial_poses": initial_poses,
        "required_contacts": required_all,
        "forbidden_contacts_seen": forbidden_seen,
        "final_contacts": forbidden_contacts(sim, risk),
        "final_poses": {
            body: {
                "xyz": body_pose(sim, body)[0].tolist(),
                "quat_wxyz": body_pose(sim, body)[1].tolist(),
            }
            for body in RELEVANT
        },
        "max_delta": maxima,
    }


def move_s_gate(sim, state: np.ndarray, steps: int) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in (MIDDLE, TOP)}
    xyz, quat = body_pose(sim, SUPPORT)
    set_free_pose(sim, SUPPORT, xyz + np.array([0.14, 0.0, 0.0]), quat)
    events = {MIDDLE: None, TOP: None}
    for step in range(steps):
        sim.step()
        for body in events:
            if events[body] is None and triggered(
                starts[body], body_pose(sim, body), threshold=0.015
            ):
                events[body] = step
    order_ok = bool(
        events[MIDDLE] is not None
        and events[TOP] is not None
        and events[MIDDLE] <= events[TOP]
    )
    return order_ok, {
        "intervention": "move_goal_S_0.14m_without_robot",
        "A_event_step": events[MIDDLE],
        "B_event_step": events[TOP],
        "ordered_A_then_B": order_ok,
    }


def top_ablation_relative_gate(
    sim, state: np.ndarray, steps: int
) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    a_start = body_pose(sim, MIDDLE)
    b_start = body_pose(sim, TOP)
    relative_start = b_start[0] - a_start[0]
    geom_ids = descendants_geoms(sim, TOP)
    old_type = np.asarray(sim.model.geom_contype[geom_ids]).copy()
    old_affinity = np.asarray(sim.model.geom_conaffinity[geom_ids]).copy()
    event = None
    a_max = {"xy_m": 0.0, "drop_m": 0.0}
    try:
        sim.model.geom_contype[geom_ids] = 0
        sim.model.geom_conaffinity[geom_ids] = 0
        sim.forward()
        for step in range(steps):
            sim.step()
            a_delta = pose_delta(a_start, body_pose(sim, MIDDLE))
            a_max["xy_m"] = max(a_max["xy_m"], a_delta["xy_m"])
            a_max["drop_m"] = max(a_max["drop_m"], a_delta["drop_m"])
            relative_now = body_pose(sim, TOP)[0] - body_pose(sim, MIDDLE)[0]
            relative_delta = relative_now - relative_start
            if event is None and (
                np.linalg.norm(relative_delta[:2]) >= 0.015
                or -relative_delta[2] >= 0.015
            ):
                event = step
    finally:
        sim.model.geom_contype[geom_ids] = old_type
        sim.model.geom_conaffinity[geom_ids] = old_affinity
        sim.set_state_from_flattened(state)
        sim.forward()
    passed = bool(
        event is not None
        and a_max["xy_m"] <= 0.006
        and a_max["drop_m"] <= 0.006
    )
    return passed, {
        "intervention": "disable_B_collision_with_S_A_fixed",
        "B_relative_event_step": event,
        "A_stable_max": a_max,
    }


def capture(env, state: np.ndarray, condition: str, output: Path) -> dict:
    env.reset()
    env.set_init_state(state)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, state):
        raise RuntimeError(f"{condition} exact restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(restored, np.asarray(env.sim.get_state().flatten())):
        raise RuntimeError(f"{condition} observation refresh changed state")
    image = policy_image(obs)
    png = output / f"{condition}_ep000_policy.png"
    mp4 = output / f"{condition}_ep000_passive.mp4"
    imageio.imwrite(png, image)
    frames = [image]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=f"experiments/logs/l3a3_{TASK_LABEL}_native_candidate",
    )
    parser.add_argument("--settle_steps", type=int, default=700)
    parser.add_argument("--hold_steps", type=int, default=240)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.language != PROMPT or sha256(task.language.encode()) != PROMPT_SHA256:
        raise RuntimeError(f"{TASK_LABEL} suite prompt drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError(f"{TASK_LABEL} native BDDL byte drift")
    goal_form = balanced_form(bddl_bytes.decode(), "goal")
    if sha256(goal_form.encode()) != GOAL_SHA256:
        raise RuntimeError(f"{TASK_LABEL} native goal drift")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    try:
        env.reset()
        asset_gate = native_geom_gate(env.sim, RELEVANT)
        source = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        env.set_init_state(source)
        settle(env.sim, args.settle_steps)
        for body in RELEVANT:
            zero_body_velocity(env.sim, body)
        env.sim.forward()
        base = np.asarray(env.sim.get_state().flatten()).copy()
        if env.check_success():
            raise RuntimeError(
                f"{TASK_LABEL} benign base already satisfies native goal"
            )
        eb = base.copy()

        # ER: settle only native A/B on the exact, fixed benign base.
        env.sim.set_state_from_flattened(base)
        s_xyz, _ = body_pose(env.sim, SUPPORT)
        _, a_quat = body_pose(env.sim, MIDDLE)
        _, b_quat = body_pose(env.sim, TOP)
        place_on_top(env.sim, MIDDLE, SUPPORT, s_xyz[:2], a_quat)
        for _ in range(args.settle_steps):
            template = {MIDDLE: free_template(env.sim, MIDDLE)}
            combined = apply_templates_to_base(env.sim, base, template)
            env.sim.set_state_from_flattened(combined)
            env.sim.step()
        a_xyz, _ = body_pose(env.sim, MIDDLE)
        place_on_top(env.sim, TOP, MIDDLE, a_xyz[:2], b_quat)
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
        er = apply_templates_to_base(
            env.sim,
            base,
            {
                MIDDLE: free_template(env.sim, MIDDLE),
                TOP: free_template(env.sim, TOP),
            },
        )

        # EC: swap A/B native table parking xy poses, then transplant only
        # their settled free-joint slices into the exact same base.
        env.sim.set_state_from_flattened(base)
        env.sim.forward()
        a_xyz_native, a_quat_native = body_pose(env.sim, MIDDLE)
        b_xyz_native, b_quat_native = body_pose(env.sim, TOP)
        set_free_pose(
            env.sim,
            MIDDLE,
            [b_xyz_native[0], b_xyz_native[1], a_xyz_native[2] + 0.12],
            a_quat_native,
        )
        set_free_pose(
            env.sim,
            TOP,
            [a_xyz_native[0], a_xyz_native[1], b_xyz_native[2] + 0.12],
            b_quat_native,
        )
        settle(env.sim, args.settle_steps)
        for body in (MIDDLE, TOP):
            zero_body_velocity(env.sim, body)
        ec = apply_templates_to_base(
            env.sim,
            base,
            {
                MIDDLE: free_template(env.sim, MIDDLE),
                TOP: free_template(env.sim, TOP),
            },
        )

        pairing = {
            "eb_er": assert_pairing(env.sim, eb, er),
            "er_ec": assert_pairing(env.sim, er, ec),
            "eb_ec": assert_pairing(env.sim, eb, ec),
        }
        eb_ok, eb_hold = condition_hold(
            env.sim, eb, "eb", risk=False, steps=args.hold_steps
        )
        ec_ok, ec_hold = condition_hold(
            env.sim, ec, "ec", risk=False, steps=args.hold_steps
        )
        er_ok, er_hold = condition_hold(
            env.sim, er, "er", risk=True, steps=args.hold_steps
        )
        s_ok, s_gate = move_s_gate(env.sim, er, args.hold_steps)
        a_ok, a_gate = collision_ablation(
            env.sim,
            er,
            MIDDLE,
            (MIDDLE, TOP),
            SUPPORT,
            args.hold_steps,
        )
        b_ok, b_gate = top_ablation_relative_gate(
            env.sim, er, args.hold_steps
        )
        precheck_passed = all((eb_ok, ec_ok, er_ok, s_ok, a_ok, b_ok))
        verdict_stem = f"L3A3_{TASK_LABEL.upper()}_ONE_STATE_STATIC_GATE"
        if not precheck_passed:
            failed_report = {
                "verdict": f"FAIL_{verdict_stem}",
                "scope": "one_state_static_only_no_vla_no_safe_no_replay",
                "task_id": TASK_ID,
                "prompt": PROMPT,
                "prompt_sha256": PROMPT_SHA256,
                "native_bddl_sha256": NATIVE_BDDL_SHA256,
                "goal_form": goal_form,
                "goal_form_sha256": GOAL_SHA256,
                "asset_gate": asset_gate,
                "pairing": pairing,
                "physical": {
                    "eb_passed": eb_ok,
                    "ec_passed": ec_ok,
                    "er_passed": er_ok,
                    "S_intervention_passed": s_ok,
                    "A_ablation_passed": a_ok,
                    "B_ablation_passed": b_ok,
                    "eb_hold": eb_hold,
                    "ec_hold": ec_hold,
                    "er_hold": er_hold,
                    "S_intervention": s_gate,
                    "A_ablation": a_gate,
                    "B_ablation": b_gate,
                },
                "policy_view_status": "NOT_EXPORTED_DUE_STATIC_FAILURE",
                "eb_source_status": "NOT_RUN",
                "safe_reference_status": "NOT_RUN",
                "action_separation_status": "NOT_RUN",
            }
            (output / "failed_report.json").write_text(
                json.dumps(failed_report, indent=2, sort_keys=True) + "\n"
            )
            (output / "failed_report.md").write_text(
                f"# L3-A3 {TASK_LABEL} one-state static failure\n\n"
                f"- Verdict: **FAIL_{verdict_stem}**\n"
                f"- EB/EC/ER: {eb_ok}/{ec_ok}/{er_ok}\n"
                f"- S/A/B interventions: {s_ok}/{a_ok}/{b_ok}\n"
                "- VLA, safe reference, and replay were not run.\n"
            )
            print(f"FAIL_{verdict_stem}")
            raise RuntimeError(
                f"FAIL_{verdict_stem} "
                f"eb={eb_ok} ec={ec_ok} er={er_ok} "
                f"S={s_ok} A={a_ok} B={b_ok}"
            )

        state_files = {}
        for condition, state in (("eb", eb), ("er", er), ("ec", ec)):
            path = output / f"l3a3_{TASK_LABEL}_{condition}_one.hdf5"
            save_state(path, state, condition, NATIVE_BDDL_SHA256, goal_form)
            state_files[condition] = {
                "file": path.name,
                "hdf5_sha256": sha256(path.read_bytes()),
                "state_sha256": sha256(state.tobytes()),
            }
        policy_evidence = {
            condition: capture(env, state, condition, output)
            for condition, state in (("eb", eb), ("er", er), ("ec", ec))
        }
    finally:
        env.close()

    report = {
        "verdict": f"PASS_{verdict_stem}",
        "scope": "one_state_static_only_no_vla_no_safe_no_replay",
        "task_id": TASK_ID,
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "native_bddl_sha256": NATIVE_BDDL_SHA256,
        "goal_form": goal_form,
        "goal_form_sha256": GOAL_SHA256,
        "native_only_bodies": {"S": SUPPORT, "A": MIDDLE, "B": TOP, "goal": TRAY},
        "asset_gate": asset_gate,
        "pairing": pairing,
        "physical": {
            "eb_hold": eb_hold,
            "ec_hold": ec_hold,
            "er_hold": er_hold,
            "S_intervention": s_gate,
            "A_ablation": a_gate,
            "B_ablation": b_gate,
        },
        "states": state_files,
        "policy_evidence": policy_evidence,
        "policy_view_status": "PENDING_MANUAL_POLICY_VIEW_REVIEW",
        "target_side_grasp_corridor_status": "PENDING_MANUAL_REVIEW",
        "eb_source_status": "NOT_RUN",
        "safe_reference_status": "NOT_RUN",
        "action_separation_status": "NOT_RUN",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(
        f"# L3-A3 {TASK_LABEL} native-only one-state static precheck\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Prompt SHA-256: `{PROMPT_SHA256}`\n"
        f"- Goal SHA-256: `{GOAL_SHA256}`\n"
        f"- Native-only S/A/B: {ROLE_SUMMARY}.\n"
        "- Pairing: outside A/B qpos/qvel bit-identical.\n"
        "- Policy view and S side-grasp corridor: pending manual review.\n"
        "- EB source, safe reference, replay: not run.\n"
    )
    print(report["verdict"])


if __name__ == "__main__":
    raise SystemExit("use a task-specific native tower generator")
