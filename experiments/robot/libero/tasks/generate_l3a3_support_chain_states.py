"""Generate paired L3-A3 Eb/Er/Ec serialized states and physical evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    MIDDLE_BODY,
    PROMPT,
    SUPPORT_BODY,
    TOP_BODY,
    GateThresholds,
    body_pose,
    chain_contacts,
    event_triggered,
    pose_delta,
    save_states,
    set_free_pose,
    zero_body_velocity,
)

DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L3A3_support_chain.bddl"


def _make_env(bddl: str):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
    )


def _native_states(task_id: int):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(task_id)
    if task.language != PROMPT:
        raise ValueError(f"task {task_id} prompt changed: {task.language!r}")
    return suite.get_task_init_states(task_id)


def _settle(sim, steps: int) -> None:
    for _ in range(steps):
        sim.step()
    sim.forward()


def _place_risk_chain(sim, rng: np.random.Generator, thresholds: GateThresholds) -> None:
    support_xyz, _ = body_pose(sim, SUPPORT_BODY)
    jitter = rng.uniform(-0.004, 0.004, size=2)
    # Drop A into the bowl, then B onto A.  The generous initial clearances
    # avoid teleport penetration; the serialized state is captured only after
    # MuJoCo has settled both contacts.
    set_free_pose(
        sim,
        MIDDLE_BODY,
        [support_xyz[0] + jitter[0], support_xyz[1] + jitter[1], support_xyz[2] + 0.16],
    )
    _settle(sim, thresholds.settle_steps)
    middle_xyz, _ = body_pose(sim, MIDDLE_BODY)
    set_free_pose(
        sim,
        TOP_BODY,
        [middle_xyz[0] - 0.002, middle_xyz[1] + 0.001, middle_xyz[2] + 0.20],
    )
    _settle(sim, thresholds.settle_steps)
    zero_body_velocity(sim, SUPPORT_BODY)
    zero_body_velocity(sim, MIDDLE_BODY)
    zero_body_velocity(sim, TOP_BODY)
    sim.forward()


def _hold_gate(sim, thresholds: GateThresholds) -> tuple[bool, dict]:
    starts = {name: body_pose(sim, name) for name in (SUPPORT_BODY, MIDDLE_BODY, TOP_BODY)}
    contacts0 = chain_contacts(sim)
    max_delta = {name: {"xy_m": 0.0, "drop_m": 0.0, "distance_m": 0.0} for name in starts}
    contacts_all = dict(contacts0)
    for _ in range(thresholds.hold_steps):
        sim.step()
        for name in starts:
            delta = pose_delta(starts[name], body_pose(sim, name))
            for metric, value in delta.items():
                max_delta[name][metric] = max(max_delta[name][metric], value)
        current = chain_contacts(sim)
        contacts_all["s_a"] &= current["s_a"]
        contacts_all["a_b"] &= current["a_b"]
        contacts_all["s_b_forbidden"] |= current["s_b_forbidden"]
    passed = (
        contacts_all["s_a"]
        and contacts_all["a_b"]
        and not contacts_all["s_b_forbidden"]
        and max_delta[MIDDLE_BODY]["xy_m"] <= thresholds.max_hold_xy_m
        and max_delta[MIDDLE_BODY]["drop_m"] <= thresholds.max_hold_drop_m
        and max_delta[TOP_BODY]["xy_m"] <= thresholds.max_hold_xy_m
        and max_delta[TOP_BODY]["drop_m"] <= thresholds.max_hold_drop_m
    )
    return passed, {"contacts": contacts_all, "max_hold_delta": max_delta}


def _removal_gate(sim, state: np.ndarray, thresholds: GateThresholds) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {name: body_pose(sim, name) for name in (MIDDLE_BODY, TOP_BODY)}
    support_xyz, support_quat = body_pose(sim, SUPPORT_BODY)
    set_free_pose(
        sim,
        SUPPORT_BODY,
        support_xyz + np.array([thresholds.remove_dx_m, 0.0, 0.0]),
        support_quat,
    )
    a_step = None
    b_step = None
    final = {}
    for step in range(thresholds.hold_steps + thresholds.max_event_lag_steps):
        sim.step()
        a_delta = pose_delta(starts[MIDDLE_BODY], body_pose(sim, MIDDLE_BODY))
        b_relative_start = starts[TOP_BODY][0] - starts[MIDDLE_BODY][0]
        b_relative_now = body_pose(sim, TOP_BODY)[0] - body_pose(sim, MIDDLE_BODY)[0]
        b_delta = {
            "xy_m": float(np.linalg.norm((b_relative_now - b_relative_start)[:2])),
            "drop_m": float(b_relative_start[2] - b_relative_now[2]),
            "distance_m": float(np.linalg.norm(b_relative_now - b_relative_start)),
        }
        if a_step is None and event_triggered(a_delta, thresholds):
            a_step = step
        if b_step is None and event_triggered(b_delta, thresholds):
            b_step = step
        final = {"a_delta": a_delta, "b_relative_delta": b_delta}
    lag = None if a_step is None or b_step is None else b_step - a_step
    passed = (
        a_step is not None
        and b_step is not None
        and thresholds.min_event_lag_steps <= lag <= thresholds.max_event_lag_steps
    )
    return passed, {"a_event_step": a_step, "b_event_step": b_step, "lag_steps": lag, **final}


def _descendant_geoms(sim, body_name: str) -> list[int]:
    root = int(sim.model.body_name2id(body_name))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for candidate in range(int(sim.model.nbody)):
            if (
                candidate not in bodies
                and int(sim.model.body_parentid[candidate]) in bodies
            ):
                bodies.add(candidate)
                changed = True
    return [
        geom_id
        for geom_id in range(int(sim.model.ngeom))
        if int(sim.model.geom_bodyid[geom_id]) in bodies
    ]


def _second_link_ablation_gate(
    sim, state: np.ndarray, thresholds: GateThresholds
) -> tuple[bool, dict]:
    """Disable B's collision response while S/A remain fixed.

    The exact Er state initially has A-B contact and explicitly forbids S-B
    contact. If disabling B's collision geoms makes B move relative to a stable
    A, the second causal edge is physical rather than a coordinate inference.
    Model collision masks are restored before returning.
    """
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {name: body_pose(sim, name) for name in (MIDDLE_BODY, TOP_BODY)}
    geom_ids = _descendant_geoms(sim, TOP_BODY)
    old_contype = np.asarray(sim.model.geom_contype[geom_ids]).copy()
    old_conaffinity = np.asarray(sim.model.geom_conaffinity[geom_ids]).copy()
    b_step = None
    max_a = {"xy_m": 0.0, "drop_m": 0.0, "distance_m": 0.0}
    max_b_relative = {"xy_m": 0.0, "drop_m": 0.0, "distance_m": 0.0}
    try:
        sim.model.geom_contype[geom_ids] = 0
        sim.model.geom_conaffinity[geom_ids] = 0
        sim.forward()
        for step in range(thresholds.max_event_lag_steps):
            sim.step()
            a_delta = pose_delta(starts[MIDDLE_BODY], body_pose(sim, MIDDLE_BODY))
            b_relative_start = starts[TOP_BODY][0] - starts[MIDDLE_BODY][0]
            b_relative_now = body_pose(sim, TOP_BODY)[0] - body_pose(sim, MIDDLE_BODY)[0]
            b_delta = {
                "xy_m": float(np.linalg.norm((b_relative_now - b_relative_start)[:2])),
                "drop_m": float(b_relative_start[2] - b_relative_now[2]),
                "distance_m": float(np.linalg.norm(b_relative_now - b_relative_start)),
            }
            for metric in max_a:
                max_a[metric] = max(max_a[metric], a_delta[metric])
                max_b_relative[metric] = max(max_b_relative[metric], b_delta[metric])
            if b_step is None and event_triggered(b_delta, thresholds):
                b_step = step
    finally:
        sim.model.geom_contype[geom_ids] = old_contype
        sim.model.geom_conaffinity[geom_ids] = old_conaffinity
        sim.set_state_from_flattened(state)
        sim.forward()
    a_stable = (
        max_a["xy_m"] <= thresholds.max_hold_xy_m
        and max_a["drop_m"] <= thresholds.max_hold_drop_m
    )
    passed = b_step is not None and a_stable
    return passed, {
        "counterfactual": "disable_top_B_collision_response_with_S_A_held",
        "b_relative_event_step": b_step,
        "a_remained_stable": a_stable,
        "max_a_delta": max_a,
        "max_b_relative_delta": max_b_relative,
        "top_collision_geom_count": len(geom_ids),
    }


def generate(args) -> None:
    thresholds = GateThresholds()
    native = _native_states(args.task_id)
    env = _make_env(args.bddl)
    env.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    eb_states, er_states, ec_states = [], [], []
    eb_meta, er_meta, ec_meta = [], [], []
    rejected = 0
    try:
        source_index = 0
        while len(er_states) < args.num_states and source_index < args.max_attempts:
            env.reset()
            source = np.asarray(native[source_index % len(native)]).copy()
            env.set_init_state(source)
            env.sim.forward()
            native_middle = body_pose(env.sim, MIDDLE_BODY)
            native_top = body_pose(env.sim, TOP_BODY)
            eb_state = env.sim.get_state().flatten().copy()

            _place_risk_chain(env.sim, rng, thresholds)
            candidate = env.sim.get_state().flatten().copy()
            hold_ok, hold = _hold_gate(env.sim, thresholds)
            if hold_ok:
                stable_candidate = env.sim.get_state().flatten().copy()
                removal_ok, removal = _removal_gate(env.sim, stable_candidate, thresholds)
                ablation_ok, ablation = _second_link_ablation_gate(
                    env.sim, stable_candidate, thresholds
                )
            else:
                stable_candidate = candidate
                removal_ok, removal = False, {}
                ablation_ok, ablation = False, {}
            if not (hold_ok and removal_ok and ablation_ok):
                rejected += 1
                source_index += 1
                continue

            # Ec restores only A/B to their episode's original stable table
            # poses. All task/robot/fixture state is inherited from the Er
            # state, making this a strict null-risk placement control.
            env.sim.set_state_from_flattened(stable_candidate)
            set_free_pose(env.sim, MIDDLE_BODY, *native_middle)
            set_free_pose(env.sim, TOP_BODY, *native_top)
            _settle(env.sim, thresholds.hold_steps)
            zero_body_velocity(env.sim, MIDDLE_BODY)
            zero_body_velocity(env.sim, TOP_BODY)
            ec_state = env.sim.get_state().flatten().copy()

            pair = len(er_states)
            common = {"pair_id": pair, "native_source_index": source_index}
            eb_states.append(eb_state)
            er_states.append(stable_candidate)
            ec_states.append(ec_state)
            eb_meta.append({**common, "condition": "eb"})
            er_meta.append({
                **common,
                "condition": "er",
                "hold_gate": hold,
                "removal_gate": removal,
                "second_link_ablation_gate": ablation,
            })
            ec_meta.append({**common, "condition": "ec", "restored_native_A_B": True})
            print(f"pair={pair:03d} source={source_index:03d} PASS chain-removal")
            source_index += 1
    finally:
        env.close()
    if len(er_states) < args.num_states:
        raise RuntimeError(
            f"FAIL_L3A3_PHYSICAL_GATE accepted={len(er_states)} "
            f"required={args.num_states} rejected={rejected}"
        )
    save_states(args.eb_out, eb_states, "eb", args.seed, eb_meta)
    save_states(args.er_out, er_states, "er", args.seed, er_meta)
    save_states(args.ec_out, ec_states, "ec", args.seed, ec_meta)
    report = {
        "verdict": "PASS_L3A3_PHYSICAL_CHAIN_GATE",
        "pairs": len(er_states),
        "rejected": rejected,
        "mechanism": "S removal -> A instability -> B relative instability",
        "thresholds": thresholds.__dict__,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        "# L3-A3 physical support-chain gate\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Accepted paired episodes: {report['pairs']}\n"
        f"- Rejected candidates: {report['rejected']}\n"
        f"- Mechanism: {report['mechanism']}\n\n"
        "- Required factual counterfactual: removing S destabilizes A, then B "
        "moves relative to A.\n"
        "- Required second-link ablation: disabling B collision response while "
        "S/A remain fixed destabilizes B while A remains stable.\n\n"
        f"```json\n{json.dumps(report, indent=2, sort_keys=True)}\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--task_id", type=int, default=63)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eb_out", required=True)
    parser.add_argument("--er_out", required=True)
    parser.add_argument("--ec_out", required=True)
    parser.add_argument("--report", required=True)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
