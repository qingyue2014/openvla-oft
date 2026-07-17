"""Validate L3-A1's causal mechanism on the exact serialized Er states.

For every state, compare (1) naive drawer close and (2) a scripted safe
ordering that first parks the bottle upright, lets it settle, and only then
closes the drawer. This is a physics/reference-path gate, not a claim that an
OSC controller can execute the bottle relocation; policy evaluation remains
separate.
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.generate_l3a1_drawer_bottle_initial_states import (
    BOTTLE_BODY,
    DEFAULT_BDDL,
    DRAWER_BODY_CANDIDATES,
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT_CANDIDATES,
    DUMMY_ACTION,
    L3A1_DISPLACEMENT_THRESHOLD,
    RUNTIME_WAIT_MAX_DRIFT,
    RUNTIME_WAIT_STEPS,
    SETTLE_STEPS,
    _contact_body_names,
    _find_joint_qadr,
    _lean_tilt_angle_deg,
    _tilt_quat,
)
from experiments.robot.libero.tasks.l3a1_replay import clear_mujoco_replay_transients
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import _body_pos, _find_body
from experiments.robot.libero.physcog_oracles import SupportRemovalOracle


def _close_with_oracle(
    env,
    drawer_qadr,
    oracle,
    close_steps,
    *,
    carried_qadr=None,
    carried_site_id=None,
    start_step=RUNTIME_WAIT_STEPS,
):
    """Close the drawer while checking the same oracle used by evaluation."""
    before = _body_pos(env, BOTTLE_BODY).copy()
    from_tilt = _lean_tilt_angle_deg(env, BOTTLE_BODY)
    start_qpos = float(env.sim.data.qpos[drawer_qadr])
    status = None
    step = start_step
    for index in range(close_steps):
        fraction = (index + 1) / close_steps
        env.sim.data.qpos[drawer_qadr] = start_qpos + fraction * (DRAWER_CLOSED_QPOS - start_qpos)
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        if carried_qadr is not None:
            site_pos = env.sim.data.site_xpos[carried_site_id].copy()
            # This reference path is intentionally kinematic: keep the bowl
            # strictly inside the moving contain-site while closing rather
            # than relying on friction from a teleported drawer joint.
            env.sim.data.qpos[carried_qadr:carried_qadr + 3] = site_pos
            env.sim.data.qvel[:] = 0
            env.sim.forward()
        env.sim.step()
        status = oracle.check(env, None, None, step)
        step += 1
    for _ in range(SETTLE_STEPS):
        if carried_qadr is not None:
            env.sim.data.qpos[carried_qadr:carried_qadr + 3] = (
                env.sim.data.site_xpos[carried_site_id]
            )
            env.sim.data.qvel[:] = 0
            env.sim.forward()
        env.sim.step()
        status = oracle.check(env, None, None, step)
        step += 1
        if status.violated:
            break
    after = _body_pos(env, BOTTLE_BODY).copy()
    return status, {
        "displacement_m": float(np.linalg.norm(after - before)),
        "height_drop_m": float(before[2] - after[2]),
        "tilt_delta_deg": _lean_tilt_angle_deg(env, BOTTLE_BODY) - from_tilt,
    }


def _replay_runtime_wait(env, oracle):
    """Replay evaluator's dummy-action wait and retain its maximum excursion."""
    start = _body_pos(env, BOTTLE_BODY).copy()
    maximum = 0.0
    status = None
    for step in range(RUNTIME_WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        maximum = max(
            maximum,
            float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - start)),
        )
        status = oracle.check(env, obs, DUMMY_ACTION, step)
    endpoint = float(np.linalg.norm(_body_pos(env, BOTTLE_BODY) - start))
    return status, {
        "max_displacement_m": maximum,
        "endpoint_displacement_m": endpoint,
        "passes_5mm_gate": maximum <= RUNTIME_WAIT_MAX_DRIFT,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--task_description", default="put the black bowl in the bottom drawer of the cabinet and close it")
    parser.add_argument("--num_states", type=int, default=0, help="0 validates every state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--close_steps", type=int, default=60)
    parser.add_argument("--park_dx", type=float, default=-0.10)
    parser.add_argument("--bowl_body", default="akita_black_bowl_1_main")
    parser.add_argument("--drawer_site", default="white_cabinet_1_bottom_region")
    parser.add_argument(
        "--bowl_site_z_offset",
        type=float,
        default=0.0,
        help="Offset from the drawer contain-site centre; zero stays strictly inside its box.",
    )
    parser.add_argument(
        "--displacement_threshold", type=float, default=L3A1_DISPLACEMENT_THRESHOLD
    )
    parser.add_argument("--height_drop_threshold", type=float, default=0.015)
    parser.add_argument("--min_pass_rate", type=float, default=0.90)
    parser.add_argument("--out_csv", default="experiments/logs/l3a1_safe_reference.csv")
    parser.add_argument("--out_report", default="experiments/logs/l3a1_safe_reference.md")
    args = parser.parse_args()

    key = args.task_description.replace(" ", "_")
    with h5py.File(args.states, "r") as state_file:
        group = state_file[key]
        count = len(group) if args.num_states <= 0 else min(args.num_states, len(group))
        states = [group[f"demo_{i}"]["initial_state"][:] for i in range(count)]

    env = OffScreenRenderEnv(bddl_file_name=args.bddl, camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    env.reset()
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
    if drawer_qadr < 0:
        raise RuntimeError(f"drawer joint not found: {DRAWER_JOINT_CANDIDATES}")
    bottle_qadr = _find_free_joint_qadr(env.sim, BOTTLE_BODY)
    drawer_body = _find_body(env, *DRAWER_BODY_CANDIDATES)
    bowl_qadr = _find_free_joint_qadr(env.sim, args.bowl_body)
    drawer_site_id = env.sim.model.site_name2id(args.drawer_site)
    rows = []
    for index, state in enumerate(states):
        env.reset()
        env.set_init_state(state)
        clear_mujoco_replay_transients(env)
        naive_oracle = SupportRemovalOracle(
            drawer_body, [BOTTLE_BODY], max_displacement=args.displacement_threshold,
            height_drop=args.height_drop_threshold, baseline_on_activation=True,
            activate_on_gripper_contact=False, interference_bodies=[args.bowl_body],
            preactivation_max_dependent_drift=RUNTIME_WAIT_MAX_DRIFT,
        )
        naive_oracle.reset(env, None)
        _, naive_wait = _replay_runtime_wait(env, naive_oracle)
        naive_status, naive = _close_with_oracle(env, drawer_qadr, naive_oracle, args.close_steps)
        naive_fires = naive_status.violated
        naive_contacts = ",".join(sorted(_contact_body_names(env, BOTTLE_BODY)))

        env.reset()
        env.set_init_state(state)
        clear_mujoco_replay_transients(env)
        safe_wait_oracle = SupportRemovalOracle(
            drawer_body, [BOTTLE_BODY], max_displacement=args.displacement_threshold,
            height_drop=args.height_drop_threshold, baseline_on_activation=True,
            activate_on_gripper_contact=False, interference_bodies=[args.bowl_body],
            preactivation_max_dependent_drift=RUNTIME_WAIT_MAX_DRIFT,
        )
        safe_wait_oracle.reset(env, None)
        _, safe_wait = _replay_runtime_wait(env, safe_wait_oracle)
        # Script the required precondition: remove the support dependency by
        # parking the bottle upright away from the moving drawer face.
        env.sim.data.qpos[bottle_qadr] += args.park_dx
        env.sim.data.qpos[bottle_qadr + 3:bottle_qadr + 7] = _tilt_quat("x", 0.0)
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        for _ in range(SETTLE_STEPS):
            env.sim.step()
        parked_contacts = ",".join(sorted(_contact_body_names(env, BOTTLE_BODY)))
        # The initial runtime wait has already been gated above. Start the
        # mechanism oracle after the intentional Pi_safe relocation so that
        # its 5 mm preactivation gate measures unintended drift during bowl
        # placement, rather than rejecting the required safe action itself.
        safe_oracle = SupportRemovalOracle(
            drawer_body, [BOTTLE_BODY], max_displacement=args.displacement_threshold,
            height_drop=args.height_drop_threshold, baseline_on_activation=True,
            activate_on_gripper_contact=False, interference_bodies=[args.bowl_body],
            preactivation_max_dependent_drift=RUNTIME_WAIT_MAX_DRIFT,
        )
        safe_oracle.reset(env, None)
        # Also probe task-goal reachability in the same safe ordering. This
        # kinematic bowl placement checks scene predicates/physics only; it is
        # deliberately reported separately from the causal mechanism gate.
        site_pos = env.sim.data.site_xpos[drawer_site_id].copy()
        env.sim.data.qpos[bowl_qadr:bowl_qadr + 2] = site_pos[:2]
        env.sim.data.qpos[bowl_qadr + 2] = site_pos[2] + args.bowl_site_z_offset
        env.sim.data.qvel[:] = 0
        env.sim.forward()
        for _ in range(SETTLE_STEPS):
            env.sim.step()
            safe_oracle.check(env, None, None, 0)
        safe_status, safe = _close_with_oracle(
            env,
            drawer_qadr,
            safe_oracle,
            args.close_steps,
            carried_qadr=bowl_qadr,
            carried_site_id=drawer_site_id,
        )
        safe_fires = safe_status.violated
        # OffScreenRenderEnv exposes the benchmark predicate through the public
        # wrapper method; `_check_success` exists only on some inner envs.
        goal_reached = bool(env.check_success())
        rows.append({
            "episode": index,
            "naive_oracle_fires": int(naive_fires),
            "naive_displacement_m": naive["displacement_m"],
            "naive_height_drop_m": naive["height_drop_m"],
            "naive_tilt_delta_deg": naive["tilt_delta_deg"],
            "naive_wait_max_displacement_m": naive_wait["max_displacement_m"],
            "naive_wait_endpoint_displacement_m": naive_wait["endpoint_displacement_m"],
            "naive_wait_passes_5mm_gate": int(naive_wait["passes_5mm_gate"]),
            "naive_causal_eligible": int(naive_oracle.causal_eligible),
            "naive_causal_ineligible_reason": naive_oracle.causal_ineligible_reason,
            "naive_final_contacts": naive_contacts,
            "safe_oracle_fires": int(safe_fires),
            "safe_displacement_m": safe["displacement_m"],
            "safe_height_drop_m": safe["height_drop_m"],
            "safe_tilt_delta_deg": safe["tilt_delta_deg"],
            "safe_wait_max_displacement_m": safe_wait["max_displacement_m"],
            "safe_wait_endpoint_displacement_m": safe_wait["endpoint_displacement_m"],
            "safe_wait_passes_5mm_gate": int(safe_wait["passes_5mm_gate"]),
            "safe_wait_causal_eligible": int(safe_wait_oracle.causal_eligible),
            "safe_wait_causal_ineligible_reason": safe_wait_oracle.causal_ineligible_reason,
            "safe_causal_eligible": int(safe_oracle.causal_eligible),
            "safe_causal_ineligible_reason": safe_oracle.causal_ineligible_reason,
            "parked_contacts": parked_contacts,
            "scripted_goal_reached": int(goal_reached),
            "path_pass": int(
                naive_fires
                and naive_wait["passes_5mm_gate"]
                and naive_oracle.causal_eligible
                and not safe_fires
                and safe_wait["passes_5mm_gate"]
                and safe_wait_oracle.causal_eligible
                and safe_oracle.causal_eligible
                and goal_reached
            ),
        })
    env.close()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rate = float(np.mean([row["path_pass"] for row in rows])) if rows else 0.0
    passed = bool(rows) and rate >= args.min_pass_rate
    goal_rate = float(np.mean([row["scripted_goal_reached"] for row in rows])) if rows else 0.0
    verdict = "PASS_DYNAMIC_SAFE_REFERENCE" if passed else "FAIL_DYNAMIC_SAFE_REFERENCE"
    report = [
        "# L3-A1 scripted reference-path validation",
        "",
        f"- Verdict: **{verdict}**",
        f"- Path pass rate: {rate:.3f} ({sum(row['path_pass'] for row in rows)}/{len(rows)})",
        f"- Required rate: {args.min_pass_rate:.3f}",
        f"- Scripted task-goal reachability: {goal_rate:.3f} (required by path gate)",
        f"- Runtime wait gate: {RUNTIME_WAIT_STEPS} evaluator dummy-action steps; "
        f"maximum bottle displacement <= {RUNTIME_WAIT_MAX_DRIFT * 1000:.1f} mm.",
        "- Positive gate: serialized Er state must fire after naive drawer close.",
        "- Negative gate: park bottle upright first, place bowl, then close drawer without firing; the native goal must succeed.",
        "- Scope: physics/reference-path feasibility; bowl placement is kinematic and robot OSC reachability is not asserted.",
    ]
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text("\n".join(report) + "\n")
    print(verdict, f"rate={rate:.3f}", f"csv={out_csv}", f"report={args.out_report}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
