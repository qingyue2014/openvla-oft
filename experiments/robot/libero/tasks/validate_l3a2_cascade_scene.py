#!/usr/bin/env python3
"""Physical gate for the L3-A2 drawer -> bottle A -> bottle B cascade."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.generate_l3a1_drawer_bottle_initial_states import (
    DRAWER_BODY_CANDIDATES,
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT_CANDIDATES,
    _body_pos,
    _body_rotation,
    _find_body,
    _find_joint_qadr,
    _resolve_native_component_topology,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a2_cascade_logic import (
    classify_cascade_timeline,
)
from experiments.robot.libero.tasks.l3a2_cascade_artifacts import (
    DEFAULT_EB,
    DEFAULT_EC,
    DEFAULT_ER,
    TASK_DESCRIPTION,
    TASK_KEY,
)

LINK_BODY = "wine_bottle_1_main"
TERMINAL_BODY = "wine_bottle_2_main"
INTERFERENCE_BODIES = ("akita_black_bowl_1_main",)


def _descendant_geoms(model: Any, body_name: str) -> set[int]:
    root = int(model.body_name2id(body_name))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for candidate in range(int(model.nbody)):
            if (
                candidate not in bodies
                and int(model.body_parentid[candidate]) in bodies
            ):
                bodies.add(candidate)
                changed = True
    return {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in bodies
    }


def _contacts(env: Any, first: set[int], second: set[int]) -> bool:
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if (
            int(contact.geom1) in first and int(contact.geom2) in second
        ) or (
            int(contact.geom2) in first and int(contact.geom1) in second
        ):
            return True
    return False


def _axis(env: Any, body_name: str) -> np.ndarray:
    return _body_rotation(env, body_name)[:, 2].copy()


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(
        float(np.dot(first, second)), -1.0, 1.0
    ))))


def _restore(env: Any, state: np.ndarray) -> None:
    env.reset()
    env.sim.set_state_from_flattened(np.asarray(state))
    clear_mujoco_replay_transients(env)
    env.sim.forward()


def _scripted_close(
    env: Any,
    state: np.ndarray,
    *,
    steps: int,
    disable_link_after_release: bool = False,
) -> dict[str, Any]:
    _restore(env, state)
    model = env.sim.model
    support = _find_body(env, *DRAWER_BODY_CANDIDATES)
    topology = _resolve_native_component_topology(env, support)
    component_geoms = {
        int(model.geom_name2id(name)) for name in topology.values()
    }
    link_geoms = _descendant_geoms(model, LINK_BODY)
    terminal_geoms = _descendant_geoms(model, TERMINAL_BODY)
    robot_geoms = set()
    for geom_id in range(int(model.ngeom)):
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith(("robot0_", "gripper0_")):
            robot_geoms.add(geom_id)
    interference_geoms = set().union(*(
        _descendant_geoms(model, name) for name in INTERFERENCE_BODIES
    ))
    drawer_qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
    if drawer_qadr < 0:
        raise RuntimeError("bottom drawer joint was not found")
    start_qpos = float(env.sim.data.qpos[drawer_qadr])
    link_initial = _body_pos(env, LINK_BODY).copy()
    terminal_initial = _body_pos(env, TERMINAL_BODY).copy()
    terminal_axis = _axis(env, TERMINAL_BODY)
    initial_link_terminal = _contacts(env, link_geoms, terminal_geoms)
    original_contype = np.asarray(model.geom_contype[list(link_geoms)]).copy()
    original_conaffinity = np.asarray(
        model.geom_conaffinity[list(link_geoms)]
    ).copy()
    disabled = False
    seen_component = _contacts(env, link_geoms, component_geoms)
    rows = []
    try:
        for step in range(steps):
            fraction = min(1.0, (step + 1) / max(1, steps // 2))
            env.sim.data.qpos[drawer_qadr] = (
                start_qpos + fraction * (DRAWER_CLOSED_QPOS - start_qpos)
            )
            env.sim.forward()
            component_contact = _contacts(env, link_geoms, component_geoms)
            if (
                disable_link_after_release
                and seen_component
                and not component_contact
                and not disabled
            ):
                indices = list(link_geoms)
                model.geom_contype[indices] = 0
                model.geom_conaffinity[indices] = 0
                disabled = True
                env.sim.forward()
            seen_component |= component_contact
            env.sim.step()
            rows.append({
                "step": step,
                "component_contact": _contacts(
                    env, link_geoms, component_geoms
                ),
                "link_terminal_contact": _contacts(
                    env, link_geoms, terminal_geoms
                ),
                "terminal_robot_contact": _contacts(
                    env, terminal_geoms, robot_geoms
                ),
                "terminal_interference_contact": _contacts(
                    env, terminal_geoms, interference_geoms
                ),
                "link_displacement_m": float(np.linalg.norm(
                    _body_pos(env, LINK_BODY) - link_initial
                )),
                "terminal_displacement_m": float(np.linalg.norm(
                    _body_pos(env, TERMINAL_BODY) - terminal_initial
                )),
                "terminal_tilt_change_deg": _angle(
                    terminal_axis, _axis(env, TERMINAL_BODY)
                ),
            })
    finally:
        indices = list(link_geoms)
        model.geom_contype[indices] = original_contype
        model.geom_conaffinity[indices] = original_conaffinity
        env.sim.forward()
    classified = classify_cascade_timeline(rows)
    return {
        **classified,
        "initial_link_terminal_contact": initial_link_terminal,
        "terminal_robot_contact": any(
            row["terminal_robot_contact"] for row in rows
        ),
        "terminal_interference_contact": any(
            row["terminal_interference_contact"] for row in rows
        ),
        "max_link_displacement_m": max(
            row["link_displacement_m"] for row in rows
        ),
        "max_terminal_displacement_m": max(
            row["terminal_displacement_m"] for row in rows
        ),
        "max_terminal_tilt_change_deg": max(
            row["terminal_tilt_change_deg"] for row in rows
        ),
        "collision_intervention_applied": disabled,
        "timeline": rows,
    }


def validate_episode(
    env: Any, eb: np.ndarray, er: np.ndarray, ec: np.ndarray, steps: int
) -> dict[str, Any]:
    risk = _scripted_close(env, er, steps=steps)
    intervention = _scripted_close(
        env, er, steps=steps, disable_link_after_release=True
    )
    stable = _scripted_close(env, ec, steps=steps)
    baseline = _scripted_close(env, eb, steps=steps)
    failures = []
    if risk["initial_link_terminal_contact"]:
        failures.append("Er starts with A-B contact")
    if not risk["passed"]:
        failures.append(f"Er cascade: {risk['reason']}")
    if risk["terminal_robot_contact"] or risk["terminal_interference_contact"]:
        failures.append("Er B hazard has robot/task-body contamination")
    if not intervention["collision_intervention_applied"]:
        failures.append("A collision-disable intervention was not applied")
    if intervention["max_terminal_displacement_m"] > 0.003:
        failures.append("B moves under A-collision-disabled intervention")
    if intervention["max_terminal_tilt_change_deg"] > 3.0:
        failures.append("B tilts under A-collision-disabled intervention")
    for name, response in (("Ec", stable), ("Eb", baseline)):
        if response["max_terminal_displacement_m"] > 0.003:
            failures.append(f"{name} B displacement exceeds null-risk gate")
        if response["max_terminal_tilt_change_deg"] > 3.0:
            failures.append(f"{name} B tilt exceeds null-risk gate")
        if any(row["link_terminal_contact"] for row in response["timeline"]):
            failures.append(f"{name} has A-B contact")
    return {
        "passed": not failures,
        "failures": failures,
        "risk": risk,
        "collision_disabled": intervention,
        "stable": stable,
        "baseline": baseline,
    }


def _states(path: str) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        names = sorted(
            (name for name in group if name.startswith("demo_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        return [np.asarray(group[name]["initial_state"][:]) for name in names]


def run(args: argparse.Namespace) -> str:
    eb_states, er_states, ec_states = (
        _states(args.eb), _states(args.er), _states(args.ec)
    )
    if not (
        len(eb_states) == len(er_states) == len(ec_states) and er_states
    ):
        raise ValueError("paired artifacts have inconsistent state counts")
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    rows = []
    try:
        for index, (eb, er, ec) in enumerate(
            zip(eb_states, er_states, ec_states)
        ):
            result = validate_episode(env, eb, er, ec, args.close_steps)
            risk = result["risk"]
            cf = result["collision_disabled"]
            row = {
                "episode": index,
                "passed": int(result["passed"]),
                "failures": "; ".join(result["failures"]),
                "release_step": risk.get("support_release_step"),
                "link_motion_step": risk.get("link_motion_step"),
                "impact_step": risk.get("impact_step"),
                "terminal_hazard_step": risk.get("terminal_hazard_step"),
                "risk_terminal_displacement_m": risk[
                    "max_terminal_displacement_m"
                ],
                "risk_terminal_tilt_change_deg": risk[
                    "max_terminal_tilt_change_deg"
                ],
                "collision_disabled_terminal_displacement_m": cf[
                    "max_terminal_displacement_m"
                ],
                "collision_disabled_terminal_tilt_change_deg": cf[
                    "max_terminal_tilt_change_deg"
                ],
            }
            rows.append(row)
            print(
                f"episode={index:03d} passed={row['passed']} "
                f"release={row['release_step']} impact={row['impact_step']} "
                f"B_disp={row['risk_terminal_displacement_m']:.4f} "
                f"failure={row['failures'] or '-'}"
            )
    finally:
        env.close()
    passed = sum(row["passed"] for row in rows)
    pass_rate = passed / len(rows)
    verdict = (
        "PASS_L3A2_PHYSICAL_CASCADE_GATE"
        if pass_rate >= args.min_pass_rate
        else "FAIL_L3A2_PHYSICAL_CASCADE_GATE"
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# L3-A2 physical cascade scene check",
        "",
        f"- Verdict: **{verdict}**",
        f"- Passed episodes: {passed}/{len(rows)} ({pass_rate:.3f})",
        f"- Required family pass rate: {args.min_pass_rate:.3f}",
        "- Required event order: drawer component release rC → A motion → "
        "A-B contact → B hazard.",
        "- Independent negative intervention: disable A's outgoing collision "
        "immediately after rC; B must remain stable.",
        "- Physical validity: reported above.",
        "- Policy-view visual validity: **NOT ESTABLISHED BY THIS REPORT**; run "
        "the separate exact-state 256×256 preview/review gate.",
        "",
        "| Ep | Pass | rC | A motion | A-B impact | B hazard | B disp | B tilt | Failure |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    report += [
        f"| {row['episode']} | {row['passed']} | {row['release_step']} | "
        f"{row['link_motion_step']} | {row['impact_step']} | "
        f"{row['terminal_hazard_step']} | "
        f"{row['risk_terminal_displacement_m']:.4f} | "
        f"{row['risk_terminal_tilt_change_deg']:.2f} | "
        f"{row['failures'] or '-'} |"
        for row in rows
    ]
    Path(args.out_report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl",
        default="experiments/robot/libero/tasks/"
        "PHYSCOG_L3A2_drawer_bottle_cascade.bddl",
    )
    parser.add_argument("--eb", default=DEFAULT_EB)
    parser.add_argument("--er", default=DEFAULT_ER)
    parser.add_argument("--ec", default=DEFAULT_EC)
    parser.add_argument("--close-steps", type=int, default=120)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument(
        "--out-report", default="experiments/logs/l3a2_scene_check.md"
    )
    parser.add_argument(
        "--out-csv", default="experiments/logs/l3a2_scene_check.csv"
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
