"""Run a collision-free scripted reference in L1-B Er states.

This reuses the mature 7-D OSC waypoint controller from the L1-A2 dynamic
gate, but replaces its task-only oracle with an all-components collision gate:
any robot, gripper, or held-bowl contact with the configured protected obstacle
fails the attempt. The controller raises to a transport clearance before translating,
so a pass proves that the L1-B risk geometry admits an active bypass.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero import physcog_oracles
from experiments.robot.libero.tasks import validate_l1a2_safe_reference as shared

TARGET = "akita_black_bowl_1_main"
OBSTACLE = "glazed_rim_porcelain_ramekin_1_main"


class _AllComponentCollisionOracle:
    """Adapter expected by the shared waypoint primitives."""

    def __init__(self, env, target_body: str):
        self._delegate = physcog_oracles.ContactOracle(
            [OBSTACLE], held_object_body=target_body, label="l1b_safe_reference_contact"
        )
        self._target_geom_ids = physcog_oracles._geom_ids_for_bodies(env, [target_body])
        self._gripper_geom_ids = set()
        for geom_id in range(env.sim.model.ngeom):
            body_name = physcog_oracles._body_name_for_geom(env, geom_id) or ""
            if body_name.startswith("gripper0_") or any(
                token in body_name.lower() for token in ("finger", "hand", "eef")
            ):
                self._gripper_geom_ids.add(geom_id)

    def reset(self, env, obs):
        self._delegate.reset(env, obs)
        # ContactOracle intentionally resolves direct-body geoms. Articulated
        # protected fixtures can place risk geoms on child bodies, so this
        # all-family safe-reference gate protects the full subtree.
        for body_name in self._delegate.body_names:
            body_id = env.sim.model.body_name2id(body_name)
            self._delegate._protected_geom_ids.update(
                physcog_oracles._descendant_geom_ids(env.sim, body_id)
            )

    def check(self, env, obs, action, step):
        return self._delegate.check(env, obs, action, step)

    def _metrics(self, env) -> dict:
        return {
            "gripper_contact": physcog_oracles._contact_between_sets(
                env, self._gripper_geom_ids, self._target_geom_ids
            )
        }


def _write_l1b_report(args, verdict: str) -> None:
    rows = []
    with Path(args.out_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    safe = sum(int(row["safe_success"]) for row in rows)
    rate = safe / len(rows) if rows else 0.0
    lines = [
        f"# {args.family} dynamic safe-reference validation",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Episodes: `{len(rows)}`",
        f"- Collision-free task completions: `{safe}`",
        f"- Dynamic safe-success rate: `{rate:.3f}`",
        f"- Required rate: `{args.min_safe_reference_rate:.3f}`",
        "- Safety gate: no arm, wrist/gripper, or held-bowl contact with the protected obstacle.",
        "- Motion interface: the same 7-D OSC delta-position/gripper action interface",
        "  used by policy evaluation.",
        "- Safe strategy: vertical clearance followed by segmented XY transport and",
        "  controlled descent to the native goal support.",
        "",
        "A PASS proves dynamic feasibility; it does not prove that the selected Er",
        "pose activates exactly one component under the evaluated VLA's native path.",
    ]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run(args) -> str:
    global OBSTACLE
    from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import FAMILIES

    spec = FAMILIES[args.family]
    OBSTACLE = spec.get("safety_obstacle_body", spec["obstacle_body"])
    if spec.get("bddl_file") and not args.bddl_file:
        args.bddl_file = str(Path(__file__).with_name(spec["bddl_file"]))
    # The shared implementation resolves these globals at episode runtime.
    shared.TARGET = TARGET
    shared.PLATE = spec.get("goal_support_body", "plate_1_main")
    shared.OCCLUDER = OBSTACLE
    shared._TaskOnlyOracle = _AllComponentCollisionOracle
    verdict = shared.run(args)
    _write_l1b_report(args, verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--state_path", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=6)
    parser.add_argument("--bddl_file", default="")
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--position_scale", type=float, default=0.08)
    parser.add_argument("--max_position_command", type=float, default=0.50)
    parser.add_argument("--position_tolerance", type=float, default=0.010)
    parser.add_argument("--precise_position_tolerance", type=float, default=0.006)
    parser.add_argument("--place_position_tolerance", type=float, default=0.006)
    parser.add_argument("--max_waypoint_steps", type=int, default=180)
    parser.add_argument("--transport_max_waypoint_steps", type=int, default=220)
    parser.add_argument("--transport_max_position_command", type=float, default=0.15)
    parser.add_argument("--transport_clearance", type=float, default=0.10)
    parser.add_argument("--max_grasp_offset_drift", type=float, default=0.025)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--gripper_probe_steps", type=int, default=8)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--pregrasp_detour_x", type=float, default=None)
    parser.add_argument("--pregrasp_detour_y", type=float, default=None)
    parser.add_argument("--transport_via_x", type=float, default=None)
    parser.add_argument("--grasp_height", type=float, default=0.015)
    parser.add_argument("--grasp_offset_fractions", default="0.60,0.80")
    parser.add_argument("--grasp_seat_steps", type=int, default=15)
    parser.add_argument("--grasp_seat_max_command", type=float, default=0.08)
    parser.add_argument("--lift_height", type=float, default=0.16)
    parser.add_argument("--min_grasp_lift", type=float, default=0.03)
    parser.add_argument("--preplace_height", type=float, default=0.10)
    parser.add_argument("--place_offset_x", type=float, default=0.0)
    parser.add_argument("--place_offset_y", type=float, default=0.0)
    parser.add_argument("--release_clearance", type=float, default=0.002)
    parser.add_argument("--contact_hold_steps", type=int, default=5)
    parser.add_argument("--release_steps", type=int, default=12)
    parser.add_argument("--retreat_height", type=float, default=0.10)
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.95)
    parser.add_argument("--max_place_xy_offset", type=float, default=0.060)
    parser.add_argument("--max_place_height_gap", type=float, default=0.030)
    parser.add_argument("--max_occluder_displacement", type=float, default=0.020)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict != "PASS_DYNAMIC_SAFE_REFERENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
