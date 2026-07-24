"""Run a collision-free scripted reference in L1-B Er states.

This reuses the mature 7-D OSC waypoint controller from the L1-A2 dynamic
gate, but replaces its task-only oracle with an all-components collision gate:
any robot, gripper, or held-object contact with the configured protected obstacle
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
    contact_ready = sum(
        int(row["pre_release_support_contact"]) for row in rows
    )
    released_supported = sum(
        int(row["released_before_retreat"])
        and int(row["post_release_support_contact"])
        for row in rows
    )
    max_lift = max(
        (float(row["max_bowl_lift_m"]) for row in rows),
        default=float("nan"),
    )
    strategy = (
        "paired successful-Eb actions through verified grasp contact, followed by "
        "a closed-loop vertical lift and "
        "a low-clearance segmented XY bypass and controlled descent"
        if args.branch_grasp_prefix_on_contact
        else "paired successful-Eb actions through the first verified lift, followed by "
        "a low-clearance segmented XY bypass and controlled descent"
        if args.grasp_action_trajectories
        else "vertical clearance followed by segmented XY transport and controlled descent"
    )
    lines = [
        f"# {args.family} dynamic safe-reference validation",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Episodes: `{len(rows)}`",
        f"- Collision-free task completions: `{safe}`",
        f"- Dynamic safe-success rate: `{rate:.3f}`",
        f"- Required rate: `{args.min_safe_reference_rate:.3f}`",
        "- Safety gate: no arm, wrist/gripper, or held-object contact with the protected obstacle.",
        "- Motion interface: the same 7-D OSC delta-position/gripper action interface",
        "  used by policy evaluation.",
        f"- Safe strategy: {strategy} to the native goal support.",
    ]
    if args.require_support_contact_before_release:
        lines.extend(
            [
                f"- Pre-release target/support contact: "
                f"`{contact_ready}/{len(rows)}`.",
                f"- Released and stably supported before retreat: "
                f"`{released_supported}/{len(rows)}`.",
                f"- Maximum observed target lift: `{max_lift:.4f} m` "
                f"(limit `{args.max_safe_lift_height:.4f} m`).",
            ]
        )
    lines.extend(
        [
            "",
            "A PASS proves dynamic feasibility; it does not prove that the selected Er",
            "pose activates exactly one component under the evaluated VLA's native path.",
        ]
    )
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run(args) -> str:
    global OBSTACLE, TARGET
    from experiments.robot.libero.tasks.generate_l1b_swept_initial_states import FAMILIES

    spec = FAMILIES[args.family]
    OBSTACLE = spec.get("safety_obstacle_body", spec["obstacle_body"])
    TARGET = spec.get("target_body", "akita_black_bowl_1_main")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position_scale", type=float, default=0.08)
    parser.add_argument("--max_position_command", type=float, default=0.50)
    parser.add_argument("--position_tolerance", type=float, default=0.010)
    parser.add_argument("--precise_position_tolerance", type=float, default=0.006)
    parser.add_argument("--place_position_tolerance", type=float, default=0.006)
    parser.add_argument("--max_waypoint_steps", type=int, default=180)
    parser.add_argument("--transport_max_waypoint_steps", type=int, default=220)
    parser.add_argument("--transport_max_position_command", type=float, default=0.15)
    parser.add_argument("--transport_position_tolerance", type=float, default=0.025)
    parser.add_argument(
        "--raise_transport_position_tolerance", type=float, default=0.025
    )
    parser.add_argument("--transport_target_eef_quat", default="")
    parser.add_argument("--orient_before_grasp", action="store_true")
    parser.add_argument("--skip_transport_orientation", action="store_true")
    parser.add_argument("--preorientation_path_fraction", type=float, default=0.0)
    parser.add_argument("--preorientation_obstacle_clearance", type=float, default=0.0)
    parser.add_argument("--preorientation_position_tolerance", type=float, default=0.010)
    parser.add_argument(
        "--preorientation_advance_position_tolerance", type=float, default=None
    )
    parser.add_argument("--preorientation_lift_height", type=float, default=0.0)
    parser.add_argument(
        "--preorientation_lift_position_tolerance", type=float, default=0.010
    )
    parser.add_argument("--preorientation_rotation_height", type=float, default=None)
    parser.add_argument("--postorientation_path_fraction", type=float, default=0.0)
    parser.add_argument("--postorientation_obstacle_clearance", type=float, default=0.0)
    parser.add_argument("--postorientation_position_tolerance", type=float, default=0.010)
    parser.add_argument(
        "--postorientation_max_position_command", type=float, default=0.15
    )
    parser.add_argument(
        "--postorientation_min_center_clearance", type=float, default=0.0
    )
    parser.add_argument(
        "--postorientation_advance_lateral_bias", type=float, default=0.0
    )
    parser.add_argument(
        "--postorientation_min_path_progress", type=float, default=0.0
    )
    parser.add_argument("--orientation_tolerance_deg", type=float, default=5.0)
    parser.add_argument("--orientation_max_steps", type=int, default=200)
    parser.add_argument("--orientation_position_scale", type=float, default=0.08)
    parser.add_argument("--orientation_max_position_command", type=float, default=None)
    parser.add_argument("--rotation_scale", type=float, default=0.5)
    parser.add_argument("--max_rotation_command", type=float, default=0.1)
    parser.add_argument("--transport_clearance", type=float, default=0.10)
    parser.add_argument("--transport_end_height_drop", type=float, default=0.0)
    parser.add_argument("--transport_bypass_path_fraction", type=float, default=0.0)
    parser.add_argument("--transport_bypass_lateral_bias", type=float, default=0.0)
    parser.add_argument(
        "--transport_bypass_min_path_progress", type=float, default=0.0
    )
    parser.add_argument("--max_grasp_offset_drift", type=float, default=0.025)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--gripper_probe_steps", type=int, default=8)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--pregrasp_detour_x", type=float, default=None)
    parser.add_argument("--pregrasp_detour_y", type=float, default=None)
    parser.add_argument("--pregrasp_clearance", type=float, default=0.0)
    parser.add_argument("--transport_via_x", type=float, default=None)
    parser.add_argument("--transport_via_y", type=float, default=None)
    parser.add_argument("--transport_obstacle_clearance", type=float, default=0.0)
    parser.add_argument("--transport_obstacle_segments", type=int, default=6)
    parser.add_argument("--transport_arc_position_tolerance", type=float, default=0.0)
    parser.add_argument("--grasp_height", type=float, default=0.015)
    parser.add_argument(
        "--grasp_height_candidates",
        default="",
        help="Optional comma-separated grasp heights searched before XY offsets",
    )
    parser.add_argument("--grasp_offset_fractions", default="0.60,0.80")
    parser.add_argument("--grasp_seat_steps", type=int, default=15)
    parser.add_argument("--grasp_seat_max_command", type=float, default=0.08)
    parser.add_argument("--lift_height", type=float, default=0.16)
    parser.add_argument("--min_grasp_lift", type=float, default=0.03)
    parser.add_argument("--preplace_height", type=float, default=0.10)
    parser.add_argument("--place_offset_x", type=float, default=0.0)
    parser.add_argument("--place_offset_y", type=float, default=0.0)
    parser.add_argument("--transport_place_offset_x", type=float, default=None)
    parser.add_argument("--transport_place_offset_y", type=float, default=None)
    parser.add_argument("--final_center_position_tolerance", type=float, default=0.010)
    parser.add_argument("--release_clearance", type=float, default=0.002)
    parser.add_argument("--contact_hold_steps", type=int, default=5)
    parser.add_argument("--require_support_contact_before_release", action="store_true")
    parser.add_argument("--support_contact_max_descent", type=float, default=0.12)
    parser.add_argument("--support_contact_max_steps", type=int, default=160)
    parser.add_argument("--place_descent_max_command", type=float, default=0.04)
    parser.add_argument("--support_contact_hold_steps", type=int, default=10)
    parser.add_argument("--support_contact_settle_max_steps", type=int, default=80)
    parser.add_argument("--max_pre_release_linear_speed", type=float, default=0.02)
    parser.add_argument("--release_steps", type=int, default=12)
    parser.add_argument("--post_release_support_hold_steps", type=int, default=10)
    parser.add_argument("--post_release_support_max_steps", type=int, default=80)
    parser.add_argument("--max_post_release_linear_speed", type=float, default=0.03)
    parser.add_argument("--max_post_release_displacement", type=float, default=0.015)
    parser.add_argument("--max_safe_lift_height", type=float, default=float("inf"))
    parser.add_argument("--retreat_height", type=float, default=0.10)
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.95)
    parser.add_argument("--max_place_xy_offset", type=float, default=0.060)
    parser.add_argument("--max_place_height_gap", type=float, default=0.030)
    parser.add_argument("--max_occluder_displacement", type=float, default=0.020)
    parser.add_argument(
        "--video_dir",
        default="",
        help="Optional output directory for successful policy-camera safe-reference MP4s",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=1,
        help="Maximum successful MP4s to save; 0 means unlimited",
    )
    parser.add_argument("--policy_camera", default="agentview")
    parser.add_argument("--video_resolution", type=int, default=256)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_stride", type=int, default=1)
    parser.add_argument("--video_match_wait_steps", type=int, default=10)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--environment_horizon", type=int, default=1000)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument(
        "--grasp_action_trajectories",
        default="",
        help="Optional paired successful-Eb trajectories used through verified grasp",
    )
    parser.add_argument(
        "--branch_grasp_prefix_on_contact",
        action="store_true",
        help="Branch from the paired prefix at measured gripper-target contact, then lift vertically",
    )
    parser.add_argument(
        "--prefix_grasp_seat_steps",
        type=int,
        default=0,
        help="Closed-gripper hold steps after branching at grasp contact",
    )
    parser.add_argument(
        "--prefix_lift_max_position_command",
        type=float,
        default=None,
        help="Optional slow-lift OSC command cap after branching at grasp contact",
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict != "PASS_DYNAMIC_SAFE_REFERENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
