"""Real-action access probe for the native two-moka-pot LIBERO task.

The probe starts from two partial-goal states that share one preregistered final
arrangement:

* ``far_first``: the far pot is already on the stove; place the near pot.
* ``near_first``: the near pot is already on the stove; place the far pot.

Only native moka-pot poses and velocities are changed while constructing the
partial states.  The remaining placement is performed through LIBERO's 7-D
OSC action interface.  This is a falsification probe: safe success in both
orders rejects the simple "near-first blocks far" hypothesis.  A one-sided
failure is only a candidate effect, not proof, because alternate safe routes
must still be tested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks.l3b_order_candidates import (
    CANDIDATES,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    SUITE,
    _refresh_observation,
    _trusted_native_states,
    body_measurement,
    native_bddl_path,
    static_preflight,
)
from experiments.robot.libero.tasks.probe_l3b_moka_pair_fit import (
    COOK_SITE,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_TILT_DEG,
    POT_BODIES,
    _free_joint_addresses,
    _set_free_body_pose,
    yaw_quaternion_wxyz,
)


GRASP_OFFSETS_XY = (
    (0.0, 0.0),
    (0.01, 0.0),
    (-0.01, 0.0),
    (0.0, 0.01),
    (0.0, -0.01),
)


def _policy_rgb(observation) -> np.ndarray:
    image = np.asarray(observation["agentview_image"])
    return np.ascontiguousarray(image[::-1, ::-1]).copy()


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    if not frames:
        raise ValueError("cannot write an empty review video")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"review video was not written: {path}")


def _runtime_task():
    from libero.libero import benchmark, get_libero_path

    candidate = CANDIDATES["moka"]
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(candidate.task_id)
    if task.bddl_file != candidate.task_file or task.language != candidate.prompt:
        raise ValueError("runtime native moka task does not match the lock")
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if bddl_path.resolve() != native_bddl_path(candidate).resolve():
        raise ValueError("runtime BDDL is not the locked native BDDL")
    return candidate, suite, task, bddl_path


def _load_zero_yaw_final_layout(path: Path) -> dict[str, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_STABLE_NATIVE_MOKA_PAIR_FOUND":
        raise ValueError("moka pair-fit report did not pass")
    for record in report.get("accepted", []):
        if all(
            abs(float(record["yaw_deg"][body_name])) < 1e-9
            for body_name in POT_BODIES
        ):
            return {
                body_name: np.asarray(
                    record["stability"][body_name]["first_policy"]["position"],
                    dtype=float,
                )
                for body_name in POT_BODIES
            }
    raise ValueError("pair-fit report has no accepted zero-yaw layout")


def _state_with_first_placement(
    env,
    native_state: np.ndarray,
    *,
    placed_body: str,
    placed_position: np.ndarray,
    settle_steps: int,
) -> tuple[np.ndarray, dict]:
    env.reset()
    env.set_init_state(native_state)
    # Normalize both pot yaws so that the two partial states lead to the exact
    # same preregistered final arrangement under a zero-rotation OSC transport.
    for body_name in POT_BODIES:
        current = body_measurement(env, body_name)
        position = np.asarray(current["position"], dtype=float)
        _set_free_body_pose(
            env,
            body_name,
            position,
            yaw_quaternion_wxyz(0.0),
        )
    spawn = np.asarray(placed_position, dtype=float).copy()
    spawn[2] += 0.03
    _set_free_body_pose(
        env,
        placed_body,
        spawn,
        yaw_quaternion_wxyz(0.0),
    )
    env.sim.forward()
    for _ in range(settle_steps):
        env.step(DUMMY_ACTION)
    state = np.asarray(env.sim.get_state().flatten(), dtype=float)
    placed = body_measurement(env, placed_body)
    other_body = POT_BODIES[1] if placed_body == POT_BODIES[0] else POT_BODIES[0]
    other = body_measurement(env, other_body)
    if placed["tilt_deg"] > MAX_TILT_DEG:
        raise ValueError(f"{placed_body} tipped while constructing partial state")
    if not any(
        str(name).startswith("flat_stove_1_") for name in placed["contacts"]
    ):
        raise ValueError(f"{placed_body} lacks stove support in partial state")
    if any(str(name).startswith("moka_pot_") for name in placed["contacts"]):
        raise ValueError(f"{placed_body} contacts the remaining pot")
    return state, {"placed": placed, "remaining": other}


class AccessRollout:
    def __init__(
        self,
        env,
        observation,
        *,
        placed_body: str,
        moving_body: str,
        args,
    ):
        self.env = env
        self.observation = observation
        self.placed_body = placed_body
        self.moving_body = moving_body
        self.args = args
        self.frames = [_policy_rgb(observation)]
        self.steps = 0
        self.forbidden_contacts: list[dict] = []

    def _observe_forbidden_contacts(self, phase: str) -> None:
        placed = body_measurement(self.env, self.placed_body)
        moving = body_measurement(self.env, self.moving_body)
        contacts = []
        for name in placed["contacts"]:
            text = str(name)
            if text.startswith(("robot0_", "gripper0_", self.moving_body[:-4])):
                contacts.append(f"{self.placed_body}::{text}")
        for name in moving["contacts"]:
            text = str(name)
            if text.startswith(self.placed_body[:-4]):
                contacts.append(f"{self.moving_body}::{text}")
        if contacts:
            self.forbidden_contacts.append(
                {
                    "step": self.steps,
                    "phase": phase,
                    "contacts": sorted(set(contacts)),
                }
            )

    def advance(self, action: np.ndarray, phase: str) -> None:
        self.observation, _, _, _ = self.env.step(action.tolist())
        self._observe_forbidden_contacts(phase)
        if self.steps % self.args.video_stride == 0:
            self.frames.append(_policy_rgb(self.observation))
        self.steps += 1

    def hold(self, gripper: float, count: int, phase: str) -> None:
        for _ in range(count):
            action = np.zeros(7, dtype=float)
            action[-1] = gripper
            self.advance(action, phase)

    def seat_grasp(self, target: np.ndarray) -> None:
        for _ in range(self.args.grasp_seat_steps):
            current = np.asarray(
                self.observation["robot0_eef_pos"], dtype=float
            )
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(
                (np.asarray(target) - current) / self.args.action_scale,
                -self.args.grasp_seat_max_command,
                self.args.grasp_seat_max_command,
            )
            action[-1] = 1.0
            self.advance(action, "grasp_seat")

    def move(
        self,
        target: np.ndarray,
        gripper: float,
        phase: str,
        *,
        max_steps: int | None = None,
        tolerance: float | None = None,
    ) -> tuple[bool, float]:
        max_steps = self.args.max_waypoint_steps if max_steps is None else max_steps
        tolerance = (
            self.args.position_tolerance if tolerance is None else tolerance
        )
        best = float("inf")
        for _ in range(max_steps):
            current = np.asarray(
                self.observation["robot0_eef_pos"], dtype=float
            )
            error = float(np.linalg.norm(np.asarray(target) - current))
            best = min(best, error)
            if error <= tolerance:
                return True, best
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(
                (np.asarray(target) - current) / self.args.action_scale,
                -1.0,
                1.0,
            )
            action[-1] = gripper
            self.advance(action, phase)
        return False, best


def _complete_remaining_placement(
    env,
    partial_state: np.ndarray,
    *,
    order: str,
    placed_body: str,
    moving_body: str,
    target_position: np.ndarray,
    grasp_offset_xy: np.ndarray,
    args,
) -> tuple[dict, list[np.ndarray]]:
    observation = env.reset()
    observation = env.set_init_state(partial_state)
    for _ in range(FORMAL_WAIT_STEPS):
        observation, _, _, _ = env.step(DUMMY_ACTION)
    rollout = AccessRollout(
        env,
        observation,
        placed_body=placed_body,
        moving_body=moving_body,
        args=args,
    )
    failure_reason = ""
    failure_stage = ""

    moving_start = np.asarray(
        body_measurement(env, moving_body)["position"], dtype=float
    )
    above = moving_start + np.asarray(
        [grasp_offset_xy[0], grasp_offset_xy[1], args.approach_height],
        dtype=float,
    )
    grasp = moving_start + np.asarray(
        [grasp_offset_xy[0], grasp_offset_xy[1], args.grasp_height],
        dtype=float,
    )
    for stage, target in (("approach", above), ("descend", grasp)):
        reached, best = rollout.move(target, -1.0, stage)
        contact_seated = bool(
            stage == "descend" and best <= args.grasp_contact_tolerance
        )
        if not reached and not contact_seated:
            failure_reason = f"waypoint_timeout_best_{best:.4f}"
            failure_stage = stage
            break

    if not failure_reason:
        rollout.seat_grasp(grasp)
        rollout.hold(1.0, args.grasp_steps, "grasp")
        grasped_offset = np.asarray(
            rollout.observation["robot0_eef_pos"], dtype=float
        ) - np.asarray(
            body_measurement(env, moving_body)["position"], dtype=float
        )
        lift_target = np.asarray(
            rollout.observation["robot0_eef_pos"], dtype=float
        )
        lift_target[2] += args.lift_height
        reached, best = rollout.move(lift_target, 1.0, "lift")
        lifted = float(
            body_measurement(env, moving_body)["position"][2] - moving_start[2]
        )
        if not reached:
            failure_reason = f"waypoint_timeout_best_{best:.4f}"
            failure_stage = "lift"
        elif lifted < args.minimum_lift:
            failure_reason = f"grasp_failed_lift_{lifted:.4f}"
            failure_stage = "lift"
    else:
        grasped_offset = np.zeros(3, dtype=float)
        lifted = 0.0

    if not failure_reason:
        desired = np.asarray(target_position, dtype=float)
        transit_body = np.asarray(
            body_measurement(env, moving_body)["position"], dtype=float
        )
        transit_body[2] = max(transit_body[2], desired[2]) + args.transport_height
        targets = (
            ("raise", transit_body + grasped_offset),
            (
                "translate",
                np.asarray([desired[0], desired[1], transit_body[2]])
                + grasped_offset,
            ),
            (
                "preplace",
                desired
                + np.asarray([0.0, 0.0, args.release_clearance])
                + grasped_offset,
            ),
        )
        for stage, target in targets:
            reached, best = rollout.move(
                target,
                1.0,
                stage,
                max_steps=args.transport_max_waypoint_steps,
            )
            if not reached:
                failure_reason = f"waypoint_timeout_best_{best:.4f}"
                failure_stage = stage
                break

    if not failure_reason:
        rollout.hold(-1.0, args.release_steps, "release")
        retreat = np.asarray(
            rollout.observation["robot0_eef_pos"], dtype=float
        )
        retreat[2] += args.retreat_height
        reached, best = rollout.move(retreat, -1.0, "retreat")
        if not reached:
            failure_reason = f"waypoint_timeout_best_{best:.4f}"
            failure_stage = "retreat"
        rollout.hold(-1.0, args.final_settle_steps, "settle")

    final_placed = body_measurement(env, placed_body)
    final_moving = body_measurement(env, moving_body)
    task_success = bool(env.check_success())
    stable = all(
        measurement["tilt_deg"] <= MAX_TILT_DEG
        and measurement["linear_speed_mps"] is not None
        and measurement["linear_speed_mps"] <= MAX_FINAL_LINEAR_SPEED_MPS
        and measurement["angular_speed_radps"] is not None
        and measurement["angular_speed_radps"]
        <= MAX_FINAL_ANGULAR_SPEED_RADPS
        and any(
            str(name).startswith("flat_stove_1_")
            for name in measurement["contacts"]
        )
        for measurement in (final_placed, final_moving)
    )
    safe_success = bool(
        not failure_reason
        and not rollout.forbidden_contacts
        and task_success
        and stable
    )
    if not failure_reason and rollout.forbidden_contacts:
        failure_reason = "forbidden_order_contact"
        failure_stage = rollout.forbidden_contacts[0]["phase"]
    elif not failure_reason and not task_success:
        failure_reason = "native_goal_false"
        failure_stage = "final"
    elif not failure_reason and not stable:
        failure_reason = "unstable_final_state"
        failure_stage = "final"
    result = {
        "order": order,
        "placed_body": placed_body,
        "moving_body": moving_body,
        "grasp_offset_xy": grasp_offset_xy.tolist(),
        "grasp_lift_m": lifted,
        "steps": rollout.steps,
        "task_success": task_success,
        "stable_final": stable,
        "safe_success": safe_success,
        "failure_reason": failure_reason,
        "failure_stage": failure_stage,
        "forbidden_contacts": rollout.forbidden_contacts,
        "final": {
            placed_body: final_placed,
            moving_body: final_moving,
        },
    }
    return result, rollout.frames


def run_probe(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    candidate, suite, task, bddl_path = _runtime_task()
    native_states = _trusted_native_states(suite, task)
    final_layout = _load_zero_yaw_final_layout(Path(args.pair_fit_report))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(0)
    results = {}
    try:
        order_specs = {
            "far_first": {
                "placed_body": POT_BODIES[0],
                "moving_body": POT_BODIES[1],
            },
            "near_first": {
                "placed_body": POT_BODIES[1],
                "moving_body": POT_BODIES[0],
            },
        }
        for order, spec in order_specs.items():
            partial_state, partial_measurements = _state_with_first_placement(
                env,
                native_states[0],
                placed_body=spec["placed_body"],
                placed_position=final_layout[spec["placed_body"]],
                settle_steps=args.construction_settle_steps,
            )
            attempts = []
            for attempt_index, offset in enumerate(GRASP_OFFSETS_XY):
                result, frames = _complete_remaining_placement(
                    env,
                    partial_state,
                    order=order,
                    placed_body=spec["placed_body"],
                    moving_body=spec["moving_body"],
                    target_position=final_layout[spec["moving_body"]],
                    grasp_offset_xy=np.asarray(offset, dtype=float),
                    args=args,
                )
                result["attempt_index"] = attempt_index
                attempts.append(result)
                if result["safe_success"]:
                    video_path = (
                        Path(args.output_dir)
                        / "success"
                        / f"{order}_safe_success.mp4"
                    )
                    _write_video(video_path, frames, args.video_fps)
                    result["review_video"] = str(video_path.resolve())
                    break
            results[order] = {
                "partial_state": partial_measurements,
                "attempts": attempts,
                "safe_success": any(item["safe_success"] for item in attempts),
            }
    finally:
        env.close()

    far_success = results["far_first"]["safe_success"]
    near_success = results["near_first"]["safe_success"]
    if far_success and near_success:
        verdict = "REJECT_SIMPLE_ORDER_CONSTRAINT_BOTH_ORDERS_SAFE"
    elif far_success and not near_success:
        verdict = "POTENTIAL_ORDER_EFFECT_NEEDS_ALTERNATE_ROUTE_TESTS"
    elif not far_success and near_success:
        verdict = "REJECT_HYPOTHESIZED_DIRECTION_NEAR_FIRST_NOT_WORSE"
    else:
        verdict = "INCONCLUSIVE_OSC_CONTROLLER_FAILED_BOTH_ORDERS"
    return {
        **static_preflight(candidate),
        "diagnostic_only": True,
        "formal_authorized": False,
        "pair_fit_report": str(Path(args.pair_fit_report).resolve()),
        "same_final_layout": {
            body_name: position.tolist()
            for body_name, position in final_layout.items()
        },
        "intervention": {
            "changed_bodies": list(POT_BODIES),
            "changed_fields": ["free_joint_qpos", "free_joint_qvel"],
            "asset_inventory_changed": False,
            "prompt_changed": False,
            "bddl_changed": False,
        },
        "results": results,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe real-action order access for the native moka task"
    )
    parser.add_argument(
        "--pair-fit-report",
        default=(
            "review/L3-B_order_candidates_task/moka_pair_fit/"
            "moka_pair_fit.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="review/L3-B_order_candidates_task/moka_order_access",
    )
    parser.add_argument("--construction-settle-steps", type=int, default=80)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.006)
    parser.add_argument("--max-waypoint-steps", type=int, default=160)
    parser.add_argument("--transport-max-waypoint-steps", type=int, default=220)
    parser.add_argument("--approach-height", type=float, default=0.18)
    parser.add_argument("--grasp-height", type=float, default=0.035)
    parser.add_argument("--grasp-contact-tolerance", type=float, default=0.045)
    parser.add_argument("--grasp-seat-steps", type=int, default=20)
    parser.add_argument("--grasp-seat-max-command", type=float, default=0.2)
    parser.add_argument("--grasp-steps", type=int, default=15)
    parser.add_argument("--lift-height", type=float, default=0.13)
    parser.add_argument("--minimum-lift", type=float, default=0.04)
    parser.add_argument("--transport-height", type=float, default=0.10)
    parser.add_argument("--release-clearance", type=float, default=0.008)
    parser.add_argument("--release-steps", type=int, default=15)
    parser.add_argument("--retreat-height", type=float, default=0.12)
    parser.add_argument("--final-settle-steps", type=int, default=30)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_probe(args)
    report_path = output_dir / "moka_order_access.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{report['verdict']} report={report_path}")


if __name__ == "__main__":
    main()
