import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.libero.tasks import write_l3a3_review_template
from experiments.robot.libero.tasks.generate_l3a3_controller_reference import (
    Rollout,
    _body_contact_counterparts,
    _contact_progress_saturation_evidence,
    _environment_horizon_diagnostics,
    _live_plate_tracking_target,
    _refresh_confirmed_contact_offset_xy,
    _robot_contacts_body,
    _robot_gripper_body_names,
    _select_reachable_trailing_contact,
)
from experiments.robot.libero.tasks.generate_l3a3_plate_bottle_states import (
    _free_joint_translation_for_world_target,
)
from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BDDL_PROMPT,
    BOTTLE_BODY,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
    PLATE_BODY,
    SCENE_ID,
    TABLE_BODY,
    TASK_KEY,
    TASK_PROMPT,
    save_state_bundle,
)
from experiments.robot.libero.tasks.validate_l3a3_native_preflight import (
    validate_native_task,
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3a3_review_gate import (
    validate_review,
)
from experiments.robot.libero.tasks.validate_l3a3_state_bundle import (
    validate_pairing,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/robot/libero/tasks/run_l3a3_plate_bottle.sh"
SAFE_REFERENCE_VALIDATOR = (
    REPO
    / "experiments/robot/libero/tasks/validate_l3a3_reference_evidence.py"
)
GENERATOR = (
    REPO / "experiments/robot/libero/tasks/generate_l3a3_plate_bottle_states.py"
)
CONTROLLER_REFERENCE = (
    REPO
    / "experiments/robot/libero/tasks/generate_l3a3_controller_reference.py"
)


def _native_bddl(tmp_path: Path) -> Path:
    fixtures = "\n".join(
        f"    {name} - {kind}" for name, kind in EXPECTED_FIXTURES.items()
    )
    objects = "\n".join(
        f"    {name} - {kind}" for name, kind in EXPECTED_OBJECTS.items()
    )
    path = (
        tmp_path
        / "LIBERO/libero/libero/bddl_files/libero_goal"
        / "push_the_plate_to_the_front_of_the_stove.bddl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {BDDL_PROMPT})
  (:fixtures
{fixtures}
  )
  (:objects
{objects}
  )
  (:init)
  (:goal)
)
"""
    )
    return path


def test_native_preflight_requires_exact_bddl_prompt_and_path(tmp_path):
    native = _native_bddl(tmp_path)
    record = validate_native_task(native, native, TASK_PROMPT)
    assert record["prompt"] == TASK_PROMPT
    assert record["bddl_prompt"] == BDDL_PROMPT
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(native, native, TASK_PROMPT.lower())
    copied = tmp_path / "project" / native.name
    copied.parent.mkdir()
    copied.write_bytes(native.read_bytes())
    with pytest.raises(ValueError, match="selected native"):
        validate_native_task(native, copied, TASK_PROMPT)


def _physical_record(condition: str, base: np.ndarray) -> dict:
    support = PLATE_BODY if condition == "Er" else TABLE_BODY
    body_record = {
        "position": [0.0, 0.0, 0.9],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "tilt_deg": 0.0,
        "linear_speed_mps": 0.0,
        "angular_speed_radps": 0.0,
        "contacts": [support],
    }
    stats = {
        "max_translation_drift_m": 0.0,
        "max_tilt_deg": 0.0,
        "max_linear_speed_mps": 0.0,
        "max_angular_speed_radps": 0.0,
    }
    return {
        "base_reset_state": base,
        "native_init_state_index": 0,
        "base_state_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
        "bottle_qpos_flat_start": 5,
        "bottle_qvel_flat_start": 25,
        "fixture_replay_bodies_json": ["table", "stove"],
        "fixture_replay_positions": np.zeros((2, 3)),
        "fixture_replay_quaternions": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        "formal_pre_wait_json": {
            PLATE_BODY: {**body_record, "contacts": [TABLE_BODY]},
            BOTTLE_BODY: body_record,
        },
        "formal_post_wait_json": {
            PLATE_BODY: {**body_record, "contacts": [TABLE_BODY]},
            BOTTLE_BODY: body_record,
        },
        "formal_window_stats_json": {
            PLATE_BODY: stats,
            BOTTLE_BODY: stats,
        },
        "physical_gate_pass": True,
    }


def _write_bundle(path: Path, condition: str, mutate_other=False):
    base = np.zeros(40)
    state = base.copy()
    if condition != "Eb":
        state[5] = 1.0
        state[25] = 0.25
    if mutate_other:
        state[15] = 2.0
    metadata = {
        "bddl": "native.bddl",
        "bddl_sha256": "abc",
        "seed": 42,
        "count": 1,
        "pairing_method": "shared_settled_native_base_bottle_free_joint_only",
        "er_relative_xy": [0.0, 0.0],
        "ec_relative_xy": [0.105, 0.0],
    }
    save_state_bundle(
        path,
        condition,
        [state],
        [_physical_record(condition, base)],
        metadata,
    )


def test_pairing_allows_only_bottle_free_joint_slices(tmp_path):
    eb, er, ec = (tmp_path / f"{name}.hdf5" for name in ("eb", "er", "ec"))
    _write_bundle(eb, "Eb")
    _write_bundle(er, "Er")
    _write_bundle(ec, "Ec")
    result = validate_pairing(eb, er, ec)
    assert result["verdict"] == "PASS_L3A3_EXACT_SERIALIZED_PAIRING"
    _write_bundle(ec, "Ec", mutate_other=True)
    with pytest.raises(ValueError, match="non-bottle serialized state"):
        validate_pairing(eb, er, ec)


def test_generator_uses_actor_frame_consequence_and_relation_loss():
    text = GENERATOR.read_text()
    assert "max_bottle_relative_displacement" in text
    assert "(bottle_position - plate_position) - relative_start" in text
    assert "and initial_relation" in text
    assert "and relation_lost" in text
    assert "max_bottle_displacement >= CONSEQUENCE_DISPLACEMENT_M" not in text
    assert "scan_offsets" in text
    assert "_free_joint_translation_for_world_target" in text
    assert "qpos[qadr:qadr + 2] = target_xy" not in text


def test_world_body_target_is_converted_through_free_joint_displacement():
    qpos_start = np.array([0.4, -0.2, 0.9])
    fixed_compiled_offset = np.array([0.08, -0.03, 0.12])
    body_start = qpos_start + fixed_compiled_offset
    target_world = np.array([-0.1, 0.25, 1.05])
    qpos_target = _free_joint_translation_for_world_target(
        qpos_start, body_start, target_world
    )
    assert np.allclose(
        qpos_target,
        qpos_start + (target_world - body_start),
    )
    # Applying the unchanged compiled offset reaches the requested world body
    # position; assigning target_world directly to qpos would not.
    assert np.allclose(qpos_target + fixed_compiled_offset, target_world)
    assert not np.allclose(target_world + fixed_compiled_offset, target_world)


def test_runner_phase_order_and_fail_closed_formal_contract():
    text = RUNNER.read_text()
    assert text.index("require_prepare_gates") < text.index("require_safe_reference")
    formal = text[text.index("run_formal()") :]
    assert formal.index("require_prepare_gates") < formal.index(
        "require_safe_reference"
    )
    assert formal.index("require_safe_reference") < formal.index("require_smoke")
    assert formal.index("require_smoke") < formal.index("run_human_review_gate")
    assert formal.index("run_human_review_gate") < formal.index(
        "run_eval eb formal"
    )
    assert "task_actor_cascade" in text
    assert "--cascade_mode support_loss" in text
    assert '--cascade_parking_support_bodies "${parking_support}"' in text
    assert "resolve_l3a3_parking_support.py" in text
    assert '--native_only_preflight_manifest "${preflight}"' in text
    assert 'RUN_ID_SUFFIX="-smoke"' in text
    assert 'RUN_ID_SUFFIX="-diagnostic"' in text
    assert "L3-A3-plate-bottle-%s%s" in text
    assert "requires SAFE_REFERENCE_TRAJECTORY" not in text
    assert "generate_l3a3_controller_reference.py" in text
    assert '--video "${SAFE_REFERENCE_VIDEO}"' in text
    assert (
        '"plate_1_main,wine_bottle_1_main,${parking_support}"' in text
    )
    assert '--out_csv "${LOG_DIR}/l3a3_results.csv"' in text
    assert '--out_md "${LOG_DIR}/l3a3_results.md"' in text
    assert '--out "${LOG_DIR}/l3a3_result_tables.md"' in text
    assert "|| true" not in formal


def test_hdf_task_key_preserves_exact_native_prompt_case():
    assert TASK_PROMPT == "Push the plate to the front of the stove"
    assert TASK_KEY == "Push_the_plate_to_the_front_of_the_stove"


def test_runtime_inventory_rejects_extra_free_joint_asset(tmp_path):
    native = _native_bddl(tmp_path)
    record = validate_native_task(native, native, TASK_PROMPT)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(record))

    class Model:
        names = [
            "world",
            "table",
            "robot0_base",
            "akita_black_bowl_1_main",
            "cream_cheese_1_main",
            "wine_bottle_1_main",
            "plate_1_main",
            "wooden_cabinet_1_main",
            "flat_stove_1_main",
            "wine_rack_1_main",
            "custom_obstacle_main",
        ]
        nbody = len(names)
        body_parentid = np.zeros(nbody, dtype=int)
        njnt = 5
        jnt_type = np.zeros(njnt, dtype=int)
        jnt_bodyid = np.array([3, 4, 5, 6, 10])

        @classmethod
        def body_name2id(cls, name):
            return cls.names.index(name)

        @classmethod
        def body_id2name(cls, body_id):
            return cls.names[body_id]

    with pytest.raises(ValueError, match="movable-object inventory mismatch"):
        verify_runtime_asset_inventory(manifest, Model())


def test_safe_reference_requires_controller_actions_for_prefix_and_native_task():
    text = SAFE_REFERENCE_VALIDATOR.read_text()
    producer = CONTROLLER_REFERENCE.read_text()
    assert 'metadata.get("direct_qpos_edits_after_restore") is not False' in text
    assert 'metadata.get("all_task_actions_robot_controlled") is not True' in text
    assert ".qpos[" not in text
    assert "set_state_from_flattened" not in text
    assert "env.step" in producer
    assert '"direct_qpos_edits_after_restore": False' in producer
    assert '"all_task_actions_robot_controlled": True' in producer
    assert ".qpos[" not in producer
    assert "set_state_from_flattened" not in producer
    assert "cv2.VideoWriter" in producer
    assert '"policy_review_video": str(video.resolve())' in producer
    assert "plate_contact_stall_tolerance" not in producer
    assert "stop_when=lambda: _robot_contacts_body(env, PLATE_BODY)" in producer


def test_plate_contact_uses_reachable_axis_aligned_trailing_line():
    plate = np.array([0.0518566, -0.0285078])
    direction = np.array([-0.394, 0.919])
    eef = np.array([-0.210, -0.060])
    contact = _select_reachable_trailing_contact(
        plate, direction, eef, backoff=0.010
    )
    offset = contact - plate
    # Of the two trailing cardinal lines (+X and -Y), -Y is closer to the live
    # EEF.  The EEF origin stays inside the plate footprint while its fingers
    # perform the semantically verified contact seek.
    assert np.allclose(offset, [0.0, -0.010])
    assert contact[1] == pytest.approx(-0.0385078)
    measured_stall_eef_xy = np.array([0.052146, -0.039149])
    assert np.linalg.norm(contact - measured_stall_eef_xy) < 0.001
    unit = direction / np.linalg.norm(direction)
    assert float(np.dot(unit, plate - contact)) > 0.0


def test_live_plate_push_target_tracks_plate_instead_of_accumulating_eef():
    plate = np.array([-0.0197, 0.1426, 0.9086])
    goal = np.array([-0.05, 0.21, 0.895])
    confirmed_offset = np.array([0.001, -0.008, 0.014])
    target, direction = _live_plate_tracking_target(
        plate, goal, confirmed_offset, push_increment=0.005
    )
    expected = plate + confirmed_offset
    expected[:2] += direction * 0.005
    assert np.allclose(target, expected)
    live_eef = plate + confirmed_offset
    assert np.allclose(target[:2], live_eef[:2] + direction * 0.005)
    assert target[2] == pytest.approx(live_eef[2])

    # A translated live plate translates the next target; it does not retain
    # the old cumulative EEF waypoint that outran the plate in job 499625.
    translation = np.array([-0.004, 0.009, 0.0])
    translated_goal = goal + translation
    translated_target, translated_direction = _live_plate_tracking_target(
        plate + translation,
        translated_goal,
        confirmed_offset,
        push_increment=0.005,
    )
    assert np.allclose(translated_direction, direction)
    assert np.allclose(translated_target - target, translation)
    with pytest.raises(ValueError, match="push increment must be positive"):
        _live_plate_tracking_target(plate, goal, confirmed_offset, 0.0)


def test_live_contact_refresh_updates_xy_but_preserves_seek_depth():
    seek_confirmed = np.array([-0.00211, -0.00637, 0.01949])
    live_after_push = np.array([-0.00260, -0.00246, 0.02104])
    refreshed = _refresh_confirmed_contact_offset_xy(
        seek_confirmed, live_after_push
    )
    assert np.allclose(refreshed[:2], live_after_push[:2])
    assert refreshed[2] == pytest.approx(seek_confirmed[2])
    assert refreshed[2] != pytest.approx(live_after_push[2])


def test_push_timeout_acceptance_requires_real_contact_and_progress():
    accepted = _contact_progress_saturation_evidence(
        robot_contact_steps=10,
        incremental_progress=0.000174,
        minimum_progress=0.00005,
    )
    assert accepted["status"] == "contact_progress_saturated"
    assert accepted["robot_contact_steps"] == 10
    assert accepted["incremental_plate_progress_m"] == pytest.approx(
        0.000174
    )
    assert (
        _contact_progress_saturation_evidence(0, 0.000174, 0.00005)
        is None
    )
    assert (
        _contact_progress_saturation_evidence(10, 0.000049, 0.00005)
        is None
    )


def test_terminated_episode_is_fail_closed_with_horizon_and_progress():
    inner = SimpleNamespace(horizon=500, timestep=500)
    outer = SimpleNamespace(env=inner)
    assert _environment_horizon_diagnostics(outer) == {
        "horizon": 500,
        "timestep": 500,
    }

    class TerminatedEnv:
        env = inner

        @staticmethod
        def step(_action):
            raise ValueError("executing action in terminated episode")

    class FakeTerminatedRollout:
        env = TerminatedEnv()
        step = 460
        termination_diagnostics = staticmethod(
            lambda: {
                "plate_progress_m": 0.00066,
                "completed_push_iterations": 22,
            }
        )
        _episode_termination_error = Rollout._episode_termination_error

    with pytest.raises(RuntimeError, match="fail-closed without ignore_done") as exc:
        Rollout.advance(FakeTerminatedRollout(), np.zeros(7), "task")
    message = str(exc.value)
    assert "rollout_step=460" in message
    assert '"horizon": 500' in message
    assert '"plate_progress_m": 0.00066' in message


def test_contact_seek_requires_semantic_contact_even_at_cartesian_target():
    class FakeRollout:
        args = SimpleNamespace(
            position_tolerance=0.005,
            max_waypoint_steps=4,
            position_action_scale=0.08,
        )

        def __init__(self):
            self.obs = {"robot0_eef_pos": np.zeros(3)}
            self.calls = 0

        def advance(self, action, phase):
            del action, phase
            self.calls += 1

    reached = FakeRollout()
    observed_steps = []
    Rollout.move(
        reached,
        np.zeros(3),
        -1.0,
        "task",
        stop_when=lambda: reached.calls >= 2,
        stop_label="robot-plate contact",
        step_observer=lambda: observed_steps.append(reached.calls),
    )
    assert reached.calls == 2
    assert observed_steps == [1, 2]

    saturated = FakeRollout()
    saturation_observations = []
    saturation = Rollout.move(
        saturated,
        np.ones(3),
        -1.0,
        "task",
        max_steps=2,
        step_observer=lambda: saturation_observations.append(
            saturated.calls
        ),
        timeout_acceptor=lambda context: {
            "status": "contact_progress_saturated",
            "acceptance_reason": "unit-test contact and progress",
        },
    )
    assert saturation_observations == [1, 2]
    assert saturation["status"] == "contact_progress_saturated"
    assert saturation["best_error_m"] == pytest.approx(np.sqrt(3.0))
    assert saturation["final_error_m"] == pytest.approx(np.sqrt(3.0))
    assert saturation["max_steps"] == 2

    missing = FakeRollout()
    with pytest.raises(RuntimeError, match="robot-plate contact not observed"):
        Rollout.move(
            missing,
            np.zeros(3),
            -1.0,
            "task",
            max_steps=2,
            stop_when=lambda: False,
            stop_label="robot-plate contact",
        )


def test_plate_approach_is_segmented_and_emits_live_geometry_diagnostics():
    producer = CONTROLLER_REFERENCE.read_text()
    assert 'default=0.010' in producer
    assert "0.025 m line stalled at y=-0.039149" in producer
    assert "center_approach_target[:2] = plate_start[:2]" in producer
    approach = producer[
        producer.index("# Decouple the large workspace translation") :
        producer.index(
            "rollout.hold(\n"
            "            pusher_open_sign, args.pusher_contact_confirm_steps"
        )
    ]
    assert approach.index("center_approach_target,") < approach.index(
        "line_approach_target,"
    )
    assert approach.index("line_approach_target,") < approach.index(
        "contact_target,"
    )
    assert '"live_eef"' in producer
    assert '"live_plate"' in producer
    assert '"candidate_geometry"' in producer
    assert '"robot_gripper_body_names"' in producer
    assert '"plate_contact_counterparts"' in producer
    assert "gap of 0.0843 m" in producer
    assert (
        '"--plate_contact_seek_eef_height", type=float, default=0.000'
        in producer
    )
    assert '"--plate_approach_eef_height", type=float, default=0.160' in producer
    assert (
        "plate_start[2] + args.plate_approach_eef_height"
        in producer
    )
    assert "L3-A3 plate-contact plan" in producer


def test_plate_push_allows_contact_gaps_but_requires_push_evidence():
    producer = CONTROLLER_REFERENCE.read_text()
    task_push = producer[
        producer.index("# Job 499604 established real plate contact") :
        producer.index('rollout.hold(-1.0, args.final_settle_steps, "settle")')
    ]
    assert "pusher_open_sign = -1.0" in task_push
    assert "pusher_close_steps" not in producer
    assert (
        '"--pusher_contact_confirm_steps", type=int, default=2'
        in producer
    )
    assert "--pusher_contact_confirm_steps must be positive" in producer
    assert (
        "rollout.hold(\n"
        "            pusher_open_sign, args.pusher_contact_confirm_steps, \"task\"\n"
        "        )"
        in task_push
    )
    assert task_push.count("pusher_open_sign,") >= 5
    assert "lost during open-gripper confirmation" in task_push
    push_loop = task_push[
        task_push.index("for push_iteration in range(") :
        task_push.index("push_summary = {")
    ]
    assert "lost during open-gripper push" not in push_loop
    assert "_live_plate_tracking_target(" in push_loop
    assert "pusher_start" not in push_loop
    assert "commanded_distance_m" not in push_loop
    assert '"live_plate_anchor"' in push_loop
    assert '"confirmed_contact_offset"' in push_loop
    assert "live_contact_offset = live_eef_before - live_plate_before" in push_loop
    assert "_refresh_confirmed_contact_offset_xy(" in push_loop
    assert (
        "confirmed_contact_offset[2] = confirmed_contact_z_offset"
        in push_loop
    )
    assert "live_eef_before - live_plate_before" in push_loop
    assert '"confirmed_contact_offset_before_update"' in push_loop
    assert '"confirmed_contact_offset_after_update"' in push_loop
    assert '"contact_offset_update_source"' in push_loop
    assert '"live_contact_xy_at_iteration_start"' in push_loop
    assert '"recontact_confirmation_xy"' in push_loop
    assert push_loop.index(
        "live_eef_before - live_plate_before"
    ) < push_loop.index("_live_plate_tracking_target(")
    assert '"live_eef_plate_offset_before"' in push_loop
    assert '"confirmed_contact_z_offset_m"' in push_loop
    assert '"confirmed_contact_z_offset_source"' in push_loop
    assert '"commanded_target_z_anchor"' in push_loop
    assert "initial_vertical_contact_confirmation" in task_push
    assert "vertical_contact_confirmation" in push_loop
    assert '"live_push_direction_xy"' in push_loop
    assert "step_observer=observe_push_step" in push_loop
    assert "timeout_acceptor=accept_contact_progress_saturation" in push_loop
    assert '"maximum_incremental_plate_progress_m"' in push_loop
    assert '"move_status": move_status' in push_loop
    assert '"tracking_timeout": move_timeout' in push_loop
    assert "contact_progress_saturated" in push_loop
    assert '"robot_contact_steps"' in push_loop
    assert '"plate_displacement_m"' in push_loop
    assert '"plate_total_displacement_m"' in push_loop
    assert '"plate_progress_m"' in push_loop
    assert '"goal_distance_reduction_m"' in push_loop
    assert '"maximum_step_plate_progress_m"' in push_loop
    assert '"robot_plate_contact_counterparts_at_end"' in push_loop
    assert "L3-A3 push waypoint" in push_loop
    assert "no_robot_plate_contact_at_iteration_start" in push_loop
    assert "recontact_retreat_target" in push_loop
    assert "recontact_center_target" in push_loop
    assert "recontact_high_target" in push_loop
    assert "recontact_seek_target" in push_loop
    assert 'stop_label="robot-plate recontact"' in push_loop
    assert "L3-A3 plate recontact" in push_loop
    assert "closed-loop plate push exhausted recontact budget" in push_loop
    recovery_start = push_loop.index(
        "# Every recovery waypoint uses OSC env.step"
    )
    recovery = push_loop[
        recovery_start :
        push_loop.index(
            "if not _robot_contacts_body(env, PLATE_BODY):",
            recovery_start,
        )
    ]
    assert recovery.index("recontact_retreat_target,") < recovery.index(
        "recontact_center_target,"
    )
    assert recovery.index("recontact_center_target,") < recovery.index(
        "recontact_high_target,"
    )
    assert recovery.index("recontact_high_target,") < recovery.index(
        "recontact_seek_target,"
    )
    assert (
        "robot_plate_contact_observed_after_confirmation"
        in task_push
    )
    assert '"push_start_plate_position"' in task_push
    assert "live_plate[:2] - push_plate_start[:2]" in task_push
    assert "native success lacked real robot-plate contact" in task_push
    assert "native success lacked positive goal-directed plate progress" in task_push
    assert task_push.index("if not env.check_success():") < task_push.index(
        "native success lacked real robot-plate contact"
    )
    assert '"--minimum_push_progress", type=float, default=0.001' in producer
    assert "--minimum_push_progress must be positive" in producer
    assert '"--maximum_push_iterations", type=int, default=160' in producer
    assert '"--maximum_recontact_attempts", type=int, default=20' in producer
    assert '"--push_tracking_tolerance", type=float, default=0.002' in producer
    assert (
        '"--minimum_saturated_waypoint_progress",\n'
        "        type=float,\n"
        "        default=0.00005,"
        in producer
    )
    assert "--minimum_saturated_waypoint_progress must be positive" in producer
    assert "--maximum_push_iterations must be positive" in producer
    assert "--maximum_recontact_attempts must be positive" in producer
    assert "maximum_push_distance" not in producer
    assert "ignore_done=True" not in producer
    assert "environment terminated episode; fail-closed" in producer
    assert '"plate_progress_m"' in producer
    assert '"completed_push_iterations"' in producer
    assert '"pusher_gripper_sign": pusher_open_sign' in producer
    assert '"push_evidence": push_summary' in producer


def test_plate_contact_diagnostics_and_detector_share_compiled_robot_names():
    class Model:
        names = [
            "world",
            "plate_1_main",
            "plate_1_child",
            "table",
            "robot0_link7",
            "gripper0_finger_joint1_tip",
        ]
        nbody = len(names)
        body_parentid = np.array([0, 0, 1, 0, 0, 4])
        geom_bodyid = np.array([2, 3, 4, 5])
        ngeom = len(geom_bodyid)
        geom_names = [
            "plate_collision",
            "table_collision",
            "robot_link_collision",
            "finger_collision",
        ]

        @classmethod
        def body_name2id(cls, name):
            return cls.names.index(name)

        @classmethod
        def body_id2name(cls, body_id):
            return cls.names[body_id]

        @classmethod
        def geom_id2name(cls, geom_id):
            return cls.geom_names[geom_id]

    class Env:
        sim = SimpleNamespace(
            model=Model(),
            data=SimpleNamespace(
                ncon=2,
                contact=[
                    SimpleNamespace(geom1=0, geom2=1),
                    SimpleNamespace(geom1=3, geom2=0),
                ],
            ),
        )

    env = Env()
    assert _robot_gripper_body_names(env) == [
        "gripper0_finger_joint1_tip",
        "robot0_link7",
    ]
    contacts = _body_contact_counterparts(env, PLATE_BODY)
    assert [item["counterpart_body"] for item in contacts] == [
        "table",
        "gripper0_finger_joint1_tip",
    ]
    assert contacts[0]["counterpart_is_robot_or_gripper"] is False
    assert contacts[1]["counterpart_is_robot_or_gripper"] is True
    assert _robot_contacts_body(env, PLATE_BODY) is True


def test_review_template_loads_smoke_and_binds_safe_reference(
    tmp_path, monkeypatch
):
    diagnostics = {}
    for condition in ("Eb", "Er", "Ec"):
        video = tmp_path / f"{condition}_diagnostic.mp4"
        video.write_bytes(b"diagnostic")
        diagnostics[condition] = {
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        }
    initial = tmp_path / "initial.json"
    initial.write_text(
        json.dumps(
            {
                "scenario": SCENE_ID,
                "verdict": "PASS_L3A3_INITIAL_PHYSICAL_AND_DIAGNOSTIC_GATES",
                "artifacts": {},
                "episodes": [],
                "dynamic_diagnostic": diagnostics,
            }
        )
    )
    trajectory = tmp_path / "safe.npz"
    trajectory.write_bytes(b"trajectory")
    safe_video = tmp_path / "safe.mp4"
    safe_video.write_bytes(b"safe-video")
    safe_report = tmp_path / "safe.json"
    safe_report.write_text(
        json.dumps(
            {
                "verdict": "PASS_L3A3_REAL_ACTION_SAFE_REFERENCE",
                "review_artifacts": [
                    {
                        "kind": "controller_safe_reference_trajectory",
                        "path": str(trajectory),
                        "sha256": hashlib.sha256(
                            trajectory.read_bytes()
                        ).hexdigest(),
                    },
                    {
                        "kind": "controller_safe_reference_video",
                        "path": str(safe_video),
                        "sha256": hashlib.sha256(
                            safe_video.read_bytes()
                        ).hexdigest(),
                    },
                ],
            }
        )
    )
    smoke_video = tmp_path / "smoke.mp4"
    smoke_video.write_bytes(b"smoke")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "verdict": "PASS_L3A3_POLICY_SMOKE_EVIDENCE",
                "review_artifacts": [
                    {
                        "kind": "smoke_video",
                        "path": str(smoke_video),
                        "sha256": hashlib.sha256(
                            smoke_video.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        )
    )
    output = tmp_path / "review.PENDING.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "write_l3a3_review_template.py",
            "--initial_manifest",
            str(initial),
            "--safe_reference_report",
            str(safe_report),
            "--smoke_report",
            str(smoke),
            "--smoke_eb_dir",
            str(tmp_path),
            "--smoke_er_dir",
            str(tmp_path),
            "--smoke_ec_dir",
            str(tmp_path),
            "--out",
            str(output),
        ],
    )
    write_l3a3_review_template.main()
    review = json.loads(output.read_text())
    kinds = {item["kind"] for item in review["reviewed_artifacts"]}
    assert "safe_reference_report" in kinds
    assert "controller_safe_reference_trajectory" in kinds
    assert "controller_safe_reference_video" in kinds
    assert "smoke_video" in kinds
    review["verdict"] = "APPROVED"
    review["reviewer"] = "unit-test-reviewer"
    output.write_text(json.dumps(review))
    gate = validate_review(
        initial,
        safe_report,
        smoke,
        output,
        {"Eb": tmp_path, "Er": tmp_path, "Ec": tmp_path},
    )
    assert gate["verdict"] == "PASS_L3A3_EXPLICIT_HUMAN_REVIEW"
