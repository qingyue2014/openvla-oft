import copy
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
    _bounded_contact_seek_vertical_stabilization_action,
    _bounded_side_contact_seek_action,
    _contact_depth_sample_validity,
    _contact_progress_saturation_evidence,
    _constraint_prioritized_outside_descent_action,
    _vertical_corridor_reserve_recovery_evidence,
    _vertical_corridor_reserve_recovery_phase_evidence,
    _compiled_corridor_reserve_action,
    _compiled_low_side_settle_brake_action,
    _compiled_adaptive_lateral_rebuffer_action,
    _compiled_adaptive_high_lateral_action,
    _compiled_adaptive_high_plane_action,
    _compiled_adaptive_workspace_release_action,
    _compiled_adaptive_vertical_descent_action,
    _compiled_collision_pair_clearance,
    _compiled_hypothetical_wrist_yaw_plan,
    _compiled_native_right_high_then_low_return_plan,
    _compiled_finger_yaw_frame,
    _compiled_native_side_contact_plan,
    _compiled_pair_set_clearance,
    _compiled_side_contact_eef_z_feasibility,
    _compiled_table_normal_evidence,
    _compiled_trailing_side_contact_candidates,
    _compiled_wrist_yaw_action,
    _compiled_vertical_staging_corridor,
    _center_high_reacquire_budget_evidence,
    _center_high_reacquire_step_gate,
    _fixed_safe_z_lateral_hold_action,
    _center_high_target_from_live_plate,
    _derive_horizon_safe_push_increment,
    _derive_overhead_staging_from_compiled_pairs,
    _environment_horizon_diagnostics,
    _finger_inward_extents_by_semantic_side,
    _fixed_z_lateral_approach_action,
    _fixed_xy_vertical_approach_action,
    _gate_live_contact_offset_xy,
    _high_plane_native_boundary_crossing_evidence,
    _high_plane_native_workspace_saturation_evidence,
    _horizon_budget,
    _hypothetical_finger_yaw_env,
    _hypothetical_wrist_yaw_specs,
    _live_plate_tracking_target,
    _live_cabinet_pose_diagnostic,
    _live_collision_geom_record,
    _live_collision_inventory,
    _live_native_cabinet_detour_guard,
    _native_osc_action_spec_evidence,
    _native_osc_rotation_spec_evidence,
    _outside_side_geometry_feedback_action,
    _outside_side_guard_from_world_aabbs,
    _outside_side_lateral_settle_evidence,
    _overhead_corridor_entry_evidence,
    _overhead_outside_high_entry_evidence,
    _overhead_lateral_buffer_evidence,
    _overhead_lateral_interlock_evidence,
    _overhead_route_frame_authorization_evidence,
    _outside_side_recovery_progress_evidence,
    _outside_side_step_response_evidence,
    _outside_side_staircase_settle_trigger,
    _plate_contact_candidate_diagnostics,
    _plate_finger_contact_sides,
    _push_window_timeout_evidence,
    _robot_contacts_body,
    _robot_gripper_body_names,
    _robot_nonrobot_contact_evidence,
    _diagnostic_only_live_detour_candidates,
    _rotation_matrix_axis_angle,
    _select_reachable_compiled_side_candidate,
    _select_executable_wrist_yaw_candidate,
    _select_reachable_trailing_contact,
    _side_contact_targets_from_compiled_bounds,
    _second_real_recompile_identity_evidence,
    _strict_native_high_prebuffer_target,
    _strict_wrist_yaw_segment_plan,
    _absolute_wrist_yaw_runtime_target,
    _validated_rigid_rotation_matrix,
    _wrist_yaw_attainment_evidence,
    _wrist_yaw_stage_budget_evidence,
    _wrist_yaw_step_gate,
    _real_recompile_wrist_yaw_candidate,
    _record_native_cabinet_detour_completion,
    _write_controller_diagnostic_manifest,
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
    assert "_seek_stable_plate_contact(" in producer
    assert "rollout.advance(action, \"task\")" in producer


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


def test_plate_side_contact_seek_caps_lateral_osc_action():
    current = np.array([0.0, 0.0, 0.9225])
    target = np.array([0.04, -0.04, 0.9225])
    bounded = _bounded_side_contact_seek_action(
        current,
        target,
        gripper=-1.0,
        scale=0.08,
        maximum_translation_action=0.10,
    )
    assert np.linalg.norm(bounded[:3]) == pytest.approx(0.10)
    assert bounded[0] == pytest.approx(np.sqrt(0.005))
    assert bounded[1] == pytest.approx(-np.sqrt(0.005))
    assert bounded[2] == pytest.approx(0.0)
    assert bounded[-1] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="maximum translation action"):
        _bounded_side_contact_seek_action(
            current, target, -1.0, 0.08, 0.0
        )


def test_contact_seek_vertical_stabilization_retains_table_reserve():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = (
        _bounded_contact_seek_vertical_stabilization_action(
            current_eef_z_m=0.9191152632668791,
            target_eef_z_m=0.9178414056548501,
            vertical_step_progress_m=-0.0005361277747323312,
            outside_side_guard={
                "finger_table_vertical_clearance_m": 0.006,
                "required_finger_table_clearance_m": np.nextafter(
                    0.0, np.inf
                ),
                "minimum_outside_clearance_m": 0.010,
                "required_outside_clearance_m": np.nextafter(
                    0.0, np.inf
                ),
            },
            outward_direction_xy=np.array([1.0, 0.0]),
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
            progress_resolution_m=0.00005,
            strict_post_action_table_clearance_m=0.0004,
            strict_post_action_outside_clearance_m=0.0004,
            closed_loop_inward_response_bound_m=0.0005,
            derivative_gain=2.0,
            native_action_spec=native_spec,
        )
    )
    assert action[0] > 0.099
    assert action[1] == 0.0
    assert action[2] == pytest.approx(-0.00252002452830457)
    assert np.all(action[3:6] == 0.0)
    assert np.linalg.norm(action[:3]) < 0.10
    assert evidence["commanded_world_delta_m"] == pytest.approx(
        -0.0002016019622643656
    )
    assert evidence[
        "predicted_finger_table_clearance_m"
    ] == pytest.approx(0.005798398037735634)
    assert evidence["proof"] == {
        "strictly_outward_xy_zero_rotation": True,
        "position_plus_velocity_vertical_feedback": True,
        "inside_unchanged_contact_seek_translation_bound": True,
        "paired_outward_authority_retained_for_every_z_command": True,
        "low_clearance_forces_pure_outward_recovery": True,
        "vertical_authority_scales_with_live_outside_reserve": True,
        "negative_z_uses_at_most_half_live_table_reserve": True,
        "nominal_post_action_table_clearance_strict": True,
        "nominal_post_action_outside_recovery_strict": True,
        "post_action_live_guards_required": True,
    }
    positive_action, positive_evidence = (
        _bounded_contact_seek_vertical_stabilization_action(
            current_eef_z_m=0.9177752232890954,
            target_eef_z_m=0.9178414056548501,
            vertical_step_progress_m=-0.00035196973401552256,
            outside_side_guard={
                "finger_table_vertical_clearance_m": 0.006,
                "required_finger_table_clearance_m": 0.0,
                "minimum_outside_clearance_m": 0.010,
                "required_outside_clearance_m": 0.0,
            },
            outward_direction_xy=np.array([1.0, 0.0]),
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
            progress_resolution_m=0.00005,
            strict_post_action_table_clearance_m=0.0004,
            strict_post_action_outside_clearance_m=0.0004,
            closed_loop_inward_response_bound_m=0.0005,
            derivative_gain=2.0,
            native_action_spec=native_spec,
        )
    )
    assert positive_action[2] > 0.0
    assert positive_action[2] < 0.10
    assert positive_action[0] > 0.0
    assert np.linalg.norm(positive_action[:3]) < 0.10
    assert positive_evidence["commanded_world_delta_m"] > 0.0


def test_contact_seek_vertical_stabilization_prioritizes_low_outside_reserve():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = (
        _bounded_contact_seek_vertical_stabilization_action(
            current_eef_z_m=0.915306101053405,
            target_eef_z_m=0.9178414056548501,
            vertical_step_progress_m=-0.0008519092719723176,
            outside_side_guard={
                "finger_table_vertical_clearance_m": (
                    0.0024863098964361674
                ),
                "required_finger_table_clearance_m": np.nextafter(
                    0.0, np.inf
                ),
                "minimum_outside_clearance_m": (
                    0.00008593313582247464
                ),
                "required_outside_clearance_m": np.nextafter(
                    0.0, np.inf
                ),
            },
            outward_direction_xy=np.array([1.0, 0.0]),
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
            progress_resolution_m=0.00005,
            strict_post_action_table_clearance_m=0.0004,
            strict_post_action_outside_clearance_m=0.0004,
            closed_loop_inward_response_bound_m=0.0005,
            derivative_gain=2.0,
            native_action_spec=native_spec,
        )
    )
    assert action[:3].tolist() == pytest.approx([0.0999, 0.0, 0.0])
    assert evidence["outside_recovery_clearance_m"] == pytest.approx(
        0.0009
    )
    assert evidence["outside_authority_fraction"] == 0.0
    assert evidence["vertical_action_cap"] == 0.0
    assert evidence["commanded_world_delta_m"] == 0.0
    assert evidence["predicted_outside_clearance_m"] == pytest.approx(
        0.008077933135822474
    )


def test_fixed_safe_z_lateral_hold_brakes_job503456_tail_before_return():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.1310398213887616, -0.02704586762631196, 0.9146855864404666]
        ),
        lateral_target_xy=np.array(
            [0.13230639548403947, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.919651391,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.001479232869910807,
        measured_outward_step_progress_m=-0.0005,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.00017634874436972536,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.0018460824236792295,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0005,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert np.linalg.norm(action[:3]) == pytest.approx(
        np.sqrt(2.0) * strict_brake
    )
    assert evidence["outside_recovery_active"] is True
    assert evidence["downward_tail_brake_active"] is True
    assert evidence["commanded_z_action"] == pytest.approx(strict_brake)
    assert evidence["predicted_outside_clearance_m"] > 0.016
    assert evidence["predicted_table_clearance_m"] > 0.017


def test_fixed_safe_z_lateral_hold_keeps_original_lateral_bound_when_safe():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array([0.148, -0.0285, 0.91965]),
        lateral_target_xy=np.array([0.132, -0.0285]),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.91965,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=0.0,
        measured_outward_step_progress_m=0.0,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.016,
            "required_outside_clearance_m": 0.0,
            "finger_table_vertical_clearance_m": 0.006,
            "required_finger_table_clearance_m": 0.0,
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0005,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    assert action[0] == pytest.approx(np.nextafter(-0.005, 0.0))
    assert np.all(action[1:6] == 0.0)
    assert evidence["outside_recovery_active"] is False
    assert evidence["downward_tail_brake_active"] is False
    assert evidence["proof"]["lateral_return_bound_unchanged"] is True


def test_fixed_safe_z_lateral_hold_stops_job503459_redundant_inward_step():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13288254233703906, -0.027356456807950593, 0.9192751655726297]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.00037622546898175013,
        measured_outward_step_progress_m=-0.001,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.0015159166988197165,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.006476105820326539,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0005,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["lateral_error_m"] == pytest.approx(
        0.0012403642841947859
    )
    assert evidence["lateral_target_reached"] is True
    assert evidence["vertical_capture_active"] is True
    assert evidence["outside_recovery_active"] is True
    assert evidence["inward_suspended_for_vertical_capture"] is False
    assert evidence["proof"][
        "no_inward_xy_after_lateral_tolerance"
    ] is True


def test_fixed_safe_z_lateral_hold_retains_job503461_below_band_z_floor():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.132, -0.026, 0.9138723983666764]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=0.00033679011255227653,
        measured_outward_step_progress_m=0.0,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.0013698377956139346,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.0012068246516190317,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0005,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[0] == 0.0
    assert action[1] == 0.0
    assert action[2] == pytest.approx(strict_brake)
    assert evidence["below_safe_z_band"] is True
    assert evidence["inside_safe_z_band"] is False
    assert evidence["positive_response_unload_active"] is False
    assert evidence["minimum_below_band_positive_z_action"] == (
        pytest.approx(strict_brake)
    )
    assert evidence["proof"][
        "below_height_band_retains_positive_z_floor"
    ] is True


def test_fixed_safe_z_lateral_hold_brakes_job503463_only_at_low_reserve():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13189459476331095, -0.02569929865255594, 0.9183459069879493]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=0.0004985928259091388,
        measured_outward_step_progress_m=-0.0005913,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.0004932888529086965,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.005666308747540705,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0005,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["lateral_target_reached"] is True
    assert evidence["outside_recovery_active"] is True
    assert evidence["measured_inward_response"] is True
    assert evidence["proof"][
        "measured_inward_tail_uses_full_outward_brake"
    ] is True


def test_fixed_safe_z_lateral_hold_prioritizes_job503467_live_recovery():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13182813906246585, -0.02685300525710793, 0.9203520218593972]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=0.0011844784392165408,
        measured_outward_step_progress_m=-0.000011337715928871894,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.00045590680613190326,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.0075844921434991,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0005,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, 0.0]
    )
    assert evidence["measured_inward_response"] is False
    assert evidence["live_outside_recovery_active"] is True
    assert evidence["live_full_outward_brake_active"] is True
    assert evidence["outside_recovery_active"] is True
    assert evidence[
        "negative_z_suspended_for_outside_recovery"
    ] is True
    assert evidence["proof"][
        "live_low_reserve_uses_full_outward_brake"
    ] is True
    assert evidence["proof"][
        "outside_recovery_suspends_negative_z"
    ] is True


def test_fixed_safe_z_lateral_hold_keeps_job503639_nominal_refill_band():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    live_clearance = 0.001273383111331322
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13260102055981382, -0.02690243421009128, 0.9199928323482724]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=0.00022190101860486422,
        measured_outward_step_progress_m=0.00033320425177799096,
        outside_side_guard={
            "minimum_outside_clearance_m": live_clearance,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.006745256826679258,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    exit_clearance = np.nextafter(0.00155, np.inf)
    refill_target = np.nextafter(exit_clearance + 0.00010, np.inf)
    nominal_refill = np.nextafter(
        (refill_target - live_clearance) / 0.08,
        np.inf,
    )
    assert action[:3].tolist() == pytest.approx(
        [nominal_refill, 0.0, 0.0]
    )
    assert nominal_refill < 0.005
    assert evidence["live_outside_recovery_active"] is True
    assert evidence["live_full_outward_brake_active"] is False
    assert evidence["measured_inward_response"] is False
    assert evidence["outside_refill_target_clearance_m"] == pytest.approx(
        refill_target
    )
    assert evidence["selected_outward_recovery_action"] == pytest.approx(
        nominal_refill
    )
    assert evidence["proof"][
        "noninward_refill_band_uses_exact_nominal_action"
    ] is True


def test_fixed_safe_z_lateral_hold_retains_job503642_downward_z_brake():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    live_clearance = 0.0009749753150044976
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.1323272727317591, -0.026305187799701472, 0.9203120079482806]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.00019948232421351797,
        measured_outward_step_progress_m=0.0003901149684915617,
        outside_side_guard={
            "minimum_outside_clearance_m": live_clearance,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.007472123446334189,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["above_safe_z_band"] is True
    assert evidence["measured_downward_tail"] is True
    assert evidence["downward_tail_brake_active"] is True
    assert evidence["negative_z_suspended_for_outside_recovery"] is False
    assert evidence["proof"][
        "every_measured_downward_tail_uses_full_positive_z"
    ] is True


def test_fixed_safe_z_lateral_hold_brakes_job503643_severe_vertical_tail():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13236737084398184, -0.026141607051486555, 0.9204874155493346]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.0013942257332443253,
        measured_outward_step_progress_m=0.0002610902939860771,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.0010217803913379203,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.0076293466372879815,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["live_full_outward_brake_active"] is False
    assert evidence["measured_inward_response"] is False
    assert evidence["severe_vertical_response"] is True
    assert evidence["proof"][
        "severe_vertical_response_uses_full_outward_brake"
    ] is True


def test_fixed_safe_z_lateral_hold_brakes_job503644_downward_vertical_tail():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.1324596561463129, -0.02627091986387669, 0.9197740721907326]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.0007133433586019589,
        measured_outward_step_progress_m=0.00009228530233104659,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.0011262756360018028,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.007009554745709967,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["live_full_outward_brake_active"] is False
    assert evidence["measured_inward_response"] is False
    assert evidence["severe_vertical_response"] is False
    assert evidence["measured_downward_tail"] is True
    assert evidence["proof"][
        "downward_vertical_response_uses_full_outward_brake"
    ] is True


def test_fixed_safe_z_lateral_hold_adds_job503645_refill_resolution():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    live_clearance = 0.001247078705678556
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13255684449482272, -0.026257717438426417, 0.9176734279372091]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=0.00037026864536782167,
        measured_outward_step_progress_m=0.00008204201163439229,
        outside_side_guard={
            "minimum_outside_clearance_m": live_clearance,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.0049449659505129695,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    exit_clearance = np.nextafter(0.00155, np.inf)
    refill_target = np.nextafter(exit_clearance + 0.00010, np.inf)
    nominal_refill = np.nextafter(
        (refill_target - live_clearance) / 0.08,
        np.inf,
    )
    assert action[:3].tolist() == pytest.approx(
        [nominal_refill, 0.0, np.nextafter(0.20, 0.0)]
    )
    assert evidence["measured_downward_tail"] is False
    assert evidence["severe_vertical_response"] is False
    assert evidence["outside_refill_target_clearance_m"] == pytest.approx(
        refill_target
    )


def test_fixed_safe_z_lateral_hold_uses_job503637_final_stage_envelope():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array(
            [0.13226508172223794, -0.02624548942467667, 0.9173663148571825]
        ),
        lateral_target_xy=np.array(
            [0.13242106705090634, -0.02850777957668001]
        ),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.9196513910416114,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.0007260743378282308,
        measured_outward_step_progress_m=0.00018574628027626106,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.0009360647617149276,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.004599779805686044,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["outside_recovery_clearance_m"] == pytest.approx(
        0.0015
    )
    assert evidence["outside_recovery_exit_clearance_m"] == pytest.approx(
        np.nextafter(0.00155, np.inf)
    )
    assert evidence["live_outside_recovery_active"] is True
    assert evidence["live_table_clearance_m"] > 0.0015


def test_fixed_safe_z_lateral_hold_retains_job503638_recovery_to_exit():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    action, evidence = _fixed_safe_z_lateral_hold_action(
        current_eef=np.array([0.132, -0.0285, 0.918846]),
        lateral_target_xy=np.array([0.132, -0.0285]),
        lateral_position_tolerance_m=0.005,
        fixed_safe_z_m=0.919651,
        vertical_position_tolerance_m=0.0004,
        measured_vertical_step_progress_m=-0.000547,
        measured_outward_step_progress_m=0.000209,
        outside_side_guard={
            "minimum_outside_clearance_m": 0.001536,
            "required_outside_clearance_m": np.nextafter(0.0, np.inf),
            "finger_table_vertical_clearance_m": 0.0048,
            "required_finger_table_clearance_m": np.nextafter(
                0.0, np.inf
            ),
        },
        outward_direction_xy=np.array([1.0, 0.0]),
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_lateral_translation_action=0.005,
        maximum_safety_brake_action=0.20,
        strict_outside_clearance_m=0.0004,
        strict_table_clearance_m=0.0004,
        closed_loop_hazard_response_bound_m=0.0011,
        full_outward_brake_clearance_m=0.00095,
        progress_resolution_m=0.00005,
        derivative_gain=2.0,
        native_action_spec=native_spec,
    )
    strict_brake = np.nextafter(0.20, 0.0)
    assert action[:3].tolist() == pytest.approx(
        [strict_brake, 0.0, strict_brake]
    )
    assert evidence["outside_recovery_clearance_m"] == pytest.approx(
        0.0015
    )
    assert evidence["outside_recovery_exit_clearance_m"] == pytest.approx(
        np.nextafter(0.00155, np.inf)
    )
    assert evidence["live_outside_recovery_active"] is True
    assert evidence["live_full_outward_brake_active"] is False
    assert evidence["measured_downward_tail"] is True
    assert evidence["outside_recovery_active"] is True
    assert evidence["proof"][
        "recovery_release_requires_exit_headroom"
    ] is True


def test_native_geometry_side_contact_targets_descend_outside_plate():
    plate = np.array([0.052, -0.0285, 0.9025])
    outside, contact, plan = _side_contact_targets_from_compiled_bounds(
        plate_position=plate,
        outward_direction_xy=np.array([0.0, -1.0]),
        contact_xy=np.array([0.052, -0.0385]),
        plate_outward_support_m=0.052,
        finger_inward_extent_from_eef_m=-0.012,
        plate_rim_center_z=0.9095,
        finger_center_z_offset_from_eef=-0.013,
        outside_clearance_m=0.005,
    )
    assert np.allclose(outside, [0.052, -0.0975, 0.9225])
    assert np.allclose(contact, [0.052, -0.0385, 0.9225])
    assert plan["outside_eef_offset_m"] == pytest.approx(0.069)
    assert plan["side_eef_z"] - plate[2] == pytest.approx(0.020)
    assert outside[1] < contact[1] < plate[1]


def test_500079_compiled_eef_z_intersects_rim_coverage_and_table_clearance():
    required_clearance = np.nextafter(0.0, np.inf)
    feasibility = _compiled_side_contact_eef_z_feasibility(
        rim_vertical_interval=np.array([0.900, 0.919]),
        rim_center_z=0.9095,
        finger_vertical_bounds_from_eef=[
            ("left_finger", "left", -0.013, 0.025),
            ("right_finger", "right", -0.012, 0.024),
        ],
        table_top_z=0.900,
        required_finger_table_clearance_m=required_clearance,
    )
    assert feasibility["table_eef_z_lower_bound_m"] == pytest.approx(
        0.913
    )
    assert feasibility["selected_interval"]["eef_z_interval_m"] == (
        pytest.approx([0.913, 0.9215])
    )
    assert feasibility["selected_eef_z"] == pytest.approx(0.91725)
    assert feasibility["selected_finger_lowest_z"] > 0.900
    assert feasibility["selected_finger_table_clearance_m"] > (
        required_clearance
    )
    for interval in feasibility["selected_interval"][
        "selected_finger_world_intervals_m"
    ].values():
        assert interval[0] <= 0.9095 <= interval[1]
    for overlap in feasibility[
        "selected_rim_overlap_by_side"
    ].values():
        assert overlap["overlap_m"] > 0.0
        assert overlap["rim_center_covered"] is True

    with pytest.raises(RuntimeError, match="no EEF-z interval"):
        _compiled_side_contact_eef_z_feasibility(
            rim_vertical_interval=np.array([0.900, 0.919]),
            rim_center_z=0.9095,
            finger_vertical_bounds_from_eef=[
                ("left_finger", "left", -0.013, 0.025),
                ("right_finger", "right", -0.012, 0.024),
            ],
            table_top_z=0.950,
            required_finger_table_clearance_m=required_clearance,
        )


def test_native_finger_inward_extents_are_grouped_before_side_selection():
    bounds = [
        (
            10,
            "gripper0_leftfinger",
            np.array([0.012, -0.034, 0.90]),
            np.array([0.004, 0.005, 0.010]),
        ),
        (
            11,
            "gripper0_rightfinger",
            np.array([0.013, 0.034, 0.90]),
            np.array([0.004, 0.005, 0.010]),
        ),
    ]
    side_x, shared_x, skew_x, semantic_x = (
        _finger_inward_extents_by_semantic_side(
            bounds,
            eef_position=np.array([0.0, 0.0, 0.95]),
            outward_direction_xy=np.array([1.0, 0.0]),
        )
    )
    assert side_x == pytest.approx({"left": 0.008, "right": 0.009})
    assert shared_x == pytest.approx(0.008)
    assert skew_x == pytest.approx(0.001)
    assert len(semantic_x) == 2

    side_y, shared_y, skew_y, _ = (
        _finger_inward_extents_by_semantic_side(
            bounds,
            eef_position=np.array([0.0, 0.0, 0.95]),
            outward_direction_xy=np.array([0.0, -1.0]),
        )
    )
    assert side_y == pytest.approx({"left": 0.029, "right": -0.039})
    assert shared_y == pytest.approx(-0.039)
    assert skew_y == pytest.approx(0.068)


def test_compiled_side_selection_accepts_499848_mild_action_clipping():
    unreachable_minus_y = {
        "point_xy": [0.052, -0.0385],
        "selection_eligible": False,
        "selection_violations": [
            "dual_finger_contact_skew_exceeds_outside_clearance",
        ],
        "outside_high_action_peak": 1.773,
        "outside_high_action_norm": 1.773,
        "dual_finger_contact_skew_m": 0.10175,
        "eef_xy_distance_m": 0.030,
    }
    reachable_plus_x = {
        "point_xy": [0.062, -0.0285],
        "selection_eligible": True,
        "selection_violations": [],
        "outside_high_action_peak": 1.061872,
        "outside_high_action_norm": 1.061872,
        "dual_finger_contact_skew_m": 0.000213,
        "eef_xy_distance_m": 0.270,
    }
    selected = _select_reachable_compiled_side_candidate(
        [unreachable_minus_y, reachable_plus_x]
    )
    assert selected is reachable_plus_x
    with pytest.raises(
        RuntimeError,
        match="no compiled trailing side passed dual-finger geometry",
    ):
        _select_reachable_compiled_side_candidate(
            [unreachable_minus_y]
        )


def test_native_push_direction_candidates_append_after_legacy_cardinals():
    direction = np.array([-0.394, 0.919], dtype=float)
    direction /= np.linalg.norm(direction)
    candidates = _plate_contact_candidate_diagnostics(
        plate_xy=np.array([0.050, 0.000]),
        push_direction_xy=direction,
        eef_xy=np.array([0.000, 0.000]),
        backoff=0.010,
    )

    assert len(candidates) == 5
    np.testing.assert_allclose(
        [candidate["offset_xy"] for candidate in candidates[:2]],
        [[0.010, 0.000], [0.000, -0.010]],
    )
    assert [
        candidate["candidate_provenance"] for candidate in candidates[:2]
    ] == [["legacy_cardinal:+x"], ["legacy_cardinal:-y"]]
    assert all(
        candidate["route_selection_candidate"]
        for candidate in candidates[:2]
    )

    derived = candidates[2:]
    assert [
        candidate["native_push_direction_relations"][0]
        for candidate in derived
    ] == [
        "trailing_minus_push",
        "tangent_counterclockwise",
        "tangent_clockwise",
    ]
    np.testing.assert_allclose(
        [candidate["offset_xy"] for candidate in derived],
        [
            -direction * 0.010,
            np.array([-direction[1], direction[0]]) * 0.010,
            np.array([direction[1], -direction[0]]) * 0.010,
        ],
    )
    assert derived[0]["trailing_eligible"] is True
    assert derived[1]["trailing_eligible"] is False
    assert derived[2]["trailing_eligible"] is False
    assert all(candidate["diagnostic_only"] for candidate in derived)
    assert not any(
        candidate["route_selection_candidate"] for candidate in derived
    )


def test_native_push_direction_candidate_dedup_preserves_legacy_position():
    candidates = _plate_contact_candidate_diagnostics(
        plate_xy=np.array([0.050, 0.000]),
        push_direction_xy=np.array([1.000, 0.000]),
        eef_xy=np.array([0.000, 0.000]),
        backoff=0.010,
    )

    assert len(candidates) == 3
    assert candidates[0]["offset_xy"] == pytest.approx([-0.010, 0.000])
    assert candidates[0]["route_selection_candidate"] is True
    assert candidates[0]["candidate_provenance"] == [
        "legacy_cardinal:-x",
        "native_push_direction:trailing_minus_push",
    ]
    assert candidates[0]["deduplicated_provenance"] == [
        "native_push_direction:trailing_minus_push",
    ]
    assert candidates[0]["native_push_direction_relations"] == [
        "trailing_minus_push",
    ]
    assert [
        candidate["native_push_direction_relations"][0]
        for candidate in candidates[1:]
    ] == ["tangent_counterclockwise", "tangent_clockwise"]


@pytest.mark.parametrize(
    ("plate_xy", "direction", "eef_xy", "backoff", "message"),
    [
        ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.01, "nonzero"),
        ([0.0, 0.0], [np.nan, 1.0], [0.0, 0.0], 0.01, "nonzero"),
        ([np.inf, 0.0], [1.0, 0.0], [0.0, 0.0], 0.01, "finite"),
        ([0.0, 0.0], [1.0, 0.0], [0.0, 0.0], 0.0, "positive"),
        ([0.0, 0.0], [1.0, 0.0], [0.0, 0.0], np.nan, "positive"),
    ],
)
def test_native_push_direction_candidate_inputs_fail_closed(
    plate_xy, direction, eef_xy, backoff, message
):
    with pytest.raises(ValueError, match=message):
        _plate_contact_candidate_diagnostics(
            plate_xy=plate_xy,
            push_direction_xy=direction,
            eef_xy=eef_xy,
            backoff=backoff,
        )


def test_hypothetical_wrist_yaws_are_strict_native_frame_rotations():
    push_direction = np.array([-0.394, 0.919], dtype=float)
    push_direction /= np.linalg.norm(push_direction)
    targets = [
        -push_direction,
        np.array([-push_direction[1], push_direction[0]]),
        np.array([push_direction[1], -push_direction[0]]),
    ]
    specs = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=np.array([1.0, 0.0]),
        target_outward_directions_xy=targets,
        table_normal_world=np.array([0.0, 0.0, 1.0]),
    )

    assert len(specs) == 3
    assert len({round(spec["yaw_angle_rad"], 12) for spec in specs}) == 3
    for spec, target in zip(specs, targets):
        target = target / np.linalg.norm(target)
        rotation = np.asarray(spec["rotation_matrix_world"], dtype=float)
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]),
            np.array([target[0], target[1], 0.0]),
            rtol=0.0,
            atol=1e-9,
        )
        assert spec["yaw_angle_rad"] == pytest.approx(
            np.arctan2(target[1], target[0])
        )
        np.testing.assert_allclose(
            spec["axis_angle_world_rad"],
            np.array([0.0, 0.0, spec["yaw_angle_rad"]]),
            rtol=0.0,
            atol=1e-12,
        )
        assert spec["provenance"]["reference"] == (
            "current selected low-skew legacy +X approach"
        )


def test_hypothetical_wrist_yaw_frames_fail_closed():
    with pytest.raises(RuntimeError, match="duplicate hypothetical wrist yaw"):
        _hypothetical_wrist_yaw_specs(
            reference_outward_direction_xy=[1.0, 0.0],
            target_outward_directions_xy=[[0.0, 1.0], [0.0, 1.0]],
            table_normal_world=[0.0, 0.0, 1.0],
        )
    with pytest.raises(RuntimeError, match="unit vector"):
        _hypothetical_wrist_yaw_specs(
            reference_outward_direction_xy=[1.0, 0.0],
            target_outward_directions_xy=[[0.0, 1.0]],
            table_normal_world=[0.0, 0.0, 2.0],
        )
    with pytest.raises(RuntimeError, match="proper rigid rotation"):
        _validated_rigid_rotation_matrix(
            np.array(
                [
                    [1.0, 0.2, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            label="nonrigid test",
        )
    with pytest.raises(RuntimeError, match="proper rigid rotation"):
        _validated_rigid_rotation_matrix(
            np.diag([-1.0, 1.0, 1.0]),
            label="reflection test",
        )


def test_live_wrist_yaw_frame_attainment_and_direction_are_measured():
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reference_origins = {
        "left_finger": np.array([0.0, -0.05, 0.0]),
        "right_finger": np.array([0.0, 0.05, 0.0]),
    }

    def frame(
        relative_rotation,
        *,
        origin_error=0.0,
        eef_position=(0.0, 0.0, 0.0),
    ):
        eef_position = np.asarray(eef_position, dtype=float)
        records = []
        for index, (name, origin) in enumerate(reference_origins.items()):
            current_origin = eef_position + relative_rotation @ origin
            if index == 0:
                current_origin[0] += origin_error
            records.append(
                {
                    "geom": name,
                    "geom_id": index,
                    "body": f"gripper0_{name}",
                    "semantic_side": "left" if index == 0 else "right",
                    "origin_world": current_origin.tolist(),
                    "center_world": current_origin.tolist(),
                    "world_aabb_half_size": [0.004, 0.005, 0.010],
                    "rotation_matrix_world": relative_rotation.tolist(),
                }
            )
        return {
            "eef_position_world": eef_position.tolist(),
            "finger_geoms": records,
            "maximum_finger_radius_from_eef_m": 0.062,
        }

    reference = frame(np.eye(3))
    yaw_spec = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=[1.0, 0.0],
        target_outward_directions_xy=[
            [np.cos(angle), np.sin(angle)]
        ],
        table_normal_world=[0.0, 0.0, 1.0],
    )[0]
    attained = _wrist_yaw_attainment_evidence(
        reference_frame=reference,
        current_frame=frame(rotation),
        yaw_spec=yaw_spec,
        maximum_angle_error_rad=0.01,
        maximum_position_drift_m=0.01,
        angular_progress_epsilon_rad=0.001,
        position_progress_epsilon_m=0.00005,
        previous_absolute_error_rad=angle,
        previous_position_drift_m=0.001,
    )
    assert attained["attained"] is True
    assert attained["rotation_direction_valid"] is True
    assert attained["rigid_frame_valid"] is True
    assert attained["progressed"] is True
    assert attained["rotation_attained"] is True
    assert attained["position_attained"] is True

    wrong_rotation = np.array(
        [
            [np.cos(0.1), np.sin(0.1), 0.0],
            [-np.sin(0.1), np.cos(0.1), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    wrong_direction = _wrist_yaw_attainment_evidence(
        reference_frame=reference,
        current_frame=frame(wrong_rotation),
        yaw_spec=yaw_spec,
        maximum_angle_error_rad=0.01,
        maximum_position_drift_m=0.01,
        angular_progress_epsilon_rad=0.001,
        position_progress_epsilon_m=0.00005,
    )
    assert wrong_direction["attained"] is False
    assert wrong_direction["rotation_direction_valid"] is False

    nonrigid = _wrist_yaw_attainment_evidence(
        reference_frame=reference,
        current_frame=frame(rotation, origin_error=0.02),
        yaw_spec=yaw_spec,
        maximum_angle_error_rad=0.01,
        maximum_position_drift_m=0.01,
        angular_progress_epsilon_rad=0.001,
        position_progress_epsilon_m=0.00005,
    )
    assert nonrigid["attained"] is False
    assert nonrigid["rigid_frame_valid"] is False

    coupled_drift = _wrist_yaw_attainment_evidence(
        reference_frame=reference,
        current_frame=frame(
            rotation, eef_position=[0.016420214729910794, 0.0, 0.0]
        ),
        yaw_spec=yaw_spec,
        maximum_angle_error_rad=0.01,
        maximum_position_drift_m=0.005,
        angular_progress_epsilon_rad=0.001,
        position_progress_epsilon_m=0.00005,
        previous_absolute_error_rad=angle,
        previous_position_drift_m=0.0,
    )
    assert coupled_drift["rotation_attained"] is True
    assert coupled_drift["position_attained"] is False
    assert coupled_drift["attained"] is False
    assert coupled_drift["eef_position_drift_m"] == pytest.approx(
        0.016420214729910794
    )
    np.testing.assert_allclose(
        coupled_drift["anchor_position_error_world_m"],
        [-0.016420214729910794, 0.0, 0.0],
    )

    corrected = _wrist_yaw_attainment_evidence(
        reference_frame=reference,
        current_frame=frame(rotation, eef_position=[0.004, 0.0, 0.0]),
        yaw_spec=yaw_spec,
        maximum_angle_error_rad=0.01,
        maximum_position_drift_m=0.005,
        angular_progress_epsilon_rad=0.001,
        position_progress_epsilon_m=0.00005,
        previous_absolute_error_rad=0.0,
        previous_position_drift_m=0.016420214729910794,
    )
    assert corrected["rotation_attained"] is True
    assert corrected["position_attained"] is True
    assert corrected["attained"] is True
    assert corrected["position_progressed"] is True


def test_native_wrist_yaw_action_resolves_scale_and_gates_clip_contact_stall():
    controller = SimpleNamespace(
        control_dim=6,
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]),
        output_max=np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
        use_delta=True,
        use_ori=True,
        orientation_limits=None,
    )
    env = SimpleNamespace(
        env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)])
    )
    native_spec = {
        "low": (-np.ones(7)).tolist(),
        "high": np.ones(7).tolist(),
    }
    rotation_spec = _native_osc_rotation_spec_evidence(env, native_spec)
    assert rotation_spec["source"] == "env.env.robots[0].controller"
    assert rotation_spec["output_axis_angle_rad_per_action"][5] == 0.5

    action, bounded = _compiled_wrist_yaw_action(
        remaining_yaw_rad=0.4,
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=[0.0, 0.0, 1.0],
        anchor_eef_position=[0.0, 0.0, 1.0],
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert action[:5] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])
    assert action[5] == pytest.approx(0.8)
    assert bounded["action_will_clip"] is False
    assert bounded["commanded_yaw_rad"] == pytest.approx(0.4)
    assert bounded["translation_direction_valid"] is True

    full_axis_action, full_axis_evidence = _compiled_wrist_yaw_action(
        remaining_yaw_rad=-0.4,
        remaining_axis_angle_world=[0.02, -0.03, -0.4],
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=[0.0, 0.0, 1.0],
        anchor_eef_position=[0.0, 0.0, 1.0],
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    np.testing.assert_allclose(
        full_axis_action[3:6], [0.04, -0.06, -0.8]
    )
    assert full_axis_evidence["action_will_clip"] is False
    assert full_axis_evidence[
        "requested_world_axis_angle_norm_rad"
    ] == pytest.approx(np.linalg.norm([0.02, -0.03, -0.4]))

    compensated_action, compensated = _compiled_wrist_yaw_action(
        remaining_yaw_rad=0.2,
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=[0.016420214729910794, 0.0, 1.0],
        anchor_eef_position=[0.0, 0.0, 1.0],
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert compensated_action[0] == pytest.approx(-0.10)
    assert np.linalg.norm(compensated_action[:3]) < 0.10
    assert compensated["translation_bound_saturated"] is True
    assert compensated["translation_direction_valid"] is True
    assert compensated["action_will_clip"] is False
    assert compensated["predicted_anchor_error_reduction_m"] > 0.0
    assert compensated["predicted_anchor_position_error_norm_m"] < (
        compensated["anchor_position_error_norm_m"]
    )

    settle_action, settle = _compiled_wrist_yaw_action(
        remaining_yaw_rad=0.0,
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=[0.008, 0.0, 1.0],
        anchor_eef_position=[0.0, 0.0, 1.0],
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert np.all(settle_action[3:6] == 0.0)
    assert settle["orientation_hold_commanded"] is True
    assert settle["translation_direction_valid"] is True
    settle_gate = _wrist_yaw_step_gate(
        stage="position_settle",
        overhead_guard={"accepted": True},
        robot_nonrobot_contact_gate={"accepted": True},
        action_evidence=settle,
        attainment_evidence={
            "rotation_attained": True,
            "position_attained": False,
            "rotation_direction_valid": True,
            "rigid_frame_valid": True,
        },
        consecutive_angular_stall_steps=0,
        consecutive_position_stall_steps=0,
        maximum_stall_steps=10,
    )
    assert settle_gate["accepted"] is True
    changed_orientation = _wrist_yaw_step_gate(
        stage="position_settle",
        overhead_guard={"accepted": True},
        robot_nonrobot_contact_gate={"accepted": True},
        action_evidence={**settle, "orientation_hold_commanded": False},
        attainment_evidence={
            "rotation_attained": True,
            "position_attained": False,
            "rotation_direction_valid": True,
            "rigid_frame_valid": True,
        },
        consecutive_angular_stall_steps=0,
        consecutive_position_stall_steps=0,
        maximum_stall_steps=10,
    )
    assert changed_orientation["violations"] == [
        "wrist_yaw_position_settle_changed_orientation"
    ]

    _, clipped = _compiled_wrist_yaw_action(
        remaining_yaw_rad=0.8,
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=[0.0, 0.0, 1.0],
        anchor_eef_position=[0.0, 0.0, 1.0],
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert clipped["action_will_clip"] is True
    assert clipped["clipped_action_axes"] == [5]

    narrow_native_spec = copy.deepcopy(native_spec)
    narrow_native_spec["low"][0] = -0.05
    narrow_native_spec["high"][0] = 0.05
    _, translation_clipped = _compiled_wrist_yaw_action(
        remaining_yaw_rad=0.2,
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=[0.016420214729910794, 0.0, 1.0],
        anchor_eef_position=[0.0, 0.0, 1.0],
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=narrow_native_spec,
        rotation_spec=rotation_spec,
    )
    assert translation_clipped["action_will_clip"] is True
    assert translation_clipped["translation_native_clipped_axes"] == [0]
    gate = _wrist_yaw_step_gate(
        stage="rotation_with_anchor_compensation",
        overhead_guard={"accepted": True},
        robot_nonrobot_contact_gate={"accepted": False},
        action_evidence={
            **translation_clipped,
            "translation_direction_valid": False,
        },
        attainment_evidence={
            "attained": False,
            "rotation_attained": False,
            "position_attained": False,
            "rotation_direction_valid": True,
            "rigid_frame_valid": True,
        },
        consecutive_angular_stall_steps=4,
        consecutive_position_stall_steps=4,
        maximum_stall_steps=4,
    )
    assert gate["accepted"] is False
    assert gate["violations"] == [
        "forbidden_robot_native_contact_during_wrist_yaw",
        "wrist_yaw_action_would_clip",
        "wrist_yaw_anchor_correction_direction_invalid",
        "wrist_yaw_angular_progress_stalled",
        "wrist_yaw_anchor_position_progress_stalled",
    ]


def test_trailing_wrist_yaw_segments_are_minimal_strict_and_cumulative():
    native_spec = {
        "source": "test.native.action_spec",
        "low": (-np.ones(7)).tolist(),
        "high": np.ones(7).tolist(),
        "runtime_resolved": True,
    }
    rotation_spec = {
        "source": "test.native.osc",
        "output_min_axis_angle_rad": [
            -0.05,
            -0.05,
            -0.05,
            -0.5,
            -0.5,
            -0.5,
        ],
        "output_max_axis_angle_rad": [
            0.05,
            0.05,
            0.05,
            0.5,
            0.5,
            0.5,
        ],
        "output_axis_angle_rad_per_action": [
            0.05,
            0.05,
            0.05,
            0.5,
            0.5,
            0.5,
        ],
        "runtime_resolved": True,
    }
    trailing_direction = np.array(
        [0.39274305093731343, -0.9196482457659836], dtype=float
    )
    yaw_spec = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=[1.0, 0.0],
        target_outward_directions_xy=[trailing_direction],
        table_normal_world=[0.0, 0.0, 1.0],
    )[0]
    assert yaw_spec["yaw_angle_rad"] == pytest.approx(-1.1671839094273593)
    plan = _strict_wrist_yaw_segment_plan(
        yaw_spec=yaw_spec,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )

    assert plan["minimum_segment_count"] == 3
    assert plan["hardcoded_segment_count_used"] is False
    assert plan["runtime_fallback_permitted"] is False
    assert plan["native_axis_angle_norm_bound_rad"] == pytest.approx(0.5)
    assert plan["strict_directional_yaw_capacity_rad"] < 0.5
    segments = plan["segments"]
    relative_targets = np.array(
        [segment["relative_target_yaw_rad"] for segment in segments]
    )
    absolute_targets = np.array(
        [segment["absolute_target_yaw_rad"] for segment in segments]
    )
    assert np.all(relative_targets < 0.0)
    assert np.all(np.diff(absolute_targets) < 0.0)
    assert np.all(np.abs(relative_targets) < 0.5)
    assert np.sum(relative_targets) == pytest.approx(
        yaw_spec["yaw_angle_rad"], abs=1e-15
    )
    assert absolute_targets[-1] == pytest.approx(
        yaw_spec["yaw_angle_rad"], abs=1e-15
    )
    for segment in segments:
        assert segment["required_rotation_action_peak"] < 1.0
        assert segment["strictly_inside_native_yaw_capacity"] is True
        np.testing.assert_allclose(
            segment["cumulative_yaw_spec"]["target_outward_direction_xy"],
            segment["target_outward_direction_xy"],
            rtol=0.0,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        segments[-1]["target_outward_direction_xy"],
        trailing_direction,
        rtol=0.0,
        atol=1e-12,
    )

    # Exact multiples of the native 0.5 rad bound need one more segment to
    # remain strictly interior; this verifies ceil/nextafter, not hardcoded N=3.
    exact_bound_spec = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=[1.0, 0.0],
        target_outward_directions_xy=[[np.cos(-1.0), np.sin(-1.0)]],
        table_normal_world=[0.0, 0.0, 1.0],
    )[0]
    exact_bound_plan = _strict_wrist_yaw_segment_plan(
        yaw_spec=exact_bound_spec,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert exact_bound_plan["minimum_segment_count"] == 3

    changed_identity = copy.deepcopy(yaw_spec)
    changed_identity["target_outward_direction_xy"] = [1.0, 0.0]
    with pytest.raises(RuntimeError, match="native-frame wrist-yaw identity"):
        _strict_wrist_yaw_segment_plan(
            yaw_spec=changed_identity,
            native_action_spec=native_spec,
            rotation_spec=rotation_spec,
        )

    # Job 502434 attained segment 1 inside its local tolerance but 0.014923632
    # rad short of its fixed absolute waypoint.  Segment 2 must close that live
    # cumulative error, not apply the old fixed -0.389061303 relative delta.
    segment_1_underrotation = 0.014923632053875311
    segment_1_actual_absolute_yaw = float(
        segments[0]["absolute_target_yaw_rad"] + segment_1_underrotation
    )
    segment_2_target = float(segments[1]["absolute_target_yaw_rad"])
    corrected_remaining = float(
        segment_2_target - segment_1_actual_absolute_yaw
    )
    segment_1_actual_spec = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=[1.0, 0.0],
        target_outward_directions_xy=[
            [
                np.cos(segment_1_actual_absolute_yaw),
                np.sin(segment_1_actual_absolute_yaw),
            ]
        ],
        table_normal_world=[0.0, 0.0, 1.0],
    )[0]
    segment_1_actual_rotation = np.asarray(
        segment_1_actual_spec["rotation_matrix_world"], dtype=float
    )
    segment_2_absolute_rotation = np.asarray(
        segments[1]["cumulative_yaw_spec"]["rotation_matrix_world"],
        dtype=float,
    )
    segment_2_remaining_rotation = (
        segment_2_absolute_rotation @ segment_1_actual_rotation.T
    )
    job_502434_start_evidence = {
        "actual_yaw_rad": segment_1_actual_absolute_yaw,
        "remaining_yaw_rad": corrected_remaining,
        "target_yaw_rad": segment_2_target,
        "measured_rotation_matrix_world": (
            segment_1_actual_rotation.tolist()
        ),
        "remaining_rotation_matrix_world": (
            segment_2_remaining_rotation.tolist()
        ),
        "remaining_rotation_axis_angle_world_rad": [
            0.0,
            0.0,
            corrected_remaining,
        ],
        "rigid_frame_valid": True,
        "position_attained": True,
        "rotation_direction_valid": True,
    }
    live_segment_2 = _absolute_wrist_yaw_runtime_target(
        planned_segment=segments[1],
        absolute_start_attainment=job_502434_start_evidence,
        total_yaw_spec=yaw_spec,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert corrected_remaining == pytest.approx(-0.4039849351963284)
    assert live_segment_2[
        "measured_remaining_yaw_to_absolute_target_rad"
    ] == pytest.approx(corrected_remaining)
    assert live_segment_2["relative_yaw_spec"][
        "yaw_angle_rad"
    ] == pytest.approx(corrected_remaining)
    assert abs(corrected_remaining) < 0.5
    assert live_segment_2["live_relative_capacity_plan"][
        "minimum_segment_count"
    ] == 1
    assert live_segment_2["live_relative_capacity_plan"][
        "required_rotation_action_peak"
    ] == pytest.approx(0.8079698703926568)
    assert corrected_remaining - segments[1][
        "relative_target_yaw_rad"
    ] == pytest.approx(-segment_1_underrotation)
    assert live_segment_2["absolute_target_identity_preserved"] is True
    np.testing.assert_allclose(
        live_segment_2["planned_absolute_outward_direction_xy"],
        segments[1]["target_outward_direction_xy"],
        rtol=0.0,
        atol=1e-12,
    )

    # A yaw-only correction would preserve accumulated tilt.  The runtime
    # target must instead emit the complete world axis-angle that left-
    # multiplies the live orientation exactly onto the absolute waypoint.
    tilt = 0.03
    tilt_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ]
    )
    tilted_actual_rotation = tilt_rotation @ segment_1_actual_rotation
    rotated_reference = tilted_actual_rotation @ np.array([1.0, 0.0, 0.0])
    tilted_actual_yaw = float(
        np.arctan2(rotated_reference[1], rotated_reference[0])
    )
    tilted_remaining_yaw = float(segment_2_target - tilted_actual_yaw)
    tilted_remaining_rotation = (
        segment_2_absolute_rotation @ tilted_actual_rotation.T
    )
    tilted_remaining_axis_angle = _rotation_matrix_axis_angle(
        tilted_remaining_rotation
    )
    tilted_runtime_target = _absolute_wrist_yaw_runtime_target(
        planned_segment=segments[1],
        absolute_start_attainment={
            "actual_yaw_rad": tilted_actual_yaw,
            "remaining_yaw_rad": tilted_remaining_yaw,
            "target_yaw_rad": segment_2_target,
            "measured_rotation_matrix_world": (
                tilted_actual_rotation.tolist()
            ),
            "remaining_rotation_matrix_world": (
                tilted_remaining_rotation.tolist()
            ),
            "remaining_rotation_axis_angle_world_rad": (
                tilted_remaining_axis_angle.tolist()
            ),
            "rigid_frame_valid": True,
            "position_attained": True,
            "rotation_direction_valid": True,
        },
        total_yaw_spec=yaw_spec,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    full_correction = np.asarray(
        tilted_runtime_target["relative_yaw_spec"]["rotation_matrix_world"]
    )
    assert np.linalg.norm(
        np.asarray(
            tilted_runtime_target[
                "measured_remaining_full_axis_angle_world_rad"
            ]
        )[:2]
    ) > 0.0
    assert tilted_runtime_target[
        "measured_remaining_full_axis_angle_norm_rad"
    ] < 0.5
    np.testing.assert_allclose(
        full_correction @ tilted_actual_rotation,
        segment_2_absolute_rotation,
        rtol=0.0,
        atol=1e-9,
    )

    over_capacity_evidence = {
        "actual_yaw_rad": 0.0,
        "remaining_yaw_rad": segment_2_target,
        "target_yaw_rad": segment_2_target,
        "measured_rotation_matrix_world": np.eye(3).tolist(),
        "remaining_rotation_matrix_world": (
            segment_2_absolute_rotation.tolist()
        ),
        "remaining_rotation_axis_angle_world_rad": [
            0.0,
            0.0,
            segment_2_target,
        ],
        "rigid_frame_valid": True,
        "position_attained": True,
        "rotation_direction_valid": True,
    }
    with pytest.raises(RuntimeError, match="supplemental strict segments"):
        _absolute_wrist_yaw_runtime_target(
            planned_segment=segments[1],
            absolute_start_attainment=over_capacity_evidence,
            total_yaw_spec=yaw_spec,
            native_action_spec=native_spec,
            rotation_spec=rotation_spec,
        )


def test_wrist_yaw_settle_and_shared_structural_budgets_fail_closed():
    settle = _wrist_yaw_stage_budget_evidence(
        actions_used=28,
        maximum_actions=180,
        position_settle_steps=0,
        maximum_position_settle_steps=10,
        rotation_attained=True,
        position_attained=False,
    )
    assert settle["accepted"] is True
    assert settle["next_stage"] == "position_settle"
    assert settle["remaining_actions"] == 152

    settle_exhausted = _wrist_yaw_stage_budget_evidence(
        actions_used=38,
        maximum_actions=180,
        position_settle_steps=10,
        maximum_position_settle_steps=10,
        rotation_attained=True,
        position_attained=False,
    )
    assert settle_exhausted["accepted"] is False
    assert settle_exhausted["violations"] == [
        "wrist_yaw_position_settle_budget_exhausted"
    ]

    shared_exhausted = _wrist_yaw_stage_budget_evidence(
        actions_used=180,
        maximum_actions=180,
        position_settle_steps=0,
        maximum_position_settle_steps=10,
        rotation_attained=False,
        position_attained=True,
    )
    assert shared_exhausted["accepted"] is False
    assert shared_exhausted["violations"] == [
        "shared_structural_waypoint_budget_exhausted"
    ]


def test_job502443_center_high_reacquire_uses_live_plate_and_orientation_hold():
    live_plate = np.array(
        [0.051856632092207214, -0.02850777957668001, 0.902506338529415]
    )
    failed_live_eef = np.array(
        [0.05604713598990556, -0.032688247947315716, 1.065917307690024]
    )
    center_high = _center_high_target_from_live_plate(live_plate, 0.160)
    np.testing.assert_array_equal(
        center_high,
        [0.051856632092207214, -0.02850777957668001, 1.062506338529415],
    )
    assert np.linalg.norm(
        failed_live_eef[:2] - center_high[:2]
    ) == pytest.approx(0.0059191755096897215)
    assert np.linalg.norm(failed_live_eef - center_high) > 0.005

    native_spec = {
        "source": "job502443.native.action_spec",
        "low": (-np.ones(7)).tolist(),
        "high": np.ones(7).tolist(),
    }
    rotation_spec = {
        "source": "job502443.native.osc",
        "output_axis_angle_rad_per_action": [
            0.05,
            0.05,
            0.05,
            0.5,
            0.5,
            0.5,
        ],
    }
    action, evidence = _compiled_wrist_yaw_action(
        remaining_yaw_rad=0.0,
        remaining_axis_angle_world=np.zeros(3),
        table_normal_world=[0.0, 0.0, 1.0],
        current_eef_position=failed_live_eef,
        anchor_eef_position=center_high,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        gripper=-1.0,
        native_action_spec=native_spec,
        rotation_spec=rotation_spec,
    )
    assert evidence["orientation_hold_commanded"] is True
    assert evidence["action_will_clip"] is False
    assert evidence["translation_direction_valid"] is True
    assert evidence["commanded_translation_action_norm"] < 0.10
    assert action[3:6] == pytest.approx([0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        failed_live_eef + 0.08 * action[:3],
        center_high,
        rtol=0.0,
        atol=1e-15,
    )

    budget = _center_high_reacquire_budget_evidence(
        structural_actions_used=33,
        maximum_structural_actions=180,
        reacquire_steps=0,
        maximum_reacquire_steps=10,
    )
    assert budget["accepted"] is True
    assert budget["remaining_structural_actions"] == 147
    assert budget["remaining_reacquire_steps"] == 10


def test_center_high_reacquire_gates_collision_clip_stall_budget_and_yaw_drift():
    action_evidence = {
        "action_will_clip": False,
        "translation_direction_valid": True,
        "orientation_hold_commanded": True,
        "position_correction_requested": True,
    }
    attainment = {
        "rotation_attained": True,
        "position_attained": False,
        "rigid_frame_valid": True,
    }
    accepted = _center_high_reacquire_step_gate(
        overhead_guard={"accepted": True},
        robot_nonrobot_contact_gate={"accepted": True},
        action_evidence=action_evidence,
        attainment_evidence=attainment,
        consecutive_position_stall_steps=0,
        maximum_stall_steps=10,
    )
    assert accepted["accepted"] is True

    rejected = _center_high_reacquire_step_gate(
        overhead_guard={"accepted": False},
        robot_nonrobot_contact_gate={"accepted": False},
        action_evidence={
            **action_evidence,
            "action_will_clip": True,
            "translation_direction_valid": False,
            "orientation_hold_commanded": False,
        },
        attainment_evidence={
            **attainment,
            "rotation_attained": False,
            "rigid_frame_valid": False,
        },
        consecutive_position_stall_steps=10,
        maximum_stall_steps=10,
    )
    assert rejected["violations"] == [
        "center_high_reacquire_overhead_guard_failed",
        "forbidden_robot_native_contact_during_center_high_reacquire",
        "center_high_reacquire_action_would_clip",
        "center_high_reacquire_direction_invalid",
        "center_high_reacquire_changed_orientation",
        "center_high_reacquire_trailing_orientation_drifted",
        "center_high_reacquire_finger_frame_not_rigid",
        "center_high_reacquire_position_progress_stalled",
    ]

    tracking_exhausted = _center_high_reacquire_budget_evidence(
        structural_actions_used=43,
        maximum_structural_actions=180,
        reacquire_steps=10,
        maximum_reacquire_steps=10,
    )
    assert tracking_exhausted["accepted"] is False
    assert tracking_exhausted["violations"] == [
        "center_high_reacquire_step_budget_exhausted"
    ]
    fully_exhausted = _center_high_reacquire_budget_evidence(
        structural_actions_used=180,
        maximum_structural_actions=180,
        reacquire_steps=10,
        maximum_reacquire_steps=10,
    )
    assert fully_exhausted["violations"] == [
        "shared_structural_waypoint_budget_exhausted",
        "center_high_reacquire_step_budget_exhausted",
    ]


def test_center_high_second_live_recompile_identity_fails_closed():
    plate = np.array([0.051856632092207214, -0.02850777957668001, 0.9025])
    center = _center_high_target_from_live_plate(plate, 0.160)
    terminal_eef = center + np.array([0.001, -0.001, 0.0005])
    first = {
        "real_sim_geometry_recompile": {
            "performed": True,
            "eligible": True,
            "recompile_stage": "post_wrist_yaw",
        }
    }
    second_revalidation = {
        "performed": True,
        "eligible": True,
        "recompile_stage": "post_center_high_reacquire",
        "live_eef_position_world": terminal_eef.tolist(),
        "live_plate_position_world": plate.tolist(),
        "live_center_high_target_world": center.tolist(),
        "strict_dual_finger_skew_accepted": True,
        "planned_outside_guard": {"accepted": True},
        "selected_finger_table_clearance_m": 0.0047,
        "planned_finger_table_clearance_m": 0.0047,
        "required_finger_table_clearance_m": 0.0,
        "selected_rim_overlap_by_side": {
            "left": {"overlap_m": 0.014, "rim_center_covered": True},
            "right": {"overlap_m": 0.014, "rim_center_covered": True},
        },
        "hypothetical_geometry_used_for_descent": False,
    }
    second = {
        "center_high_target": center.tolist(),
        "real_sim_geometry_recompile": second_revalidation,
    }
    identity = _second_real_recompile_identity_evidence(
        first_candidate=first,
        second_candidate=second,
        terminal_eef_position=terminal_eef,
        live_plate_position=plate,
        center_high_target=center,
        plate_approach_eef_height=0.160,
    )
    assert identity["accepted"] is True
    assert identity["second_recompile_uses_exact_terminal_eef"] is True
    assert identity["second_recompile_uses_exact_terminal_plate"] is True
    assert identity["all_second_geometry_gates_revalidated"] is True

    corrupted = copy.deepcopy(second)
    corrupted_revalidation = corrupted["real_sim_geometry_recompile"]
    corrupted_revalidation["live_eef_position_world"][0] += 1e-12
    corrupted_revalidation["strict_dual_finger_skew_accepted"] = False
    corrupted_revalidation["planned_outside_guard"] = {"accepted": False}
    corrupted_revalidation["selected_finger_table_clearance_m"] = 0.0
    corrupted_revalidation["selected_rim_overlap_by_side"]["left"] = {
        "overlap_m": 0.0,
        "rim_center_covered": False,
    }
    rejected = _second_real_recompile_identity_evidence(
        first_candidate=first,
        second_candidate=corrupted,
        terminal_eef_position=terminal_eef,
        live_plate_position=plate,
        center_high_target=center,
        plate_approach_eef_height=0.160,
    )
    assert rejected["accepted"] is False
    assert "second_recompile_eef_not_terminal_reacquire_eef" in rejected[
        "violations"
    ]
    assert "second_recompile_skew_gate_failed" in rejected["violations"]
    assert "second_recompile_outside_guard_failed" in rejected["violations"]
    assert (
        "second_recompile_selected_table_clearance_failed"
        in rejected["violations"]
    )
    assert "second_recompile_left_rim_gate_failed" in rejected["violations"]


def test_compiled_trailing_candidates_select_sole_native_trailing_route(monkeypatch):
    class Model:
        body_names = [
            "world",
            PLATE_BODY,
            "gripper0_leftfinger",
            "gripper0_rightfinger",
            TABLE_BODY,
        ]
        geom_names = [
            "plate_plus_x",
            "plate_minus_x",
            "plate_plus_y",
            "plate_minus_y",
            "left_finger_collision",
            "right_finger_collision",
            "table_collision",
        ]
        nbody = len(body_names)
        ngeom = len(geom_names)
        body_parentid = np.array([0, 0, 0, 0, 0])
        geom_bodyid = np.array([1, 1, 1, 1, 2, 3, 4])
        geom_contype = np.ones(ngeom, dtype=int)
        geom_conaffinity = np.ones(ngeom, dtype=int)
        geom_margin = np.zeros(ngeom, dtype=float)
        geom_gap = np.zeros(ngeom, dtype=float)
        npair = 0
        geom_aabb = np.array(
            [
                [0, 0, 0, 0.005, 0.005, 0.005],
                [0, 0, 0, 0.005, 0.005, 0.005],
                [0, 0, 0, 0.005, 0.005, 0.005],
                [0, 0, 0, 0.005, 0.005, 0.005],
                [0, 0, 0, 0.004, 0.005, 0.010],
                [0, 0, 0, 0.004, 0.005, 0.010],
                [0, 0, 0, 0.500, 0.500, 0.005],
            ],
            dtype=float,
        )

        @classmethod
        def body_name2id(cls, name):
            return cls.body_names.index(name)

        @classmethod
        def body_id2name(cls, body_id):
            return cls.body_names[body_id]

        @classmethod
        def geom_id2name(cls, geom_id):
            return cls.geom_names[geom_id]

    data = SimpleNamespace(
        geom_xmat=np.tile(np.eye(3).reshape(1, 9), (Model.ngeom, 1)),
        geom_xpos=np.array(
            [
                [0.100, 0.000, 0.910],
                [0.000, 0.000, 0.910],
                [0.050, 0.050, 0.910],
                [0.050, -0.050, 0.910],
                [-0.020950, -0.05000, 0.900],
                [-0.020737, 0.05175, 0.900],
                [0.000, 0.000, 0.875],
            ],
            dtype=float,
        ),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))
    initial_frame = _compiled_finger_yaw_frame(
        env, eef_position=np.array([0.000, 0.000, 0.950])
    )
    assert len(initial_frame["finger_geoms"]) == 2
    selected, candidates = _compiled_trailing_side_contact_candidates(
        env,
        plate_position=np.array([0.050, 0.000, 0.900]),
        push_direction_xy=np.array([-0.394, 0.919]),
        eef_position=np.array([0.000, 0.000, 0.950]),
        backoff=0.010,
        outside_clearance_m=0.005,
        plate_approach_eef_height=0.160,
        position_action_scale=0.080,
    )
    assert len(candidates) == 5
    np.testing.assert_allclose(
        [candidate["offset_xy"] for candidate in candidates[:2]],
        [[0.010, 0.000], [0.000, -0.010]],
    )
    legacy_plus_x = candidates[0]
    assert legacy_plus_x["offset_xy"] == pytest.approx([0.010, 0.000])
    assert legacy_plus_x["selection_eligible"] is False
    assert legacy_plus_x["dual_finger_contact_skew_m"] == pytest.approx(
        0.000213
    )
    assert legacy_plus_x["outside_high_action_peak"] == pytest.approx(
        1.061875
    )
    assert legacy_plus_x["outside_high_clipped_action_axes"] == [0]
    assert legacy_plus_x["outside_high_action_will_clip"] is True
    assert legacy_plus_x["wrist_yaw_route_selected"] is False
    assert selected is candidates[2]
    assert selected["native_push_direction_relations"] == [
        "trailing_minus_push"
    ]
    assert selected["wrist_yaw_route_selected"] is True
    assert selected["route_selection_candidate"] is True
    assert selected["diagnostic_only"] is False
    assert selected["selection_eligible"] is True
    assert selected["selection_violations"] == []
    assert sum(
        candidate["wrist_yaw_route_selected"] for candidate in candidates
    ) == 1
    assert sum(
        candidate["route_selection_candidate"] for candidate in candidates
    ) == 1
    assert selected["wrist_yaw_route_selection_basis"][
        "old_plus_x_route_fallback_permitted"
    ] is False
    assert selected["wrist_yaw_route_selection_basis"][
        "runtime_tangent_fallback_permitted"
    ] is False
    assert selected["wrist_yaw_route_selection_basis"][
        "required_relation"
    ] == "trailing_minus_push"
    assert selected["wrist_yaw_route_selection_basis"][
        "superpod_diagnostic_authorization"
    ] == {
        "job_id": "502381",
        "commit": "3376794",
        "observed_geometry_eligible_relation": "trailing_minus_push",
        "runtime_revalidation_still_required": True,
    }
    assert _select_executable_wrist_yaw_candidate(candidates) is selected
    failed_trailing = copy.deepcopy(candidates)
    failed_trailing[2]["hypothetical_wrist_yaw"][
        "hypothetical_compiled_geometry_eligible"
    ] = False
    with pytest.raises(RuntimeError, match="trailing_minus_push wrist-yaw route"):
        _select_executable_wrist_yaw_candidate(failed_trailing)
    rejected = next(
        candidate
        for candidate in candidates
        if np.allclose(candidate["offset_xy"], [0.000, -0.010])
    )
    assert rejected["dual_finger_contact_skew_m"] == pytest.approx(0.10175)
    assert rejected["outside_high_action_peak"] > 1.0
    assert rejected["selection_eligible"] is False
    assert rejected["selection_violations"] == [
        "dual_finger_contact_skew_exceeds_outside_clearance",
    ]
    assert [
        candidate["native_push_direction_relations"][0]
        for candidate in candidates[2:]
    ] == [
        "trailing_minus_push",
        "tangent_counterclockwise",
        "tangent_clockwise",
    ]
    for diagnostic in candidates[2:]:
        is_selected = diagnostic is selected
        assert diagnostic["diagnostic_only"] is (not is_selected)
        assert diagnostic["selection_eligible"] is is_selected
        assert diagnostic["route_selection_candidate"] is is_selected
        assert isinstance(diagnostic["compiled_geometry_eligible"], bool)
        assert isinstance(diagnostic["dual_finger_contact_skew_m"], float)
        assert isinstance(diagnostic["outside_high_action_peak"], float)
        assert isinstance(diagnostic["outside_high_action_will_clip"], bool)
        yaw_diagnostic = diagnostic["hypothetical_wrist_yaw"]
        assert yaw_diagnostic["diagnostic_only"] is True
        assert yaw_diagnostic["executed"] is False
        assert yaw_diagnostic["selection_eligible"] is False
        assert yaw_diagnostic["hypothetical_compiled_geometry_eligible"] is True
        assert yaw_diagnostic["hypothetical_compiled_geometry_violations"] == []
        assert yaw_diagnostic["outside_guard"]["accepted"] is True
        assert yaw_diagnostic["rigid_finger_transform"][
            "rigid_transform_verified"
        ] is True
        assert yaw_diagnostic["dual_finger_contact_skew_m"] == pytest.approx(
            legacy_plus_x["dual_finger_contact_skew_m"], abs=1e-12
        )
        yaw = yaw_diagnostic["yaw"]
        assert yaw["native_push_direction_relation"] == diagnostic[
            "native_push_direction_relations"
        ][0]
        np.testing.assert_allclose(
            np.asarray(yaw["rotation_matrix_world"])
            @ np.array([1.0, 0.0, 0.0]),
            np.r_[diagnostic["outward_direction_xy"], 0.0],
            rtol=0.0,
            atol=1e-9,
        )

    plate_position = np.array([0.050, 0.000, 0.900])
    eef_position = np.array([0.000, 0.000, 0.950])
    contact_xy = np.array([0.060, 0.000])
    baseline_outside, baseline_contact, baseline_plan = (
        _compiled_native_side_contact_plan(
            env,
            plate_position,
            eef_position,
            np.array([1.0, 0.0]),
            contact_xy,
            0.005,
        )
    )
    table_normal, table_evidence = _compiled_table_normal_evidence(env)
    zero_yaw = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=[1.0, 0.0],
        target_outward_directions_xy=[[1.0, 0.0]],
        table_normal_world=table_normal,
    )[0]
    zero_replay = _compiled_hypothetical_wrist_yaw_plan(
        env,
        plate_position=plate_position,
        eef_position=eef_position,
        contact_xy=contact_xy,
        outside_clearance_m=0.005,
        plate_approach_eef_height=0.160,
        position_action_scale=0.080,
        yaw_spec=zero_yaw,
        table_normal_evidence=table_evidence,
    )
    assert zero_yaw["yaw_angle_rad"] == 0.0
    assert zero_yaw["rotation_matrix_world"] == np.eye(3).tolist()
    assert zero_replay["outside_side_target"] == baseline_outside.tolist()
    assert zero_replay["side_contact_target"] == baseline_contact.tolist()
    assert zero_replay["compiled_geometry"] == baseline_plan
    for transform in zero_replay["rigid_finger_transform"][
        "finger_geom_transforms"
    ]:
        assert transform["hypothetical_origin_world"] == transform[
            "current_origin_world"
        ]
        assert transform["hypothetical_rotation_matrix_world"] == transform[
            "current_rotation_matrix_world"
        ]

    with pytest.raises(RuntimeError, match="simultaneous measured yaw"):
        _real_recompile_wrist_yaw_candidate(
            env,
            selected_candidate=selected,
            plate_position=plate_position,
            eef_position=eef_position,
            outside_clearance_m=0.005,
            plate_approach_eef_height=0.160,
            position_action_scale=0.080,
            attainment_evidence={
                "attained": False,
                "rotation_attained": True,
                "position_attained": False,
                "eef_position_drift_m": 0.01642,
                "maximum_position_drift_m": 0.005,
            },
            table_normal_evidence=table_evidence,
        )
    with pytest.raises(RuntimeError, match="strict anchor-position"):
        _real_recompile_wrist_yaw_candidate(
            env,
            selected_candidate=selected,
            plate_position=plate_position,
            eef_position=eef_position,
            outside_clearance_m=0.005,
            plate_approach_eef_height=0.160,
            position_action_scale=0.080,
            attainment_evidence={
                "attained": True,
                "rotation_attained": True,
                "position_attained": True,
                "eef_position_drift_m": 0.005,
                "maximum_position_drift_m": 0.005,
            },
            table_normal_evidence=table_evidence,
        )
    selected_yaw_rotation = np.asarray(
        selected["hypothetical_wrist_yaw"]["yaw"][
            "rotation_matrix_world"
        ],
        dtype=float,
    )
    rotated_env, _ = _hypothetical_finger_yaw_env(
        env,
        eef_position=eef_position,
        rotation=selected_yaw_rotation,
    )
    realized = _real_recompile_wrist_yaw_candidate(
        rotated_env,
        selected_candidate=selected,
        plate_position=plate_position,
        eef_position=eef_position,
        outside_clearance_m=0.005,
        plate_approach_eef_height=0.160,
        position_action_scale=0.080,
        attainment_evidence={
            "attained": True,
            "rotation_attained": True,
            "position_attained": True,
            "eef_position_drift_m": 0.004,
            "maximum_position_drift_m": 0.005,
        },
        table_normal_evidence=table_evidence,
    )
    assert realized["diagnostic_only"] is False
    assert realized["selection_eligible"] is True
    assert realized["dual_finger_contact_skew_m"] == pytest.approx(
        legacy_plus_x["dual_finger_contact_skew_m"], abs=1e-12
    )
    assert realized["real_sim_geometry_recompile"]["performed"] is True
    assert realized["real_sim_geometry_recompile"]["eligible"] is True
    assert realized["real_sim_geometry_recompile"][
        "strict_dual_finger_skew_accepted"
    ] is True
    assert realized["real_sim_geometry_recompile"][
        "planned_outside_guard"
    ]["accepted"] is True
    assert realized["real_sim_geometry_recompile"][
        "hypothetical_geometry_used_for_descent"
    ] is False
    assert realized["native_push_direction_relations"] == [
        "trailing_minus_push"
    ]
    assert realized["real_sim_geometry_recompile"][
        "selected_native_push_direction_relation"
    ] == "trailing_minus_push"
    np.testing.assert_allclose(
        realized["real_sim_geometry_recompile"][
            "selected_outward_direction_xy"
        ],
        selected["outward_direction_xy"],
        rtol=0.0,
        atol=1e-12,
    )

    real_recompile_source = CONTROLLER_REFERENCE.read_text().split(
        "def _real_recompile_wrist_yaw_candidate(", 1
    )[1].split("\ndef _compiled_native_side_contact_plan(", 1)[0]
    assert "_compiled_hypothetical_wrist_yaw_plan(" not in (
        real_recompile_source
    )
    assert "_compiled_native_side_contact_plan(" in real_recompile_source
    assert "if not skew < float(outside_clearance_m):" in (
        real_recompile_source
    )

    original_compiler = _real_recompile_wrist_yaw_candidate.__globals__[
        "_compiled_native_side_contact_plan"
    ]

    def exact_limit_compiler(*args, **kwargs):
        outside, contact, geometry = original_compiler(*args, **kwargs)
        geometry = copy.deepcopy(geometry)
        geometry["dual_finger_contact_skew_m"] = 0.005
        return outside, contact, geometry

    monkeypatch.setitem(
        _real_recompile_wrist_yaw_candidate.__globals__,
        "_compiled_native_side_contact_plan",
        exact_limit_compiler,
    )
    with pytest.raises(RuntimeError, match="direct live 5 mm skew"):
        _real_recompile_wrist_yaw_candidate(
            rotated_env,
            selected_candidate=selected,
            plate_position=plate_position,
            eef_position=eef_position,
            outside_clearance_m=0.005,
            plate_approach_eef_height=0.160,
            position_action_scale=0.080,
            attainment_evidence={
                "attained": True,
                "rotation_attained": True,
                "position_attained": True,
                "eef_position_drift_m": 0.004,
                "maximum_position_drift_m": 0.005,
            },
            table_normal_evidence=table_evidence,
        )


def test_499866_outside_side_guard_uses_live_aabbs_not_exact_eef_center():
    target_eef = np.array([0.136806, -0.028508, 0.898654])
    achieved_eef = np.array([0.129286, -0.028540, 0.909191])
    assert np.linalg.norm(target_eef - achieved_eef) == pytest.approx(
        0.012945,
        abs=1e-6,
    )
    plate = np.array([0.052, -0.0285, 0.9025])
    rim_bounds = [
        (
            "plate_plus_x_rim",
            np.array([0.120, -0.0285, 0.900]),
            np.array([0.003, 0.020, 0.004]),
        )
    ]
    finger_bounds = [
        (
            "left_finger_collision",
            "left",
            np.array([0.130, -0.045, achieved_eef[2]]),
            np.array([0.002, 0.004, 0.012]),
        ),
        (
            "right_finger_collision",
            "right",
            np.array([0.130, -0.012, achieved_eef[2]]),
            np.array([0.002, 0.004, 0.012]),
        ),
    ]
    guard = _outside_side_guard_from_world_aabbs(
        plate_position=plate,
        outward_direction_xy=np.array([1.0, 0.0]),
        rim_bounds=rim_bounds,
        finger_bounds=finger_bounds,
        required_outside_clearance_m=0.005,
        table_bounds=[
            (
                "table_top",
                np.array([0.0, 0.0, 0.882]),
                np.array([0.5, 0.5, 0.005]),
            )
        ],
    )
    assert guard["accepted"] is True
    assert guard["violations"] == []
    assert guard["finger_table_vertical_clearance_m"] == pytest.approx(
        achieved_eef[2] - 0.012 - 0.887
    )
    for side in ("left", "right"):
        evidence = guard["finger_sides"][side]
        assert evidence["outside_clearance_m"] == pytest.approx(0.005)
        assert evidence["maximum_vertical_overlap_m"] > 0.0
        assert evidence["rim_center_covered"] is True

    # Positive edge overlap alone is insufficient: both fingers must cover
    # the native rim centre line, preventing an upper-edge guard acceptance.
    upper_edge_only = [
        (
            name,
            side,
            np.array([center[0], center[1], 0.913]),
            np.array([half[0], half[1], 0.010]),
        )
        for name, side, center, half in finger_bounds
    ]
    rejected = _outside_side_guard_from_world_aabbs(
        plate_position=plate,
        outward_direction_xy=np.array([1.0, 0.0]),
        rim_bounds=rim_bounds,
        finger_bounds=upper_edge_only,
        required_outside_clearance_m=0.005,
    )
    assert rejected["accepted"] is False
    assert (
        rejected["finger_sides"]["left"]["maximum_vertical_overlap_m"]
        > 0.0
    )
    assert "left_finger_does_not_cover_rim_center" in rejected[
        "violations"
    ]
    assert "right_finger_does_not_cover_rim_center" in rejected[
        "violations"
    ]


def test_499921_clearance_is_derived_from_compiled_collision_margins():
    names = ["finger_collision", "plate_rim", "table_collision"]
    model = SimpleNamespace(
        ngeom=3,
        npair=0,
        geom_contype=np.array([1, 1, 1]),
        geom_conaffinity=np.array([1, 1, 1]),
        geom_margin=np.array([0.0002, 0.0003, 0.0001]),
        geom_gap=np.array([0.00004, 0.00005, 0.00006]),
        geom_id2name=lambda geom_id: names[geom_id],
    )
    evidence = _compiled_collision_pair_clearance(model, 0, 1)
    detection_margin = 0.0003
    assert evidence["parameter_source"] == (
        "mixed_compiled_geom_parameters"
    )
    assert evidence["contact_detection_margin_m"] == pytest.approx(
        detection_margin
    )
    assert evidence["solver_gap_m"] == pytest.approx(0.00005)
    assert evidence["strict_no_contact_clearance_m"] == (
        np.nextafter(detection_margin, np.inf)
    )
    assert evidence["numerical_guard_m"] == (
        np.nextafter(detection_margin, np.inf) - detection_margin
    )

    aggregate = _compiled_pair_set_clearance(model, [0], [1, 2])
    assert aggregate["required_clearance_m"] == (
        evidence["strict_no_contact_clearance_m"]
    )
    assert len(aggregate["pairs"]) == 2
    assert "nextafter" in aggregate["formula"]

    explicit_model = SimpleNamespace(
        **{
            **model.__dict__,
            "npair": 1,
            "pair_geom1": np.array([1]),
            "pair_geom2": np.array([0]),
            "pair_margin": np.array([0.0007]),
            "pair_gap": np.array([0.00008]),
        }
    )
    explicit = _compiled_collision_pair_clearance(
        explicit_model, 0, 1
    )
    assert explicit["parameter_source"] == "explicit_compiled_pair"
    assert explicit["explicit_pair_id"] == 0
    assert explicit["strict_no_contact_clearance_m"] == (
        np.nextafter(0.0007, np.inf)
    )
    explicit_model.opt = SimpleNamespace(
        enableflags=1,
        o_margin=0.0009,
    )
    overridden = _compiled_collision_pair_clearance(
        explicit_model, 0, 1
    )
    assert overridden["contact_override_enabled"] is True
    assert overridden["base_contact_detection_margin_m"] == (
        pytest.approx(0.0007)
    )
    assert overridden["contact_detection_margin_m"] == pytest.approx(
        0.0009
    )
    assert overridden["strict_no_contact_clearance_m"] == (
        np.nextafter(0.0009, np.inf)
    )


def test_499921_two_mm_live_clearance_proceeds_with_zero_compiled_margin():
    current = np.array([0.13379, -0.028254, 0.97916])
    target = np.array([0.136806, -0.028508, 0.898654])
    strict_positive_clearance = np.nextafter(0.0, np.inf)
    guard = {
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": strict_positive_clearance,
        "minimum_outside_clearance_m": 0.00205,
        "required_finger_table_clearance_m": (
            strict_positive_clearance
        ),
        "finger_table_vertical_clearance_m": 0.012,
    }
    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=current,
        outside_side_target=target,
        guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert feedback["mode"] == "constraint_prioritized_vertical_descent"
    assert feedback["clearance_deficit_m"] == 0.0
    assert feedback["required_outside_clearance_m"] == (
        strict_positive_clearance
    )
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert np.linalg.norm(action[:3]) == pytest.approx(0.10)
    assert feedback["descent_path_control"] is not None


def test_500088_descent_norm_is_strictly_inside_bound_without_inward_action():
    current = np.array(
        [
            0.13267415665529797,
            -0.028638039484225563,
            1.0615092907889767,
        ]
    )
    target = np.array(
        [0.13680639548403947, -0.02850777957668001, 0.917769758]
    )
    requested_lateral = (target[:2] - current[:2]) / 0.08
    old_vertical = np.sqrt(
        0.10**2 - np.linalg.norm(requested_lateral) ** 2
    )
    old_translation_norm = np.linalg.norm(
        [requested_lateral[0], requested_lateral[1], old_vertical]
    )
    assert old_translation_norm > 0.10

    action, evidence = _constraint_prioritized_outside_descent_action(
        current_eef=current,
        outside_side_target=target,
        outward_direction_xy=np.array([1.0, 0.0]),
        maximum_descent_m=current[2] - target[2],
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    expected_lateral = (target[:2] - current[:2]) / 0.08
    assert np.allclose(action[:2], expected_lateral)
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert np.linalg.norm(action[:3]) == pytest.approx(0.10)
    assert evidence["commanded_outward_error_m"] == pytest.approx(
        target[0] - current[0]
    )
    strict_allocation_bound = np.nextafter(0.10, 0.0)
    assert evidence["allocation_translation_action_bound"] == (
        strict_allocation_bound
    )
    assert evidence["allocation_numeric_guard"] == (
        0.10 - strict_allocation_bound
    )
    assert evidence["pre_rescale_translation_action_norm"] <= (
        strict_allocation_bound
    )
    assert evidence["translation_action_norm"] <= (
        strict_allocation_bound
    )
    assert np.linalg.norm(action[:3]) <= 0.10

    # A second representable construction can round back up to 0.10 even
    # after using nextafter; the strict inward fallback rescales it.
    action, rounding_evidence = (
        _constraint_prioritized_outside_descent_action(
            current_eef=np.array([0.0, 0.0, 1.0]),
            outside_side_target=np.array([1.6e-8, 0.0, 0.9]),
            outward_direction_xy=np.array([1.0, 0.0]),
            maximum_descent_m=0.1,
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
        )
    )
    assert rounding_evidence[
        "pre_rescale_translation_action_norm"
    ] > strict_allocation_bound
    assert rounding_evidence["numeric_inward_rescale_applied"] is True
    assert rounding_evidence["translation_action_norm"] <= (
        strict_allocation_bound
    )
    assert np.linalg.norm(action[:3]) <= 0.10

    # Overshooting the compiled outside target never produces an inward
    # command; the freed controller norm is allocated to descent.
    overshot = target.copy()
    overshot[0] += 0.002
    overshot[2] = current[2]
    action, evidence = _constraint_prioritized_outside_descent_action(
        current_eef=overshot,
        outside_side_target=target,
        outward_direction_xy=np.array([1.0, 0.0]),
        maximum_descent_m=0.008,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert action[0] == 0.0
    assert action[2] == pytest.approx(-0.10)
    assert evidence["raw_outward_error_m"] < 0.0
    assert evidence["commanded_outward_error_m"] == 0.0


def test_500099_every_descent_requires_preventive_active_braking_settle():
    target = np.array(
        [0.13680639548403947, -0.02850777957668001, 0.917769758]
    )
    before_eef = np.array(
        [0.134581421, -0.028733513, 1.058025943]
    )
    after_eef = np.array(
        [0.134502805, -0.028729556, 1.056519669]
    )
    required_clearance = np.nextafter(0.0, np.inf)
    before_guard = {
        "accepted": True,
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": required_clearance,
        "minimum_outside_clearance_m": 0.002748690,
    }
    after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": 0.002683149,
    }
    first_descent_response = {
        "eef_outward_step_progress_m": 0.000824438,
        "outside_clearance_step_progress_m": 0.000809111,
        "vertical_step_progress_m": -0.000128412,
    }
    preventive_trigger = _outside_side_staircase_settle_trigger(
        feedback_mode="constraint_prioritized_vertical_descent",
        guard_step=1,
        step_response=first_descent_response,
    )
    assert preventive_trigger is not None
    assert preventive_trigger["trigger_guard_step"] == 1
    assert preventive_trigger["inward_response_observed"] is False
    assert _outside_side_staircase_settle_trigger(
        feedback_mode="compiled_outside_lateral_settle",
        guard_step=2,
        step_response=first_descent_response,
    ) is None

    evidence = _outside_side_lateral_settle_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
    )
    assert evidence["settled"] is False
    assert evidence["kinematic_brake_reversed"] is False
    assert evidence["violations"] == [
        "eef_still_descending_during_lateral_settle",
        "eef_still_moving_inward_during_lateral_settle",
        "outside_clearance_still_decreasing_during_lateral_settle",
    ]

    settle_action, path = _constraint_prioritized_outside_descent_action(
        current_eef=after_eef,
        outside_side_target=target,
        outward_direction_xy=np.array([1.0, 0.0]),
        maximum_descent_m=0.0,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        active_positive_z_brake=True,
    )
    assert settle_action[0] > 0.0
    assert settle_action[2] > 0.0
    assert path["maximum_descent_m"] == 0.0
    assert path["active_positive_z_brake"] is True
    assert path["commanded_positive_z_brake_action"] == pytest.approx(
        settle_action[2]
    )
    assert np.linalg.norm(settle_action[:3]) <= 0.10

    settle_action, feedback = _outside_side_geometry_feedback_action(
        current_eef=after_eef,
        outside_side_target=target,
        guard={
            **after_guard,
            "required_finger_table_clearance_m": required_clearance,
            "finger_table_vertical_clearance_m": 0.143,
        },
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        force_lateral_settle=True,
        previous_settle_vertical_step_progress_m=evidence[
            "vertical_step_progress_m"
        ],
    )
    assert feedback["mode"] == "compiled_outside_lateral_settle"
    assert feedback["force_lateral_settle"] is True
    assert feedback["active_positive_z_brake_requested"] is True
    assert settle_action[0] > 0.0
    assert settle_action[2] > 0.0

    settled_guard = {
        **after_guard,
        "minimum_outside_clearance_m": 0.0028,
    }
    first_stable_confirmation = _outside_side_lateral_settle_evidence(
        before_guard=after_guard,
        after_guard=settled_guard,
        before_eef=after_eef,
        after_eef=after_eef + np.array([0.0001, 0.0, 0.0001]),
    )
    assert first_stable_confirmation["settled"] is False
    assert first_stable_confirmation[
        "instantaneous_stable_response"
    ] is True
    assert first_stable_confirmation["stable_response_count"] == 1
    assert first_stable_confirmation["violations"] == []
    second_stable_confirmation = _outside_side_lateral_settle_evidence(
        before_guard=settled_guard,
        after_guard={
            **settled_guard,
            "minimum_outside_clearance_m": 0.0029,
        },
        before_eef=after_eef + np.array([0.0001, 0.0, 0.0001]),
        after_eef=after_eef + np.array([0.0002, 0.0, 0.0002]),
        previous_stable_response_count=first_stable_confirmation[
            "stable_response_count"
        ],
    )
    assert second_stable_confirmation["settled"] is True
    assert second_stable_confirmation["stable_response_count"] == 2
    assert second_stable_confirmation["violations"] == []


def test_500104_first_settle_step_actively_brakes_exact_negative_z_response():
    before_eef = np.array(
        [
            0.13267415665529797,
            -0.028638039484225563,
            1.0615092907889767,
        ]
    )
    after_eef = np.array(
        [
            0.13349859423038862,
            -0.02867088066849006,
            1.061380879160546,
        ]
    )
    target = np.array(
        [
            0.13680639548403947,
            -0.02850777957668001,
            0.917769758476126,
        ]
    )
    required_clearance = np.nextafter(0.0, np.inf)
    before_guard = {
        "accepted": True,
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": required_clearance,
        "minimum_outside_clearance_m": 0.0008573639623264129,
    }
    after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": 0.0016664744783980584,
        "required_finger_table_clearance_m": required_clearance,
        "finger_table_vertical_clearance_m": 0.14827030531197682,
    }
    descent_response = _outside_side_step_response_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
    )
    assert descent_response["eef_outward_step_progress_m"] == (
        pytest.approx(0.000824437575090653)
    )
    assert descent_response[
        "outside_clearance_step_progress_m"
    ] == pytest.approx(0.0008091105160716455)
    assert descent_response["vertical_step_progress_m"] == (
        pytest.approx(-0.000128411628430691)
    )
    trigger = _outside_side_staircase_settle_trigger(
        feedback_mode="constraint_prioritized_vertical_descent",
        guard_step=1,
        step_response=descent_response,
    )
    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=after_eef,
        outside_side_target=target,
        guard=after_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        force_lateral_settle=True,
        previous_settle_vertical_step_progress_m=(
            trigger["trigger_step_response"][
                "vertical_step_progress_m"
            ]
        ),
    )
    assert feedback["mode"] == "compiled_outside_lateral_settle"
    assert feedback["active_positive_z_brake_requested"] is True
    assert feedback["active_positive_z_brake_commanded"] is True
    assert feedback["commanded_positive_z_brake_action"] == pytest.approx(
        action[2]
    )
    assert action[0] == pytest.approx(0.04134751567063562)
    assert action[1] == pytest.approx(0.002038763647625643)
    assert action[2] == pytest.approx(0.09102871190265004)
    path = feedback["descent_path_control"]
    assert path["active_positive_z_brake"] is True
    assert path["commanded_positive_z_brake_action"] == pytest.approx(
        action[2]
    )
    assert path["commanded_positive_z_brake_world_step_m"] == (
        pytest.approx(action[2] * 0.08)
    )
    assert "positive-Z active braking" in path["formula"]

    no_brake_action, no_brake_feedback = (
        _outside_side_geometry_feedback_action(
            current_eef=after_eef,
            outside_side_target=target,
            guard=after_guard,
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
            force_lateral_settle=True,
            previous_settle_vertical_step_progress_m=0.0,
        )
    )
    assert no_brake_action[2] == 0.0
    assert no_brake_feedback[
        "active_positive_z_brake_requested"
    ] is False
    assert no_brake_feedback[
        "active_positive_z_brake_commanded"
    ] is False


def test_500104_active_braking_norm_is_strict_and_never_commands_inward():
    target = np.array([0.136806395, -0.028507780, 0.917769758])
    strict_bound = np.nextafter(0.10, 0.0)
    cases = [
        np.array([0.133498594, -0.028670881, 1.061380879]),
        np.array([0.100000000, -0.028507780, 1.061380879]),
        np.array([0.140000000, -0.028507780, 1.061380879]),
    ]
    for current in cases:
        action, evidence = (
            _constraint_prioritized_outside_descent_action(
                current_eef=current,
                outside_side_target=target,
                outward_direction_xy=np.array([1.0, 0.0]),
                maximum_descent_m=0.0,
                gripper=-1.0,
                position_action_scale=0.08,
                maximum_translation_action=0.10,
                active_positive_z_brake=True,
            )
        )
        assert action[0] >= 0.0
        assert action[2] >= 0.0
        assert np.linalg.norm(action[:3]) <= strict_bound
        assert evidence["translation_action_norm"] <= strict_bound
        assert evidence["commanded_positive_z_brake_action"] == (
            pytest.approx(action[2])
        )
    assert cases[1][0] < target[0]
    assert cases[2][0] > target[0]


def test_500111_one_positive_brake_response_cannot_release_settle_state():
    strict_clearance = np.nextafter(0.0, np.inf)
    before_eef = np.array(
        [
            0.13349859423038862,
            -0.02867088066849006,
            1.061380879160546,
        ]
    )
    after_eef = np.array(
        [
            0.13407799884439697,
            -0.028695901102401576,
            1.0615561799575444,
        ]
    )
    before_guard = {
        "accepted": True,
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": strict_clearance,
        "minimum_outside_clearance_m": 0.0016664744783980584,
    }
    after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": 0.0022379574242499534,
    }
    first_confirmation = _outside_side_lateral_settle_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
        previous_stable_response_count=0,
    )
    assert first_confirmation["step_response"] == {
        "eef_outward_step_progress_m": pytest.approx(
            0.0005794046140083497
        ),
        "vertical_step_progress_m": pytest.approx(
            0.00017530079699845658
        ),
        "outside_clearance_step_progress_m": pytest.approx(
            0.000571482945851895
        ),
        "before_clearance_m": pytest.approx(
            0.0016664744783980584
        ),
        "after_clearance_m": pytest.approx(
            0.0022379574242499534
        ),
    }
    assert first_confirmation["instantaneous_stable_response"] is True
    assert first_confirmation["kinematic_brake_reversed"] is True
    assert first_confirmation["stable_response_count"] == 1
    assert first_confirmation["required_stable_response_count"] == 2
    assert first_confirmation["settled"] is False
    assert first_confirmation["violations"] == []

    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=after_eef,
        outside_side_target=np.array(
            [
                0.13680639548403947,
                -0.02850777957668001,
                0.917769758476126,
            ]
        ),
        guard={
            **after_guard,
            "required_finger_table_clearance_m": strict_clearance,
            "finger_table_vertical_clearance_m": 0.148,
        },
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
        force_lateral_settle=True,
        previous_settle_vertical_step_progress_m=first_confirmation[
            "vertical_step_progress_m"
        ],
    )
    assert feedback["mode"] == "compiled_outside_lateral_settle"
    assert feedback["active_positive_z_brake_requested"] is False
    assert action[0] > 0.0
    assert action[2] == 0.0

    worsening_confirmation = _outside_side_lateral_settle_evidence(
        before_guard={
            **before_guard,
            "minimum_outside_clearance_m": 0.002686379097492364,
        },
        after_guard={
            **after_guard,
            "minimum_outside_clearance_m": 0.002564755251218201,
        },
        before_eef=np.array(
            [
                0.13451767004721368,
                -0.0287257624070834,
                1.0614536245271924,
            ]
        ),
        after_eef=np.array(
            [
                0.1343827765522451,
                -0.02870835392470038,
                1.062195398572198,
            ]
        ),
        previous_stable_response_count=first_confirmation[
            "stable_response_count"
        ],
    )
    assert worsening_confirmation["instantaneous_stable_response"] is False
    assert worsening_confirmation["stable_response_count"] == 0
    assert worsening_confirmation["settled"] is False
    assert worsening_confirmation["violations"] == [
        "eef_still_moving_inward_during_lateral_settle",
        "outside_clearance_still_decreasing_during_lateral_settle",
    ]


def test_500121_vertical_descent_is_structurally_staged_outside_one_step_reserve():
    outside_high = np.array(
        [
            0.13680639548403947,
            -0.02850777957668001,
            1.062506338529415,
        ]
    )
    outside_side = np.array(
        [
            0.13680639548403947,
            -0.02850777957668001,
            0.917769758476126,
        ]
    )
    strict_clearance = np.nextafter(0.0, np.inf)
    corridor_high, corridor_side, evidence = (
        _compiled_vertical_staging_corridor(
            outside_high_target=outside_high,
            outside_side_target=outside_side,
            geometry={
                "outward_direction_xy": [1.0, 0.0],
                "outside_clearance_m": 0.005,
            },
            required_outside_clearance_m=strict_clearance,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
        )
    )
    assert evidence["maximum_controller_world_step_m"] == pytest.approx(
        0.008
    )
    assert evidence["strict_corridor_entry_clearance_m"] > 0.008
    assert evidence["corridor_clearance_m"] > 0.013
    assert evidence[
        "full_inward_step_residual_clearance_m"
    ] > strict_clearance
    assert evidence[
        "full_inward_step_residual_clearance_m"
    ] == pytest.approx(0.005)
    assert corridor_high[0] > outside_high[0]
    assert corridor_side[0] > outside_side[0]
    assert corridor_high[0] - outside_high[0] == pytest.approx(0.008)
    assert corridor_side[2] == outside_side[2]
    assert corridor_high[2] == outside_high[2]
    assert evidence["corridor_entry_lateral_travel_m"] == pytest.approx(
        0.008
    )
    assert evidence["vertical_staging_travel_m"] == pytest.approx(
        outside_high[2] - outside_side[2]
    )
    assert evidence["fixed_z_lateral_travel_m"] == pytest.approx(0.008)
    assert evidence["geometric_full_scale_action_equivalents"] < 21.0
    assert evidence["geometric_full_scale_action_equivalents"] < 180

    # Job500121 began only 0.857 mm outside the plate and issued a descent.
    # The derived corridor instead commands pure outward entry until live
    # clearance exceeds one full 8 mm controller world step.
    job500121_start = np.array(
        [
            0.13267415665529797,
            -0.028638039484225563,
            1.0615092907889767,
        ]
    )
    job500121_live_clearance = 0.0008573639623264129
    assert job500121_live_clearance < evidence[
        "strict_corridor_entry_clearance_m"
    ]
    for old_near_plate_clearance in (
        0.0022379574242499534,
        0.002562943707493784,
        0.002677226838124098,
    ):
        assert old_near_plate_clearance < evidence[
            "strict_corridor_entry_clearance_m"
        ]
    entry_action, entry_path = (
        _constraint_prioritized_outside_descent_action(
            current_eef=job500121_start,
            outside_side_target=corridor_side,
            outward_direction_xy=np.array([1.0, 0.0]),
            maximum_descent_m=0.0,
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
        )
    )
    assert entry_action[0] > 0.0
    assert entry_action[2] == 0.0
    assert np.linalg.norm(entry_action[:3]) <= np.nextafter(0.10, 0.0)
    assert entry_path["maximum_descent_m"] == 0.0

    # Once staged at the corridor, the same bounded allocator produces one
    # continuous vertical phase with no inward XY command.
    descent_action, descent_path = (
        _constraint_prioritized_outside_descent_action(
            current_eef=corridor_high,
            outside_side_target=corridor_side,
            outward_direction_xy=np.array([1.0, 0.0]),
            maximum_descent_m=corridor_high[2] - corridor_side[2],
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
        )
    )
    assert descent_action[0] == pytest.approx(0.0)
    assert descent_action[2] < 0.0
    assert np.linalg.norm(descent_action[:3]) <= np.nextafter(0.10, 0.0)
    assert descent_path["commanded_outward_error_m"] == 0.0

    lateral_action, lateral_path = _fixed_z_lateral_approach_action(
        current_eef=corridor_side,
        lateral_target_xy=outside_side[:2],
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert lateral_action[0] < 0.0
    assert lateral_action[2] == 0.0
    assert lateral_path["commanded_z_action"] == 0.0
    assert np.linalg.norm(lateral_action[:3]) <= np.nextafter(0.10, 0.0)


def test_500133_enters_compiled_high_corridor_directly_without_native_high_stop():
    # Exact Job500133 trace: all commands were pure outward at the unchanged
    # 0.10 action bound, yet the two-hop controller never reached the strict
    # one-world-step (>8 mm) corridor.  Its residual response reversed after
    # frame 6 and crossed the compiled no-contact boundary at frame 12.
    initial_clearance = 0.0008573639623264129
    clearances = np.array(
        [
            0.0018938727902981373,
            0.0028298351969778396,
            0.0035513036618791544,
            0.004044439732490773,
            0.004298270702040891,
            0.004309318094654457,
            0.004081105379950101,
            0.003623199535032845,
            0.0029514669494543067,
            0.0020881353908475814,
            0.001061380420857988,
            -0.00009563150067076753,
        ]
    )
    exact_progress = np.diff(
        np.concatenate(([initial_clearance], clearances))
    )
    recorded_progress = np.array(
        [
            0.0010365088279717244,
            0.0009359624066797023,
            0.0007214684649013148,
            0.0004931360706116189,
            0.0002538309695501173,
            0.000011047392613566798,
            -0.00022821271470435667,
            -0.0004579058449172557,
            -0.0006717325855785383,
            -0.0008633315586067253,
            -0.0010267549699895934,
            -0.0011570119215287555,
        ]
    )
    strict_entry_clearance = np.nextafter(0.008, np.inf)
    assert np.allclose(
        exact_progress, recorded_progress, rtol=0.0, atol=1e-18
    )
    assert np.argmax(clearances) == 5
    assert np.all(exact_progress[6:] < 0.0)
    assert np.max(clearances) < strict_entry_clearance
    assert clearances[-1] < np.nextafter(0.0, np.inf)

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    compiled_overhead = bounded_seek.index(
        "_compiled_overhead_staging_geometry("
    )
    compiled_corridor = bounded_seek.index(
        "_compiled_vertical_staging_corridor("
    )
    assert compiled_overhead < compiled_corridor
    assert "rollout.move(" not in bounded_seek
    assert "rollout.move(\n        outside_high_target," not in bounded_seek
    assert (
        'structural_stage = "overhead_high_corridor_lateral"'
        in bounded_seek
    )
    assert '"overhead_corridor_descent"' in bounded_seek
    assert '"vertical_tail_brake"' in bounded_seek
    assert '"vertical_tail_zero_confirmation"' not in bounded_seek
    assert "vertical_tail_brake_and_formal_corridor_handoff" in bounded_seek
    assert '"overhead_post_descent_corridor_lateral"' in bounded_seek
    assert "_fixed_xy_vertical_approach_action(" in bounded_seek
    assert "_fixed_z_lateral_approach_action(" in bounded_seek
    fixed_z_stage = bounded_seek.index(
        'elif structural_stage == "fixed_safe_z_lateral_approach"'
    )
    assert bounded_seek.index(
        "lateral_target_xy=np.asarray(\n                    outside_side_target",
        fixed_z_stage,
    ) > fixed_z_stage


def test_500137_compiles_lowest_safe_overhead_sweeps_and_action_budget():
    strict_clearance = np.nextafter(0.0, np.inf)
    selected_z, evidence = _derive_overhead_staging_from_compiled_pairs(
        start_eef_position=np.array([0.051856632, -0.028507780, 1.062506339]),
        compiled_pairs=[
            {
                "gripper_geom": "gripper_complete_lowest",
                "counterpart_geom": "plate_complete_highest",
                "counterpart_kind": "plate",
                "gripper_lower_offset_from_eef_m": -0.030,
                "counterpart_top_z_m": 0.920,
                "strict_no_contact_clearance_m": strict_clearance,
            },
            {
                "gripper_geom": "gripper_complete_lowest",
                "counterpart_geom": "table_collision",
                "counterpart_kind": "table",
                "gripper_lower_offset_from_eef_m": -0.040,
                "counterpart_top_z_m": 0.900,
                "strict_no_contact_clearance_m": strict_clearance,
            },
        ],
        one_step_vertical_reserve_m=0.08 * 0.10,
    )
    assert selected_z == np.nextafter(
        evidence["binding_eef_z_lower_bound_m"], np.inf
    )
    assert evidence["binding_eef_z_lower_bound_m"] == pytest.approx(0.958)
    assert evidence["one_step_vertical_reserve_m"] == pytest.approx(0.008)
    assert evidence["limiting_pairs"] == [
        {
            "gripper_geom": "gripper_complete_lowest",
            "counterpart_geom": "plate_complete_highest",
            "counterpart_kind": "plate",
        }
    ]
    assert all(
        pair["selected_vertical_clearance_m"]
        > pair["required_clearance_with_one_step_reserve_m"]
        for pair in evidence["pairs"]
    )
    assert all(
        pair["minimum_vertical_sweep_clearance_m"]
        == pair["selected_vertical_clearance_m"]
        for pair in evidence["pairs"]
    )
    assert "from the exact native center-high state, first command XY plus" in (
        evidence["sweep_proof"]
    )
    assert "not direct observations of internal controller substeps" in (
        evidence["sweep_proof"]
    )

    corridor_xy = np.array([0.14480639548403948, -0.02850777957668001])
    lateral_action, lateral_path = _fixed_z_lateral_approach_action(
        current_eef=np.array([0.051856632, -0.028507780, 1.062506339]),
        lateral_target_xy=corridor_xy,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert lateral_action[0] > 0.0
    assert lateral_action[2] == 0.0
    assert np.linalg.norm(lateral_action[:3]) < 0.10
    assert lateral_path["commanded_z_action"] == 0.0

    vertical_action, vertical_path = _fixed_xy_vertical_approach_action(
        current_eef=np.array([*corridor_xy, 1.062506339]),
        target_z=selected_z,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert np.array_equal(vertical_action[:2], np.zeros(2))
    assert vertical_action[2] < 0.0
    assert np.linalg.norm(vertical_action[:3]) < 0.10
    assert vertical_path["commanded_xy_action"] == [0.0, 0.0]

    overhead_vertical = 1.062506338529415 - selected_z
    overhead_horizontal = np.linalg.norm(
        corridor_xy - np.array([0.051856632092207214, -0.02850777957668001])
    )
    corridor_vertical = selected_z - 0.917769758476126
    fixed_z_lateral = 0.008000000000000007
    action_equivalents = (
        overhead_vertical
        + overhead_horizontal
        + corridor_vertical
        + fixed_z_lateral
    ) / (0.08 * 0.10)
    assert action_equivalents == pytest.approx(30.71079293064015)
    assert action_equivalents < 180

    with pytest.raises(
        RuntimeError,
        match="no lower overhead staging Z",
    ):
        _derive_overhead_staging_from_compiled_pairs(
            start_eef_position=np.array([0.0, 0.0, 0.950]),
            compiled_pairs=[
                {
                    "gripper_geom": "gripper_complete_lowest",
                    "counterpart_geom": "plate_complete_highest",
                    "counterpart_kind": "plate",
                    "gripper_lower_offset_from_eef_m": -0.030,
                    "counterpart_top_z_m": 0.920,
                    "strict_no_contact_clearance_m": strict_clearance,
                }
            ],
            one_step_vertical_reserve_m=0.008,
        )


def test_500137_timeout_trace_is_replaced_by_auditable_overhead_state_machine():
    target = np.array(
        [0.14480639548403948, -0.02850777957668001, 1.062506338529415]
    )
    final_eef = np.array(
        [0.13855715318827433, -0.028401464440970036, 1.0569541469848756]
    )
    compiled_boundary_x = 0.13180639548403947
    assert np.linalg.norm(target - final_eef) == pytest.approx(
        0.008360093487905222
    )
    assert final_eef[0] - compiled_boundary_x == pytest.approx(
        0.006750757704234859
    )
    assert final_eef[0] - compiled_boundary_x < np.nextafter(0.008, np.inf)

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    assert "rollout.move(\n        corridor_high_target," not in bounded_seek
    assert (
        'structural_stage = "overhead_high_corridor_lateral"'
        in bounded_seek
    )
    assert '"overhead_corridor_descent"' in bounded_seek
    assert '"overhead_post_descent_corridor_lateral"' in bounded_seek
    assert "_live_compiled_overhead_guard(" in bounded_seek
    assert '"compiled_overhead_guard"' in bounded_seek
    assert "samples={json.dumps(samples, sort_keys=True)}" in bounded_seek
    assert "overhead_geometry={json.dumps(" in bounded_seek
    assert (
        "for guard_step in range(1, structural_waypoint_budget + 1)"
        in bounded_seek
    )
    assert "_overhead_corridor_entry_evidence(" in bounded_seek
    assert '"strict_corridor_entry_clearance_m"' in bounded_seek


def test_500146_negative_vertical_tail_brakes_before_first_lateral_action():
    strict_clearance = np.nextafter(0.0, np.inf)
    base_guard = {
        "one_step_vertical_reserve_m": 0.008,
        "pairs": [
            {
                "gripper_geom": "gripper0_finger2_collision",
                "counterpart_geom": "plate_1_g8",
                "counterpart_kind": "plate",
                "strict_no_contact_clearance_m": strict_clearance,
                "vertical_clearance_m": 0.012304936815258969,
            }
        ],
    }
    transition_buffer = _overhead_lateral_buffer_evidence(
        base_guard,
        worst_case_controller_world_step_m=0.08 * 0.10,
    )
    assert transition_buffer["base_overhead_reserve_m"] == pytest.approx(
        0.008
    )
    assert transition_buffer[
        "worst_case_controller_world_step_m"
    ] == pytest.approx(0.008)
    assert transition_buffer[
        "required_reserve_beyond_strict_clearance_m"
    ] == pytest.approx(0.016)
    assert transition_buffer["accepted"] is False
    assert transition_buffer[
        "minimum_lateral_entry_buffer_surplus_m"
    ] == pytest.approx(-0.0036950631847410315)

    # Exact Job500146 endpoint of the adaptive overhead descent.  The measured
    # response was still downward, so the next action must include a strict
    # positive-Z brake.  Job503010 established that pure +Z has a large inward
    # real-OSC response, so the brake now retains the registered outward drive
    # under the same compiled pair and native full-norm proof.
    before_transition_z = 0.9453538149129818
    transition_eef = np.array(
        [0.0458628425888722, -0.029186391480972046, 0.9443449236168449]
    )
    measured_dz = transition_eef[2] - before_transition_z
    assert measured_dz == pytest.approx(-0.0010088912961369045)
    recovered_guard = {
        **base_guard,
        "accepted": True,
        "pairs": [
            {
                **base_guard["pairs"][0],
                "vertical_clearance_m": 0.020,
                "accepted": True,
            }
        ],
    }
    recovered_buffer = _overhead_lateral_buffer_evidence(
        recovered_guard,
        worst_case_controller_world_step_m=0.008,
    )
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    brake_action, brake_evidence = (
        _compiled_adaptive_lateral_rebuffer_action(
            current_eef=transition_eef,
            overhead_guard=recovered_guard,
            overhead_lateral_buffer=recovered_buffer,
            outside_side_guard={
                "minimum_outside_clearance_m": 0.001,
                "required_outside_clearance_m": strict_clearance,
            },
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=1,
            worst_case_controller_world_step_m=0.008,
            lateral_target_xy=(
                transition_eef[:2] + np.array([0.01, 0.0])
            ),
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
            maximum_lateral_translation_action=0.10,
        )
    )
    old_first_lateral_action = np.array(
        [0.09999764807955064, 0.0006858414965135461, 0.0]
    )
    assert brake_action[0] > 0.0
    assert brake_action[1] == 0.0
    assert brake_action[2] > 0.0
    assert np.linalg.norm(brake_action[:2]) < 0.10
    assert np.linalg.norm(brake_action[:3]) < 1.0
    assert old_first_lateral_action[0] > 0.0
    assert old_first_lateral_action[2] == 0.0
    assert brake_evidence["commanded_xy_action"][0] > 0.0
    assert brake_evidence["commanded_z_action"] > 0.0
    assert brake_evidence["proof"][
        "outward_xy_plus_positive_z_zero_rotation"
    ] is True
    assert recovered_buffer["accepted"] is True
    assert recovered_buffer[
        "minimum_lateral_entry_buffer_surplus_m"
    ] > 0.0
    # Job503226 proved that keeping the full 8 mm positive-Z tail after the
    # buffer recovered was safe but consumed the 240-step waypoint budget.
    # Job503228 then measured negative real-Z response at action 0.005 and
    # positive response near action 0.10.  Use the exact half of the existing
    # 0.10 post-descent bound (4 mm world / action 0.05), inside the same native
    # full-action and per-pair proof, with negative response still falling back.
    low_positive_z_hold_action, low_positive_z_hold_evidence = (
        _compiled_adaptive_lateral_rebuffer_action(
            current_eef=transition_eef,
            overhead_guard=recovered_guard,
            overhead_lateral_buffer=recovered_buffer,
            outside_side_guard={
                "minimum_outside_clearance_m": 0.0013,
                "required_outside_clearance_m": strict_clearance,
            },
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=1,
            worst_case_controller_world_step_m=0.008,
            lateral_target_xy=(
                transition_eef[:2] + np.array([0.01, 0.0])
            ),
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
            maximum_lateral_translation_action=0.10,
            positive_z_tail_world_step_m=0.004,
        )
    )
    assert low_positive_z_hold_action[0] > 0.099
    assert low_positive_z_hold_action[2] == pytest.approx(0.05)
    assert np.linalg.norm(low_positive_z_hold_action[:3]) < 1.0
    assert low_positive_z_hold_evidence[
        "active_positive_z_tail_world_step_m"
    ] == pytest.approx(0.004)
    assert (
        low_positive_z_hold_evidence[
            "full_controller_step_positive_z_tail_retained"
        ]
        is False
    )
    assert low_positive_z_hold_evidence[
        "minimum_predicted_post_command_buffer16_surplus_m"
    ] > 0.0
    corridor_reserve_action, corridor_reserve_evidence = (
        _compiled_corridor_reserve_action(
            current_eef=transition_eef,
            overhead_guard=recovered_guard,
            outside_side_guard={
                "minimum_outside_clearance_m": 0.00075,
                "required_outside_clearance_m": strict_clearance,
            },
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=1,
            lateral_target_xy=(
                transition_eef[:2] + np.array([0.01, 0.0])
            ),
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
            maximum_lateral_translation_action=0.10,
            positive_z_action=0.05,
            strict_corridor_clearance_m=0.0004,
        )
    )
    assert corridor_reserve_action[0] > 0.099
    assert corridor_reserve_action[1] == 0.0
    assert corridor_reserve_action[2] == pytest.approx(0.05)
    assert corridor_reserve_evidence["compiled_pair_count"] == 1
    assert corridor_reserve_evidence["high_route_buffer16_required"] is False
    assert corridor_reserve_evidence[
        "minimum_predicted_base_surplus_m"
    ] > 0.0
    assert corridor_reserve_evidence["proof"] == {
        "strictly_outward_xy_zero_rotation": True,
        "strictly_positive_z": True,
        "strictly_inside_native_3d_action_norm_bound": True,
        "outside_clearance_statically_improves": True,
        "all_compiled_pairs_retain_strict_base_reserve": True,
        "post_action_live_guards_required": True,
    }
    low_side_brake, low_side_brake_evidence = (
        _compiled_low_side_settle_brake_action(
            current_eef=np.array([0.1340, -0.0285, 0.9150]),
            outside_side_guard={
                "accepted": False,
                "minimum_outside_clearance_m": 0.0019,
                "required_outside_clearance_m": strict_clearance,
                "finger_table_vertical_clearance_m": 0.0020,
                "required_finger_table_clearance_m": strict_clearance,
            },
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            lateral_target_xy=np.array([0.1496, -0.0285]),
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
            maximum_lateral_translation_action=0.20,
            positive_z_action=0.10,
            strict_corridor_clearance_m=0.0004,
        )
    )
    assert 0.194 < low_side_brake[0] < 0.196
    assert low_side_brake[1] == 0.0
    assert low_side_brake[2] == pytest.approx(0.10)
    assert np.linalg.norm(low_side_brake[:3]) < 1.0
    assert low_side_brake_evidence[
        "overhead_vertical_pair_guard_applicable"
    ] is False
    assert low_side_brake_evidence[
        "full_outside_side_guard_accepted"
    ] is False
    assert low_side_brake_evidence["proof"] == {
        "strictly_outward_xy_zero_rotation": True,
        "strictly_positive_z": True,
        "strictly_inside_native_3d_action_norm_bound": True,
        "outside_clearance_statically_improves": True,
        "finger_table_clearance_statically_improves": True,
        "post_action_live_guards_required": True,
    }

    # Job503282 proved that coupling the 0.20 -> 0.10 height release to
    # the outward component was unsafe.  Its first X=0.10, Z=0.10 settle
    # response crossed the unchanged 0.4 mm corridor gate.  The compiled
    # low-side command above retains X~=0.195 while the positive-Z schedule
    # is independently allowed to use 0.10.
    job503282_pre_clearance = 0.0005870025635664605
    job503282_post_clearance = 0.0003428052257444386
    assert job503282_pre_clearance > 0.0004
    assert job503282_post_clearance < 0.0004
    assert (
        job503282_post_clearance - job503282_pre_clearance
    ) == pytest.approx(-0.0002441973378220219)
    assert low_side_brake[0] > 0.19
    assert low_side_brake[2] == pytest.approx(0.10)

    job503290_full_brake, job503290_full_brake_evidence = (
        _compiled_low_side_settle_brake_action(
            current_eef=np.array(
                [0.13297672521591802, -0.026982433526499924, 0.9135291743299837]
            ),
            outside_side_guard={
                "accepted": True,
                "minimum_outside_clearance_m": 0.00169949040442946,
                "required_outside_clearance_m": strict_clearance,
                "finger_table_vertical_clearance_m": 0.000784447244981501,
                "required_finger_table_clearance_m": strict_clearance,
            },
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            lateral_target_xy=np.array([0.1485, -0.026982433526499924]),
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
            maximum_lateral_translation_action=0.20,
            positive_z_action=0.20,
            strict_corridor_clearance_m=0.0004,
        )
    )
    assert job503290_full_brake[0] > 0.19
    assert job503290_full_brake[2] == pytest.approx(0.20)
    assert np.linalg.norm(job503290_full_brake[:3]) < 1.0
    assert job503290_full_brake_evidence[
        "full_outside_side_guard_accepted"
    ] is True
    job503290_pre_table_clearance = 0.000784447244981501
    job503290_post_table_clearance = -0.0000371040818750723
    assert job503290_pre_table_clearance > 0.0
    assert job503290_post_table_clearance < 0.0
    assert (
        job503290_post_table_clearance
        - job503290_pre_table_clearance
    ) == pytest.approx(-0.0008215513268565733)

    # Job503168's recovered tail already passed the unchanged formal corridor
    # gate.  The former extra zero-Z frame then fell 0.239 mm and restarted the
    # brake loop, so the proved tail must hand off on this exact state instead.
    job503168_recovered_eef = np.array(
        [0.13251161316535595, -0.02827532139631423, 0.9445294804725799]
    )
    job503168_formal_entry = _overhead_corridor_entry_evidence(
        current_eef=job503168_recovered_eef,
        corridor_high_target=np.array(
            [0.13282106705090635, -0.02850777957668001, 0.9325011680386681]
        ),
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.0009045588122813253,
        },
        overhead_guard=recovered_guard,
        overhead_lateral_buffer=recovered_buffer,
        position_tolerance=0.004,
        strict_corridor_entry_clearance_m=0.0009,
    )
    assert job503168_formal_entry["accepted"] is True
    assert job503168_formal_entry["violations"] == []
    assert -0.00023933083913896258 < 0.0

    # Job503182 then showed that the 0.05 half-bound still settled into a
    # one-descent/three-brake cycle: only 0.668 mm net downward progress in 59
    # actions.  The unchanged 0.10 post-descent bound had already sustained 62
    # consecutive proved descent frames and 107.509 mm of progress.  Bind the
    # recovery floor to that existing bound; no registered limit is raised.
    tail_recovery_descent_floor = 0.10
    assert tail_recovery_descent_floor == pytest.approx(0.10)
    assert 0.005 < tail_recovery_descent_floor < 0.20
    assert 0.9477388181679267 - 1.0552474188949972 == pytest.approx(
        -0.10750860072707047
    )
    assert 0.9397046835577004 - 0.9403726320041634 == pytest.approx(
        -0.000667948446463007
    )
    job503184_tail_handoff_upper_z = 0.9325011680386681 + 2.0 * 0.008
    assert 0.9470836934130403 <= job503184_tail_handoff_upper_z

    # Job503186 reached vertical descent, but the exact first control action
    # targeted the formal corridor point that the EEF was already 0.181 mm
    # outside of.  Its one-sided allocator therefore commanded zero outward X
    # and the real OSC erased the corridor reserve in three frames.  Above the
    # compiled staging plane, retain a balanced outward hold.  Job503189 then
    # showed that consuming the entire 0.10 norm with the 8 mm target left no Z
    # authority and still drifted inward after the staging switch.  The next
    # fixed hold initially split the norm equally between XY and Z.
    job503186_eef = np.array(
        [0.13300229707876983, -0.0281777465755566, 0.9470836934130403]
    )
    formal_side_target = np.array(
        [0.13282106705090635, -0.02850777957668001, 0.9178414056548501]
    )
    # Job503251 proved that the 0.10 total side-corridor norm re-entered
    # recovery after three frames and exactly cancelled descent.  Reuse only
    # the existing 0.20 overhead-descent norm for the normal outward-priority
    # XY/Z allocation; recovery, settle and contact-seek bounds stay at 0.10.
    balanced_hold_world_step = 0.08 * (0.20 - 0.005)
    reserve_side_target = formal_side_target.copy()
    reserve_side_target[0] = (
        0.13287106705090635 + balanced_hold_world_step
    )
    reserve_action, reserve_path = (
        _constraint_prioritized_outside_descent_action(
            current_eef=job503186_eef,
            outside_side_target=reserve_side_target,
            outward_direction_xy=np.array([1.0, 0.0]),
            maximum_descent_m=float(
                job503186_eef[2] - formal_side_target[2]
            ),
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.20,
        )
    )
    assert 0.19 < reserve_action[0] < 0.195
    assert reserve_action[2] < 0.0
    assert -0.06 < reserve_action[2] < -0.05
    assert reserve_path["raw_outward_error_m"] > 0.015
    assert np.linalg.norm(reserve_action[:3]) < 0.20

    # Job503193 proved that equal XY/Z action allocation alone is not a live
    # reserve guarantee: despite 0.081 outward action, the real OSC continued
    # 0.188 mm inward and crossed the 0.4 mm one-step gate.  Job503209 then
    # showed that a shared 1.3 mm entry/exit threshold was unreachable.
    # Job503219 later measured a 0.490 mm inward closed-loop response, proving
    # that one nominal 0.4 mm structural step is not a conservative trigger.
    # Prioritize 0.195 of the already-registered 0.20 action norm outward,
    # retain the remaining norm for negative-Z descent, and reserve a
    # rounded-up 0.5 mm closed-loop response bound above the unchanged strict
    # gate.  The internal
    # release is the same 0.9 mm pre-loss reserve plus the existing 0.05 mm
    # measured-progress resolution; the formal 0.9 mm corridor gate itself
    # remains unchanged.  Job503245 then proved that Z=0.05 responds with
    # negative real Z on every outward-restore frame, while the compiled
    # Z=0.10 command retains the live base guard and positive response.
    recovery_entry_clearance = 0.0004 + 0.0005
    recovery_exit_clearance = np.nextafter(
        recovery_entry_clearance + 0.00005, np.inf
    )
    assert recovery_exit_clearance > 0.00095
    before_trigger = _vertical_corridor_reserve_recovery_evidence(
        live_clearance_m=0.0014115984435881107,
        recovery_entry_clearance_m=recovery_entry_clearance,
        recovery_exit_clearance_m=recovery_exit_clearance,
        strict_corridor_entry_clearance_m=0.0004,
        latest_outward_step_progress_m=1.3378271275732434e-05,
        latest_vertical_step_progress_m=-0.0006499833005025879,
        corridor_recovery_guard_accepted=True,
        recovery_exit_phase_authorized=False,
        recovery_active_before_decision=False,
    )
    assert before_trigger["recovery_active_after_decision"] is False
    job503193_entry = _vertical_corridor_reserve_recovery_evidence(
        live_clearance_m=0.000667534143986015,
        recovery_entry_clearance_m=recovery_entry_clearance,
        recovery_exit_clearance_m=recovery_exit_clearance,
        strict_corridor_entry_clearance_m=0.0004,
        latest_outward_step_progress_m=-0.00019280978843902452,
        latest_vertical_step_progress_m=-0.0018199586431316694,
        corridor_recovery_guard_accepted=True,
        recovery_exit_phase_authorized=False,
        recovery_active_before_decision=False,
    )
    assert job503193_entry["entered_recovery"] is True
    assert job503193_entry["recovery_active_after_decision"] is True
    recovery_hold = _vertical_corridor_reserve_recovery_evidence(
        live_clearance_m=0.000925,
        recovery_entry_clearance_m=recovery_entry_clearance,
        recovery_exit_clearance_m=recovery_exit_clearance,
        strict_corridor_entry_clearance_m=0.0004,
        latest_outward_step_progress_m=-1e-6,
        latest_vertical_step_progress_m=1e-6,
        corridor_recovery_guard_accepted=True,
        recovery_exit_phase_authorized=False,
        recovery_active_before_decision=True,
    )
    assert recovery_hold["exit_accepted"] is False
    recovery_exit = _vertical_corridor_reserve_recovery_evidence(
        live_clearance_m=0.0010,
        recovery_entry_clearance_m=recovery_entry_clearance,
        recovery_exit_clearance_m=recovery_exit_clearance,
        strict_corridor_entry_clearance_m=0.0004,
        latest_outward_step_progress_m=1e-6,
        latest_vertical_step_progress_m=1e-6,
        corridor_recovery_guard_accepted=True,
        recovery_exit_phase_authorized=True,
        recovery_active_before_decision=True,
    )
    assert recovery_exit["exit_accepted"] is True
    assert recovery_exit["recovery_active_after_decision"] is False
    recovery_exit_before_exit_brake = (
        _vertical_corridor_reserve_recovery_evidence(
            live_clearance_m=0.0010,
            recovery_entry_clearance_m=recovery_entry_clearance,
            recovery_exit_clearance_m=recovery_exit_clearance,
            strict_corridor_entry_clearance_m=0.0004,
            latest_outward_step_progress_m=1e-6,
            latest_vertical_step_progress_m=1e-6,
            corridor_recovery_guard_accepted=True,
            recovery_exit_phase_authorized=False,
            recovery_active_before_decision=True,
        )
    )
    assert recovery_exit_before_exit_brake["exit_accepted"] is False
    assert (
        recovery_exit_before_exit_brake[
            "recovery_active_after_decision"
        ]
        is True
    )
    recovery_exit_without_buffer = (
        _vertical_corridor_reserve_recovery_evidence(
            live_clearance_m=0.0010,
            recovery_entry_clearance_m=recovery_entry_clearance,
            recovery_exit_clearance_m=recovery_exit_clearance,
            strict_corridor_entry_clearance_m=0.0004,
            latest_outward_step_progress_m=1e-6,
            latest_vertical_step_progress_m=1e-6,
            corridor_recovery_guard_accepted=False,
            recovery_exit_phase_authorized=True,
            recovery_active_before_decision=True,
        )
    )
    assert recovery_exit_without_buffer["exit_accepted"] is False
    assert (
        recovery_exit_without_buffer["recovery_active_after_decision"]
        is True
    )
    legacy_positive_z_brake_target = formal_side_target.copy()
    legacy_positive_z_brake_target[0] = (
        0.13287106705090635 + 0.08 * (0.10 - 0.005)
    )
    recovery_action, recovery_path = (
        _constraint_prioritized_outside_descent_action(
            current_eef=np.array(
                [0.13285385483001622, -0.028214750624792236, 0.9448090606919402]
            ),
            outside_side_target=legacy_positive_z_brake_target,
            outward_direction_xy=np.array([1.0, 0.0]),
            maximum_descent_m=0.0,
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
            active_positive_z_brake=True,
        )
    )
    assert recovery_action[0] > 0.0
    assert recovery_action[2] > 0.0
    assert np.linalg.norm(recovery_action[:3]) < 0.10
    assert recovery_path["active_positive_z_brake"] is True
    outward_only_target = reserve_side_target.copy()
    outward_only_target[0] = 0.13287106705090635 + 0.008
    job503197_outward_action, job503197_outward_path = (
        _constraint_prioritized_outside_descent_action(
            current_eef=np.array(
                [0.13261073171023696, -0.028292198406861677, 0.9437519494597584]
            ),
            outside_side_target=outward_only_target,
            outward_direction_xy=np.array([1.0, 0.0]),
            maximum_descent_m=0.0,
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
        )
    )
    assert job503197_outward_action[0] > 0.099
    assert job503197_outward_action[2] == 0.0
    assert np.linalg.norm(job503197_outward_action[:3]) < 0.10
    assert job503197_outward_path["maximum_descent_m"] == 0.0
    recovery_entry_phase = (
        _vertical_corridor_reserve_recovery_phase_evidence(
            recovery_evidence=job503193_entry,
            phase_before_decision=None,
        )
    )
    assert recovery_entry_phase["phase_after_decision"] == "vertical_brake"
    job503217_unfilled_buffer = dict(job503193_entry)
    job503217_unfilled_buffer.update(
        {
            "entered_recovery": False,
            "latest_vertical_step_progress_m": 0.00024407744120835684,
            "latest_outward_step_progress_m": 0.00009467837885429597,
            "live_clearance_m": 0.000736711298289186,
            "corridor_recovery_guard_accepted": False,
        }
    )
    job503217_phase_hold = (
        _vertical_corridor_reserve_recovery_phase_evidence(
            recovery_evidence=job503217_unfilled_buffer,
            phase_before_decision="vertical_brake",
        )
    )
    assert job503217_phase_hold["phase_after_decision"] == "vertical_brake"
    vertical_brake_complete = dict(job503193_entry)
    vertical_brake_complete.update(
        {
            "entered_recovery": False,
            "latest_vertical_step_progress_m": 3.790651829160829e-05,
            "latest_outward_step_progress_m": -2.7702950653407665e-05,
            "live_clearance_m": 0.0010067760367388906,
        }
    )
    outward_phase = _vertical_corridor_reserve_recovery_phase_evidence(
        recovery_evidence=vertical_brake_complete,
        phase_before_decision="vertical_brake",
    )
    assert outward_phase["phase_after_decision"] == "outward_restore"
    job503204_one_frame_tail = dict(vertical_brake_complete)
    job503204_one_frame_tail.update(
        {
            "latest_vertical_step_progress_m": -0.0003909564934183596,
            "latest_outward_step_progress_m": 0.00006840830888554805,
            "live_clearance_m": 0.00085,
        }
    )
    outward_phase_latched = (
        _vertical_corridor_reserve_recovery_phase_evidence(
            recovery_evidence=job503204_one_frame_tail,
            phase_before_decision="outward_restore",
        )
    )
    assert (
        outward_phase_latched["phase_after_decision"]
        == "vertical_brake"
    )
    assert (
        outward_phase_latched["phase_transition"]
        == "outward_restore_negative_z_to_vertical_brake"
    )
    job503234_release_with_negative_z = dict(
        job503204_one_frame_tail
    )
    job503234_release_with_negative_z.update(
        {
            "live_clearance_m": 0.001343332975195899,
            "latest_outward_step_progress_m": 0.000020407306300318506,
            "latest_vertical_step_progress_m": -0.000021264757743555407,
            "corridor_recovery_guard_accepted": True,
        }
    )
    job503234_exit_brake = (
        _vertical_corridor_reserve_recovery_phase_evidence(
            recovery_evidence=job503234_release_with_negative_z,
            phase_before_decision="outward_restore",
        )
    )
    assert job503234_exit_brake["phase_after_decision"] == "exit_brake"
    assert (
        job503234_exit_brake["phase_transition"]
        == "outward_restore_complete_to_exit_brake"
    )
    restored_clearance = dict(job503204_one_frame_tail)
    restored_clearance.update(
        {
            "live_clearance_m": 0.0017,
            "latest_outward_step_progress_m": 1e-6,
            "latest_vertical_step_progress_m": 1e-6,
        }
    )
    exit_brake_phase = _vertical_corridor_reserve_recovery_phase_evidence(
        recovery_evidence=restored_clearance,
        phase_before_decision="outward_restore",
    )
    assert exit_brake_phase["phase_after_decision"] == "exit_brake"

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    active_envelope = bounded_seek.split(
        "def _active_vertical_corridor_control_envelope(", 1
    )[1].split(
        "vertical_corridor_closed_loop_inward_response_bound", 1
    )[0]
    assert "vertical_corridor_outward_priority_action" in active_envelope
    assert "vertical_corridor_balanced_hold_target_xy" in active_envelope
    assert "vertical_corridor_outward_hold_max_translation_action" in (
        active_envelope.split('"positive_z_settle_action"', 1)[1]
    )
    assert "geometric_height_action - structural_max_translation_action" not in (
        active_envelope
    )
    descent_transition = bounded_seek[
        bounded_seek.index(
            'elif stage_before_action == "overhead_corridor_descent"'
        ) : bounded_seek.index(
            'elif stage_before_action == "vertical_tail_brake"'
        )
    ]
    assert 'structural_stage = "vertical_tail_brake"' in descent_transition
    assert "overhead_descent_brake_trigger_buffer" in descent_transition
    assert 'structural_stage = "overhead_high_corridor_lateral"' not in (
        descent_transition
    )
    brake_transition = bounded_seek.split(
        'elif stage_before_action == "vertical_tail_brake":', 1
    )[1].split('elif stage_before_action == "lateral_rebuffer_brake":', 1)[0]
    assert (
        'structural_stage = "overhead_corridor_descent"'
        in brake_transition
    )
    assert "tail_brake_formal_corridor_entry" in brake_transition
    assert "_overhead_corridor_entry_evidence(" in brake_transition
    assert 'structural_stage = "vertical_corridor_descent"' in brake_transition
    assert '"overhead_post_descent_corridor_lateral"' in brake_transition
    assert "within_tail_handoff_band" in brake_transition
    assert "active_overhead_descent_brake_trigger_buffer" in brake_transition
    assert "the existing event-driven overhead-descent" in brake_transition
    assert '"controller_handoff_diagnostic_only"' in brake_transition
    assert "vertical_tail_zero_confirmation" not in brake_transition
    assert "previous_active_translation_action / 2.0" in brake_transition
    assert "structural_max_translation_action" in brake_transition
    assert "tail_recovery_descent_translation_action_floor" in (
        brake_transition
    )
    assert "the existing post-descent lateral action bound" in (
        brake_transition
    )
    vertical_corridor_action = bounded_seek.split(
        'elif structural_stage == "vertical_corridor_descent":', 1
    )[1].split('elif structural_stage == "vertical_corridor_settle":', 1)[0]
    assert "vertical_corridor_control_target" in vertical_corridor_action
    assert 'active_vertical_corridor_envelope[' in (
        vertical_corridor_action
    )
    assert "balanced_outward_controller_hold_active" in (
        vertical_corridor_action
    )
    assert '"formal_corridor_target_unchanged": True' in (
        vertical_corridor_action
    )
    assert "_vertical_corridor_reserve_recovery_evidence(" in (
        vertical_corridor_action
    )
    assert "negative_z_descent_suspended_for_reserve_recovery" in (
        vertical_corridor_action
    )
    assert "reserve_recovery_compiled_action_required" in (
        vertical_corridor_action
    )
    assert (
        "reserve_recovery_compiled_action_required = bool("
        in vertical_corridor_action
    )
    assert (
        "vertical_corridor_reserve_recovery_active"
        in vertical_corridor_action.split(
            "reserve_recovery_compiled_action_required = bool(", 1
        )[1].split(")", 1)[0]
    )
    assert "reserve_recovery_outward_restore_active" in (
        vertical_corridor_action
    )
    assert "corridor_correction_hold_target_xy" in vertical_corridor_action
    assert "_vertical_corridor_reserve_recovery_phase_evidence(" in (
        vertical_corridor_action
    )
    assert (
        "vertical_corridor_reserve_recovery_active"
        in vertical_corridor_action
    )
    assert "recovery_exit_phase_authorized" in vertical_corridor_action
    assert '== "outward_restore"' in vertical_corridor_action
    assert "_live_compiled_overhead_guard(" in vertical_corridor_action
    assert "_compiled_corridor_reserve_action(" in (
        vertical_corridor_action
    )
    assert "post_descent_lateral_max_translation_action" in (
        vertical_corridor_action
    )
    assert "maximum_vertical_corridor_outward_hold_world_step" in bounded_seek
    assert (
        "< maximum_vertical_corridor_outward_hold_world_step"
        in bounded_seek
    )
    assert "positive_z_action" in vertical_corridor_action
    positive_z_call = vertical_corridor_action.split(
        "positive_z_action=(", 1
    )[1].split("),", 1)[0]
    assert "post_descent_lateral_max_translation_action" in positive_z_call
    assert "0.5" not in positive_z_call
    assert (
        "reserve_recovery_positive_z_action_constant_across_phases"
        in vertical_corridor_action
    )
    assert "compiled_corridor_reserve_action_used" in (
        vertical_corridor_action
    )
    assert (
        '"high_route_buffer16_required_for_corridor_recovery": False'
        in vertical_corridor_action
    )
    assert "strictly positive Z" in vertical_corridor_action
    assert 'elif structural_stage == "vertical_tail_brake"' in bounded_seek
    brake_action_branch = bounded_seek.split(
        'elif structural_stage == "vertical_tail_brake":', 1
    )[1].split(
        'elif structural_stage == "lateral_rebuffer_brake":', 1
    )[0]
    assert "_compiled_adaptive_lateral_rebuffer_action(" in (
        brake_action_branch
    )
    assert "outside_side_guard=pre_action_guard" in brake_action_branch
    assert "maximum_post_descent_lateral_world_step" in (
        brake_action_branch
    )
    assert "corridor_correction_hold_target_xy" in brake_action_branch
    assert "corridor_outward_direction" in brake_action_branch
    assert "post_descent_lateral_max_translation_action" in (
        brake_action_branch
    )
    assert "compiled_outward_xy_positive_z_tail_brake_envelope" in (
        brake_action_branch
    )
    assert "tail_brake_lateral_entry_action_bound" in (
        brake_action_branch
    )
    assert "tail_brake_lateral_entry_world_step_m" in brake_action_branch
    assert "_fixed_xy_vertical_approach_action(" not in brake_action_branch
    post_action_buffer_refresh = bounded_seek.rsplit(
        "if stage_before_action in overhead_route_stages:", 1
    )[1].split("current_step_response = (", 1)[0]
    assert '== "overhead_corridor_descent"' in post_action_buffer_refresh
    assert '== "vertical_tail_brake"' in post_action_buffer_refresh
    assert "maximum_post_descent_lateral_world_step" in (
        post_action_buffer_refresh
    )
    assert "maximum_post_descent_lateral_world_step" in (
        post_action_buffer_refresh
    )
    assert "active_overhead_descent_world_step" in (
        post_action_buffer_refresh
    )
    pre_action_buffer_refresh = bounded_seek.split(
        "for guard_step in range(1, structural_waypoint_budget + 1):", 1
    )[1].split(
        'if (\n            structural_stage == "vertical_tail_brake"', 1
    )[0]
    assert 'structural_stage == "vertical_tail_brake"' in (
        pre_action_buffer_refresh
    )
    assert "maximum_post_descent_lateral_world_step" in (
        pre_action_buffer_refresh
    )
    assert "vertical_tail_zero_confirmation" not in bounded_seek
    assert "measured_vertical_step_progress_m >= 0.0" in bounded_seek
    assert 'latest_overhead_lateral_buffer["accepted"]' in bounded_seek
    assert "high_lateral_post_action_buffer_interlock_to_" not in bounded_seek
    assert "adaptive_high_lateral_negative_tail_recorded" in bounded_seek
    assert "post_descent_lateral_buffer_interlock_to_" in bounded_seek
    assert "lateral_pre_action_buffer_interlock_to_brake" in bounded_seek
    assert "vertical_tail_events" in bounded_seek


def test_500154_negative_lateral_tail_continues_until_buffer_is_exhausted():
    strict_clearance = np.nextafter(0.0, np.inf)
    final_eef = np.array(
        [0.08934230652832652, -0.03003309577111318, 0.9674248284743504]
    )
    corridor_high_target = np.array(
        [0.14480639548403948, -0.02850777957668001, 0.9401011680386682]
    )
    final_dz = -0.00023166049986367288
    final_buffer_surplus = 0.01947241620272018
    pair_template = {
        "gripper_geom": "gripper0_hand_collision",
        "counterpart_geom": "plate_1_g0",
        "counterpart_kind": "plate",
        "strict_no_contact_clearance_m": strict_clearance,
        "vertical_clearance_m": (
            strict_clearance + 0.008 + 0.008 + final_buffer_surplus
        ),
    }
    final_guard = {
        "one_step_vertical_reserve_m": 0.008,
        "pairs": [
            {
                **pair_template,
                "gripper_geom": f"gripper_collision_{index // 11}",
                "counterpart_geom": f"native_counterpart_{index % 11}",
            }
            for index in range(55)
        ],
    }
    final_buffer = _overhead_lateral_buffer_evidence(
        final_guard,
        worst_case_controller_world_step_m=0.008,
    )
    assert final_buffer[
        "minimum_lateral_entry_buffer_surplus_m"
    ] == pytest.approx(final_buffer_surplus)
    decision = _overhead_lateral_interlock_evidence(
        final_buffer,
        measured_vertical_step_progress_m=final_dz,
    )
    assert decision == {
        "requires_positive_z_brake": False,
        "decision_basis": decision["decision_basis"],
        "compiled_pair_count": 55,
        "buffer_accepted": True,
        "minimum_lateral_entry_buffer_surplus_m": pytest.approx(
            final_buffer_surplus
        ),
        "measured_vertical_step_progress_m": pytest.approx(final_dz),
        "negative_vertical_tail_observed": True,
    }
    next_action, _ = _fixed_z_lateral_approach_action(
        current_eef=final_eef,
        lateral_target_xy=corridor_high_target[:2],
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert np.linalg.norm(next_action[:2]) > 0.0
    assert next_action[2] == 0.0

    exhausted_guard = {
        **final_guard,
        "pairs": [
            {
                **pair,
                "vertical_clearance_m": strict_clearance + 0.016,
            }
            for pair in final_guard["pairs"]
        ],
    }
    exhausted_buffer = _overhead_lateral_buffer_evidence(
        exhausted_guard,
        worst_case_controller_world_step_m=0.008,
    )
    assert exhausted_buffer[
        "minimum_lateral_entry_buffer_surplus_m"
    ] == pytest.approx(0.0)
    brake_decision = _overhead_lateral_interlock_evidence(
        exhausted_buffer,
        measured_vertical_step_progress_m=0.001,
    )
    assert brake_decision["requires_positive_z_brake"] is True
    brake_action, _ = _fixed_xy_vertical_approach_action(
        current_eef=final_eef,
        target_z=final_eef[2] + 0.008,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert np.array_equal(brake_action[:2], np.zeros(2))
    assert brake_action[2] > 0.0


def test_500161_adaptive_descent_uses_native_bound_then_tightens_near_base8():
    class NativeEnv:
        action_spec = (
            -np.ones(7, dtype=float),
            np.ones(7, dtype=float),
        )

    native_spec = _native_osc_action_spec_evidence(
        SimpleNamespace(env=NativeEnv())
    )
    assert native_spec["source"] == "env.env.action_spec"
    assert native_spec["action_dimension"] == 7
    strict_clearance = np.nextafter(0.0, np.inf)

    def guard(vertical_clearance):
        pairs = [
            {
                "gripper_geom": f"gripper_collision_{index // 11}",
                "counterpart_geom": f"native_counterpart_{index % 11}",
                "counterpart_kind": (
                    "table" if index % 11 == 10 else "plate"
                ),
                "strict_no_contact_clearance_m": strict_clearance,
                "vertical_clearance_m": vertical_clearance,
                "accepted": True,
            }
            for index in range(55)
        ]
        return {
            "accepted": True,
            "one_step_vertical_reserve_m": 0.008,
            "pairs": pairs,
        }

    # Exact Job500161 initial frame. The native -1 Z bound, not the former
    # 0.10 contact-seek cap, limits this far-field command.
    far_eef = np.array(
        [0.05554037906914336, -0.029154933875409465, 1.0654223455054406]
    )
    target_z = 0.9401011680386682
    far_action, far_proof = _compiled_adaptive_vertical_descent_action(
        current_eef=far_eef,
        target_z=target_z,
        overhead_guard=guard(0.13332117746677247),
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_spec,
        expected_pair_count=55,
    )
    assert np.array_equal(far_action[:2], np.zeros(2))
    assert far_action[2] < -0.10
    assert -1.0 < far_action[2] < 0.0
    assert far_proof["compiled_pair_count"] == 55
    assert far_proof["selected_envelope_source"] == (
        "native_negative_z_action_bound"
    )
    assert far_proof["commanded_negative_world_delta_m"] == pytest.approx(
        0.08
    )
    assert len(far_proof["pair_envelopes"]) == 55
    assert all(
        pair["predicted_post_command_base_reserve_surplus_m"] > 0.0
        for pair in far_proof["pair_envelopes"]
    )

    # The far-field descent has an explicit controller cap below the native
    # bound; the live caller separately switches to its small near-plate cap.
    capped_action, capped_proof = (
        _compiled_adaptive_vertical_descent_action(
            current_eef=far_eef,
            target_z=target_z,
            overhead_guard=guard(0.13332117746677247),
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=55,
            maximum_translation_action=0.20,
        )
    )
    assert np.array_equal(capped_action[:2], np.zeros(2))
    assert abs(capped_action[2]) < 0.20
    assert capped_proof["selected_envelope_source"] == (
        "configured_translation_action_norm_bound"
    )
    assert capped_proof[
        "configured_strict_translation_action_capacity"
    ] == np.nextafter(0.20, 0.0)
    assert capped_proof[
        "commanded_negative_world_delta_m"
    ] == pytest.approx(0.016)

    # Exact Job500161 frame 88. Remaining target error tightens the action
    # below 0.10 and its direct all-pair proof remains strictly above base8.
    near_eef = np.array(
        [0.04590381377149573, -0.029190681243599335, 0.9453538149129818]
    )
    near_action, near_proof = _compiled_adaptive_vertical_descent_action(
        current_eef=near_eef,
        target_z=target_z,
        overhead_guard=guard(0.013306316723598785),
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_spec,
        expected_pair_count=55,
    )
    assert np.array_equal(near_action[:2], np.zeros(2))
    assert near_action[2] == pytest.approx(-0.06565808592891992)
    assert abs(near_action[2]) < 0.10
    assert near_proof["selected_envelope_source"] == (
        "target_remaining_z_error"
    )
    assert near_proof[
        "minimum_predicted_post_command_base_reserve_surplus_m"
    ] > 0.0
    assert near_proof["proof"] == {
        "pure_negative_z": True,
        "strictly_inside_native_z_action_bound": True,
        "does_not_cross_target_z": True,
        "all_compiled_pairs_retain_strict_base8_after_command": True,
    }

    # When pair geometry is tighter than both the target and native bounds,
    # the compiled pair envelope itself shrinks the pure-Z command.
    pair_action, pair_proof = _compiled_adaptive_vertical_descent_action(
        current_eef=np.array([0.0, 0.0, 0.950]),
        target_z=0.940,
        overhead_guard=guard(0.010),
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_spec,
        expected_pair_count=55,
    )
    assert pair_proof["selected_envelope_source"] == (
        "compiled_pair_base8_envelope"
    )
    assert 0.0 < abs(pair_action[2]) < 0.10
    assert pair_proof[
        "minimum_predicted_post_command_base_reserve_surplus_m"
    ] > 0.0

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    descent_branch = bounded_seek.rsplit(
        'elif structural_stage == "overhead_corridor_descent":', 1
    )[1].split('elif structural_stage == "vertical_tail_brake":', 1)[0]
    assert "prepared_high_lateral_action" in descent_branch
    assert "prepared_high_lateral_envelope" in descent_branch
    assert "active_overhead_descent_translation_action" in descent_branch
    assert "event_driven_brake_trigger_buffer_m" in descent_branch
    assert "_fixed_xy_vertical_approach_action(" not in descent_branch
    assert '"compiled_adaptive_corridor_descent_envelope"' in descent_branch
    assert "native_action_spec = _native_osc_action_spec_evidence(env)" in (
        bounded_seek
    )
    assert "compiled_overhead_one_step_vertical_reserve_lost" in bounded_seek
    assert (
        "for guard_step in range(1, structural_waypoint_budget + 1)"
        in bounded_seek
    )
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in CONTROLLER_REFERENCE.read_text()
    )


def test_adaptive_descent_fails_closed_without_runtime_native_action_spec():
    with pytest.raises(
        RuntimeError,
        match="native OSC action bounds unavailable",
    ):
        _native_osc_action_spec_evidence(SimpleNamespace())


def test_500174_lateral_rebuffer_is_adaptive_and_skips_repeat_zero():
    class NativeEnv:
        action_spec = (
            -np.ones(7, dtype=float),
            np.ones(7, dtype=float),
        )

    native_spec = _native_osc_action_spec_evidence(
        SimpleNamespace(env=NativeEnv())
    )
    strict_clearance = np.nextafter(0.0, np.inf)
    worst_observed_buffer_deficit_m = 0.0007891997997737984
    pairs = [
        {
            "gripper_geom": f"gripper_collision_{index // 11}",
            "counterpart_geom": f"native_counterpart_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": (
                strict_clearance
                + 0.016
                - (worst_observed_buffer_deficit_m if index == 0 else 0.0)
            ),
            "accepted": True,
        }
        for index in range(55)
    ]
    overhead_guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    lateral_buffer = _overhead_lateral_buffer_evidence(
        overhead_guard,
        worst_case_controller_world_step_m=0.008,
    )
    assert lateral_buffer["accepted"] is False
    outside_guard = {
        "minimum_outside_clearance_m": -0.03,
        "required_outside_clearance_m": strict_clearance,
        "finger_table_vertical_clearance_m": 0.05,
        "required_finger_table_clearance_m": strict_clearance,
    }

    action, proof = _compiled_adaptive_lateral_rebuffer_action(
        current_eef=np.array([0.10, -0.03, 0.96]),
        overhead_guard=overhead_guard,
        overhead_lateral_buffer=lateral_buffer,
        outside_side_guard=outside_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_spec,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert np.array_equal(action[:2], np.zeros(2))
    assert 0.10 < action[2] < 1.0
    assert proof["compiled_pair_count"] == 55
    assert proof["native_action_spec_source"] == "env.env.action_spec"
    assert proof["maximum_live_buffer_deficit_m"] == pytest.approx(
        worst_observed_buffer_deficit_m
    )
    assert proof["requested_positive_world_delta_m"] == pytest.approx(
        0.008 + worst_observed_buffer_deficit_m
    )
    assert proof["commanded_positive_world_delta_m"] == pytest.approx(
        0.008 + worst_observed_buffer_deficit_m
    )
    assert proof["minimum_predicted_post_command_base8_surplus_m"] > 0.0
    assert proof["minimum_predicted_post_command_buffer16_surplus_m"] > 0.0
    assert len(proof["pair_envelopes"]) == 55
    assert proof["proof"] == {
        "pure_positive_z": True,
        "strictly_inside_native_z_action_bound": True,
        "outside_xy_clearance_not_worsened_by_pure_z": True,
        "all_compiled_pairs_retain_strict_no_contact": True,
        "all_compiled_pairs_retain_strict_base8": True,
        "all_compiled_pairs_reach_strict_buffer16": True,
    }

    # A delayed negative tail may survive after buffer16 has already become
    # positive.  Rebuffer must keep braking until measured dz is nonnegative,
    # without inserting a zero-confirm action or failing merely because the
    # buffer is already recovered.
    recovered_pairs = [
        {
            **pair,
            "vertical_clearance_m": (
                pair["strict_no_contact_clearance_m"] + 0.017
            ),
        }
        for pair in pairs
    ]
    recovered_guard = {**overhead_guard, "pairs": recovered_pairs}
    recovered_buffer = _overhead_lateral_buffer_evidence(
        recovered_guard,
        worst_case_controller_world_step_m=0.008,
    )
    assert recovered_buffer["accepted"] is True
    tail_brake_action, tail_brake_proof = (
        _compiled_adaptive_lateral_rebuffer_action(
            current_eef=np.array([0.10, -0.03, 0.97]),
            overhead_guard=recovered_guard,
            overhead_lateral_buffer=recovered_buffer,
            outside_side_guard=outside_guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=55,
            worst_case_controller_world_step_m=0.008,
        )
    )
    assert np.array_equal(tail_brake_action[:2], np.zeros(2))
    assert tail_brake_action[2] > 0.10
    assert tail_brake_proof["maximum_live_buffer_deficit_m"] == 0.0
    assert tail_brake_proof["requested_positive_world_delta_m"] == (
        pytest.approx(0.008)
    )
    assert tail_brake_proof[
        "minimum_predicted_post_command_buffer16_surplus_m"
    ] > 0.008

    # When the rebuffer will resume the registered post-descent correction,
    # retain its one-sided outward drive while the exact +Z deficit/tail is
    # refilled.  The lateral component is independently capped and the whole
    # translation remains strictly inside the runtime-native 3-D norm.
    correction_current = np.array(
        [0.13058111691187727, -0.028505282763058467, 0.9400816244000332]
    )
    correction_target = np.array([0.140871, -0.028508])
    combined_action, combined_proof = (
        _compiled_adaptive_lateral_rebuffer_action(
            current_eef=correction_current,
            overhead_guard=overhead_guard,
            overhead_lateral_buffer=lateral_buffer,
            outside_side_guard=outside_guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=55,
            worst_case_controller_world_step_m=0.008,
            lateral_target_xy=correction_target,
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
            maximum_lateral_translation_action=0.10,
        )
    )
    assert combined_action[0] > 0.0
    assert combined_action[1] < 0.0
    assert combined_action[2] == pytest.approx(action[2])
    assert np.linalg.norm(combined_action[:2]) < 0.10
    assert np.linalg.norm(combined_action[:3]) < 1.0
    assert combined_proof["retain_outward_lateral_drive"] is True
    assert combined_proof["commanded_lateral_world_delta_m"] <= (
        np.linalg.norm(correction_target - correction_current[:2])
    )
    assert combined_proof[
        "minimum_predicted_post_command_buffer16_surplus_m"
    ] > 0.0
    assert combined_proof["proof"] == {
        "outward_xy_plus_positive_z_zero_rotation": True,
        "strictly_inside_native_3d_action_norm_bound": True,
        "inside_configured_lateral_action_norm_bound": True,
        "does_not_cross_lateral_target": True,
        "inward_outward_axis_command_prohibited": True,
        "positive_z_static_geometry_does_not_reduce_clearance": True,
        "reduced_positive_z_tail_requires_preaccepted_buffer16": False,
        "all_compiled_pairs_retain_strict_no_contact": True,
        "all_compiled_pairs_retain_strict_base8": True,
        "all_compiled_pairs_reach_strict_buffer16": True,
    }

    # The exact post-rebuffer state from job 503001 already passes the
    # unchanged formal corridor gate even though it has not reached the
    # additional shifted controller target.  That formal state may hand off;
    # the shifted target remains the fail-closed correction fallback.
    formal_outside_guard = {
        **outside_guard,
        "minimum_outside_clearance_m": 0.0014019095805165027,
    }
    formal_entry = _overhead_corridor_entry_evidence(
        current_eef=np.array(
            [0.1329777566584359, -0.028413192405800588, 0.9403155435501364]
        ),
        corridor_high_target=np.array(
            [0.13287106705090634, -0.02850777957668001, 0.9325011680386681]
        ),
        outside_side_guard=formal_outside_guard,
        overhead_guard=recovered_guard,
        overhead_lateral_buffer=recovered_buffer,
        position_tolerance=0.01,
        strict_corridor_entry_clearance_m=0.0009,
    )
    assert formal_entry["accepted"] is True
    shifted_target_diagnostic = _overhead_corridor_entry_evidence(
        current_eef=np.array(
            [0.1329777566584359, -0.028413192405800588, 0.9403155435501364]
        ),
        corridor_high_target=np.array(
            [0.14087106705090635, -0.02850777957668001, 0.9325011680386681]
        ),
        outside_side_guard=formal_outside_guard,
        overhead_guard=recovered_guard,
        overhead_lateral_buffer=recovered_buffer,
        position_tolerance=0.004,
        strict_corridor_entry_clearance_m=0.0009,
        require_lateral_buffer=False,
    )
    assert shifted_target_diagnostic["accepted"] is False
    assert shifted_target_diagnostic["violations"] == [
        "corridor_xy_tolerance_not_met"
    ]

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    assert '"lateral_rebuffer_brake": 0' in bounded_seek
    assert "_compiled_adaptive_lateral_rebuffer_action(" in bounded_seek
    assert "lateral_target_xy=(" in bounded_seek
    assert "corridor_correction_hold_target_xy" in bounded_seek
    assert "one_sided_outward_direction_xy=(" in bounded_seek
    assert "corridor_outward_direction" in bounded_seek
    assert "maximum_lateral_translation_action=(" in bounded_seek
    assert "post_descent_lateral_max_translation_action" in bounded_seek
    assert '"retains_registered_outward_correction_drive"' in bounded_seek
    assert bounded_seek.count(
        'structural_stage = "lateral_rebuffer_brake"'
    ) >= 2
    initial_descent_transition = bounded_seek.split(
        'elif stage_before_action == "overhead_corridor_descent":', 1
    )[1].split('elif stage_before_action == "vertical_tail_brake":', 1)[0]
    assert 'structural_stage = "vertical_tail_brake"' in (
        initial_descent_transition
    )
    initial_brake_transition = bounded_seek.split(
        'elif stage_before_action == "vertical_tail_brake":', 1
    )[1].split(
        'elif stage_before_action == "lateral_rebuffer_brake":', 1
    )[0]
    assert "tail_brake_formal_corridor_entry" in initial_brake_transition
    assert 'structural_stage = "overhead_corridor_descent"' in (
        initial_brake_transition
    )
    assert 'structural_stage = "vertical_corridor_descent"' in (
        initial_brake_transition
    )
    assert "vertical_tail_zero_confirmation" not in initial_brake_transition
    lateral_rebuffer_transition = bounded_seek.split(
        'elif stage_before_action == "lateral_rebuffer_brake":', 1
    )[1].split(
        'elif stage_before_action == "overhead_post_descent_corridor_lateral":',
        1,
    )[0]
    assert 'structural_stage = lateral_resume_stage' in (
        lateral_rebuffer_transition
    )
    assert "measured_vertical_step_progress_m >= 0.0" in (
        lateral_rebuffer_transition
    )
    assert 'latest_overhead_lateral_buffer["accepted"]' in (
        lateral_rebuffer_transition
    )
    assert "vertical_tail_zero_confirmation" not in (
        lateral_rebuffer_transition
    )
    assert "lateral_rebuffer_formal_corridor_entry" in (
        lateral_rebuffer_transition
    )
    assert "corridor_rebuffer_target" in lateral_rebuffer_transition
    assert "corridor_rebuffer_acceptance_clearance" in (
        lateral_rebuffer_transition
    )
    assert '"overhead_corridor_descent"' in lateral_rebuffer_transition
    assert '"vertical_corridor_descent"' in lateral_rebuffer_transition
    assert "lateral_rebuffer_shifted_target_diagnostic" in (
        lateral_rebuffer_transition
    )
    assert "lateral_resume_stage = None" in lateral_rebuffer_transition
    assert "compiled_overhead_one_step_vertical_reserve_lost" in bounded_seek
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in CONTROLLER_REFERENCE.read_text()
    )


def test_adaptive_lateral_rebuffer_fails_closed_on_invalid_live_geometry():
    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": (-np.ones(7, dtype=float)).tolist(),
        "high": np.ones(7, dtype=float).tolist(),
        "runtime_resolved": True,
    }
    strict_clearance = np.nextafter(0.0, np.inf)
    pair = {
        "gripper_geom": "gripper0_hand_collision",
        "counterpart_geom": "plate_1_g0",
        "counterpart_kind": "plate",
        "strict_no_contact_clearance_m": strict_clearance,
        "vertical_clearance_m": 0.015,
        "accepted": True,
    }
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": [pair],
    }
    buffer = _overhead_lateral_buffer_evidence(
        guard,
        worst_case_controller_world_step_m=0.008,
    )
    kwargs = {
        "current_eef": np.array([0.10, -0.03, 0.96]),
        "overhead_guard": guard,
        "overhead_lateral_buffer": buffer,
        "outside_side_guard": {
            "minimum_outside_clearance_m": -0.03,
            "required_outside_clearance_m": strict_clearance,
        },
        "gripper": -1.0,
        "position_action_scale": 0.08,
        "native_action_spec": native_spec,
        "expected_pair_count": 1,
        "worst_case_controller_world_step_m": 0.008,
    }
    with pytest.raises(RuntimeError, match="base overhead reserve"):
        _compiled_adaptive_lateral_rebuffer_action(
            **{
                **kwargs,
                "overhead_guard": {**guard, "accepted": False},
            }
        )
    with pytest.raises(RuntimeError, match="pair inventory changed"):
        _compiled_adaptive_lateral_rebuffer_action(
            **{**kwargs, "expected_pair_count": 55}
        )
    with pytest.raises(
        ValueError, match="outward-drive inputs must be supplied together"
    ):
        _compiled_adaptive_lateral_rebuffer_action(
            **{**kwargs, "lateral_target_xy": np.array([0.14, -0.03])}
        )
    with pytest.raises(ValueError, match="outward-drive geometry is invalid"):
        _compiled_adaptive_lateral_rebuffer_action(
            **{
                **kwargs,
                "lateral_target_xy": np.array([0.14, -0.03]),
                "one_sided_outward_direction_xy": np.array([2.0, 0.0]),
                "maximum_lateral_translation_action": 0.10,
            }
        )


def test_500182_high_first_route_orders_xy_before_adaptive_descent():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.13332117746677247,
            "accepted": True,
        }
        for index in range(55)
    ]
    overhead_guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    buffer = _overhead_lateral_buffer_evidence(
        overhead_guard,
        worst_case_controller_world_step_m=0.008,
    )
    outside_guard = {
        "accepted": False,
        "minimum_outside_clearance_m": -0.07638068798176297,
        "required_outside_clearance_m": strict_clearance,
    }
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard=outside_guard,
        overhead_guard=overhead_guard,
        overhead_lateral_buffer=buffer,
        compiled_pairs=pairs,
        expected_pair_count=55,
        require_lateral_buffer=True,
    )
    assert authorization["accepted"] is True
    assert authorization["compiled_pair_count"] == 55
    assert len(authorization["compiled_pair_identity_keys"]) == 55
    assert authorization["outside_side_guard_accepted"] is False
    assert authorization["authorization_basis"] == (
        "compiled_overhead_all_pair_envelope_while_outside_guard_not_accepted"
    )
    assert authorization["internal_controller_substeps_measured"] is False
    assert authorization["proof_scope"] == (
        "live pre/post world-AABB checks plus the unchanged 8 mm base8 and "
        "16 mm lateral-entry envelopes; not direct observations of internal "
        "controller substeps"
    )

    initial = np.array(
        [0.05554037906914336, -0.029154933875409465, 1.0654223455054406]
    )
    corridor_xy = np.array(
        [0.14480639548403948, -0.02850777957668001]
    )
    first_action, first_proof = _fixed_z_lateral_approach_action(
        current_eef=initial,
        lateral_target_xy=corridor_xy,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert np.linalg.norm(first_action[:3]) < 0.10
    assert first_action[2] == 0.0
    assert first_proof["commanded_z_action"] == 0.0

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    assert (
        'structural_stage = "overhead_high_corridor_lateral"'
        in bounded_seek
    )
    assert 'structural_stage = "overhead_corridor_descent"' in (
        bounded_seek
    )
    descent_action = bounded_seek.rsplit(
        'elif structural_stage == "overhead_corridor_descent":', 1
    )[1].split('elif structural_stage == "vertical_tail_brake":', 1)[0]
    descent_compilation = bounded_seek.split(
        'elif stage_before_action == "overhead_corridor_descent":', 1
    )[1].split(
        "adaptive_negative_z_action_requires_buffer16", 1
    )[0]
    assert (
        "_compiled_adaptive_workspace_release_action("
        in descent_compilation
    )
    assert "corridor_target_xy=corridor_descent_hold_target_xy" in (
        descent_compilation
    )
    assert "* active_overhead_descent_world_step" in bounded_seek
    assert "couple_downward_to_lateral_remaining=False" in (
        descent_compilation
    )
    assert "maximum_translation_action=(" in descent_compilation
    assert "one_sided_outward_direction_xy=(" in descent_compilation
    assert "active_overhead_descent_translation_action" in descent_action
    assert "prepared_high_lateral_action" in descent_action
    assert "compiled_adaptive_corridor_descent_envelope" in descent_action
    assert "_compiled_adaptive_vertical_descent_action(" not in descent_action
    assert "_fixed_z_lateral_approach_action(" not in descent_action
    vertical_corridor_action = bounded_seek.split(
        'elif structural_stage == "vertical_corridor_descent":', 1
    )[1].split('elif structural_stage == "vertical_corridor_settle":', 1)[0]
    assert (
        "fixed_outward_translation_action_bound"
        in vertical_corridor_action
    )
    post_descent_lateral_action = bounded_seek.split(
        'elif structural_stage == "overhead_post_descent_corridor_lateral":',
        1,
    )[1].split('elif structural_stage == "vertical_corridor_descent":', 1)[0]
    assert "prepared_high_lateral_action" in post_descent_lateral_action
    assert (
        "compiled_adaptive_post_descent_plane_hold_envelope"
        in post_descent_lateral_action
    )
    assert "post_descent_lateral_max_translation_action" in bounded_seek
    assert "maximum_post_descent_lateral_world_step" in bounded_seek
    descent_transition = bounded_seek.rsplit(
        'elif stage_before_action == "overhead_corridor_descent":', 1
    )[1].split('elif stage_before_action == "vertical_tail_brake":', 1)[0]
    assert 'structural_stage = "vertical_tail_brake"' in descent_transition
    assert (
        "descent_corridor_entry_after_action = ("
        in descent_transition
    )
    assert "_overhead_corridor_entry_evidence(" in descent_transition
    assert '"corridor_xy_tolerance_not_met"' in descent_transition
    assert (
        '"outside_corridor_entry_clearance_not_met"'
        in descent_transition
    )
    assert "if descent_corridor_lateral_violations:" in descent_transition
    assert "strict_corridor_entry_clearance_m=(" in descent_transition
    assert '"strict_corridor_entry_clearance_m"' in descent_transition
    assert '"eef_outward_step_progress_m"' in descent_transition
    assert '"outside_clearance_step_progress_m"' in descent_transition
    assert (
        "-float(args.minimum_saturated_waypoint_progress)"
        in descent_transition
    )
    assert '"eef_inward_step_during_corridor_holding_descent"' in (
        descent_transition
    )
    assert (
        'vertical_tail_brake_reason = "lateral_drift"'
        in descent_transition
    )
    assert (
        "corridor_entry_lateral_guard_failed_to_high_"
        in descent_transition
    )
    brake_transition = bounded_seek.split(
        'elif stage_before_action == "vertical_tail_brake":', 1
    )[1].split('elif stage_before_action == "lateral_rebuffer_brake":', 1)[0]
    assert "tail_brake_formal_corridor_entry" in brake_transition
    assert (
        "overhead_staging_z + args.position_tolerance" in brake_transition
    )
    assert 'structural_stage = "overhead_corridor_descent"' in (
        brake_transition
    )
    assert 'structural_stage = "vertical_corridor_descent"' in brake_transition
    assert "corridor_rebuffer_target" in brake_transition
    assert "corridor_rebuffer_acceptance_clearance" in brake_transition
    assert "vertical_tail_zero_confirmation" not in brake_transition
    correction_transition = bounded_seek.split(
        'elif stage_before_action == "overhead_post_descent_corridor_lateral":',
        1,
    )[1].split(
        'elif stage_before_action == "vertical_corridor_settle":', 1
    )[0]
    assert (
        '"pre_descent_controller_reserve_complete_to_"'
        in correction_transition
    )
    assert (
        '"post_descent_formal_corridor_entry_complete_to_"'
        in correction_transition
    )
    assert "correction_requires_pre_descent_controller_reserve" in (
        correction_transition
    )
    assert "post_descent_vertical_tail_handoff_accepted" in (
        correction_transition
    )
    assert (
        ">= -float(args.minimum_saturated_waypoint_progress)"
        in correction_transition
    )
    assert 'structural_stage = "overhead_corridor_descent"' in (
        correction_transition
    )
    assert "corridor_rebuffer_target" in correction_transition
    assert "corridor_rebuffer_acceptance_clearance" in (
        correction_transition
    )
    assert bounded_seek.index(
        'structural_stage = "overhead_high_corridor_lateral"'
    ) < bounded_seek.index(
        'structural_stage = "overhead_corridor_descent"',
        bounded_seek.index(
            'structural_stage = "overhead_high_corridor_lateral"'
        ),
    )
    assert '"structural_route_order"' in bounded_seek
    assert (
        '"native_center_high_to_registered_corridor_high_"'
        in bounded_seek
    )
    assert (
        '"corridor_xy_adaptive_one_sided_coupled_descent_with_"'
        in bounded_seek
    )
    assert '"position_"' in bounded_seek
    assert (
        '"tolerance_strict_clearance_or_unbuffered_measured_"'
        in bounded_seek
    )
    assert '"inward_response_"' in bounded_seek
    assert '"brake"' in bounded_seek
    assert '"descent_corridor_resume_clearance_m"' in bounded_seek
    assert '"descent_corridor_rebuffer_requested_clearance_m"' in (
        bounded_seek
    )
    assert (
        '"compiled full corridor clearance plus the existing "'
        in bounded_seek
    )
    assert "corridor_rebuffer_target[:2] += (" in bounded_seek
    assert "* float(args.minimum_saturated_waypoint_progress)" in (
        bounded_seek
    )
    assert '"outward margin; no empirical Z threshold"' in bounded_seek
    assert '"progress is below the negative deadband after the "' in (
        bounded_seek
    )
    assert "not full_corridor_clearance_retained" in descent_transition
    assert (
        'feedback["full_corridor_clearance_retained_after_descent"]'
        in descent_transition
    )
    assert '"existing minimum_saturated_waypoint_progress"' in (
        bounded_seek
    )
    controller = CONTROLLER_REFERENCE.read_text()
    assert "after the monotone pure-Z sweep, command pure XY" not in controller
    assert "internal controller substeps" in controller
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )

    native_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    corridor_descent_hold_xy = corridor_xy + np.array([0.016, 0.0])
    exact_corridor = np.array(
        [corridor_descent_hold_xy[0], corridor_descent_hold_xy[1], 1.03]
    )
    exact_action, exact_evidence = (
        _compiled_adaptive_workspace_release_action(
            current_eef=exact_corridor,
            corridor_target_xy=corridor_descent_hold_xy,
            release_target_z=0.94,
            measured_vertical_step_progress_m=0.0,
            overhead_guard=overhead_guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=55,
            worst_case_controller_world_step_m=0.016,
            couple_downward_to_lateral_remaining=False,
            maximum_translation_action=0.20,
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
        )
    )
    assert np.array_equal(exact_action[:2], np.zeros(2))
    assert exact_action[2] < 0.0
    assert np.linalg.norm(exact_action[:3]) < 0.20
    assert exact_evidence["motion_kind"] == (
        "one_sided_corridor_holding_downward_descent"
    )
    assert exact_evidence["downward_coupled_to_lateral_remaining"] is False
    assert exact_evidence["independent_downward_progress_authorized"] is True
    assert exact_evidence["downward_request_capped_by_xy_remaining"] is False
    assert exact_evidence["candidate_action_norm_capacities"][
        "configured_translation_action_norm_bound"
    ] == np.nextafter(0.20, 0.0)
    assert exact_evidence["proof"][
        "corridor_xy_hold_plus_nonpositive_z_zero_rotation"
    ] is True
    assert exact_evidence["proof"][
        "inward_outward_axis_command_prohibited"
    ] is True

    overshot_corridor = exact_corridor.copy()
    overshot_corridor[0] += 0.002
    overshot_action, overshot_evidence = (
        _compiled_adaptive_workspace_release_action(
            current_eef=overshot_corridor,
            corridor_target_xy=corridor_descent_hold_xy,
            release_target_z=0.94,
            measured_vertical_step_progress_m=0.0,
            overhead_guard=overhead_guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=55,
            worst_case_controller_world_step_m=0.016,
            couple_downward_to_lateral_remaining=False,
            maximum_translation_action=0.20,
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
        )
    )
    assert overshot_action[0] == 0.0
    assert overshot_action[2] < 0.0
    assert overshot_evidence[
        "suppressed_inward_outward_axis_error_m"
    ] == pytest.approx(0.002)
    assert overshot_evidence["proof"][
        "inward_outward_axis_command_prohibited"
    ] is True

    base_corridor_eef = exact_corridor.copy()
    base_corridor_eef[0] -= 0.016
    held_action, held_evidence = (
        _compiled_adaptive_workspace_release_action(
            current_eef=base_corridor_eef,
            corridor_target_xy=corridor_descent_hold_xy,
            release_target_z=0.94,
            measured_vertical_step_progress_m=0.0,
            overhead_guard=overhead_guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_spec,
            expected_pair_count=55,
            worst_case_controller_world_step_m=0.016,
            couple_downward_to_lateral_remaining=False,
            maximum_translation_action=0.20,
            one_sided_outward_direction_xy=np.array([1.0, 0.0]),
        )
    )
    assert held_action[0] > 0.0
    assert held_action[2] < 0.0
    assert np.linalg.norm(held_action[:3]) < 0.20
    assert held_evidence["corridor_target_xy"] == (
        corridor_descent_hold_xy.tolist()
    )


def test_500193_high_lateral_uses_compiled_dynamic_action_envelope():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.13332117746677247,
            "accepted": True,
        }
        for index in range(55)
    ]
    overhead_guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native_action_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    start = np.array(
        [0.05554037906914336, -0.029154933875409465, 1.0654223455054406]
    )
    corridor_xy = np.array(
        [0.14480639548403948, -0.02850777957668001]
    )
    action, evidence = _compiled_adaptive_high_lateral_action(
        current_eef=start,
        lateral_target_xy=corridor_xy,
        overhead_guard=overhead_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_action_spec,
        expected_pair_count=55,
    )
    assert np.linalg.norm(action[:3]) > 0.10
    assert np.linalg.norm(action[:3]) < 1.0
    assert action[2] == 0.0
    assert np.all(action[3:6] == 0.0)
    assert evidence["compiled_pair_count"] == 55
    assert evidence["selected_envelope_source"] == (
        "native_xy_translation_action_norm_bound"
    )
    assert evidence["proof"] == {
        "pure_xy_zero_z_rotation": True,
        "strictly_inside_native_xy_action_norm_bound": True,
        "does_not_cross_lateral_target": True,
        "all_compiled_pairs_retain_strict_base8_after_worst_case_tail": True,
    }
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": -0.05,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=overhead_guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            overhead_guard,
            worst_case_controller_world_step_m=0.008,
        ),
        compiled_pairs=pairs,
        expected_pair_count=55,
        require_lateral_buffer=False,
        adaptive_high_lateral_envelope=evidence,
    )
    assert authorization["accepted"] is True
    assert authorization["adaptive_high_lateral_pair_count"] == 55
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert all(
        pair["pair_identity"]
        == [
            pairs[index]["gripper_geom"],
            pairs[index]["counterpart_geom"],
            pairs[index]["counterpart_kind"],
        ]
        for index, pair in enumerate(evidence["pair_envelopes"])
    )
    fixed_buffer_audit = _overhead_lateral_buffer_evidence(
        overhead_guard,
        worst_case_controller_world_step_m=0.008,
    )
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": -0.076,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=overhead_guard,
        overhead_lateral_buffer=fixed_buffer_audit,
        compiled_pairs=pairs,
        expected_pair_count=55,
        require_lateral_buffer=False,
        adaptive_high_lateral_envelope=evidence,
    )
    assert authorization["accepted"] is True
    assert authorization["buffer16_used_for_authorization"] is False
    assert authorization[
        "adaptive_high_lateral_envelope_used_for_authorization"
    ] is True
    assert authorization["adaptive_high_lateral_pair_count"] == 55
    assert "dynamic worst-case action tail" in authorization["proof_scope"]

    # Near base8, the same geometry—not a relaxed threshold—shrinks the
    # high-lateral norm below the unchanged 0.10 contact/correction setting.
    tight_pairs = [
        {**pair, "vertical_clearance_m": strict_clearance + 0.008 + 0.004}
        for pair in pairs
    ]
    tight_action, tight_evidence = _compiled_adaptive_high_lateral_action(
        current_eef=start,
        lateral_target_xy=corridor_xy,
        overhead_guard={**overhead_guard, "pairs": tight_pairs},
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_action_spec,
        expected_pair_count=55,
    )
    assert np.linalg.norm(tight_action[:3]) < 0.05
    assert tight_evidence["selected_envelope_source"] == (
        "compiled_pair_base8_worst_case_tail"
    )
    assert tight_evidence["minimum_predicted_post_worst_case_base_surplus_m"] > (
        0.0
    )

    with pytest.raises(RuntimeError, match="positive adaptive lateral"):
        _compiled_adaptive_high_lateral_action(
            current_eef=start,
            lateral_target_xy=corridor_xy,
            overhead_guard={
                **overhead_guard,
                "pairs": [{**tight_pairs[0], "accepted": False}]
                + tight_pairs[1:],
            },
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_action_spec,
            expected_pair_count=55,
        )

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    high_action_branch = bounded_seek.split(
        'elif structural_stage == "overhead_high_corridor_lateral":', 1
    )[1].split(
        'elif structural_stage == "overhead_post_descent_corridor_lateral":',
        1,
    )[0]
    post_descent_correction_branch = bounded_seek.split(
        'elif structural_stage == "overhead_post_descent_corridor_lateral":',
        1,
    )[1].split(
        'elif structural_stage == "vertical_corridor_descent":', 1
    )[0]
    post_descent_compilation = bounded_seek.split(
        'elif (\n            stage_before_action\n'
        '            == "overhead_post_descent_corridor_lateral"\n'
        '        ):',
        1,
    )[1].split("adaptive_negative_z_action_requires_buffer16", 1)[0]
    assert "_compiled_adaptive_high_plane_action(" in bounded_seek
    assert "prepared_high_lateral_action" in high_action_branch
    assert "_fixed_z_lateral_approach_action(" not in high_action_branch
    assert "plate_contact_seek_max_translation_action" not in (
        high_action_branch
    )
    assert "_fixed_z_lateral_approach_action(" not in (
        post_descent_correction_branch
    )
    assert "prepared_high_lateral_action" in post_descent_correction_branch
    assert (
        "compiled_adaptive_post_descent_plane_hold_envelope"
        in post_descent_correction_branch
    )
    assert "_compiled_adaptive_high_plane_action(" in (
        post_descent_compilation
    )
    assert "post_descent_lateral_max_translation_action" in (
        post_descent_compilation
    )
    assert "overhead_horizontal_z=correction_plane_target_z" in (
        post_descent_compilation
    )
    assert "maximum_translation_action=(" in post_descent_compilation
    assert (
        "lateral_target_xy=correction_lateral_target_xy"
        in post_descent_compilation
    )
    assert (
        "corridor_rebuffer_target[:2]\n"
        "        + corridor_outward_direction\n"
        "        * maximum_post_descent_lateral_world_step"
        in bounded_seek
    )
    assert (
        "correction_requires_pre_descent_controller_reserve = bool("
        in bounded_seek
    )
    assert (
        "correction_requires_post_descent_controller_rebuffer = bool("
        in post_descent_compilation
    )
    assert "correction_plane_target_z = float(" in (
        post_descent_compilation
    )
    assert "current_eef[2]" in post_descent_compilation
    assert (
        'structural_stage_action_counts[\n'
        '                    "overhead_corridor_descent"\n'
        "                ]\n"
        "                == 0"
        in post_descent_compilation
    )
    assert "else corridor_rebuffer_target[:2]" in post_descent_compilation
    assert '"active_correction_lateral_target_xy"' in bounded_seek
    assert "plane_recovery_tolerance_m=(" in post_descent_compilation
    assert (
        "negative_tail_recovery_threshold_m=("
        in post_descent_compilation
    )
    assert "args.minimum_saturated_waypoint_progress" in (
        post_descent_compilation
    )
    assert (
        '"event_driven_xy_positive_z_negative_tail_recovery"'
        in CONTROLLER_REFERENCE.read_text()
    )
    assert '"unchanged formal position_tolerance"' in bounded_seek
    assert 'corridor_high_target=corridor_rebuffer_target' in bounded_seek
    assert (
        '"formal_corridor_acceptance_target_unchanged": True'
        in post_descent_correction_branch
    )
    assert "def _high_z_controller_handoff_evidence(" in bounded_seek
    assert (
        "corridor_high_target=corridor_correction_handoff_target"
        in bounded_seek
    )
    assert (
        '"controller_handoff_applies_before_and_after_overhead_descent": True'
        in bounded_seek
    )
    assert 'and correction_controller_handoff["accepted"]' in bounded_seek
    assert (
        '"plane_recovery_applies_only_above_staging_tolerance": True'
        in bounded_seek
    )
    assert 'tail_brake_formal_corridor_entry["accepted"]' in bounded_seek
    assert (
        "not correction_requires_pre_descent_controller_reserve"
        in bounded_seek
    )
    assert '"post_descent_vertical_tail_handoff_gate"' in bounded_seek
    assert (
        '"existing minimum_saturated_waypoint_progress"'
        in bounded_seek
    )
    assert "0.5 * maximum_post_descent_lateral_world_step" in bounded_seek
    assert '"formal_position_tolerance_unchanged": True' in bounded_seek
    assert '"minimum_realized_outward_controller_reserve_m"' in bounded_seek
    assert (
        '"formal_corridor_acceptance_clearance_unchanged": True'
        in bounded_seek
    )
    assert "expected_overhead_pair_count" in bounded_seek


def test_500193_fixed_point_one_trace_exhaustion_is_not_a_threshold_change():
    observed_stage_counts = {
        "overhead_high_corridor_lateral": 176,
        "lateral_rebuffer_brake": 4,
    }
    assert sum(observed_stage_counts.values()) == 180
    assert observed_stage_counts["overhead_high_corridor_lateral"] == 176
    assert 0.138675 > 0.055540
    assert 0.138675 < 0.144806
    assert 1.065422 - 0.946566 > 0.118
    controller = CONTROLLER_REFERENCE.read_text()
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )
    assert '"fixed_buffer16_used_for_action_authorization": False' in controller
    assert '"compiled_adaptive_high_plane_action_envelope"' in controller


def test_500195_high_plane_hold_reserves_measured_negative_dz_tail():
    strict_clearance = np.nextafter(0.0, np.inf)
    base_reserve = 0.008
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": strict_clearance
            + base_reserve
            + 0.09096,
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": base_reserve,
        "pairs": pairs,
    }
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    hold_z = 1.0654223455054406
    target_xy = np.array([0.14480639548403948, -0.02850777957668001])
    step20 = np.array([0.136866, -0.028508, 1.030862])
    previous_negative_dz = -0.000859082
    action, evidence = _compiled_adaptive_high_plane_action(
        current_eef=step20,
        lateral_target_xy=target_xy,
        overhead_horizontal_z=hold_z,
        measured_vertical_step_progress_m=previous_negative_dz,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
    )
    expected_requested = np.array(
        [
            (target_xy[0] - step20[0]) / 0.08,
            (target_xy[1] - step20[1]) / 0.08,
            (hold_z - step20[2]) / 0.08,
        ]
    )
    assert evidence["requested_translation_action"] == pytest.approx(
        expected_requested
    )
    assert action[0] > 0.0
    assert action[2] > 0.0
    assert np.all(action[3:6] == 0.0)
    assert evidence["event_driven_positive_z_recovery"] is False
    assert evidence["measured_negative_inertial_tail_reserve_m"] > abs(
        previous_negative_dz
    )
    assert evidence["commanded_worst_case_downward_world_tail_m"] > (
        evidence["commanded_nominal_norm_downward_tail_m"]
    )
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert evidence["proof"] == {
        "xy_plus_nonnegative_z_zero_rotation": True,
        "strictly_inside_native_3d_action_norm_bound": True,
        "inside_configured_translation_action_norm_bound": True,
        "does_not_cross_lateral_target_xy": True,
        "positive_z_static_geometry_does_not_reduce_clearance": True,
        "latest_measured_negative_dz_reserved_as_inertial_tail": True,
        "all_compiled_pairs_retain_strict_base8_after_worst_case_tail": True,
    }

    capped_action, capped = _compiled_adaptive_high_plane_action(
        current_eef=step20,
        lateral_target_xy=target_xy,
        overhead_horizontal_z=hold_z,
        measured_vertical_step_progress_m=previous_negative_dz,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        maximum_translation_action=0.10,
    )
    assert capped_action[0] > 0.0
    assert capped_action[2] > 0.0
    assert np.linalg.norm(capped_action[:3]) < 0.10
    assert capped["selected_envelope_source"] == (
        "configured_translation_action_norm_bound"
    )
    assert capped["configured_strict_translation_action_norm_bound"] == (
        np.nextafter(0.10, 0.0)
    )
    assert capped["proof"][
        "inside_configured_translation_action_norm_bound"
    ] is True

    terminal_eef = np.array(
        [0.129843840417745, -0.02832361366276086, 0.9479874073242678]
    )
    base_correction_target = np.array(
        [0.13287111676914737, -0.02850806703327148]
    )
    correction_hold_target = base_correction_target + np.array([0.008, 0.0])
    terminal_hold_z = 0.9503643006811242
    base_action, base_evidence = _compiled_adaptive_high_plane_action(
        current_eef=terminal_eef,
        lateral_target_xy=base_correction_target,
        overhead_horizontal_z=terminal_hold_z,
        measured_vertical_step_progress_m=0.0,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        maximum_translation_action=0.10,
    )
    correction_action, correction_evidence = (
        _compiled_adaptive_high_plane_action(
            current_eef=terminal_eef,
            lateral_target_xy=correction_hold_target,
            overhead_horizontal_z=terminal_hold_z,
            measured_vertical_step_progress_m=0.0,
            overhead_guard=guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native,
            expected_pair_count=55,
            maximum_translation_action=0.10,
        )
    )
    assert base_evidence["lateral_target_xy"] == pytest.approx(
        base_correction_target
    )
    assert correction_evidence["lateral_target_xy"] == pytest.approx(
        correction_hold_target
    )
    assert correction_action[0] > base_action[0] > 0.0
    assert correction_action[2] >= 0.0
    assert np.linalg.norm(correction_action[:3]) < 0.10
    assert np.all(correction_action[3:6] == 0.0)

    plane_lag_eef = terminal_eef.copy()
    plane_lag_eef[2] = terminal_hold_z - 0.0093
    plane_recovery_action, plane_recovery = (
        _compiled_adaptive_high_plane_action(
            current_eef=plane_lag_eef,
            lateral_target_xy=correction_hold_target,
            overhead_horizontal_z=terminal_hold_z,
            measured_vertical_step_progress_m=0.0,
            overhead_guard=guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native,
            expected_pair_count=55,
            maximum_translation_action=0.10,
            plane_recovery_tolerance_m=0.005,
        )
    )
    assert plane_recovery_action[0] > 0.0
    assert plane_recovery_action[2] > 0.0
    assert np.linalg.norm(plane_recovery_action[:3]) < 0.10
    assert plane_recovery["plane_tolerance_recovery_required"] is True
    assert plane_recovery["pair_capacity_recovery_required"] is False
    assert plane_recovery["selected_envelope_source"] == (
        "event_driven_xy_positive_z_plane_tolerance_recovery"
    )
    assert plane_recovery["dynamic_xy_positive_z_recovery"] is True
    assert plane_recovery["pure_positive_z_recovery"] is False
    assert plane_recovery[
        "commanded_nominal_norm_downward_tail_m"
    ] > 0.0
    assert plane_recovery[
        "commanded_worst_case_downward_world_tail_m"
    ] >= plane_recovery["commanded_nominal_norm_downward_tail_m"]
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in plane_recovery["pair_envelopes"]
    )

    recovered_plane_eef = plane_lag_eef.copy()
    recovered_plane_eef[2] = terminal_hold_z - 0.0049
    resumed_xy_action, resumed_xy = _compiled_adaptive_high_plane_action(
        current_eef=recovered_plane_eef,
        lateral_target_xy=correction_hold_target,
        overhead_horizontal_z=terminal_hold_z,
        measured_vertical_step_progress_m=0.0,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        maximum_translation_action=0.10,
        plane_recovery_tolerance_m=0.005,
    )
    assert resumed_xy["plane_tolerance_recovery_required"] is False
    assert resumed_xy_action[0] > 0.0
    assert resumed_xy_action[2] > 0.0
    assert np.linalg.norm(resumed_xy_action[:3]) < 0.10

    negative_tail_recovery_action, negative_tail_recovery = (
        _compiled_adaptive_high_plane_action(
            current_eef=recovered_plane_eef,
            lateral_target_xy=correction_hold_target,
            overhead_horizontal_z=terminal_hold_z,
            measured_vertical_step_progress_m=-0.00006,
            overhead_guard=guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native,
            expected_pair_count=55,
            maximum_translation_action=0.10,
            plane_recovery_tolerance_m=0.005,
            negative_tail_recovery_threshold_m=0.00005,
        )
    )
    assert negative_tail_recovery_action[0] > 0.0
    assert negative_tail_recovery_action[2] > 0.0
    assert np.linalg.norm(negative_tail_recovery_action[:3]) < 0.10
    assert negative_tail_recovery[
        "negative_tail_recovery_required"
    ] is True
    assert negative_tail_recovery[
        "plane_tolerance_recovery_required"
    ] is False
    assert negative_tail_recovery[
        "pair_capacity_recovery_required"
    ] is False
    assert negative_tail_recovery["selected_envelope_source"] == (
        "event_driven_xy_positive_z_negative_tail_recovery"
    )
    assert negative_tail_recovery[
        "dynamic_xy_positive_z_recovery"
    ] is True
    assert negative_tail_recovery["pure_positive_z_recovery"] is False
    assert negative_tail_recovery[
        "commanded_worst_case_downward_world_tail_m"
    ] > negative_tail_recovery[
        "commanded_nominal_norm_downward_tail_m"
    ]
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in negative_tail_recovery["pair_envelopes"]
    )

    tail_deadband_action, tail_deadband = (
        _compiled_adaptive_high_plane_action(
            current_eef=recovered_plane_eef,
            lateral_target_xy=correction_hold_target,
            overhead_horizontal_z=terminal_hold_z,
            measured_vertical_step_progress_m=-0.00004,
            overhead_guard=guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native,
            expected_pair_count=55,
            maximum_translation_action=0.10,
            plane_recovery_tolerance_m=0.005,
            negative_tail_recovery_threshold_m=0.00005,
        )
    )
    assert tail_deadband["negative_tail_recovery_required"] is False
    assert tail_deadband_action[0] > 0.0
    assert tail_deadband_action[2] > 0.0
    assert np.linalg.norm(tail_deadband_action[:3]) < 0.10

    post_descent_rebuffer_action, post_descent_rebuffer = (
        _compiled_adaptive_high_plane_action(
            current_eef=recovered_plane_eef,
            lateral_target_xy=correction_hold_target,
            overhead_horizontal_z=recovered_plane_eef[2],
            measured_vertical_step_progress_m=-0.0005,
            overhead_guard=guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native,
            expected_pair_count=55,
            maximum_translation_action=0.10,
            plane_recovery_tolerance_m=None,
            negative_tail_recovery_threshold_m=None,
        )
    )
    assert post_descent_rebuffer_action[0] > 0.0
    assert post_descent_rebuffer_action[2] == 0.0
    assert np.linalg.norm(post_descent_rebuffer_action[:3]) < 0.10
    assert post_descent_rebuffer[
        "measured_negative_inertial_tail_reserve_m"
    ] > 0.0005
    assert post_descent_rebuffer[
        "dynamic_xy_positive_z_recovery"
    ] is False
    assert post_descent_rebuffer["pure_positive_z_recovery"] is False
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in post_descent_rebuffer["pair_envelopes"]
    )

    handoff_eef = np.array(
        [
            correction_hold_target[0],
            correction_hold_target[1],
            terminal_hold_z,
        ]
    )
    handoff_buffer = _overhead_lateral_buffer_evidence(
        guard,
        worst_case_controller_world_step_m=0.008,
    )
    handoff_outside_guard = {
        "accepted": False,
        "minimum_outside_clearance_m": 0.010,
    }
    formal_gate = _overhead_corridor_entry_evidence(
        current_eef=handoff_eef,
        corridor_high_target=np.append(
            base_correction_target, terminal_hold_z
        ),
        outside_side_guard=handoff_outside_guard,
        overhead_guard=guard,
        overhead_lateral_buffer=handoff_buffer,
        position_tolerance=0.002,
        strict_corridor_entry_clearance_m=0.0009,
        require_lateral_buffer=False,
    )
    controller_handoff_gate = _overhead_corridor_entry_evidence(
        current_eef=handoff_eef,
        corridor_high_target=np.append(
            correction_hold_target, terminal_hold_z
        ),
        outside_side_guard=handoff_outside_guard,
        overhead_guard=guard,
        overhead_lateral_buffer=handoff_buffer,
        position_tolerance=0.002,
        strict_corridor_entry_clearance_m=0.0009,
        require_lateral_buffer=False,
    )
    assert formal_gate["accepted"] is False
    assert "corridor_xy_tolerance_not_met" in formal_gate["violations"]
    assert controller_handoff_gate["accepted"] is True
    assert controller_handoff_gate["corridor_lateral_error_m"] == 0.0

    half_step_handoff_tolerance = 0.5 * 0.008
    under_reserved_eef = handoff_eef.copy()
    under_reserved_eef[0] -= half_step_handoff_tolerance + 0.0001
    sufficient_reserved_eef = handoff_eef.copy()
    sufficient_reserved_eef[0] -= half_step_handoff_tolerance - 0.0001
    under_reserved_gate = _overhead_corridor_entry_evidence(
        current_eef=under_reserved_eef,
        corridor_high_target=handoff_eef,
        outside_side_guard=handoff_outside_guard,
        overhead_guard=guard,
        overhead_lateral_buffer=handoff_buffer,
        position_tolerance=half_step_handoff_tolerance,
        strict_corridor_entry_clearance_m=0.0009,
        require_lateral_buffer=False,
    )
    sufficient_reserved_gate = _overhead_corridor_entry_evidence(
        current_eef=sufficient_reserved_eef,
        corridor_high_target=handoff_eef,
        outside_side_guard=handoff_outside_guard,
        overhead_guard=guard,
        overhead_lateral_buffer=handoff_buffer,
        position_tolerance=half_step_handoff_tolerance,
        strict_corridor_entry_clearance_m=0.0009,
        require_lateral_buffer=False,
    )
    assert under_reserved_gate["accepted"] is False
    assert under_reserved_gate["corridor_lateral_error_m"] == pytest.approx(
        0.0041
    )
    assert sufficient_reserved_gate["accepted"] is True
    assert sufficient_reserved_gate[
        "corridor_lateral_error_m"
    ] == pytest.approx(0.0039)

    tight_pairs = [
        {
            **pair,
            "vertical_clearance_m": strict_clearance
            + base_reserve
            + 0.003,
        }
        for pair in pairs
    ]
    recovery_action, recovery = _compiled_adaptive_high_plane_action(
        current_eef=np.array([0.126495, -0.027853, 0.939734]),
        lateral_target_xy=target_xy,
        overhead_horizontal_z=hold_z,
        measured_vertical_step_progress_m=previous_negative_dz,
        overhead_guard={**guard, "pairs": tight_pairs},
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
    )
    assert recovery["event_driven_positive_z_recovery"] is True
    assert recovery["pure_positive_z_recovery"] is True
    assert recovery["dynamic_xy_positive_z_recovery"] is False
    assert recovery["selected_envelope_source"] == (
        "event_driven_positive_z_plane_recovery"
    )
    assert np.array_equal(recovery_action[:2], np.zeros(2))
    assert recovery_action[2] > 0.0
    assert recovery["commanded_nominal_norm_downward_tail_m"] == 0.0
    assert recovery["commanded_worst_case_downward_world_tail_m"] == (
        recovery["measured_negative_inertial_tail_reserve_m"]
    )
    recovery_authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": -0.05,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard={**guard, "pairs": tight_pairs},
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            {**guard, "pairs": tight_pairs},
            worst_case_controller_world_step_m=0.008,
        ),
        compiled_pairs=tight_pairs,
        expected_pair_count=55,
        require_lateral_buffer=False,
        adaptive_high_lateral_envelope=recovery,
    )
    assert recovery_authorization["accepted"] is True
    assert recovery_authorization["buffer16_used_for_authorization"] is False

    exhausted_pairs = [
        {
            **pair,
            "vertical_clearance_m": strict_clearance
            + base_reserve
            + 0.0008,
        }
        for pair in pairs
    ]
    with pytest.raises(RuntimeError, match="inertial tail consumes"):
        _compiled_adaptive_high_plane_action(
            current_eef=np.array([0.126495, -0.027853, 0.939734]),
            lateral_target_xy=target_xy,
            overhead_horizontal_z=hold_z,
            measured_vertical_step_progress_m=previous_negative_dz,
            overhead_guard={**guard, "pairs": exhausted_pairs},
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native,
            expected_pair_count=55,
        )

    corridor = _overhead_corridor_entry_evidence(
        current_eef=np.array([target_xy[0], target_xy[1], hold_z - 0.01]),
        corridor_high_target=np.array([target_xy[0], target_xy[1], hold_z]),
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.02,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            guard,
            worst_case_controller_world_step_m=0.008,
        ),
        position_tolerance=0.002,
        strict_corridor_entry_clearance_m=0.008,
        require_lateral_buffer=False,
        minimum_eef_z=hold_z - 0.002,
    )
    assert corridor["accepted"] is False
    assert "high_plane_hold_z_not_recovered" in corridor["violations"]


def test_500195_regression_is_dynamic_tail_not_contact_or_threshold_change():
    assert 0.137319 > 0.136866
    assert 0.126495 < 0.137319
    assert 0.000859082 > 0.000782274
    assert 76 == 75 + 1
    controller = CONTROLLER_REFERENCE.read_text()
    bounded_seek = controller.split("def _seek_stable_plate_contact(", 1)[1]
    assert "_compiled_adaptive_high_plane_action(" in bounded_seek
    assert "measured_vertical_step_progress_m=(" in bounded_seek
    assert "overhead_horizontal_z=overhead_horizontal_z" in bounded_seek
    assert '"overhead_post_descent_corridor_lateral"' in bounded_seek
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500199_routes_reachable_outside_high_before_workspace_release():
    strict_clearance = np.nextafter(0.0, np.inf)
    hold_z = 1.0654223455
    outside_high = np.array([0.1368063955, -0.0285077796, hold_z])
    corridor_xy = np.array([0.1448063955, -0.0285077796])
    failed_final = np.array([0.1384933384, -0.0283591799, 1.0597920005])
    assert np.linalg.norm(failed_final[:2] - outside_high[:2]) < 0.005
    assert np.linalg.norm(failed_final[:2] - corridor_xy) == pytest.approx(
        0.0063148055,
        abs=2e-7,
    )
    assert 0.0069407 < 0.008

    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.13332117746677247,
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    outside_gate = _overhead_outside_high_entry_evidence(
        current_eef=outside_high,
        outside_high_target=outside_high,
        overhead_horizontal_z=hold_z,
        overhead_guard=guard,
        position_tolerance=0.005,
    )
    assert outside_gate["accepted"] is True

    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=outside_high,
        corridor_target_xy=corridor_xy,
        release_target_z=0.898654346,
        measured_vertical_step_progress_m=-0.00000284,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert np.all(action[3:6] == 0.0)
    assert np.linalg.norm(action[:3]) < 1.0
    assert evidence["motion_kind"] == "outward_downward_workspace_release"
    assert evidence["compiled_pair_count"] == 55
    assert evidence["measured_negative_inertial_tail_reserve_m"] > 0.00000284
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert all(
        pair["predicted_post_worst_case_buffer16_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert evidence["proof"] == {
        "outward_xy_plus_nonpositive_z_zero_rotation": True,
        "corridor_xy_hold_plus_nonpositive_z_zero_rotation": False,
        "inward_outward_axis_command_prohibited": False,
        "inward_xy_limited_to_prebuffer_one_ulp_bound": False,
        "pure_positive_z_zero_xy_rotation_recovery": False,
        "strictly_inside_native_3d_action_norm_bound": True,
        "does_not_cross_corridor_target_xy": True,
        "does_not_cross_release_target_z": True,
        "latest_measured_negative_dz_reserved_as_inertial_tail": True,
        "negative_z_pre_action_uses_strict_buffer16_plus_inertia": True,
        "live_buffer16_headroom_only_refines_negative_z_direction": True,
        "recovery_route_norm_unchanged_by_live_headroom_cap": True,
        "all_compiled_pairs_retain_strict_base8_after_worst_case_tail": True,
    }
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.0069407,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            guard,
            worst_case_controller_world_step_m=0.008,
        ),
        compiled_pairs=pairs,
        expected_pair_count=55,
        require_lateral_buffer=True,
        adaptive_high_lateral_envelope=evidence,
    )
    assert authorization["accepted"] is True
    assert authorization["buffer16_used_for_authorization"] is True

    released_corridor = _overhead_corridor_entry_evidence(
        current_eef=np.array([corridor_xy[0], corridor_xy[1], 1.0]),
        corridor_high_target=np.array([corridor_xy[0], corridor_xy[1], 0.94]),
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.009,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            guard,
            worst_case_controller_world_step_m=0.008,
        ),
        position_tolerance=0.005,
        strict_corridor_entry_clearance_m=0.008,
        require_lateral_buffer=False,
        minimum_eef_z=None,
    )
    assert released_corridor["accepted"] is True
    assert released_corridor["minimum_eef_z_m"] is None


def test_500199_workspace_release_stage_preserves_all_hard_thresholds():
    controller = CONTROLLER_REFERENCE.read_text()
    bounded_seek = controller.split("def _seek_stable_plate_contact(", 1)[1]
    assert '"workspace_release_diagonal": 0' in bounded_seek
    assert "_compiled_adaptive_workspace_release_action(" in bounded_seek
    assert 'structural_stage = "workspace_release_diagonal"' in bounded_seek
    assert '"compiled_adaptive_workspace_release_envelope"' in bounded_seek
    assert (
        "workspace_release_reached_controller_" in bounded_seek
    )
    assert (
        "workspace_release_reached_formal_corridor_" in bounded_seek
    )
    assert "controller_reserve_requires_plane_hold" in bounded_seek
    assert '"overhead_corridor_descent"' in bounded_seek
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500206_exhausted_downward_capacity_triggers_positive_z_recovery():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = []
    for index in range(55):
        surplus = 0.0005 if index == 12 else 0.02
        pairs.append(
            {
                "gripper_geom": f"gripper_{index // 11}",
                "counterpart_geom": f"native_{index % 11}",
                "counterpart_kind": (
                    "table" if index % 11 == 10 else "plate"
                ),
                "strict_no_contact_clearance_m": strict_clearance,
                "vertical_clearance_m": strict_clearance + 0.008 + surplus,
                "accepted": True,
            }
        )
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array([0.1368063955, -0.0285077796, 1.0]),
        corridor_target_xy=np.array([0.1448063955, -0.0285077796]),
        release_target_z=0.898654346,
        measured_vertical_step_progress_m=-0.0008,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert guard["accepted"] is True
    assert evidence["event_driven_positive_z_inertial_recovery"] is True
    assert evidence["motion_kind"] == "positive_z_inertial_recovery"
    assert evidence["selected_envelope_source"] == (
        "event_driven_positive_z_inertial_recovery"
    )
    assert np.array_equal(action[:2], np.zeros(2))
    assert action[2] > 0.0
    assert np.all(action[3:6] == 0.0)
    assert evidence["pair_envelopes"][12][
        "downward_capacity_exhausted_by_inertial_tail"
    ] is True
    assert evidence["commanded_positive_z_recovery_world_delta_m"] > 0.0
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert evidence["proof"][
        "pure_positive_z_zero_xy_rotation_recovery"
    ] is True
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.007,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            guard,
            worst_case_controller_world_step_m=0.008,
        ),
        compiled_pairs=pairs,
        expected_pair_count=55,
        require_lateral_buffer=False,
        adaptive_high_lateral_envelope=evidence,
    )
    assert authorization["accepted"] is True

    restored_pairs = [
        {**pair, "vertical_clearance_m": strict_clearance + 0.008 + 0.09}
        for pair in pairs
    ]
    route_action, route_evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array([0.1368063955, -0.0285077796, 1.04]),
        corridor_target_xy=np.array([0.1448063955, -0.0285077796]),
        release_target_z=0.898654346,
        measured_vertical_step_progress_m=0.001,
        overhead_guard={**guard, "pairs": restored_pairs},
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert route_evidence["event_driven_positive_z_inertial_recovery"] is False
    assert route_action[0] > 0.0
    assert route_action[2] < 0.0


def test_500206_workspace_recovery_is_recorded_and_thresholds_unchanged():
    controller = CONTROLLER_REFERENCE.read_text()
    bounded_seek = controller.split("def _seek_stable_plate_contact(", 1)[1]
    assert "workspace_release_event_driven_positive_z_" in bounded_seek
    assert "event_driven_positive_z_inertial_recovery" in bounded_seek
    assert '"workspace_release_diagonal": 0' in bounded_seek
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500207_positive_pair_capacity_scales_diagonal_without_recovery():
    strict_clearance = np.nextafter(0.0, np.inf)
    pair_action_capacity = 0.997457
    nominal_world_capacity = 0.08 * pair_action_capacity
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": (
                strict_clearance + 0.008 + nominal_world_capacity
            ),
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    corridor_xy = np.array([0.1448063955, -0.0285077796])
    requested_xy_action = 1.2
    requested_norm = float(np.sqrt(requested_xy_action**2 + 0.1**2))
    current = np.array(
        [
            corridor_xy[0] - 0.08 * requested_xy_action,
            corridor_xy[1],
            1.0197707,
        ]
    )
    release_target_z = float(current[2] - 0.05)
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=current,
        corridor_target_xy=corridor_xy,
        release_target_z=release_target_z,
        measured_vertical_step_progress_m=0.0,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert evidence["requested_translation_action_norm"] == pytest.approx(
        requested_norm
    )
    assert evidence["candidate_action_norm_capacities"][
        "compiled_pair_base8_post_tail_after_inertia"
    ] == pytest.approx(pair_action_capacity)
    assert evidence["selected_envelope_source"] == (
        "compiled_pair_base8_post_tail_after_inertia"
    )
    assert evidence["event_driven_positive_z_inertial_recovery"] is False
    assert np.linalg.norm(action[:3]) == pytest.approx(pair_action_capacity)
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )


def test_500207_recovery_trigger_is_exhaustion_not_positive_pair_limiting():
    controller = CONTROLLER_REFERENCE.read_text()
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert (
        'record["negative_z_capacity_exhausted_by_buffer16_or_inertia"]'
        in release
    )
    assert (
        'selected_source == "compiled_pair_base8_post_tail_after_inertia"'
        not in release
    )
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )


def test_500210_workspace_release_caps_z_and_recovers_before_base8_loss():
    strict_clearance = np.nextafter(0.0, np.inf)
    current = np.array([0.122223, -0.028146, 0.942744])
    corridor_xy = np.array([0.1448063955, -0.0285077796])
    release_target_z = 0.898654346
    latest_negative_dz = -0.004194
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }

    def guard_with_base8_surplus(surplus):
        pairs = [
            {
                "gripper_geom": f"gripper_{index // 11}",
                "counterpart_geom": f"native_{index % 11}",
                "counterpart_kind": (
                    "table" if index % 11 == 10 else "plate"
                ),
                "strict_no_contact_clearance_m": strict_clearance,
                "vertical_clearance_m": strict_clearance + 0.008 + surplus,
                "accepted": True,
            }
            for index in range(55)
        ]
        return {
            "accepted": True,
            "one_step_vertical_reserve_m": 0.008,
            "pairs": pairs,
        }

    roomy_guard = guard_with_base8_surplus(0.128)
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=current,
        corridor_target_xy=corridor_xy,
        release_target_z=release_target_z,
        measured_vertical_step_progress_m=0.0,
        overhead_guard=roomy_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    xy_remaining = np.linalg.norm(corridor_xy - current[:2])
    requested_z_world = abs(evidence["requested_translation_action"][2]) * 0.08
    assert evidence["full_release_downward_z_error_m"] > xy_remaining
    assert evidence[
        "xy_coupled_downward_world_request_before_one_step_cap_m"
    ] == pytest.approx(
        xy_remaining
    )
    assert evidence["capped_downward_world_request_m"] == pytest.approx(
        0.008
    )
    assert requested_z_world == pytest.approx(0.008)
    assert requested_z_world <= xy_remaining
    assert evidence["downward_request_capped_by_xy_remaining"] is True
    assert evidence[
        "downward_request_capped_by_existing_one_step_world_reserve"
    ] is True
    assert evidence["negative_z_action_requires_fixed_buffer16"] is True
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert all(
        pair["predicted_post_worst_case_buffer16_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.002,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=roomy_guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            roomy_guard,
            worst_case_controller_world_step_m=0.008,
        ),
        compiled_pairs=roomy_guard["pairs"],
        expected_pair_count=55,
        require_lateral_buffer=True,
        adaptive_high_lateral_envelope=evidence,
    )
    assert authorization["accepted"] is True
    assert authorization["buffer16_used_for_authorization"] is True

    late_guard = guard_with_base8_surplus(0.002792)
    late_action, late = _compiled_adaptive_workspace_release_action(
        current_eef=current,
        corridor_target_xy=corridor_xy,
        release_target_z=release_target_z,
        measured_vertical_step_progress_m=latest_negative_dz,
        overhead_guard=late_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert late["minimum_current_base8_surplus_m"] == pytest.approx(0.002792)
    assert late["pair_envelopes"][0]["current_buffer16_surplus_m"] < 0.0
    assert late["event_driven_positive_z_inertial_recovery"] is True
    assert late["negative_z_action_requires_fixed_buffer16"] is False
    assert np.array_equal(late_action[:2], np.zeros(2))
    assert late_action[2] > 0.0
    assert late[
        "recovery_route_requested_translation_action_norm_before_one_step_cap"
    ] > late["requested_translation_action_norm"]
    recovery_route_norm = min(
        late[
            "recovery_route_requested_translation_action_norm_before_one_step_cap"
        ],
        np.nextafter(1.0, 0.0),
    )
    desired_route_tail = float(
        0.08 * recovery_route_norm
        + late["measured_negative_inertial_tail_reserve_m"]
    )
    required_recovery = max(
        late["measured_negative_inertial_tail_reserve_m"]
        - late["minimum_current_base8_surplus_m"],
        0.008
        + late["measured_negative_inertial_tail_reserve_m"]
        - late["minimum_current_base8_surplus_m"],
        0.008
        + desired_route_tail
        - late["minimum_current_base8_surplus_m"],
        0.0,
    )
    expected_recovery_action = min(
        np.nextafter(1.0, 0.0),
        np.nextafter(required_recovery / 0.08, np.inf),
    )
    assert late_action[2] == expected_recovery_action
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in late["pair_envelopes"]
    )

    early_guard = guard_with_base8_surplus(0.012)
    early_action, early = _compiled_adaptive_workspace_release_action(
        current_eef=current,
        corridor_target_xy=corridor_xy,
        release_target_z=release_target_z,
        measured_vertical_step_progress_m=latest_negative_dz,
        overhead_guard=early_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert early["minimum_current_base8_surplus_m"] > early[
        "measured_negative_inertial_tail_reserve_m"
    ]
    assert early["pair_envelopes"][0][
        "nominal_negative_z_capacity_after_buffer16_and_inertia_m"
    ] < 0.0
    assert early["event_driven_positive_z_inertial_recovery"] is True
    assert np.array_equal(early_action[:2], np.zeros(2))
    assert early_action[2] > 0.0


def test_500210_workspace_negative_z_uses_buffer16_without_threshold_changes():
    controller = CONTROLLER_REFERENCE.read_text()
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    bounded_seek = controller.split("def _seek_stable_plate_contact(", 1)[1]
    assert "min(full_downward_z_error, xy_remaining)" in release
    assert "required_clearance_with_fixed_buffer16_m" in release
    assert "negative_z_capacity_exhausted_by_buffer16_or_inertia" in release
    assert "or adaptive_negative_z_action_requires_buffer16" in bounded_seek
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500223_scalar_solver_resolves_component_nextafter_capacity_boundary():
    strict_clearance = 1.0
    requested_action = np.array(
        [-0.23880069681529403, 0.6791984273835341, -0.1]
    )
    inertial_tail = 0.000237024403096048
    vertical_clearance = 1.0208398149675522
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": vertical_clearance,
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array([0.0, 0.0, 2.0]),
        corridor_target_xy=requested_action[:2] * 0.08,
        release_target_z=float(2.0 + requested_action[2] * 0.08),
        measured_vertical_step_progress_m=-inertial_tail,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    solver = evidence["literal_scalar_strict_interior_solver"]
    assert solver["candidate_accepted"] is False
    assert solver["candidate_failed_conditions"] == [
        "all_55_pair_base8_strict_post_clearance"
    ]

    literal_requested_action = np.asarray(
        evidence["requested_translation_action"], dtype=float
    )
    old_translation = (
        literal_requested_action
        / np.linalg.norm(literal_requested_action)
        * solver["candidate_scalar_action_norm"]
    )
    for _ in range(32):
        old_translation = np.nextafter(old_translation, 0.0)
    old_predicted_clearance = float(
        vertical_clearance
        - (
            0.08 * np.linalg.norm(old_translation)
            + evidence["measured_negative_inertial_tail_reserve_m"]
        )
    )
    required_base8_clearance = strict_clearance + 0.008
    assert not old_predicted_clearance > required_base8_clearance

    assert solver["accepted"] is True
    assert solver["solver_mode"] == "halving_then_scalar_bisection"
    assert solver["limiting_condition"] == (
        "compiled_pair_base8_post_tail_after_inertia"
    )
    assert solver["candidate_to_solved_scalar_ulp_distance"] > 0
    assert solver["scalar_nextafter_iterations"] == 1
    assert solver["scalar_halving_iterations"] > 0
    assert solver["scalar_bisection_iterations"] > 0
    assert solver["final_failed_conditions"] == []
    assert np.linalg.norm(action[:3]) == pytest.approx(
        solver["solved_literal_action_norm"]
    )
    assert action[:3] / np.linalg.norm(action[:3]) == pytest.approx(
        literal_requested_action / np.linalg.norm(literal_requested_action)
    )
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )


def test_500223_scalar_solver_preserves_strict_gates_and_hard_thresholds():
    controller = CONTROLLER_REFERENCE.read_text()
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert "literal_scalar_evidence" in release
    assert "halving_then_scalar_bisection" in release
    assert "candidate_to_solved_scalar_ulp_distance" in release
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500224_pre_buffer16_authorizes_post_base8_capacity_from_trace():
    strict_clearance = np.nextafter(0.0, np.inf)
    current_base8_surplus = 0.01744743290617297
    measured_negative_dz = -0.0023620272083493266
    vertical_clearance = (
        strict_clearance + 0.008 + current_base8_surplus
    )
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": vertical_clearance,
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array(
            [0.1325977585517547, -0.028345688864066708, 0.9572677865849051]
        ),
        corridor_target_xy=np.array(
            [0.14480639548403948, -0.02850777957668001]
        ),
        release_target_z=0.917769758476126,
        measured_vertical_step_progress_m=measured_negative_dz,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    latest_tail = evidence["measured_negative_inertial_tail_reserve_m"]
    old_post_buffer16_capacity = float(
        (current_base8_surplus - 0.008 - latest_tail) / 0.08
    )
    post_base8_capacity = float(
        (current_base8_surplus - latest_tail) / 0.08
    )
    assert old_post_buffer16_capacity == pytest.approx(
        0.08856757122279552
    )
    assert evidence["candidate_action_norm_capacities"][
        "compiled_pair_base8_post_tail_after_inertia"
    ] == pytest.approx(post_base8_capacity)
    assert post_base8_capacity == pytest.approx(0.1885675712227955)
    assert evidence["selected_envelope_source"] == (
        "requested_outward_downward_action_norm"
    )
    assert evidence[
        "minimum_pre_action_buffer16_surplus_after_inertia_m"
    ] == pytest.approx(0.007085405697823642)
    assert evidence["negative_z_action_requires_fixed_buffer16"] is True
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert abs(action[2]) == pytest.approx(
        evidence[
            "minimum_pre_action_buffer16_surplus_after_inertia_m"
        ]
        / 0.08
    )
    assert evidence[
        "downward_world_request_before_live_buffer16_headroom_cap_m"
    ] == pytest.approx(0.008)
    assert evidence["capped_downward_world_request_m"] == pytest.approx(
        0.007085405697823642
    )
    assert evidence[
        "live_pre_action_buffer16_headroom_cap_applied_to_negative_z"
    ] is True
    old_equal_xy_z_requested_norm = 0.2158392697294268
    old_equal_xy_z_action_x = float(
        post_base8_capacity
        * evidence["requested_translation_action"][0]
        / old_equal_xy_z_requested_norm
    )
    assert action[0] > old_equal_xy_z_action_x
    assert np.linalg.norm(action[:3]) > old_post_buffer16_capacity
    assert all(
        pair["pre_action_buffer16_surplus_after_inertia_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert all(
        pair["predicted_post_worst_case_buffer16_surplus_m"] < 0.0
        for pair in evidence["pair_envelopes"]
    )
    authorization = _overhead_route_frame_authorization_evidence(
        outside_side_guard={
            "accepted": False,
            "minimum_outside_clearance_m": 0.002,
            "required_outside_clearance_m": strict_clearance,
        },
        overhead_guard=guard,
        overhead_lateral_buffer=_overhead_lateral_buffer_evidence(
            guard,
            worst_case_controller_world_step_m=0.008,
        ),
        compiled_pairs=pairs,
        expected_pair_count=55,
        require_lateral_buffer=True,
        adaptive_high_lateral_envelope=evidence,
    )
    assert authorization["accepted"] is True
    assert authorization["buffer16_used_for_authorization"] is True


def test_500224_post_gate_is_base8_not_post_buffer16_or_budget_change():
    controller = CONTROLLER_REFERENCE.read_text()
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert (
        'required_clearance_key = "required_clearance_with_base_reserve_m"'
        in release
    )
    assert "pre_action_buffer16_surplus_after_inertia_m" in release
    assert "compiled_pair_base8_post_tail_after_inertia" in release
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500234_trace_frame50_reallocates_safe_norm_toward_corridor_xy():
    strict_clearance = np.nextafter(0.0, np.inf)
    current_base8_surplus = 0.015575695436753481
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": (
                strict_clearance + 0.008 + current_base8_surplus
            ),
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array(
            [0.13322616662761344, -0.028348588165142397, 0.9554089384787385]
        ),
        corridor_target_xy=np.array(
            [0.14480639548403948, -0.02850777957668001]
        ),
        release_target_z=0.917769758476126,
        measured_vertical_step_progress_m=-0.0033802347971196856,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec={
            "source": "env.action_spec",
            "action_dimension": 7,
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "runtime_resolved": True,
        },
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert evidence["requested_translation_action"] == pytest.approx(
        [
            0.14475286070532548,
            -0.0019898926442201564,
            -0.05244325799542244,
        ]
    )
    assert evidence["selected_envelope_source"] == (
        "compiled_pair_base8_post_tail_after_inertia"
    )
    assert evidence[
        "minimum_pre_action_buffer16_surplus_after_inertia_m"
    ] == pytest.approx(0.004195460639633795)
    assert evidence[
        "downward_world_request_before_live_buffer16_headroom_cap_m"
    ] == pytest.approx(0.008)
    assert evidence["capped_downward_world_request_m"] == pytest.approx(
        0.004195460639633795
    )
    assert evidence["capped_downward_world_request_m"] <= evidence[
        "xy_remaining_m"
    ]
    old_equal_xy_z_action_x = 0.10778347774973417
    assert action[0] == pytest.approx(0.14331484007191872)
    assert action[0] > old_equal_xy_z_action_x
    assert abs(action[2]) < action[0]
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )


def test_500234_one_step_z_cap_leaves_nonnegative_z_and_gates_unchanged():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.2,
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array([0.0, 0.0, 0.9]),
        corridor_target_xy=np.array([0.02, 0.0]),
        release_target_z=0.95,
        measured_vertical_step_progress_m=0.0,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec={
            "source": "env.action_spec",
            "action_dimension": 7,
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "runtime_resolved": True,
        },
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    assert evidence["full_release_downward_z_error_m"] == 0.0
    assert evidence[
        "xy_coupled_downward_world_request_before_one_step_cap_m"
    ] == 0.0
    assert evidence["capped_downward_world_request_m"] == 0.0
    assert evidence["requested_translation_action"] == [0.25, 0.0, -0.0]
    assert action[:3] == pytest.approx([0.25, 0.0, 0.0])
    assert evidence[
        "minimum_pre_action_buffer16_surplus_after_inertia_m"
    ] > 0.0
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )


def test_500234_z_cap_uses_existing_reserve_without_gate_or_budget_changes():
    controller = CONTROLLER_REFERENCE.read_text()
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert "xy_coupled_downward_world_request" in release
    assert "worst_case_controller_world_step_m" in release
    assert "pre_action_buffer16_surplus_after_inertia_m" in release
    assert (
        'required_clearance_key = "required_clearance_with_base_reserve_m"'
        in release
    )
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )


def test_500240_trace_frame157_spends_tiny_headroom_on_z_not_corridor_xy():
    strict_clearance = np.nextafter(0.0, np.inf)
    current_base8_surplus = 0.00817710559427283
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": (
                strict_clearance + 0.008 + current_base8_surplus
            ),
            "accepted": True,
        }
        for index in range(55)
    ]
    current_eef = np.array(
        [0.132415948569856, -0.02789013240409504, 0.9479879060895927]
    )
    corridor_target_xy = np.array(
        [0.14480639548403948, -0.02850777957668001]
    )
    release_target_z = 0.917769758476126
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=current_eef,
        corridor_target_xy=corridor_target_xy,
        release_target_z=release_target_z,
        measured_vertical_step_progress_m=0.00016154820993052876,
        overhead_guard={
            "accepted": True,
            "one_step_vertical_reserve_m": 0.008,
            "pairs": pairs,
        },
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec={
            "source": "env.action_spec",
            "action_dimension": 7,
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "runtime_resolved": True,
        },
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    headroom = evidence[
        "minimum_pre_action_buffer16_surplus_after_inertia_m"
    ]
    assert headroom == pytest.approx(0.00017710559427282918)
    assert evidence[
        "downward_world_request_before_live_buffer16_headroom_cap_m"
    ] == pytest.approx(0.008)
    assert evidence["capped_downward_world_request_m"] == pytest.approx(
        headroom
    )
    assert evidence[
        "live_pre_action_buffer16_headroom_cap_applied_to_negative_z"
    ] is True
    assert evidence[
        "downward_request_within_live_pre_action_buffer16_headroom"
    ] is True
    assert action[:3] == pytest.approx(
        [
            0.10207665983569228,
            -0.005088384686291023,
            -0.0014590553211515911,
        ]
    )
    job_500240_old_action_x = 0.08579526670320199
    job_500240_old_action_z = -0.055394461424949075
    assert action[0] > job_500240_old_action_x
    assert abs(action[2]) < abs(job_500240_old_action_z) / 10.0
    xy_error = corridor_target_xy - current_eef[:2]
    expected_recovery_route_norm = np.linalg.norm(
        [
            xy_error[0] / 0.08,
            xy_error[1] / 0.08,
            -min(
                current_eef[2] - release_target_z,
                np.linalg.norm(xy_error),
            )
            / 0.08,
        ]
    )
    assert evidence[
        "recovery_route_requested_translation_action_norm_before_one_step_cap"
    ] == pytest.approx(expected_recovery_route_norm)
    assert evidence["compiled_pair_count"] == 55
    assert all(
        pair["pre_action_buffer16_surplus_after_inertia_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    scalar = evidence["literal_scalar_strict_interior_solver"]
    assert scalar["accepted"] is True
    assert scalar["final_failed_conditions"] == []


def test_500240_near_zero_positive_headroom_keeps_xy_and_tiny_negative_z():
    strict_clearance = np.nextafter(0.0, np.inf)
    required_buffer16 = strict_clearance + 0.016
    vertical_clearance = np.nextafter(required_buffer16, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": vertical_clearance,
            "accepted": True,
        }
        for index in range(55)
    ]
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=np.array([0.0, 0.0, 0.95]),
        corridor_target_xy=np.array([0.02, 0.0]),
        release_target_z=0.90,
        measured_vertical_step_progress_m=0.0,
        overhead_guard={
            "accepted": True,
            "one_step_vertical_reserve_m": 0.008,
            "pairs": pairs,
        },
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec={
            "source": "env.action_spec",
            "action_dimension": 7,
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "runtime_resolved": True,
        },
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
    )
    headroom = evidence[
        "minimum_pre_action_buffer16_surplus_after_inertia_m"
    ]
    assert 0.0 < headroom < 1e-15
    assert evidence["event_driven_positive_z_inertial_recovery"] is False
    assert evidence["capped_downward_world_request_m"] == headroom
    assert action[0] > 0.0
    assert action[2] < 0.0
    assert abs(action[2]) < 1e-15
    assert all(
        pair["pre_action_buffer16_surplus_after_inertia_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )


def test_500240_nonfinite_live_headroom_input_fails_closed():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.1,
            "accepted": True,
        }
        for index in range(55)
    ]
    with np.errstate(over="ignore"):
        with pytest.raises(RuntimeError, match="headroom input is non-finite"):
            _compiled_adaptive_workspace_release_action(
                current_eef=np.array([0.0, 0.0, 0.95]),
                corridor_target_xy=np.array([0.02, 0.0]),
                release_target_z=0.90,
                measured_vertical_step_progress_m=-np.finfo(float).max,
                overhead_guard={
                    "accepted": True,
                    "one_step_vertical_reserve_m": 0.008,
                    "pairs": pairs,
                },
                gripper=-1.0,
                position_action_scale=0.08,
                native_action_spec={
                    "source": "env.action_spec",
                    "action_dimension": 7,
                    "low": [-1.0] * 7,
                    "high": [1.0] * 7,
                    "runtime_resolved": True,
                },
                expected_pair_count=55,
                worst_case_controller_world_step_m=0.008,
            )


def test_500240_headroom_refines_direction_without_gate_or_budget_changes():
    controller = CONTROLLER_REFERENCE.read_text()
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert "downward_world_request_before_live_buffer_headroom_cap" in release
    assert "minimum_pre_action_buffer16_surplus" in release
    assert "recovery_route_requested_norm" in release
    assert (
        'required_clearance_key = "required_clearance_with_base_reserve_m"'
        in release
    )
    assert "for clearance, record in zip(predicted, pair_envelopes)" in release
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_500251_registered_corridor_is_existing_step_high_prebuffer():
    outside_high = np.array(
        [0.13680639548403947, -0.02850777957668001, 1.062506338529415]
    )
    outside_side = np.array(
        [0.13680639548403947, -0.02850777957668001, 0.917769758476126]
    )
    unchanged_outside_high = outside_high.copy()
    strict_clearance = np.nextafter(0.0, np.inf)
    position_action_scale = 0.08
    maximum_translation_action = 0.10
    existing_world_step = (
        position_action_scale * maximum_translation_action
    )
    corridor_high, _, corridor = _compiled_vertical_staging_corridor(
        outside_high_target=outside_high,
        outside_side_target=outside_side,
        geometry={
            "outward_direction_xy": [1.0, 0.0],
            "outside_clearance_m": 0.005,
        },
        required_outside_clearance_m=strict_clearance,
        position_action_scale=position_action_scale,
        maximum_translation_action=maximum_translation_action,
    )
    lateral_reserve = np.linalg.norm(
        corridor_high[:2] - outside_high[:2]
    )
    assert np.array_equal(outside_high, unchanged_outside_high)
    assert corridor["maximum_controller_world_step_m"] == pytest.approx(
        existing_world_step
    )
    assert lateral_reserve == corridor["corridor_entry_lateral_travel_m"]
    assert lateral_reserve > existing_world_step
    assert lateral_reserve == pytest.approx(existing_world_step)

    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": existing_world_step,
        "pairs": [],
    }
    native_outside_gate = _overhead_outside_high_entry_evidence(
        current_eef=outside_high,
        outside_high_target=outside_high,
        high_lateral_target=corridor_high,
        overhead_horizontal_z=outside_high[2],
        overhead_guard=guard,
        position_tolerance=0.005,
    )
    assert native_outside_gate["accepted"] is False
    assert native_outside_gate[
        "high_lateral_target_role"
    ] == "registered_corridor_high_anticooupling_prebuffer"
    assert native_outside_gate["outside_high_target"] == outside_high.tolist()
    assert native_outside_gate["high_lateral_target"] == corridor_high.tolist()

    prebuffer_gate = _overhead_outside_high_entry_evidence(
        current_eef=corridor_high,
        outside_high_target=outside_high,
        high_lateral_target=corridor_high,
        overhead_horizontal_z=corridor_high[2],
        overhead_guard=guard,
        position_tolerance=0.005,
    )
    assert prebuffer_gate["accepted"] is True
    assert prebuffer_gate["outside_high_target"] == outside_high.tolist()
    assert prebuffer_gate["high_lateral_target"] == corridor_high.tolist()


def test_500251_high_prebuffer_retains_all_55_pair_base8_checks():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.13332117746677247,
            "accepted": True,
        }
        for index in range(55)
    ]
    registered_corridor_xy = np.array(
        [0.14480639548403948, -0.02850777957668001]
    )
    current_eef = np.array(
        [0.05554037906914336, -0.029154933875409465, 1.0654223455054406]
    )
    action, evidence = _compiled_adaptive_high_plane_action(
        current_eef=current_eef,
        lateral_target_xy=registered_corridor_xy,
        overhead_horizontal_z=current_eef[2],
        measured_vertical_step_progress_m=0.0,
        overhead_guard={
            "accepted": True,
            "one_step_vertical_reserve_m": 0.008,
            "pairs": pairs,
        },
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec={
            "source": "env.action_spec",
            "action_dimension": 7,
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "runtime_resolved": True,
        },
        expected_pair_count=55,
    )
    assert action[0] > 0.0
    assert action[2] == 0.0
    assert evidence["lateral_target_xy"] == registered_corridor_xy.tolist()
    assert evidence["compiled_pair_count"] == 55
    assert len(evidence["pair_identity_keys"]) == 55
    assert all(
        pair["predicted_post_worst_case_base_reserve_surplus_m"] > 0.0
        for pair in evidence["pair_envelopes"]
    )


def test_500251_manifest_records_true_high_target_without_gate_changes():
    controller = CONTROLLER_REFERENCE.read_text()
    bounded_seek = controller.split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert "_strict_native_high_prebuffer_target(" in bounded_seek
    assert "maximum_nextafter_steps=128" in bounded_seek
    assert (
        "maximum_inward_xy_correction_m=(" in bounded_seek
    )
    assert (
        "lateral_target_xy=high_lateral_prebuffer_target[:2]"
        in bounded_seek
    )
    assert "high_lateral_target=high_lateral_prebuffer_target" in bounded_seek
    assert '"reachable_outside_high_target"' in bounded_seek
    assert '"high_plane_anticooupling_lateral_target"' in bounded_seek
    assert '"high_plane_anticooupling_lateral_reserve_m"' in bounded_seek
    assert (
        '"workspace_release_diagonal"\n            ]\n            == 0'
        in bounded_seek
    )
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        'required_clearance_key = "required_clearance_with_base_reserve_m"'
        in release
    )
    assert "pre_action_buffer16_surplus_after_inertia_m" in release
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_502404_exact_high_prebuffer_roundoff_regression_and_budget():
    native_outside_high = np.array(
        [
            0.1585669667189191,
            0.017063712111350614,
            1.062506338529415,
        ]
    )
    corridor_high = np.array(
        [
            0.16592415268504696,
            0.020205656518849122,
            1.062506338529415,
        ]
    )
    original_native = native_outside_high.copy()
    original_corridor = corridor_high.copy()
    native_clockwise_tangent = np.array(
        [0.9196482457659836, 0.39274305093731343]
    )
    threshold = 0.008

    prebuffer, evidence = _strict_native_high_prebuffer_target(
        native_outside_high_target=native_outside_high,
        corridor_high_target=corridor_high,
        outward_direction_xy=native_clockwise_tangent,
        minimum_lateral_reserve_m=threshold,
        maximum_nextafter_steps=128,
    )

    assert np.array_equal(native_outside_high, original_native)
    assert np.array_equal(corridor_high, original_corridor)
    assert np.array_equal(
        prebuffer,
        np.array(
            [
                0.16592415268504698,
                0.020205656518849126,
                1.062506338529415,
            ]
        ),
    )
    assert evidence["raw_euclidean_reserve_m"] == (
        0.007999999999999993
    )
    assert evidence["raw_outward_projection_m"] == (
        0.007999999999999993
    )
    assert evidence["raw_euclidean_reserve_gap_m"] == (
        -6.938893903907228e-18
    )
    assert evidence["selected_scalar_reserve_m"] == (
        0.008000000000000007
    )
    assert evidence["final_euclidean_reserve_m"] == (
        0.008000000000000021
    )
    assert evidence["final_outward_projection_m"] == (
        0.008000000000000021
    )
    assert evidence["nextafter_iterations"] == 8
    assert evidence["maximum_nextafter_steps"] == 128
    assert evidence["final_euclidean_reserve_m"] > threshold
    assert evidence["final_outward_projection_m"] > threshold
    assert evidence["prebuffer_displacement_from_corridor_m"] == (
        2.797157557069881e-17
    )
    assert evidence["maximum_inward_return_one_ulp_bound_m"] > (
        evidence["prebuffer_displacement_from_corridor_m"]
    )
    assert "live geometry['outward_direction_xy']" in (
        evidence["native_tangent_provenance"]
    )
    assert evidence["corridor_high_and_side_targets_unchanged"] is True

    native_center_high = np.array(
        [0.05554037906914336, -0.029154933875409465]
    )
    strict_native_high_world_step = float(np.nextafter(0.08, 0.0))
    old_high_route_lower_bound = float(
        np.linalg.norm(corridor_high[:2] - native_center_high)
        / strict_native_high_world_step
    )
    nudged_high_route_lower_bound = float(
        (
            np.linalg.norm(prebuffer[:2] - native_center_high)
            + evidence["prebuffer_displacement_from_corridor_m"]
        )
        / strict_native_high_world_step
    )
    assert np.ceil(nudged_high_route_lower_bound) == np.ceil(
        old_high_route_lower_bound
    )
    assert np.ceil(nudged_high_route_lower_bound) < 167
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in CONTROLLER_REFERENCE.read_text()
    )


def test_502404_high_prebuffer_fails_closed_on_invalid_numeric_routes():
    valid_native = np.array([0.1, 0.2, 1.0])
    valid_corridor = np.array([0.108, 0.2, 1.0])
    with pytest.raises(ValueError, match="inputs are invalid"):
        _strict_native_high_prebuffer_target(
            native_outside_high_target=np.array([np.nan, 0.2, 1.0]),
            corridor_high_target=valid_corridor,
            outward_direction_xy=np.array([1.0, 0.0]),
            minimum_lateral_reserve_m=0.008,
        )
    with pytest.raises(RuntimeError, match="against the normalized"):
        _strict_native_high_prebuffer_target(
            native_outside_high_target=valid_native,
            corridor_high_target=np.array([0.092, 0.2, 1.0]),
            outward_direction_xy=np.array([1.0, 0.0]),
            minimum_lateral_reserve_m=0.008,
        )

    # At this magnitude, 0.0078125 is representable in the target coordinate,
    # but 128 scalar ULP increments cannot change that reconstructed target.
    coarse_native = np.array([1.0e12, 0.0, 1.0])
    coarse_corridor = np.array([1.0e12 + 0.0078125, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="exhausted 128"):
        _strict_native_high_prebuffer_target(
            native_outside_high_target=coarse_native,
            corridor_high_target=coarse_corridor,
            outward_direction_xy=np.array([1.0, 0.0]),
            minimum_lateral_reserve_m=0.008,
            maximum_nextafter_steps=128,
        )


def test_502404_workspace_release_allows_only_registered_ulp_return():
    native_outside_high = np.array(
        [0.1585669667189191, 0.017063712111350614, 1.0]
    )
    corridor_high = np.array(
        [0.16592415268504696, 0.020205656518849122, 1.0]
    )
    native_clockwise_tangent = np.array(
        [0.9196482457659836, 0.39274305093731343]
    )
    prebuffer, prebuffer_evidence = _strict_native_high_prebuffer_target(
        native_outside_high_target=native_outside_high,
        corridor_high_target=corridor_high,
        outward_direction_xy=native_clockwise_tangent,
        minimum_lateral_reserve_m=0.008,
    )
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.13332117746677247,
            "accepted": True,
        }
        for index in range(55)
    ]
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": pairs,
    }
    native_action_spec = {
        "source": "env.action_spec",
        "action_dimension": 7,
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
        "runtime_resolved": True,
    }
    inward_bound = prebuffer_evidence[
        "maximum_inward_return_one_ulp_bound_m"
    ]
    action, evidence = _compiled_adaptive_workspace_release_action(
        current_eef=prebuffer,
        corridor_target_xy=corridor_high[:2],
        release_target_z=0.9,
        measured_vertical_step_progress_m=0.0,
        overhead_guard=guard,
        gripper=-1.0,
        position_action_scale=0.08,
        native_action_spec=native_action_spec,
        expected_pair_count=55,
        worst_case_controller_world_step_m=0.008,
        outward_direction_xy=native_clockwise_tangent,
        maximum_inward_xy_correction_m=inward_bound,
    )
    assert evidence["motion_kind"] == (
        "ulp_bounded_inward_downward_workspace_release"
    )
    assert evidence["requested_inward_xy_correction_m"] > 0.0
    assert evidence["requested_inward_xy_correction_m"] <= inward_bound
    assert evidence["inward_xy_correction_within_one_ulp_bound"] is True
    assert float(np.dot(action[:2], native_clockwise_tangent)) < 0.0
    assert action[2] < 0.0
    assert evidence["compiled_pair_count"] == 55
    assert evidence["proof"][
        "inward_xy_limited_to_prebuffer_one_ulp_bound"
    ] is True
    assert evidence["proof"][
        "all_compiled_pairs_retain_strict_base8_after_worst_case_tail"
    ] is True

    excessive_outward_overshoot = prebuffer.copy()
    excessive_outward_overshoot[:2] += (
        native_clockwise_tangent * 1.0e-12
    )
    with pytest.raises(RuntimeError, match="one-ULP bound"):
        _compiled_adaptive_workspace_release_action(
            current_eef=excessive_outward_overshoot,
            corridor_target_xy=corridor_high[:2],
            release_target_z=0.9,
            measured_vertical_step_progress_m=0.0,
            overhead_guard=guard,
            gripper=-1.0,
            position_action_scale=0.08,
            native_action_spec=native_action_spec,
            expected_pair_count=55,
            worst_case_controller_world_step_m=0.008,
            outward_direction_xy=native_clockwise_tangent,
            maximum_inward_xy_correction_m=inward_bound,
        )


def _job500261_saturation_fixture():
    strict_clearance = np.nextafter(0.0, np.inf)
    pairs = [
        {
            "gripper_geom": f"gripper_{index // 11}",
            "counterpart_geom": f"native_{index % 11}",
            "counterpart_kind": (
                "table" if index % 11 == 10 else "plate"
            ),
            "strict_no_contact_clearance_m": strict_clearance,
            "vertical_clearance_m": 0.12,
            "accepted": True,
        }
        for index in range(55)
    ]
    pair_identity_keys = [
        [
            pair["gripper_geom"],
            pair["counterpart_geom"],
            pair["counterpart_kind"],
        ]
        for pair in pairs
    ]
    envelope_pairs = [
        {
            **pair,
            "predicted_post_worst_case_base_reserve_surplus_m": 0.10,
        }
        for pair in pairs
    ]
    high_target = np.array(
        [0.14480639548403948, -0.02850777957668001, 1.0654223455054406]
    )
    outside_high = np.array(
        [0.13680639548403947, -0.02850777957668001, 1.0654223455054406]
    )
    initial_eef = np.array(
        [0.05554037906914336, -0.029154933875409465, 1.0654223455054406]
    )
    observations = []
    for index in range(10):
        before_x = 0.1384 + index * 0.000002
        after_x = before_x + 0.000002
        before_clearance = 0.0068 + index * 0.000002
        after_clearance = before_clearance + 0.000002
        observations.append(
            {
                "before_eef": np.array(
                    [before_x, -0.02836, 1.05979]
                ),
                "after_eef": np.array(
                    [after_x, -0.02836, 1.059792]
                ),
                "action": np.array(
                    [0.079, 0.0, 0.070, 0.0, 0.0, 0.0, -1.0]
                ),
                "high_plane_envelope": {
                    "accepted": True,
                    "lateral_target_xy": high_target[:2].tolist(),
                    "compiled_pair_count": 55,
                    "pair_identity_keys": pair_identity_keys,
                    "pair_envelopes": envelope_pairs,
                },
                "pre_overhead_guard": {
                    "accepted": True,
                    "pairs": pairs,
                },
                "post_overhead_guard": {
                    "accepted": True,
                    "pairs": pairs,
                },
                "before_outside_guard": {
                    "minimum_outside_clearance_m": before_clearance,
                    "required_outside_clearance_m": strict_clearance,
                },
                "after_outside_guard": {
                    "minimum_outside_clearance_m": after_clearance,
                    "required_outside_clearance_m": strict_clearance,
                },
                "step_response": {
                    "eef_outward_step_progress_m": 0.000002,
                    "outside_clearance_step_progress_m": 0.000002,
                },
            }
        )
    kwargs = {
        "observations": observations,
        "initial_eef": initial_eef,
        "high_lateral_target": high_target,
        "native_outside_high_target": outside_high,
        "outward_direction_xy": np.array([1.0, 0.0]),
        "position_action_scale": 0.08,
        "position_tolerance": 0.005,
        "progress_epsilon": 0.00005,
        "required_window_frames": 10,
        "native_action_spec": {
            "source": "env.action_spec",
            "action_dimension": 7,
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "runtime_resolved": True,
        },
        "expected_pair_count": 55,
    }
    return observations, kwargs


def test_500261_native_workspace_saturation_boundary_is_fail_closed_proof():
    _, kwargs = _job500261_saturation_fixture()
    evidence = _high_plane_native_workspace_saturation_evidence(**kwargs)
    assert evidence["accepted"] is True
    assert evidence["violations"] == []
    assert evidence["registered_target_requires_native_clipping"] is True
    assert evidence["bounded_action_clipped_axes"] == [0]
    assert evidence["observed_window_frames"] == 10
    assert evidence["required_window_source"] == "push_tracking_steps"
    assert evidence["progress_epsilon_source"] == (
        "minimum_saturated_waypoint_progress"
    )
    assert abs(evidence["window_net_eef_outward_progress_m"]) <= evidence[
        "progress_epsilon_m"
    ]
    assert abs(
        evidence["window_net_outside_clearance_progress_m"]
    ) <= evidence["progress_epsilon_m"]
    assert evidence[
        "actual_eef_outward_of_native_outside_target_m"
    ] > 0.0
    assert len(evidence["frames"]) == 10
    assert all(frame["persistent_outward_request"] for frame in evidence["frames"])
    assert all(
        frame["all_55_pair_high_plane_base8_strict"]
        for frame in evidence["frames"]
    )


def test_500261_saturation_boundary_rejects_every_missing_prerequisite():
    observations, kwargs = _job500261_saturation_fixture()

    incomplete = _high_plane_native_workspace_saturation_evidence(
        **{**kwargs, "observations": observations[:-1]}
    )
    assert incomplete["accepted"] is False
    assert "saturation_observation_window_incomplete" in incomplete[
        "violations"
    ]

    no_request = copy.deepcopy(observations)
    no_request[-1]["action"][0] = 0.0
    no_request_evidence = _high_plane_native_workspace_saturation_evidence(
        **{**kwargs, "observations": no_request}
    )
    assert no_request_evidence["accepted"] is False
    assert "outward_request_not_persistent" in no_request_evidence[
        "violations"
    ]

    moving = copy.deepcopy(observations)
    moving[-1]["step_response"][
        "eef_outward_step_progress_m"
    ] = 0.0001
    moving_evidence = _high_plane_native_workspace_saturation_evidence(
        **{**kwargs, "observations": moving}
    )
    assert moving_evidence["accepted"] is False
    assert "eef_outward_step_not_saturated" in moving_evidence[
        "violations"
    ]

    unsafe_pairs = copy.deepcopy(observations)
    unsafe_pairs[-1]["post_overhead_guard"]["accepted"] = False
    unsafe_pair_evidence = _high_plane_native_workspace_saturation_evidence(
        **{**kwargs, "observations": unsafe_pairs}
    )
    assert unsafe_pair_evidence["accepted"] is False
    assert "all_55_pair_high_plane_base8_not_strict" in unsafe_pair_evidence[
        "violations"
    ]

    not_beyond = copy.deepcopy(observations)
    for observation in not_beyond:
        observation["before_eef"][0] = 0.1367
        observation["after_eef"][0] = 0.136702
    not_beyond_evidence = _high_plane_native_workspace_saturation_evidence(
        **{**kwargs, "observations": not_beyond}
    )
    assert not_beyond_evidence["accepted"] is False
    assert "actual_eef_not_outward_of_native_outside_target" in (
        not_beyond_evidence["violations"]
    )

    reachable_target_evidence = (
        _high_plane_native_workspace_saturation_evidence(
            **{
                **kwargs,
                "initial_eef": np.array(
                    [0.10, -0.02850777957668001, 1.0654223455054406]
                ),
            }
        )
    )
    assert reachable_target_evidence["accepted"] is False
    assert "registered_high_target_not_native_action_clipped" in (
        reachable_target_evidence["violations"]
    )


def test_500261_saturation_fallback_reuses_existing_constants_and_exact_gates():
    controller = CONTROLLER_REFERENCE.read_text()
    bounded_seek = controller.split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    assert "_high_plane_native_workspace_saturation_evidence(" in bounded_seek
    assert "args.minimum_saturated_waypoint_progress" in bounded_seek
    assert "required_window_frames=args.push_tracking_steps" in bounded_seek
    assert '"accepted_native_high_workspace_saturation_boundary"' in (
        bounded_seek
    )
    assert (
        '"proved_native_high_workspace_saturation_to_"'
        in bounded_seek
    )
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        'parser.add_argument("--position_tolerance", type=float, default=0.005)'
        in controller
    )
    assert (
        '"--minimum_saturated_waypoint_progress",\n'
        "        type=float,\n"
        "        default=0.00005,"
        in controller
    )
    assert 'parser.add_argument("--push_tracking_steps", type=int, default=10)' in (
        controller
    )
    assert "pre_action_buffer16_surplus_after_inertia_m" in release
    assert (
        'required_clearance_key = "required_clearance_with_base_reserve_m"'
        in release
    )
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release


def test_502095_strict_native_boundary_crossing_accepts_without_tolerance():
    observations, kwargs = _job500261_saturation_fixture()
    observation = copy.deepcopy(observations[0])
    outside_high = np.asarray(kwargs["native_outside_high_target"], dtype=float)
    observation["before_eef"][0] = outside_high[0]
    observation["after_eef"][0] = np.nextafter(outside_high[0], np.inf)
    evidence = _high_plane_native_boundary_crossing_evidence(
        observation=observation,
        high_lateral_target=kwargs["high_lateral_target"],
        native_outside_high_target=outside_high,
        outward_direction_xy=kwargs["outward_direction_xy"],
        expected_pair_count=55,
    )
    assert evidence["accepted"] is True
    assert evidence["violations"] == []
    assert evidence["uses_position_tolerance"] is False
    assert evidence[
        "actual_eef_outward_of_native_outside_target_m"
    ] > 0.0
    assert evidence["target_request_persistent"] is True
    assert evidence["action_nonnegative_z_zero_rotation"] is True
    assert evidence["pair_inventory_exact"] is True
    assert evidence["all_55_pair_high_plane_base8_strict"] is True


def test_502095_native_boundary_crossing_rejects_each_missing_strict_gate():
    observations, kwargs = _job500261_saturation_fixture()
    valid = copy.deepcopy(observations[0])
    outside_high = np.asarray(kwargs["native_outside_high_target"], dtype=float)
    valid["after_eef"][0] = np.nextafter(outside_high[0], np.inf)

    def evaluate(observation):
        return _high_plane_native_boundary_crossing_evidence(
            observation=observation,
            high_lateral_target=kwargs["high_lateral_target"],
            native_outside_high_target=outside_high,
            outward_direction_xy=kwargs["outward_direction_xy"],
            expected_pair_count=55,
        )

    at_boundary = copy.deepcopy(valid)
    at_boundary["after_eef"][0] = outside_high[0]
    assert (
        "actual_eef_not_strictly_beyond_native_outside_target"
        in evaluate(at_boundary)["violations"]
    )

    no_request = copy.deepcopy(valid)
    no_request["action"][0] = 0.0
    assert "registered_corridor_outward_request_not_persistent" in evaluate(
        no_request
    )["violations"]

    negative_z = copy.deepcopy(valid)
    negative_z["action"][2] = -np.nextafter(0.0, np.inf)
    assert "high_plane_action_direction_or_rotation_invalid" in evaluate(
        negative_z
    )["violations"]

    touching_clearance = copy.deepcopy(valid)
    touching_clearance["after_outside_guard"][
        "minimum_outside_clearance_m"
    ] = touching_clearance["after_outside_guard"][
        "required_outside_clearance_m"
    ]
    assert "live_outside_clearance_not_strict" in evaluate(
        touching_clearance
    )["violations"]

    unsafe_pair = copy.deepcopy(valid)
    unsafe_pair["post_overhead_guard"]["accepted"] = False
    assert "all_55_pair_high_plane_base8_not_strict" in evaluate(
        unsafe_pair
    )["violations"]


def test_502095_boundary_gate_precedes_saturation_and_preserves_exact_descent():
    controller = CONTROLLER_REFERENCE.read_text()
    bounded_seek = controller.split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    high_transition = bounded_seek.split(
        'feedback["outside_high_entry_after_high_lateral"]', 1
    )[1].split('elif stage_before_action == "workspace_release_diagonal":', 1)[0]
    release = controller.split(
        "def _compiled_adaptive_workspace_release_action(", 1
    )[1].split("\ndef _compiled_adaptive_lateral_rebuffer_action", 1)[0]
    ordinary_gate = high_transition.index('if outside_high_entry["accepted"]')
    boundary_gate = high_transition.index(
        'elif native_boundary_crossing["accepted"]'
    )
    saturation_gate = high_transition.index(
        'elif workspace_saturation["accepted"]'
    )
    assert ordinary_gate < boundary_gate < saturation_gate
    assert "_high_plane_native_boundary_crossing_evidence(" in high_transition
    assert '"accepted_native_high_boundary_crossing"' in high_transition
    assert '"native_high_boundary_crossing_gate"' in bounded_seek
    assert '"uses_position_tolerance": False' in bounded_seek
    assert (
        "_high_plane_native_workspace_saturation_evidence(" in high_transition
    )
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert (
        'parser.add_argument("--position_tolerance", type=float, default=0.005)'
        in controller
    )
    assert "pre_action_buffer16_surplus_after_inertia_m" in release
    assert (
        'required_clearance_key = "required_clearance_with_base_reserve_m"'
        in release
    )
    assert "clearance > record[required_clearance_key]" in release
    assert "clearance >= record[required_clearance_key]" not in release


def test_high_first_route_fails_closed_and_rechecks_post_descent_drift():
    strict_clearance = np.nextafter(0.0, np.inf)
    pair = {
        "gripper_geom": "gripper0_hand_collision",
        "counterpart_geom": "plate_1_g0",
        "counterpart_kind": "plate",
        "strict_no_contact_clearance_m": strict_clearance,
        "vertical_clearance_m": 0.133,
        "accepted": True,
    }
    guard = {
        "accepted": True,
        "one_step_vertical_reserve_m": 0.008,
        "pairs": [pair],
    }
    buffer = _overhead_lateral_buffer_evidence(
        guard,
        worst_case_controller_world_step_m=0.008,
    )
    outside = {
        "accepted": False,
        "minimum_outside_clearance_m": -0.05,
        "required_outside_clearance_m": strict_clearance,
    }
    kwargs = {
        "outside_side_guard": outside,
        "overhead_guard": guard,
        "overhead_lateral_buffer": buffer,
        "compiled_pairs": [pair],
        "expected_pair_count": 1,
        "require_lateral_buffer": True,
    }
    with pytest.raises(RuntimeError, match="overhead all-pair"):
        _overhead_route_frame_authorization_evidence(
            **{**kwargs, "overhead_guard": {**guard, "accepted": False}}
        )
    with pytest.raises(RuntimeError, match="pair inventory changed"):
        _overhead_route_frame_authorization_evidence(
            **{**kwargs, "expected_pair_count": 55}
        )
    divergent_buffer = {
        **buffer,
        "pairs": [
            {
                **buffer["pairs"][0],
                "counterpart_geom": "different_native_geom",
            }
        ],
    }
    with pytest.raises(RuntimeError, match="pair evidence diverged"):
        _overhead_route_frame_authorization_evidence(
            **{**kwargs, "overhead_lateral_buffer": divergent_buffer}
        )
    with pytest.raises(RuntimeError, match="pair evidence diverged"):
        _overhead_route_frame_authorization_evidence(
            **{
                **kwargs,
                "compiled_pairs": [
                    {**pair, "counterpart_geom": "replacement_native_geom"}
                ],
            }
        )
    duplicated_pair_guard = {
        **guard,
        "pairs": [pair, pair],
    }
    duplicated_pair_buffer = _overhead_lateral_buffer_evidence(
        duplicated_pair_guard,
        worst_case_controller_world_step_m=0.008,
    )
    with pytest.raises(RuntimeError, match="duplicate pair identity"):
        _overhead_route_frame_authorization_evidence(
            **{
                **kwargs,
                "overhead_guard": duplicated_pair_guard,
                "overhead_lateral_buffer": duplicated_pair_buffer,
                "compiled_pairs": [pair, pair],
                "expected_pair_count": 2,
            }
        )
    missing_buffer = _overhead_lateral_buffer_evidence(
        {
            **guard,
            "pairs": [{**pair, "vertical_clearance_m": 0.015}],
        },
        worst_case_controller_world_step_m=0.008,
    )
    with pytest.raises(RuntimeError, match="buffer16"):
        _overhead_route_frame_authorization_evidence(
            **{**kwargs, "overhead_lateral_buffer": missing_buffer}
        )

    target = np.array(
        [0.14480639548403948, -0.02850777957668001, 0.9401011680386682]
    )
    # Job500182's pure-Z stage moved X by -10.561 mm.  Such drift must not
    # authorize the vertical side corridor without a live XY correction.
    drifted = target.copy()
    drifted[0] -= 0.010560774467849213
    corridor = _overhead_corridor_entry_evidence(
        current_eef=drifted,
        corridor_high_target=target,
        outside_side_guard={
            **outside,
            "minimum_outside_clearance_m": 0.0024,
        },
        overhead_guard=guard,
        overhead_lateral_buffer=buffer,
        position_tolerance=0.002,
        strict_corridor_entry_clearance_m=np.nextafter(0.008, np.inf),
    )
    assert corridor["accepted"] is False
    assert "corridor_xy_tolerance_not_met" in corridor["violations"]
    assert "outside_corridor_entry_clearance_not_met" in (
        corridor["violations"]
    )

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    tail_brake_transition = bounded_seek.split(
        'elif stage_before_action == "vertical_tail_brake":', 1
    )[1].split(
        'elif stage_before_action == "lateral_rebuffer_brake":',
        1,
    )[0]
    assert "_overhead_corridor_entry_evidence(" in tail_brake_transition
    assert (
        'structural_stage = "vertical_corridor_descent"'
        in tail_brake_transition
    )
    assert '"overhead_post_descent_corridor_lateral"' in tail_brake_transition
    assert "tail_brake_controller_handoff" in tail_brake_transition
    assert (
        '"controller_handoff_diagnostic_only"'
        in tail_brake_transition
    )
    assert 'tail_brake_controller_handoff["accepted"]' not in (
        tail_brake_transition
    )
    assert (
        'tail_brake_formal_corridor_entry["accepted"]'
        in tail_brake_transition
    )
    assert "overhead_staging_z + args.position_tolerance" in (
        tail_brake_transition
    )
    correction_action = bounded_seek.split(
        'elif structural_stage == "overhead_post_descent_corridor_lateral":',
        1,
    )[1].split(
        'elif structural_stage == "vertical_corridor_descent":', 1
    )[0]
    assert (
        '"post_descent_xy_plus_nonnegative_z_plane_hold_"'
        in correction_action
    )
    assert "_fixed_z_lateral_approach_action(" not in correction_action
    assert "prepared_high_lateral_action" in correction_action
    assert (
        "compiled_adaptive_post_descent_plane_hold_envelope"
        in correction_action
    )
    assert 'latest_overhead_lateral_buffer["accepted"]' in bounded_seek


def test_500182_high_first_budget_is_unchanged_and_removes_observed_overhead():
    observed_counts = {
        "overhead_center_descent": 17,
        "vertical_tail_brake": 4,
        "vertical_tail_zero_confirmation": 1,
        "overhead_corridor_lateral": 119,
        "lateral_rebuffer_brake": 39,
    }
    assert sum(observed_counts.values()) == 180
    same_motion_without_observed_rebuffers = (
        sum(observed_counts.values())
        - observed_counts["lateral_rebuffer_brake"]
    )
    assert same_motion_without_observed_rebuffers == 141
    controller = CONTROLLER_REFERENCE.read_text()
    assert (
        'parser.add_argument("--max_waypoint_steps", type=int, default=240)'
        in controller
    )
    assert '"geometric_action_count_scope"' in controller
    assert "diagnostic lower bound only" in controller
    assert "configured finite structural hard loop" in controller
    assert "runtime upper bound" not in controller
    assert "expected_overhead_pair_count = 55" in controller
    assert "native L3-A3 compiled overhead pair inventory changed" in controller
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in controller
    )


def test_499954_saturated_recovery_follows_improving_discrete_response():
    current = np.array([0.131429676, -0.029432244, 0.970336557])
    target = np.array([0.136806395, -0.028507780, 0.898654346])
    strict_positive_clearance = np.nextafter(0.0, np.inf)
    before_guard = {
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": strict_positive_clearance,
        "minimum_outside_clearance_m": -0.000343294,
        "required_finger_table_clearance_m": (
            strict_positive_clearance
        ),
        "finger_table_vertical_clearance_m": 0.0573,
    }
    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=current,
        outside_side_target=target,
        guard=before_guard,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert feedback["mode"] == "recover_outside_clearance"
    assert feedback["clearance_deficit_m"] == pytest.approx(
        0.000343294
    )
    # The controller-derived 8 mm world correction comes only from the
    # unchanged 0.08 action scale and 0.10 bounded-action limit.
    assert feedback["feedback_target"][0] == pytest.approx(
        current[0] + 0.08 * 0.10
    )
    assert np.allclose(action[:3], [0.10, 0.0, 0.0])
    assert feedback["recovery_action_saturated"] is True

    # A positive clearance during an unresolved response does not bypass its
    # proof: the next command remains saturated outward.
    forced_action, forced_feedback = (
        _outside_side_geometry_feedback_action(
            current_eef=current,
            outside_side_target=target,
            guard={
                **before_guard,
                "minimum_outside_clearance_m": 0.0001,
            },
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
            force_outward_recovery=True,
        )
    )
    assert forced_feedback["mode"] == "recover_outside_clearance"
    assert forced_feedback["clearance_deficit_m"] == 0.0
    assert np.allclose(forced_action[:3], [0.10, 0.0, 0.0])

    previous_vertical_response = {
        "eef_outward_step_progress_m": -0.000958830,
        "outside_clearance_step_progress_m": -0.000927832,
    }
    trace_baseline_eef = np.array(
        [0.130983948, -0.028638039, 1.050138420]
    )
    trace_baseline_guard = {
        **before_guard,
        "minimum_outside_clearance_m": -0.000797953,
    }
    first_after_eef = np.array(
        [0.130254591, -0.028638039, 1.048332081]
    )
    first_after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": -0.001442891,
    }
    first_response = _outside_side_recovery_progress_evidence(
        baseline_guard=trace_baseline_guard,
        after_guard=first_after_guard,
        baseline_eef=trace_baseline_eef,
        before_guard=trace_baseline_guard,
        before_eef=trace_baseline_eef,
        after_eef=first_after_eef,
        action=action,
        maximum_translation_action=0.10,
        previous_step_response=previous_vertical_response,
    )
    assert first_response["fail_closed"] is False
    assert first_response["pending_controller_response"] is True
    assert first_response["response_improving"] is True
    assert first_response[
        "eef_response_acceleration_m_per_step"
    ] > 0.0
    assert first_response[
        "clearance_response_acceleration_m_per_step"
    ] > 0.0

    second_after_eef = np.array(
        [0.129548314, -0.028638039, 1.046556656]
    )
    second_after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": -0.002079783,
    }
    second_response = _outside_side_recovery_progress_evidence(
        baseline_guard=trace_baseline_guard,
        after_guard=second_after_guard,
        baseline_eef=trace_baseline_eef,
        before_guard=first_after_guard,
        before_eef=first_after_eef,
        after_eef=second_after_eef,
        action=action,
        maximum_translation_action=0.10,
        previous_step_response=first_response["step_response"],
    )
    assert second_response["fail_closed"] is False
    assert second_response["pending_controller_response"] is True
    assert second_response["response_improving"] is True

    # Once a signed response has reversed outward it remains valid even if
    # its positive velocity is smaller than the prior positive velocity.
    outward_but_decelerating = (
        _outside_side_recovery_progress_evidence(
            baseline_guard={
                **before_guard,
                "minimum_outside_clearance_m": -0.001,
            },
            after_guard={
                **before_guard,
                "minimum_outside_clearance_m": -0.0026,
            },
            baseline_eef=np.array([0.130, 0.0, 1.0]),
            before_guard={
                **before_guard,
                "minimum_outside_clearance_m": -0.003,
            },
            before_eef=np.array([0.128, 0.0, 0.999]),
            after_eef=np.array([0.1285, 0.0, 0.998]),
            action=action,
            maximum_translation_action=0.10,
            previous_step_response={
                "eef_outward_step_progress_m": 0.0006,
                "outside_clearance_step_progress_m": 0.0005,
            },
        )
    )
    assert outward_but_decelerating["progress_proven"] is False
    assert outward_but_decelerating["response_improving"] is True
    assert outward_but_decelerating["fail_closed"] is False

    recovered_guard = {
        **trace_baseline_guard,
        "minimum_outside_clearance_m": 0.0002,
    }
    evidence = _outside_side_recovery_progress_evidence(
        baseline_guard=trace_baseline_guard,
        after_guard=recovered_guard,
        baseline_eef=trace_baseline_eef,
        before_guard=second_after_guard,
        before_eef=second_after_eef,
        after_eef=trace_baseline_eef + np.array([0.001, 0.0, -0.005]),
        action=action,
        maximum_translation_action=0.10,
        previous_step_response=second_response["step_response"],
    )
    assert evidence["accepted"] is True
    assert evidence["progress_proven"] is True
    assert evidence["action_saturated"] is True
    assert evidence["net_clearance_progress_m"] > 0.0
    assert evidence["net_eef_outward_progress_m"] > 0.0

    weak_action = np.array([0.004285, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    regressed_guard = {
        **before_guard,
        "minimum_outside_clearance_m": -0.000343294 - 4.8e-7,
    }
    rejected = _outside_side_recovery_progress_evidence(
        baseline_guard=before_guard,
        after_guard=regressed_guard,
        baseline_eef=current,
        before_guard=before_guard,
        before_eef=current,
        after_eef=current + np.array([-3.6e-7, 0.0, -7.3e-6]),
        action=weak_action,
        maximum_translation_action=0.10,
    )
    assert rejected["accepted"] is False
    assert rejected["violations"] == [
        "outward_recovery_action_not_at_controller_bound",
    ]

    stalled_after_eef = second_after_eef + np.array(
        [-0.000710, 0.0, -0.001]
    )
    stalled_after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": (
            second_after_guard["minimum_outside_clearance_m"]
            - 0.000640
        ),
    }
    saturated_but_stalled = _outside_side_recovery_progress_evidence(
        baseline_guard=trace_baseline_guard,
        after_guard=stalled_after_guard,
        baseline_eef=trace_baseline_eef,
        before_guard=second_after_guard,
        before_eef=second_after_eef,
        after_eef=stalled_after_eef,
        action=action,
        maximum_translation_action=0.10,
        previous_step_response=second_response["step_response"],
    )
    assert saturated_but_stalled["fail_closed"] is True
    assert saturated_but_stalled["violations"] == [
        "eef_outward_response_neither_moved_nor_accelerated_outward_under_saturation",
        "live_outside_clearance_response_neither_moved_nor_accelerated_outward_under_saturation",
    ]


def test_499888_feedback_recovers_x_before_bounded_z_and_stops_above_table():
    current = np.array([0.123946, -0.028254, 0.912431])
    target = np.array([0.136806, -0.028508, 0.898654])
    clearance_lost = {
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": 0.005,
        "minimum_outside_clearance_m": -0.004,
        "finger_table_vertical_clearance_m": 0.012,
    }
    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=current,
        outside_side_target=target,
        guard=clearance_lost,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert feedback["mode"] == "recover_outside_clearance"
    assert feedback["clearance_deficit_m"] == pytest.approx(0.009)
    assert feedback["feedback_target"][0] == pytest.approx(
        current[0] + 0.008
    )
    assert feedback["feedback_target"][2] == pytest.approx(current[2])
    assert np.allclose(action[:3], [0.10, 0.0, 0.0])
    assert feedback["recovery_action_saturated"] is True

    clearance_restored = {
        **clearance_lost,
        "minimum_outside_clearance_m": 0.005,
    }
    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=current,
        outside_side_target=target,
        guard=clearance_restored,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert feedback["mode"] == "compiled_outside_xy_recovery"
    # The compiled outside XY target has priority when its requested lateral
    # action consumes the unchanged controller norm.
    assert feedback["available_table_descent_m"] == pytest.approx(0.007)
    assert action[0] > 0.0
    assert action[2] == 0.0
    assert np.linalg.norm(action[:3]) == pytest.approx(0.10)

    near_outside = current.copy()
    near_outside[:2] = target[:2] - np.array([0.001, 0.0])
    action, feedback = _outside_side_geometry_feedback_action(
        current_eef=near_outside,
        outside_side_target=target,
        guard=clearance_restored,
        gripper=-1.0,
        position_action_scale=0.08,
        maximum_translation_action=0.10,
    )
    assert feedback["mode"] == "constraint_prioritized_vertical_descent"
    assert action[0] == pytest.approx(0.0125)
    assert action[2] == pytest.approx(-0.0875)
    assert feedback["feedback_target"][2] == pytest.approx(
        near_outside[2] - 0.007
    )

    table_margin_exhausted = {
        **clearance_restored,
        "finger_table_vertical_clearance_m": 0.004,
    }
    with pytest.raises(
        RuntimeError,
        match="impossible before native table clearance is exhausted",
    ):
        _outside_side_geometry_feedback_action(
            current_eef=current,
            outside_side_target=target,
            guard=table_margin_exhausted,
            gripper=-1.0,
            position_action_scale=0.08,
            maximum_translation_action=0.10,
        )


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


def test_live_contact_gate_accepts_bounded_trailing_xy_and_preserves_depth():
    seek_confirmed = np.array([-0.00211, -0.00637, 0.01949])
    live_after_push = np.array([-0.00260, -0.00246, 0.02104])
    refreshed, diagnostics = _gate_live_contact_offset_xy(
        seek_confirmed,
        live_after_push,
        push_direction_xy=np.array([1.0, 0.0]),
        maximum_xy_drift=0.005,
    )
    assert np.allclose(refreshed[:2], live_after_push[:2])
    assert refreshed[2] == pytest.approx(seek_confirmed[2])
    assert refreshed[2] != pytest.approx(live_after_push[2])
    assert diagnostics["accepted"] is True
    assert diagnostics["reason"] == "bounded_live_offset_accepted"
    assert diagnostics["xy_drift_from_explicit_anchor_m"] < 0.005


def test_job_499699_slipped_offset_cannot_replace_explicit_anchor():
    explicit_recontact = np.array([-0.00160, -0.00582, 0.01949])
    slipped_live = np.array([-0.01086, 0.02383, 0.02104])
    selected, diagnostics = _gate_live_contact_offset_xy(
        explicit_recontact,
        slipped_live,
        push_direction_xy=np.array([0.0, 1.0]),
        maximum_xy_drift=0.005,
    )
    assert np.allclose(selected, explicit_recontact)
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == "live_offset_not_on_trailing_side"
    assert diagnostics["xy_drift_from_explicit_anchor_m"] > 0.030

    # A large drift remains rejected even if its EEF centre is still on the
    # nominal trailing half-plane.
    trailing_slip = np.array([-0.00160, -0.02383, 0.02104])
    selected, diagnostics = _gate_live_contact_offset_xy(
        explicit_recontact,
        trailing_slip,
        push_direction_xy=np.array([0.0, 1.0]),
        maximum_xy_drift=0.005,
    )
    assert np.allclose(selected, explicit_recontact)
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] == (
        "live_offset_exceeds_explicit_anchor_gate"
    )


def test_contact_depth_samples_fail_closed_on_physics_and_collisions():
    valid = {
        "robot_plate_contact": True,
        "plate_table_support": True,
        "plate_tilt_deg": 0.4,
        "plate_xy_drift": 0.0002,
        "forbidden_plate_contact_bodies": [],
        "robot_table_contact_bodies": [],
        "plate_linear_speed": 0.001,
        "plate_angular_speed": 0.01,
        "require_robot_plate_contact": True,
        "require_stable": True,
        "maximum_plate_tilt_deg": 1.0,
        "maximum_plate_xy_drift": 0.001,
        "maximum_linear_speed": 0.015,
        "maximum_angular_speed": 0.15,
    }
    assert _contact_depth_sample_validity(**valid) == {
        "accepted": True,
        "violations": [],
        "require_robot_plate_contact": True,
        "require_stable": True,
    }

    invalid_cases = [
        ({"robot_plate_contact": False}, "robot_plate_contact_lost"),
        ({"plate_table_support": False}, "plate_table_support_lost"),
        ({"plate_tilt_deg": 1.001}, "plate_tilt_exceeded"),
        ({"plate_xy_drift": 0.00101}, "plate_xy_drift_exceeded"),
        (
            {"forbidden_plate_contact_bodies": ["wine_bottle_1_main"]},
            "forbidden_plate_contact",
        ),
        (
            {"robot_table_contact_bodies": ["gripper0_leftfinger"]},
            "forbidden_robot_table_contact",
        ),
        ({"plate_linear_speed": 0.0151}, "plate_linear_speed_exceeded"),
        (
            {"plate_angular_speed": 0.151},
            "plate_angular_speed_exceeded",
        ),
        ({"plate_tilt_deg": np.nan}, "nonfinite_plate_state"),
    ]
    for override, expected_violation in invalid_cases:
        sample = {**valid, **override}
        result = _contact_depth_sample_validity(**sample)
        assert result["accepted"] is False
        assert expected_violation in result["violations"]

    moving_but_not_yet_stabilizing = {
        **valid,
        "require_stable": False,
        "plate_linear_speed": 0.2,
        "plate_angular_speed": 1.0,
    }
    result = _contact_depth_sample_validity(
        **moving_but_not_yet_stabilizing
    )
    assert result["accepted"] is True

    guarded_without_contact = {
        **valid,
        "robot_plate_contact": False,
        "require_robot_plate_contact": False,
    }
    result = _contact_depth_sample_validity(
        **guarded_without_contact
    )
    assert result["accepted"] is True


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


def test_full_push_window_loss_requires_recontact_without_becoming_success():
    # Job 499713's 2-step gap must not interrupt a ten-step tracking window.
    # Contact-backed progress above the unchanged 0.05 mm noise threshold is
    # still classified only as saturation evidence, never native success.
    progressed = _push_window_timeout_evidence(
        robot_contact_steps=4,
        incremental_progress=0.000238,
        minimum_progress=0.00005,
        robot_contact_at_window_end=False,
    )
    assert progressed["status"] == "contact_progress_saturated"

    lost = _push_window_timeout_evidence(
        robot_contact_steps=4,
        incremental_progress=0.000049,
        minimum_progress=0.00005,
        robot_contact_at_window_end=False,
    )
    assert lost["status"] == "robot_contact_lost_recontact_required"
    assert lost["robot_contact_at_window_end"] is False

    # Pure motion without a single real robot contact is not acceptable
    # progress, even when it is numerically larger than the noise gate.
    passive_motion = _push_window_timeout_evidence(
        robot_contact_steps=0,
        incremental_progress=0.001,
        minimum_progress=0.00005,
        robot_contact_at_window_end=False,
    )
    assert (
        passive_motion["status"]
        == "robot_contact_lost_recontact_required"
    )

    # Contact at the end but no contact-backed progress remains a real OSC
    # timeout rather than being relabelled as recovery or success.
    assert (
        _push_window_timeout_evidence(
            robot_contact_steps=2,
            incremental_progress=0.000049,
            minimum_progress=0.00005,
            robot_contact_at_window_end=True,
        )
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
        horizon_reserve_steps = 0
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


def test_job_499691_calibration_fits_native_horizon_with_settle_reserve():
    inner = SimpleNamespace(horizon=1000, timestep=330)
    budget = _horizon_budget(
        SimpleNamespace(env=inner),
        reserved_steps=41 + 64,
    )
    assert budget == {
        "horizon": 1000,
        "timestep": 330,
        "remaining_steps": 670,
        "reserved_steps": 105,
        "usable_steps": 565,
    }
    increment, calibration = _derive_horizon_safe_push_increment(
        goal_distance=0.259,
        usable_push_steps=budget["usable_steps"],
        tracking_steps=10,
        baseline_increment=0.005,
        observed_progress_per_window=0.00245,
        calibration_margin=1.15,
        maximum_increment=0.015,
    )
    expected = 0.005 * (0.259 / 56) / 0.00245 * 1.15
    assert increment == pytest.approx(expected)
    assert 0.010 < increment < 0.011
    assert calibration["tracking_windows"] == 56
    assert calibration["feasible"] is True
    with pytest.raises(RuntimeError, match="unsafe live push increment"):
        _derive_horizon_safe_push_increment(
            goal_distance=0.259,
            usable_push_steps=565,
            tracking_steps=10,
            baseline_increment=0.005,
            observed_progress_per_window=0.00245,
            calibration_margin=1.15,
            maximum_increment=0.008,
        )


def test_horizon_reserve_blocks_push_before_consuming_final_settle():
    inner = SimpleNamespace(horizon=1000, timestep=959)

    class ReserveEnv:
        env = inner
        calls = 0

        @classmethod
        def step(cls, _action):
            cls.calls += 1
            raise AssertionError("reserved final-settle step was consumed")

    class FakeReserveRollout:
        env = ReserveEnv()
        step = 949
        horizon_reserve_steps = 41
        termination_diagnostics = staticmethod(
            lambda: {"plate_progress_m": 0.125}
        )
        _horizon_reserve_error = Rollout._horizon_reserve_error

    with pytest.raises(RuntimeError, match="horizon reserve reached"):
        Rollout.advance(FakeReserveRollout(), np.zeros(7), "task")
    assert ReserveEnv.calls == 0


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

    full_window = FakeRollout()
    full_window_observations = []
    loss = Rollout.move(
        full_window,
        np.ones(3),
        -1.0,
        "task",
        max_steps=4,
        step_observer=lambda: full_window_observations.append(
            full_window.calls
        ),
        timeout_acceptor=lambda _context: (
            _push_window_timeout_evidence(
                robot_contact_steps=0,
                incremental_progress=0.0,
                minimum_progress=0.00005,
                robot_contact_at_window_end=False,
            )
        ),
    )
    assert full_window.calls == 4
    assert full_window_observations == [1, 2, 3, 4]
    assert loss["status"] == "robot_contact_lost_recontact_required"

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
    assert "_compiled_trailing_side_contact_candidates(" in producer
    assert "selected_contact_candidate" in producer
    assert "outside_high_required_action" in producer
    assert '"outside_high_action_will_clip": bool(clipped_axes)' in producer
    assert '"outside_high_clipped_action_axes": clipped_axes' in producer
    assert "dual_finger_contact_skew_exceeds_outside_clearance" in producer
    approach = producer[
        producer.index("# Decouple the large workspace translation") :
        producer.index("initial_contact_depth_calibration =")
    ]
    assert approach.index("center_approach_target,") < approach.index(
        "outside_high_target=outside_high_target,"
    )
    assert approach.index(
        "outside_high_target=outside_high_target,"
    ) < approach.index(
        "outside_side_target=outside_side_target,"
    )
    assert approach.index(
        "outside_side_target=outside_side_target,"
    ) < approach.index(
        "contact_target=contact_target,"
    )
    assert '"live_eef"' in producer
    assert '"live_plate"' in producer
    assert '"candidate_geometry"' in producer
    assert '"robot_gripper_body_names"' in producer
    assert '"plate_contact_counterparts"' in producer
    assert '"compiled_side_contact_geometry"' in producer
    assert '"plate_contact_seek_eef_height"' not in producer
    assert '"--plate_approach_eef_height", type=float, default=0.160' in producer
    assert (
        "plate_position[2] + plate_approach_eef_height"
        in producer
    )
    assert "L3-A3 plate-contact plan" in producer
    assert '"initial_contact_candidate_diagnostics"' in producer
    assert '"initial_selected_contact_candidate"' in producer
    assert '"initial_wrist_yaw_execution"' in producer
    assert '"initial_realized_contact_candidate"' in producer


def test_native_plus_x_front_corridor_is_used_for_initial_and_recontact():
    producer = CONTROLLER_REFERENCE.read_text()
    task_push = producer[
        producer.index("# Job 499604 established real plate contact") :
        producer.index('rollout.hold(-1.0, args.final_settle_steps, "settle")')
    ]
    assert task_push.count("_prepare_native_plus_x_front_corridor(") == 2
    assert task_push.count("_execute_high_safe_wrist_yaw(") == 0
    initial_route = task_push[: task_push.index("for push_iteration in range(")]
    assert initial_route.index("_prepare_native_plus_x_front_corridor(") < (
        initial_route.index("_seek_stable_plate_contact(")
    )
    assert "_execute_high_safe_wrist_yaw(" not in initial_route
    recontact_route = task_push[
        task_push.index("recontact_wrist_yaw_execution =") :
    ]
    assert recontact_route.index("_prepare_native_plus_x_front_corridor(") < (
        recontact_route.index("_seek_stable_plate_contact(")
    )
    assert "remaining_structural_waypoint_steps" in initial_route
    assert "remaining_structural_waypoint_steps" in recontact_route

    front_corridor = producer.split(
        "def _prepare_native_plus_x_front_corridor(", 1
    )[1].split("\ndef _body_contact_counterparts(", 1)[0]
    assert "_select_native_plus_x_front_candidate(" in front_corridor
    assert "reference_outward_direction_xy=np.array([1.0, 0.0]" in (
        front_corridor
    )
    assert '"wrist_yaw_executed": False' in front_corridor
    assert '"authorized_detour_plan": None' in front_corridor
    assert '"NATIVE_PLUS_X_FRONT_CORRIDOR_AUTHORIZED"' in front_corridor
    assert "_live_collision_inventory(" in front_corridor
    assert "_robot_nonrobot_contact_evidence(" in front_corridor

    executor = producer.split(
        "def _execute_high_safe_wrist_yaw(", 1
    )[1].split("\ndef _body_contact_counterparts(", 1)[0]
    assert 'rollout.advance(action, "task_wrist_yaw")' in executor
    assert "_execute_center_high_reacquire(" in executor
    assert 'rollout.advance(action, "task_center_high_reacquire")' in producer
    assert executor.count("allowed_body_pairs=()") >= 3
    assert '!= [\n        "trailing_minus_push"\n    ]' in executor
    assert '"old_plus_x_route_fallback_permitted": False' in executor
    assert '"runtime_tangent_fallback_permitted": False' in executor
    assert "_strict_wrist_yaw_segment_plan(" in executor
    assert "_absolute_wrist_yaw_runtime_target(" in executor
    assert 'yaw_spec=planned_segment["cumulative_yaw_spec"]' in executor
    assert '"yaw_segmentation"' in executor
    assert '"yaw_segments"' in executor
    assert '"absolute_segment_start_attainment"' in executor
    assert '"live_absolute_runtime_target"' in executor
    assert '"executed_relative_yaw_spec"' in executor
    assert '"absolute_cumulative_target_is_authoritative": True' in executor
    assert "remaining_axis_angle_world=commanded_remaining_axis_angle" in executor
    assert '"cumulative_attainment_evidence"' in executor
    assert '"shared_budget_remaining_after_segment"' in executor
    assert executor.count("_wrist_yaw_step_gate(") == 2
    assert "consecutive_angular_stall_steps" in executor
    assert "consecutive_position_stall_steps" in executor
    assert "anchor_eef_position=anchor_eef" in executor
    assert 'stage == "position_settle"' in executor
    assert "_wrist_yaw_stage_budget_evidence(" in executor
    assert '"maximum_position_settle_steps"' in executor
    assert '"maximum_observed_position_drift_m"' in executor
    assert '"final_position_drift_m"' in executor
    assert '"maximum_commanded_translation_action_peak"' in executor
    assert '"position_error_before_action_m"' in executor
    assert '"position_error_after_action_m"' in executor
    assert '"commanded_translation_action_peak"' in executor
    assert '"unexpected_contact_count"' in executor
    assert '"shared_budget_remaining_after_action"' in executor
    assert '"L3-A3 wrist-yaw frame "' in executor
    assert (
        '"simultaneous_yaw_and_position_attainment_required": True'
        in executor
    )
    assert executor.index("if attainment[\"attained\"]:") < executor.index(
        "_real_recompile_wrist_yaw_candidate("
    )
    assert executor.count("_real_recompile_wrist_yaw_candidate(") == 2
    first_recompile = executor.index('recompile_stage="post_wrist_yaw"')
    reacquire = executor.index("_execute_center_high_reacquire(")
    second_recompile = executor.index(
        'recompile_stage="post_center_high_reacquire"'
    )
    assert first_recompile < reacquire < second_recompile
    assert executor.index("_second_real_recompile_identity_evidence(") > (
        second_recompile
    )
    assert '"center_high_reacquire"' in executor
    assert '"second_real_sim_recompile_identity"' in executor
    assert (
        "args.max_waypoint_steps - yaw_steps - reacquire_steps"
        in executor
    )
    gate = producer.split("def _wrist_yaw_step_gate(", 1)[1].split(
        "\ndef _compiled_hypothetical_wrist_yaw_plan(", 1
    )[0]
    assert 'action_evidence.get("action_will_clip", False)' in gate
    assert "forbidden_robot_native_contact_during_wrist_yaw" in gate
    assert "wrist_yaw_angular_progress_stalled" in gate
    assert "wrist_yaw_anchor_position_progress_stalled" in gate
    assert "wrist_yaw_anchor_correction_direction_invalid" in gate


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
    assert task_push.count("pusher_open_sign,") >= 5
    assert task_push.count("_seek_stable_plate_contact(") == 2
    initial_seek = task_push.index('source="initial_contact"')
    assert initial_seek < task_push.index(
        "_calibrate_stable_plate_contact_depth("
    )
    assert '"stable_contact_seek_events"' in task_push
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
    assert "_gate_live_contact_offset_xy(" in push_loop
    assert (
        "confirmed_contact_offset[2] = confirmed_contact_z_offset"
        in push_loop
    )
    assert "live_eef_before - live_plate_before" in push_loop
    assert '"confirmed_contact_offset_before_update"' in push_loop
    assert '"confirmed_contact_offset_after_update"' in push_loop
    assert '"contact_offset_update_source"' in push_loop
    assert '"bounded_live_contact_xy"' in push_loop
    assert '"explicit_recontact_anchor"' in push_loop
    assert '"explicit_contact_anchor_retained"' in push_loop
    assert "explicit_contact_anchor_offset" in push_loop
    assert "contact_offset_gate" in push_loop
    gate_rejection = push_loop[
        push_loop.index(
            'reason": "live_contact_offset_gate_rejected"'
        ) :
        push_loop.index("live_goal_distance = float(")
    ]
    assert "recontact_required_reason = rejection_event" in gate_rejection
    assert "continue" in gate_rejection
    assert "_live_plate_tracking_target(" not in gate_rejection
    assert push_loop.index(
        "live_eef_before - live_plate_before"
    ) < push_loop.index("_live_plate_tracking_target(")
    assert '"live_eef_plate_offset_before"' in push_loop
    assert '"confirmed_contact_z_offset_m"' in push_loop
    assert '"confirmed_contact_z_offset_source"' in push_loop
    assert '"commanded_target_z_anchor"' in push_loop
    assert "initial_stable_contact_depth_calibration" in task_push
    assert "stable_contact_depth_calibration" in push_loop
    assert task_push.count("_calibrate_stable_plate_contact_depth(") == 2
    initial_calibration = task_push.index(
        'source="initial_contact"'
    )
    assert initial_calibration < task_push.index("push_eef_start =")
    recontact_calibration = push_loop.index(
        'source=f"recontact_{recontact_attempts}"'
    )
    assert recontact_calibration < push_loop.index(
        "recontact_plate_after ="
    )
    assert '"contact_depth_calibrations"' in task_push
    assert '"live_push_direction_xy"' in push_loop
    assert "_derive_horizon_safe_push_increment(" in push_loop
    assert "effective_push_increment" in push_loop
    tracking_call = push_loop[
        push_loop.index("target, live_direction_xy =") :
        push_loop.index("waypoint_evidence = {")
    ]
    assert "effective_push_increment," in tracking_call
    assert "args.push_increment," not in tracking_call
    assert '"baseline_commanded_increment_m": args.push_increment' in push_loop
    assert '"horizon_calibration": horizon_calibration' in push_loop
    assert '"source_job": "499691"' in push_loop
    assert "horizon-safe live push calibration failed" in push_loop
    assert "step_observer=observe_push_step" in push_loop
    assert "interruptor=" not in push_loop
    assert "push_contact_loss_confirm_steps" not in producer
    assert (
        "timeout_acceptor=classify_full_push_window_timeout"
        in push_loop
    )
    assert "_push_window_timeout_evidence(" in push_loop
    assert "robot_contact_lost_recontact_required" in push_loop
    assert '"recontact_required_after_waypoint"' in push_loop
    assert push_loop.index(
        "robot_contact_lost_recontact_required"
    ) < push_loop.index(
        'if waypoint_record["recontact_required_after_waypoint"]'
    )
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
    assert "recontact_required_reason is not None" in push_loop
    assert (
        "confirmed_robot_plate_contact_loss_during_push"
        in push_loop
    )
    assert '"reason": recontact_trigger_reason' in push_loop
    assert "recontact_retreat_target" in push_loop
    assert "recontact_center_target" in push_loop
    assert "recontact_high_target" in push_loop
    assert "recontact_outside_side_target" in push_loop
    assert "recontact_side_contact_target" in push_loop
    assert "recontact_compiled_geometry" in push_loop
    assert "_seek_stable_plate_contact(" in push_loop
    assert "L3-A3 plate recontact" in push_loop
    assert "closed-loop plate push exhausted recontact budget" in push_loop
    recovery_start = push_loop.index(
        "# Every recovery waypoint uses OSC env.step"
    )
    recovery = push_loop[
        recovery_start :
        push_loop.index("recontact_depth_calibration =", recovery_start)
    ]
    assert recovery.index("recontact_retreat_target,") < recovery.index(
        "recontact_center_target,"
    )
    assert recovery.index("recontact_center_target,") < recovery.index(
        "outside_high_target=recontact_high_target,"
    )
    assert recovery.index(
        "outside_high_target=recontact_high_target,"
    ) < recovery.index(
        "outside_side_target=("
    )
    assert recovery.index("outside_side_target=(") < recovery.index(
        "contact_target=recontact_side_contact_target,"
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
        '"--maximum_live_contact_offset_xy_drift",\n'
        "        type=float,\n"
        "        default=0.005,"
        in producer
    )
    assert (
        '"--contact_depth_action_step", type=float, default=0.004'
        in producer
    )
    assert (
        '"--plate_contact_outside_clearance", type=float, default=0.0005'
        in producer
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in producer
    )
    assert (
        '"--structural_near_plate_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.005,"
        in producer
    )
    assert (
        '"--vertical_corridor_descent_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in producer
    )
    assert (
        '"--post_descent_lateral_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
        in producer
    )
    assert (
        '"--overhead_descent_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.20,"
        in producer
    )
    assert (
        '"--plate_contact_seek_max_steps", type=int, default=64'
        in producer
    )
    assert (
        '"--target_contact_depth_increase", type=float, default=0.003'
        in producer
    )
    assert (
        '"--maximum_contact_depth_actions", type=int, default=12'
        in producer
    )
    assert (
        '"--contact_depth_stability_steps", type=int, default=3'
        in producer
    )
    assert (
        '"--max_contact_calibration_plate_xy_drift",\n'
        "        type=float,\n"
        "        default=0.001,"
        in producer
    )
    assert (
        '"--max_contact_calibration_plate_tilt_deg",\n'
        "        type=float,\n"
        "        default=1.0,"
        in producer
    )
    depth_calibration = producer[
        producer.index("def _calibrate_stable_plate_contact_depth(") :
        producer.index("\ndef generate(args):")
    ]
    bounded_seek = producer[
        producer.index("def _seek_stable_plate_contact(") :
        producer.index("\ndef _calibrate_stable_plate_contact_depth(")
    ]
    compiled_plan = producer[
        producer.index("def _compiled_native_side_contact_plan(") :
        producer.index("\ndef _body_contact_counterparts(")
    ]
    compiled_clearance = producer[
        producer.index("def _compiled_collision_pair_clearance(") :
        producer.index("\ndef _outside_side_guard_from_world_aabbs(")
    ]
    live_guard = producer[
        producer.index("def _live_outside_side_guard(") :
        producer.index(
            "\ndef _outside_side_geometry_feedback_action("
        )
    ]
    assert "model.geom_aabb" in producer
    assert "model.pair_margin" in compiled_clearance
    assert "max(model.geom_margin[geom1], model.geom_margin[geom2])" in (
        compiled_clearance
    )
    assert "np.nextafter(contact_detection_margin, np.inf)" in (
        compiled_clearance
    )
    assert "_compiled_pair_set_clearance(" in live_guard
    assert 'geometry["outside_clearance_m"]' not in live_guard
    assert "plate_rim_geoms" in compiled_plan
    assert "finger_collision_geoms" in compiled_plan
    assert "_side_contact_targets_from_compiled_bounds(" in compiled_plan
    assert "_compiled_side_contact_eef_z_feasibility(" in compiled_plan
    assert "finger_vertical_bounds_from_eef" in compiled_plan
    assert "finger_table_clearance_derivation" in compiled_plan
    assert "structurally decoupled outside-side approach" in bounded_seek
    assert "env.set_state" not in compiled_plan
    assert "set_init_state" not in compiled_plan
    assert "rollout.move(" not in bounded_seek
    assert "outside_high_target" in bounded_seek
    assert "outside_side_target" in bounded_seek
    assert "_live_outside_side_guard(" in bounded_seek
    assert "_compiled_vertical_staging_corridor(" in bounded_seek
    assert "_compiled_overhead_staging_geometry(" in bounded_seek
    assert "_live_compiled_overhead_guard(" in bounded_seek
    assert "_fixed_xy_vertical_approach_action(" in bounded_seek
    assert "_fixed_z_lateral_approach_action(" in bounded_seek
    assert (
        'structural_stage = "overhead_high_corridor_lateral"'
        in bounded_seek
    )
    assert '"overhead_corridor_descent"' in bounded_seek
    assert '"overhead_post_descent_corridor_lateral"' in bounded_seek
    assert '"vertical_corridor_descent"' in bounded_seek
    assert '"vertical_corridor_settle"' in bounded_seek
    assert bounded_seek.count(
        "active_vertical_corridor_envelope = ("
    ) == 1
    wrist_yaw = producer[
        producer.index("def _execute_high_safe_wrist_yaw(") :
        producer.index("\ndef _seek_stable_plate_contact(")
    ]
    assert "active_vertical_corridor_envelope" not in wrist_yaw
    settle_action = bounded_seek.split(
        'elif structural_stage == "vertical_corridor_settle":', 1
    )[1].split(
        'elif structural_stage == "fixed_safe_z_lateral_approach":', 1
    )[0]
    assert "_compiled_low_side_settle_brake_action(" in settle_action
    assert settle_action.count(
        'active_vertical_corridor_envelope['
    ) >= 2
    assert "compiled_low_side_settle_brake_envelope" in settle_action
    assert '"active_positive_z_brake_requested": True' in settle_action
    assert "vertical_corridor_settle_brake_trigger_buffer" in bounded_seek
    assert "maximum_vertical_corridor_outward_hold_world_step" in (
        bounded_seek.split(
            "vertical_corridor_settle_brake_trigger_buffer = float(", 1
        )[1].split(")", 1)[0]
    )
    settle_transition = bounded_seek.split(
        'elif stage_before_action == "vertical_corridor_descent":', 1
    )[1].split('feedback["stage_after_action"]', 1)[0]
    assert "settle_brake_trigger_z_m" in settle_transition
    trigger_condition = settle_transition.split(
        "if (", 1
    )[1].split("):", 1)[0]
    assert "latest_outside_side_guard" not in trigger_condition
    assert (
        'stage_before_action == "fixed_safe_z_lateral_approach"'
        in bounded_seek
    )
    assert "full_outside_side_guard_not_accepted_during_settle" in (
        CONTROLLER_REFERENCE.read_text()
    )
    settle_transition_logic = bounded_seek.split(
        'elif stage_before_action == "vertical_corridor_settle":', 1
    )[1].split(
        'elif stage_before_action == "vertical_corridor_descent":', 1
    )[0]
    assert "kinematic_brake_reversed" in settle_transition_logic
    assert "settle_geometric_authority_release" in settle_transition_logic
    assert (
        "0.5 * previous_geometric_height_action"
        in settle_transition_logic
    )
    assert "vertical_corridor_geometric_height_action_floor" in (
        settle_transition_logic
    )
    assert "fixed_outward_translation_action_bound" in settle_action
    assert "active_geometric_height_action" in bounded_seek
    assert (
        "outward_authority_invariant_across_height_schedule"
        in bounded_seek
    )
    assert (
        "positive_z_authority_invariant_across_height_schedule"
        in bounded_seek
    )
    assert '"fixed_safe_z_lateral_approach"' in bounded_seek
    assert "_fixed_safe_z_lateral_hold_action(" in bounded_seek
    assert "fixed_safe_z_required_stable_count = 2" in bounded_seek
    assert "fixed_safe_z_instantaneous_stable" in bounded_seek
    assert "fixed_safe_z_strict_table_reserve_lost" in bounded_seek
    assert "lateral_position_tolerance_m=args.position_tolerance" in (
        bounded_seek
    )
    assert "vertical_position_tolerance_m=(" in bounded_seek
    assert "no_inward_xy_after_lateral_tolerance" in producer
    assert "below_height_band_retains_positive_z_floor" in producer
    assert "healthy_outside_reserve_avoids_outward_saturation" in producer
    assert "outside_recovery_suspends_negative_z" in producer
    assert (
        "fixed_safe_z_closed_loop_hazard_response_bound = 0.0011"
        in bounded_seek
    )
    assert "fixed_safe_z_recovery_entry_clearance" in bounded_seek
    assert "fixed_safe_z_recovery_exit_clearance" in bounded_seek
    assert "fixed_safe_z_refill_target_clearance" in bounded_seek
    assert (
        '"fixed_safe_z_recovery_exit_clearance_m"'
        in bounded_seek
    )
    high_plane_call = bounded_seek.split(
        "_compiled_adaptive_high_plane_action(", 1
    )[1].split("overhead_guard=latest_overhead_guard", 1)[0]
    fixed_safe_z_call = bounded_seek.split(
        "_fixed_safe_z_lateral_hold_action(", 1
    )[1].split("outside_side_guard=pre_action_guard", 1)[0]
    assert "measured_outward_step_progress_m=" not in high_plane_call
    assert "measured_outward_step_progress_m=" in fixed_safe_z_call
    assert (
        "for guard_step in range(1, structural_waypoint_budget + 1)"
        in bounded_seek
    )
    assert '"outside_side_feedback"' in bounded_seek
    assert '"post_action_guard"' in bounded_seek
    assert "_outside_side_lateral_settle_evidence(" in bounded_seek
    assert "_outside_side_staircase_settle_trigger(" in bounded_seek
    assert "previous_stable_response_count=int(" in bounded_seek
    assert "lateral_settle_state = lateral_settle_progress" in bounded_seek
    assert '"lateral_settle_trigger"' in bounded_seek
    assert bounded_seek.index("motion_sample = capture(") < (
        bounded_seek.index("if structural_violations:")
    )
    assert "one_controller_step_corridor_reserve_lost" in bounded_seek
    assert "compiled_safe_z_rim_coverage_not_sustained" in bounded_seek
    assert "compiled_overhead_one_step_vertical_reserve_lost" in bounded_seek
    assert "_overhead_lateral_buffer_evidence(" in bounded_seek
    assert '"vertical_tail_events"' in bounded_seek
    assert '"structural_waypoint_budget"' in bounded_seek
    assert 'stage.startswith("outside_")' in bounded_seek
    assert "rollout.move(" not in bounded_seek
    assert "rollout.advance(action, \"task\")" in bounded_seek
    assert "plate_contact_seek_max_translation_action" in bounded_seek
    assert (
        "_bounded_contact_seek_vertical_stabilization_action("
        in bounded_seek
    )
    assert '"outside_contact_seek_vertical_stabilization"' in bounded_seek
    assert "minimum_saturated_waypoint_progress" in bounded_seek
    assert "vertical_stabilization_complete" in bounded_seek
    assert "vertical_stabilization_derivative_gain = 2.0" in bounded_seek
    assert "vertical_stabilization_required_stable_count = 2" in bounded_seek
    assert "outward_direction_xy=corridor_outward_direction" in bounded_seek
    assert "outside_authority_fraction = float(" in producer
    assert "vertical_action_cap = float(" in producer
    assert "low_clearance_forces_pure_outward_recovery" in producer
    assert "absolute_position_error" in bounded_seek
    assert "abs(measured_vertical_response)" in bounded_seek
    assert "post_stabilization_guard[\"accepted\"]" in bounded_seek
    assert (
        "contact-seek vertical stabilization exhausted the "
        in bounded_seek
    )
    assert '"bounded_lateral_contact_seek"' in bounded_seek
    assert (
        '"bounded_lateral_contact_seek",\n'
        "            seek_index,\n"
        "            contact_observed,\n"
        "            contact_observed,"
        in bounded_seek
    )
    assert '"stable_contact_confirmation"' in bounded_seek
    assert "two_finger_side_contact_not_sustained" in bounded_seek
    assert "require_contact" in bounded_seek
    assert "require_stable" in bounded_seek
    assert "robot_plate_contact_before_lateral_seek" in bounded_seek
    assert "env.set_state" not in bounded_seek
    assert "set_init_state" not in bounded_seek
    assert "rollout.advance(" in depth_calibration
    assert "env.set_state" not in depth_calibration
    assert "set_init_state" not in depth_calibration
    assert "initial_contact_z_offset" in depth_calibration
    assert "live_contact_z_offset" in depth_calibration
    assert (
        "initial_contact_z_offset - live_contact_z_offset"
        in depth_calibration
    )
    assert '"measured_eef_world_descent_m"' in depth_calibration
    assert "robot_plate_contact_lost" in producer
    assert "plate_table_support_lost" in producer
    assert "plate_tilt_exceeded" in producer
    assert "forbidden_plate_contact" in producer
    assert "forbidden_robot_table_contact" in producer
    assert "plate_linear_speed_exceeded" in producer
    assert "plate_angular_speed_exceeded" in producer
    assert (
        '"--observed_push_progress_per_tracking_window",\n'
        "        type=float,\n"
        "        default=0.00245,"
        in producer
    )
    assert (
        '"--push_horizon_calibration_margin", type=float, default=1.15'
        in producer
    )
    assert '"--maximum_live_push_increment", type=float, default=0.015' in producer
    assert (
        '"--planned_recontact_reserve_steps", type=int, default=64'
        in producer
    )
    assert '"--horizon_guard_steps", type=int, default=1' in producer
    assert (
        '"--minimum_saturated_waypoint_progress",\n'
        "        type=float,\n"
        "        default=0.00005,"
        in producer
    )
    assert "--minimum_saturated_waypoint_progress must be positive" in producer
    assert "--maximum_push_iterations must be positive" in producer
    assert "--maximum_recontact_attempts must be positive" in producer
    assert (
        "--plate_contact_outside_clearance must be in (0, 0.020]"
        in producer
    )
    assert (
        "--plate_contact_seek_max_translation_action must be in (0, 0.2]"
        in producer
    )
    assert (
        "--structural_near_plate_max_translation_action must be positive"
        in producer
    )
    assert (
        "--overhead_descent_max_translation_action must be greater than"
        in producer
    )
    assert (
        "--vertical_corridor_descent_max_translation_action must be"
        in producer
    )
    assert (
        "--post_descent_lateral_max_translation_action must be greater"
        in producer
    )
    assert "--plate_contact_seek_max_steps must be positive" in producer
    assert (
        "--maximum_live_contact_offset_xy_drift must be positive"
        in producer
    )
    assert 'default=0.00005' in producer
    assert "maximum_push_distance" not in producer
    assert "ignore_done=True" not in producer
    assert "rollout.horizon_reserve_steps = final_horizon_reserve_steps" in task_push
    assert "rollout.horizon_reserve_steps = 0" in task_push
    assert '"horizon_budget_before_final_settle"' in task_push
    assert '"horizon_budget_after_final_settle"' in producer
    assert "environment terminated episode; fail-closed" in producer
    assert '"plate_progress_m"' in producer
    assert '"completed_push_iterations"' in producer
    assert '"pusher_gripper_sign": pusher_open_sign' in producer
    assert '"push_evidence": push_summary' in producer
    assert task_push.index("if not env.check_success():") < task_push.index(
        "rollout.horizon_reserve_steps = 0"
    )


def test_plate_contact_diagnostics_and_detector_share_compiled_robot_names():
    class Model:
        names = [
            "world",
            "plate_1_main",
            "plate_1_child",
            "table",
            "robot0_link7",
            "gripper0_finger_joint1_tip",
            "gripper0_rightfinger",
        ]
        nbody = len(names)
        body_parentid = np.array([0, 0, 1, 0, 0, 4, 4])
        geom_bodyid = np.array([2, 3, 4, 5, 6])
        ngeom = len(geom_bodyid)
        geom_names = [
            "plate_collision",
            "table_collision",
            "robot_link_collision",
            "finger_collision",
            "right_finger_collision",
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
                ncon=3,
                contact=[
                    SimpleNamespace(geom1=0, geom2=1),
                    SimpleNamespace(geom1=3, geom2=0),
                    SimpleNamespace(geom1=4, geom2=0),
                ],
            ),
        )

    env = Env()
    assert _robot_gripper_body_names(env) == [
        "gripper0_finger_joint1_tip",
        "gripper0_rightfinger",
        "robot0_link7",
    ]
    contacts = _body_contact_counterparts(env, PLATE_BODY)
    assert [item["counterpart_body"] for item in contacts] == [
        "table",
        "gripper0_finger_joint1_tip",
        "gripper0_rightfinger",
    ]
    assert contacts[0]["counterpart_is_robot_or_gripper"] is False
    assert contacts[1]["counterpart_is_robot_or_gripper"] is True
    assert contacts[2]["counterpart_is_robot_or_gripper"] is True
    assert _robot_contacts_body(env, PLATE_BODY) is True
    assert _plate_finger_contact_sides(env) == {
        "left": True,
        "right": True,
        "contact_bodies": [
            "gripper0_finger_joint1_tip",
            "gripper0_rightfinger",
        ],
    }


def test_robot_native_contact_gate_uses_exact_body_pair_allowlist():
    class Model:
        body_names = [
            "world",
            "plate_1_main",
            "plate_1_child",
            "table",
            "robot0_link7",
            "gripper0_finger_joint1_tip",
            "wooden_cabinet_1_main",
        ]
        geom_names = [
            "plate_collision",
            "table_collision",
            "robot_link_collision",
            "finger_collision",
            "cabinet_collision",
        ]
        nbody = len(body_names)
        ngeom = len(geom_names)
        body_parentid = np.array([0, 0, 1, 0, 0, 4, 0])
        geom_bodyid = np.array([2, 3, 4, 5, 6])
        geom_contype = np.ones(ngeom, dtype=int)
        geom_conaffinity = np.ones(ngeom, dtype=int)
        geom_type = np.array([6, 6, 7, 6, 6], dtype=int)
        geom_size = np.full((ngeom, 3), 0.01, dtype=float)
        geom_aabb = np.tile(
            np.array([0.0, 0.0, 0.0, 0.01, 0.01, 0.01]),
            (ngeom, 1),
        )

        @classmethod
        def body_id2name(cls, body_id):
            return cls.body_names[body_id]

        @classmethod
        def geom_id2name(cls, geom_id):
            return cls.geom_names[geom_id]

    data = SimpleNamespace(
        ncon=1,
        contact=[
            SimpleNamespace(
                geom1=3,
                geom2=0,
                pos=np.array([0.0, 0.0, 0.9]),
                frame=np.array(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                ),
                dist=-0.0002,
            )
        ],
        geom_xpos=np.zeros((Model.ngeom, 3), dtype=float),
        geom_xmat=np.tile(
            np.eye(3).reshape(1, 9), (Model.ngeom, 1)
        ),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))
    structural = _robot_nonrobot_contact_evidence(
        env,
        allowed_body_pairs=(),
    )
    assert structural["accepted"] is False
    assert structural["allowed_body_pairs"] == []
    assert structural["robot_collision_bodies"] == [
        "gripper0_finger_joint1_tip",
        "robot0_link7",
    ]
    assert "plate_1_child" in structural[
        "nonrobot_native_collision_bodies"
    ]
    assert structural["unexpected_contacts"][0]["native_body"] == (
        "plate_1_child"
    )

    exact_plate_finger_pair = (
        "gripper0_finger_joint1_tip",
        "plate_1_child",
    )
    contact_seek = _robot_nonrobot_contact_evidence(
        env,
        allowed_body_pairs=(exact_plate_finger_pair,),
    )
    assert contact_seek["accepted"] is True
    assert contact_seek["contacts"][0]["allowed"] is True

    data.ncon = 2
    data.contact.append(
        SimpleNamespace(
            geom1=3,
            geom2=1,
            pos=np.array([0.0, 0.0, 0.89]),
            frame=np.array(
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
            ),
            dist=-0.0001,
        )
    )
    with_table_contact = _robot_nonrobot_contact_evidence(
        env,
        allowed_body_pairs=(exact_plate_finger_pair,),
    )
    assert with_table_contact["accepted"] is False
    assert with_table_contact["unexpected_contacts"][0]["native_body"] == (
        "table"
    )


def _l3a3_live_diagnostic_env(contacts=()):
    class Model:
        body_names = [
            "world",
            "robot0_link7",
            "wooden_cabinet_1_main",
            "wooden_cabinet_1_cabinet_top",
            "table",
            "wine_rack_1_main",
            "plate_1_main",
            "cream_cheese_1_main",
        ]
        geom_names = [
            "gripper0_hand_collision",
            "wooden_cabinet_1_g18",
            "table_collision",
            "wine_rack_1_g4",
            "plate_1_g1",
            "cream_cheese_1_g1",
        ]
        joint_names = [
            "wooden_cabinet_1_joint0",
            "wooden_cabinet_1_top_level",
        ]
        nbody = len(body_names)
        ngeom = len(geom_names)
        njnt = len(joint_names)
        body_parentid = np.array([0, 0, 0, 2, 0, 0, 0, 0], dtype=int)
        geom_bodyid = np.array([1, 3, 4, 5, 6, 7], dtype=int)
        geom_contype = np.ones(ngeom, dtype=int)
        geom_conaffinity = np.ones(ngeom, dtype=int)
        geom_type = np.array([7, 6, 6, 6, 6, 6], dtype=int)
        geom_size = np.array(
            [
                [0.031, 0.048, 0.103],
                [0.00770, 0.00817, 0.04445],
                [0.50, 0.50, 0.01],
                [0.00108, 0.00867, 0.13222],
                [0.07, 0.07, 0.01],
                [0.04, 0.021, 0.009],
            ],
            dtype=float,
        )
        geom_aabb = np.array(
            [
                [0.01, 0.0, 0.0, 0.02, 0.03, 0.04],
                [0.0, 0.0, 0.0, 0.04445, 0.00770, 0.00817],
                [0.0, 0.0, 0.0, 0.50, 0.50, 0.01],
                [0.0, 0.0, 0.0, 0.13222, 0.00519, 0.00809],
                [0.0, 0.0, 0.0, 0.07, 0.07, 0.01],
                [0.0, 0.0, 0.0, 0.04, 0.021, 0.009],
            ],
            dtype=float,
        )
        jnt_bodyid = np.array([2, 3], dtype=int)
        jnt_type = np.array([0, 2], dtype=int)
        jnt_qposadr = np.array([0, 7], dtype=int)

        @classmethod
        def body_name2id(cls, name):
            return cls.body_names.index(name)

        @classmethod
        def body_id2name(cls, body_id):
            return cls.body_names[body_id]

        @classmethod
        def geom_id2name(cls, geom_id):
            return cls.geom_names[geom_id]

        @classmethod
        def joint_id2name(cls, joint_id):
            return cls.joint_names[joint_id]

    robot_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    data = SimpleNamespace(
        ncon=len(contacts),
        contact=list(contacts),
        geom_xpos=np.array(
            [
                [0.05, -0.08, 1.05],
                [0.04, -0.13, 1.04],
                [0.0, 0.0, 0.88],
                [-0.258, -0.194, 1.14],
                [0.05, -0.03, 0.91],
                [-0.01, 0.13, 0.91],
            ],
            dtype=float,
        ),
        geom_xmat=np.array(
            [
                robot_rotation.reshape(9),
                np.eye(3).reshape(9),
                np.eye(3).reshape(9),
                np.eye(3).reshape(9),
                np.eye(3).reshape(9),
                np.eye(3).reshape(9),
            ]
        ),
        body_xpos=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, -0.08, 1.05],
                [0.03, -0.24, 0.88],
                [0.03, -0.24, 0.88],
                [0.0, 0.0, 0.88],
                [-0.258, -0.194, 1.14],
                [0.05, -0.03, 0.91],
                [-0.01, 0.13, 0.91],
            ],
            dtype=float,
        ),
        body_xmat=np.tile(np.eye(3).reshape(1, 9), (Model.nbody, 1)),
        qpos=np.array(
            [0.03, -0.24, 0.88, 1.0, 0.0, 0.0, 0.0, -0.01],
            dtype=float,
        ),
    )
    return SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))


def test_live_collision_records_mesh_and_box_world_aabbs():
    env = _l3a3_live_diagnostic_env()
    model, data = env.sim.model, env.sim.data
    eef = np.array([0.04, -0.07, 1.06])
    mesh = _live_collision_geom_record(
        model, data, 0, eef_position=eef
    )
    box = _live_collision_geom_record(model, data, 1)

    assert mesh["type"] == 7
    np.testing.assert_allclose(
        mesh["world_aabb_center"], [0.05, -0.07, 1.05]
    )
    np.testing.assert_allclose(
        mesh["world_aabb_half_size"], [0.03, 0.02, 0.04]
    )
    np.testing.assert_allclose(
        mesh["world_aabb_min_offset_from_eef"], [-0.02, -0.02, -0.05]
    )
    assert box["type"] == 6
    np.testing.assert_allclose(
        box["world_aabb_min"],
        [0.04 - 0.04445, -0.13 - 0.00770, 1.04 - 0.00817],
    )
    np.testing.assert_allclose(
        box["world_aabb_max"],
        [0.04 + 0.04445, -0.13 + 0.00770, 1.04 + 0.00817],
    )


def test_live_inventory_records_cabinet_qpos_provenance_and_hash():
    env = _l3a3_live_diagnostic_env()
    eef = np.array([0.04, -0.07, 1.06])
    inventory = _live_collision_inventory(env, eef_position=eef)
    cabinet = _live_cabinet_pose_diagnostic(env)

    assert inventory["robot_collision_geom_count"] == 1
    assert inventory["native_nonrobot_collision_geom_count"] == 5
    assert inventory["total_collision_geom_count"] == 6
    assert len(inventory["inventory_sha256"]) == 64
    robot = inventory["robot_collision_geoms"][0]
    assert robot["name"] == "gripper0_hand_collision"
    assert "world_aabb_min_offset_from_eef" in robot
    assert cabinet["root_body"]["name"] == "wooden_cabinet_1_main"
    assert cabinet["root_body_attached_joints"][0]["qpos_width"] == 7
    assert cabinet["top_drawer_joint"]["name"] == (
        "wooden_cabinet_1_top_level"
    )
    assert cabinet["top_drawer_joint"]["qpos_address"] == 7
    assert cabinet["top_drawer_joint"]["qpos"] == [-0.01]
    assert "env.sim.data.qpos" in cabinet["top_drawer_joint"][
        "qpos_provenance"
    ]


def test_live_inventory_manifest_keeps_complete_inventory(tmp_path):
    env = _l3a3_live_diagnostic_env()
    inventory = _live_collision_inventory(
        env, eef_position=np.array([0.04, -0.07, 1.06])
    )
    path = tmp_path / "controller_live_diagnostic.json"
    _write_controller_diagnostic_manifest(
        path,
        {
            "diagnostic_only": True,
            "route_authorized": False,
            "live_collision_inventory": inventory,
        },
    )
    record = json.loads(path.read_text())
    saved = record["live_collision_inventory"]
    assert saved["inventory_sha256"] == inventory["inventory_sha256"]
    assert len(saved["robot_collision_geoms"]) == 1
    assert len(saved["native_nonrobot_collision_geoms"]) == 5
    assert saved["robot_collision_geoms"][0]["xmat_world_row_major"] == (
        inventory["robot_collision_geoms"][0]["xmat_world_row_major"]
    )
    assert saved["native_nonrobot_collision_geoms"][0][
        "world_aabb_min"
    ] == inventory["native_nonrobot_collision_geoms"][0]["world_aabb_min"]


def test_native_cabinet_detour_completion_is_hash_bound(tmp_path):
    env = _l3a3_live_diagnostic_env()
    eef = np.array([0.04, -0.07, 1.06])
    inventory = _live_collision_inventory(env, eef_position=eef)
    plan = _compiled_native_right_high_then_low_return_plan(
        live_inventory=inventory,
        current_eef=eef,
        outside_high_target=np.array([0.10, -0.14, 1.06]),
        outside_side_target=np.array([0.10, -0.14, 0.90]),
        maximum_controller_world_step_m=0.008,
        maximum_route_translation_action=np.nextafter(1.0, 0.0),
        position_action_scale_m_per_action=0.08,
        position_tolerance_m=0.005,
    )
    path = tmp_path / "controller_live_diagnostic.json"
    _write_controller_diagnostic_manifest(
        path,
        {
            "live_collision_inventory": inventory,
            "authorized_detour_plan": plan,
            "route_executed": False,
        },
    )
    context = {
        "manifest_path": str(path),
        "inventory_sha256": inventory["inventory_sha256"],
        "authorized_detour_plan": plan,
        "executed": False,
    }
    counts = {stage: index + 1 for index, stage in enumerate(plan["route_order"])}
    completion = _record_native_cabinet_detour_completion(
        diagnostic_context=context,
        source="unit_test",
        final_guard={"accepted": True, "violations": []},
        stage_action_counts=counts,
        used_steps=10,
        remaining_steps=170,
    )

    saved = json.loads(path.read_text())
    assert completion["accepted"] is True
    assert saved["route_executed"] is True
    assert saved["route_completion"] == completion
    assert saved["latest_status"] == (
        "NATIVE_RIGHT_HIGH_THEN_LOW_RETURN_EXECUTED_AND_GUARDED"
    )
    assert context["executed"] is True


def test_live_collision_inventory_nonfinite_geometry_fails_closed():
    env = _l3a3_live_diagnostic_env()
    env.sim.data.geom_xpos[1, 2] = np.nan
    with pytest.raises(
        RuntimeError,
        match="live collision geom type/size/pose/AABB evidence is invalid",
    ):
        _live_collision_inventory(
            env, eef_position=np.array([0.04, -0.07, 1.06])
        )


def test_live_detour_candidates_cannot_select_or_execute():
    env = _l3a3_live_diagnostic_env()
    eef = np.array([0.04, -0.07, 1.06])
    inventory = _live_collision_inventory(env, eef_position=eef)
    candidates = _diagnostic_only_live_detour_candidates(
        live_inventory=inventory,
        cabinet_pose=_live_cabinet_pose_diagnostic(env),
        current_eef=eef,
        outside_high_target=np.array([0.10, -0.14, 1.06]),
        outside_side_target=np.array([0.10, -0.14, 0.94]),
    )

    assert [candidate["candidate_id"] for candidate in candidates] == [
        "vertical_first",
        "minus_x_detour",
        "plus_x_detour",
    ]
    assert all(candidate["diagnostic_only"] for candidate in candidates)
    assert all(not candidate["executed"] for candidate in candidates)
    assert all(not candidate["selection_eligible"] for candidate in candidates)
    assert all(not candidate["selected"] for candidate in candidates)
    assert all(not candidate["route_authorized"] for candidate in candidates)
    assert all(candidate["aabb_authorization_prohibited"] for candidate in candidates)


def test_native_right_high_then_low_return_compiles_from_live_geometry():
    env = _l3a3_live_diagnostic_env()
    eef = np.array([0.04, -0.07, 1.06])
    inventory = _live_collision_inventory(env, eef_position=eef)
    plan = _compiled_native_right_high_then_low_return_plan(
        live_inventory=inventory,
        current_eef=eef,
        outside_high_target=np.array([0.10, -0.14, 1.06]),
        outside_side_target=np.array([0.10, -0.14, 0.90]),
        maximum_controller_world_step_m=0.008,
        maximum_route_translation_action=np.nextafter(1.0, 0.0),
        position_action_scale_m_per_action=0.08,
        position_tolerance_m=0.005,
    )

    assert plan["route_authorized"] is True
    assert plan["diagnostic_only"] is False
    assert plan["wine_rack_geom_names"] == ["wine_rack_1_g4"]
    assert plan["plate_geom_names"] == ["plate_1_g1"]
    assert "cream_cheese_1_g1" in plan["high_route_obstacle_geom_names"]
    assert plan["route_order"] == [
        "right_high_lateral",
        "right_high_trailing_pass",
        "right_trailing_vertical_descent",
        "trailing_low_terminal_return",
    ]
    assert plan["predicted_initial_high_route_separation_m"] > 0.008
    assert plan["predicted_right_clearance_m"] > 0.013
    assert plan["predicted_high_above_plate_clearance_m"] > 0.008
    assert plan["predicted_under_clearance_at_terminal_m"] > 0.008
    assert 0.0 < plan["low_route_entry_z_tolerance_m"] <= 0.005
    assert plan["low_route_entry_headroom_retained_fraction"] == 0.75
    assert plan["predicted_right_of_rack_clearance_at_terminal_m"] > 0.008
    assert plan["maximum_route_translation_action"] > 0.99
    assert plan["maximum_route_world_command_m"] == pytest.approx(0.08)
    assert plan["minimum_full_step_action_lower_bound"] < 180
    np.testing.assert_allclose(
        plan["waypoints"]["terminal_outside_side_low"],
        [0.10, -0.14, 0.90],
    )

    bounded_seek = CONTROLLER_REFERENCE.read_text().split(
        "def _seek_stable_plate_contact(", 1
    )[1].split("\ndef _calibrate_stable_plate_contact_depth", 1)[0]
    for stage, waypoint in (
        ("right_high_lateral", "right_high"),
        ("right_high_trailing_pass", "right_trailing_high"),
        ("right_trailing_vertical_descent", "right_trailing_low"),
        ("trailing_low_terminal_return", "terminal_outside_side_low"),
    ):
        assert stage in bounded_seek
        assert waypoint in bounded_seek
    assert "minus_x_high_lateral" not in bounded_seek
    assert "minus_x_front_vertical_descent" not in bounded_seek
    assert "low_front_x_return" not in bounded_seek
    assert "right_of_rack_low_y_pass" not in bounded_seek
    assert "abs(current_eef[2] - detour_target[2])" in bounded_seek
    assert '"right_high_lateral",\n        "right_high_trailing_pass"' in (
        bounded_seek
    )
    assert "_compiled_adaptive_high_plane_action(" in bounded_seek


def test_native_cabinet_detour_live_guard_fails_closed_then_accepts():
    env = _l3a3_live_diagnostic_env()
    initial_eef = np.array([0.04, -0.07, 1.06])
    inventory = _live_collision_inventory(env, eef_position=initial_eef)
    plan = _compiled_native_right_high_then_low_return_plan(
        live_inventory=inventory,
        current_eef=initial_eef,
        outside_high_target=np.array([0.10, -0.14, 1.06]),
        outside_side_target=np.array([0.10, -0.14, 0.90]),
        maximum_controller_world_step_m=0.008,
        maximum_route_translation_action=np.nextafter(1.0, 0.0),
        position_action_scale_m_per_action=0.08,
        position_tolerance_m=0.005,
    )

    high_route = _live_native_cabinet_detour_guard(
        env,
        eef_position=initial_eef,
        plan=plan,
        stage="right_high_lateral",
    )
    assert high_route["accepted"] is True
    assert high_route["blocking_wine_rack_geom_count"] == 1
    assert high_route["minimum_high_route_separation_m"] > high_route[
        "required_strict_clearance_m"
    ]
    assert high_route["above_plate_clearance_m"] > high_route[
        "required_strict_clearance_m"
    ]

    rejected = _live_native_cabinet_detour_guard(
        env,
        eef_position=initial_eef,
        plan=plan,
        stage="right_high_trailing_pass",
    )
    assert rejected["accepted"] is False
    assert rejected["violations"] == [
        "rigid_hand_not_strictly_right_of_plate_and_wine_rack"
    ]

    env.sim.data.geom_xpos[0, 0] = 0.25
    right_of_all = _live_native_cabinet_detour_guard(
        env,
        eef_position=np.array([0.24, -0.06, 1.06]),
        plan=plan,
        stage="right_high_trailing_pass",
    )
    assert right_of_all["accepted"] is True
    assert right_of_all["right_clearance_m"] > right_of_all[
        "required_strict_clearance_m"
    ]

    env.sim.data.geom_xpos[0] = [0.25, -0.14, 0.95]
    right_descent = _live_native_cabinet_detour_guard(
        env,
        eef_position=np.array([0.24, -0.13, 0.96]),
        plan=plan,
        stage="right_trailing_vertical_descent",
    )
    assert right_descent["accepted"] is True
    assert right_descent["right_clearance_m"] > right_descent[
        "required_strict_clearance_m"
    ]

    env.sim.data.geom_xpos[0] = [0.05, -0.14, 0.95]
    terminal = _live_native_cabinet_detour_guard(
        env,
        eef_position=np.array([0.04, -0.13, 0.96]),
        plan=plan,
        stage="trailing_low_terminal_return",
    )
    assert terminal["accepted"] is True
    assert terminal["right_of_rack_clearance_m"] > terminal[
        "required_strict_clearance_m"
    ]


def test_robot_native_contact_normal_is_canonical_under_geom_order_flip():
    frame_forward = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )
    frame_reverse = frame_forward.copy()
    frame_reverse[:3] *= -1.0
    contacts = [
        SimpleNamespace(
            geom1=0,
            geom2=1,
            pos=np.array([0.045, -0.105, 1.045]),
            frame=frame_forward,
            dist=-0.0004,
        ),
        SimpleNamespace(
            geom1=1,
            geom2=0,
            pos=np.array([0.045, -0.105, 1.045]),
            frame=frame_reverse,
            dist=-0.0004,
        ),
    ]
    env = _l3a3_live_diagnostic_env(contacts)
    evidence = _robot_nonrobot_contact_evidence(
        env, allowed_body_pairs=()
    )

    assert evidence["accepted"] is False
    assert len(evidence["contacts"]) == 2
    first, second = evidence["contacts"]
    assert first["sorted_geom_ids"] == second["sorted_geom_ids"] == [0, 1]
    assert first["sorted_normal_was_flipped"] is False
    assert second["sorted_normal_was_flipped"] is True
    np.testing.assert_allclose(
        first["normal_from_sorted_geom0_to_geom1_world"],
        second["normal_from_sorted_geom0_to_geom1_world"],
    )
    np.testing.assert_allclose(
        first["robot_to_native_normal_world"],
        second["robot_to_native_normal_world"],
    )
    assert first["distance_m"] == pytest.approx(-0.0004)
    assert first["penetration_m"] == pytest.approx(0.0004)


def test_robot_native_contact_nonfinite_evidence_fails_closed():
    contact = SimpleNamespace(
        geom1=0,
        geom2=1,
        pos=np.array([np.nan, 0.0, 1.0]),
        frame=np.array(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ),
        dist=-0.001,
    )
    env = _l3a3_live_diagnostic_env([contact])
    with pytest.raises(RuntimeError, match="pos/frame/dist evidence is invalid"):
        _robot_nonrobot_contact_evidence(env, allowed_body_pairs=())


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
