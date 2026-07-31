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
    _bounded_side_contact_seek_action,
    _contact_depth_sample_validity,
    _contact_progress_saturation_evidence,
    _constraint_prioritized_outside_descent_action,
    _compiled_collision_pair_clearance,
    _compiled_pair_set_clearance,
    _compiled_side_contact_eef_z_feasibility,
    _compiled_trailing_side_contact_candidates,
    _derive_horizon_safe_push_increment,
    _environment_horizon_diagnostics,
    _finger_inward_extents_by_semantic_side,
    _gate_live_contact_offset_xy,
    _horizon_budget,
    _live_plate_tracking_target,
    _outside_side_geometry_feedback_action,
    _outside_side_guard_from_world_aabbs,
    _outside_side_lateral_settle_evidence,
    _outside_side_recovery_progress_evidence,
    _plate_finger_contact_sides,
    _push_window_timeout_evidence,
    _robot_contacts_body,
    _robot_gripper_body_names,
    _select_reachable_compiled_side_candidate,
    _select_reachable_trailing_contact,
    _side_contact_targets_from_compiled_bounds,
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


def test_compiled_trailing_candidates_choose_dual_finger_reachable_plus_x():
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
    assert len(candidates) == 2
    assert selected["offset_xy"] == pytest.approx([0.010, 0.000])
    assert selected["selection_eligible"] is True
    assert selected["dual_finger_contact_skew_m"] == pytest.approx(
        0.000213
    )
    assert selected["outside_high_action_peak"] == pytest.approx(
        1.061875
    )
    assert selected["outside_high_clipped_action_axes"] == [0]
    assert selected["outside_high_action_will_clip"] is True
    assert selected["selection_violations"] == []
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


def test_500096_inward_coupled_descent_requires_lateral_only_settle():
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
        "outward_direction_xy": [1.0, 0.0],
        "required_outside_clearance_m": required_clearance,
        "minimum_outside_clearance_m": 0.002748690,
    }
    after_guard = {
        **before_guard,
        "minimum_outside_clearance_m": 0.002683149,
    }
    evidence = _outside_side_lateral_settle_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
    )
    assert evidence["settled"] is False
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
    )
    assert settle_action[0] > 0.0
    assert settle_action[2] == 0.0
    assert path["maximum_descent_m"] == 0.0

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
    )
    assert feedback["mode"] == "compiled_outside_lateral_settle"
    assert feedback["force_lateral_settle"] is True
    assert settle_action[0] > 0.0
    assert settle_action[2] == 0.0

    settled_guard = {
        **after_guard,
        "minimum_outside_clearance_m": 0.0028,
    }
    settled = _outside_side_lateral_settle_evidence(
        before_guard=after_guard,
        after_guard=settled_guard,
        before_eef=after_eef,
        after_eef=after_eef + np.array([0.0001, 0.0, 0.0001]),
    )
    assert settled["settled"] is True
    assert settled["violations"] == []


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
        '"--plate_contact_outside_clearance", type=float, default=0.005'
        in producer
    )
    assert (
        '"--plate_contact_seek_max_translation_action",\n'
        "        type=float,\n"
        "        default=0.10,"
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
    assert "cause_type={type(exc).__name__}" in bounded_seek
    assert "cause_message={str(exc)!r}" in bounded_seek
    assert "env.set_state" not in compiled_plan
    assert "set_init_state" not in compiled_plan
    assert "rollout.move(" in bounded_seek
    assert "outside_high_target" in bounded_seek
    assert "outside_side_target" in bounded_seek
    assert "_live_outside_side_guard(" in bounded_seek
    assert (
        "_outside_side_geometry_feedback_action("
        in bounded_seek
    )
    assert (
        "for guard_step in range(1, args.max_waypoint_steps + 1)"
        in bounded_seek
    )
    assert '"outside_side_feedback"' in bounded_seek
    assert '"post_action_guard"' in bounded_seek
    assert "_outside_side_recovery_progress_evidence(" in bounded_seek
    assert "_outside_side_lateral_settle_evidence(" in bounded_seek
    assert "force_lateral_settle=(" in bounded_seek
    assert '"lateral_settle_trigger"' in bounded_seek
    assert "recovery_progress[\"fail_closed\"]" in bounded_seek
    assert bounded_seek.index("motion_sample = capture(") < (
        bounded_seek.index("recovery_progress[\"fail_closed\"]")
    )
    assert (
        "orientation is infeasible before table contact"
        in bounded_seek
    )
    assert 'stage.startswith("outside_")' in bounded_seek
    assert "tolerance=" not in bounded_seek
    assert "rollout.advance(action, \"task\")" in bounded_seek
    assert "plate_contact_seek_max_translation_action" in bounded_seek
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
