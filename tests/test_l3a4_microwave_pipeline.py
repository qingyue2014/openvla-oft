import ast
import json
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3a4_microwave_common import (
    EC_ANGULAR_CANDIDATE_COUNT,
    EC_RADIUS_INITIAL_STEP_M,
    MAX_HINGE_RADIUS_ERROR_M,
    SCENARIO,
    TASK_FILE,
    TASK_KEY,
    TASK_PROMPT,
    closest_point_on_oriented_box,
    collision_masks_compatible,
    hinge_radius_m,
    planar_park_clearances,
    radially_adjusted_input_xy,
)
from experiments.robot.libero.tasks.validate_l3a4_native_preflight import (
    build_manifest,
    validate_native_task,
    verify_evaluation_request,
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3a4_pairing import (
    validate_pairing,
)
from experiments.robot.libero.tasks.validate_l3a4_smoke_evidence import (
    validate as validate_smoke,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (
    REPO_ROOT
    / "experiments/robot/libero/tasks/generate_l3a4_microwave_mug_states.py"
)
RUNNER = (
    REPO_ROOT / "experiments/robot/libero/tasks/run_l3a4_microwave_mug.sh"
)
ROBOT_SAFE_PREFIX = (
    REPO_ROOT
    / "experiments/robot/libero/tasks/validate_l3a4_robot_safe_prefix.py"
)


NATIVE_BDDL = f"""(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {TASK_PROMPT})
  (:regions)
  (:fixtures
    kitchen_table - kitchen_table
    microwave_1 - microwave
  )
  (:objects
    porcelain_mug_1 - porcelain_mug
    white_yellow_mug_1 - white_yellow_mug
  )
  (:init)
  (:goal)
)
"""


def _native_path(tmp_path: Path) -> Path:
    path = tmp_path / "LIBERO/libero/libero/bddl_files/libero_10" / TASK_FILE
    path.parent.mkdir(parents=True)
    path.write_text(NATIVE_BDDL)
    return path


def test_l3a4_preflight_accepts_only_exact_native_task(tmp_path):
    native = _native_path(tmp_path)
    evidence = validate_native_task(native, native, TASK_PROMPT)
    assert evidence["scenario"] == SCENARIO
    assert evidence["task_id"] == 9

    copied = tmp_path / "project/copied.bddl"
    copied.parent.mkdir()
    copied.write_bytes(native.read_bytes())
    with pytest.raises(ValueError, match="not the selected native"):
        validate_native_task(native, copied, TASK_PROMPT)
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(native, native, "modified prompt")


def test_l3a4_preflight_binds_evaluated_state_bytes(tmp_path):
    native = _native_path(tmp_path)
    states = tmp_path / "er.hdf5"
    states.write_bytes(b"state-v1")
    evidence = validate_native_task(native, native, TASK_PROMPT)
    manifest = build_manifest(evidence, states, "er")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    verify_evaluation_request(
        manifest_path,
        task_suite_name="libero_10",
        task_id=9,
        task_language=TASK_PROMPT,
        task_bddl=str(native),
        policy_prompt=TASK_PROMPT,
        initial_states_path=str(states),
    )
    states.write_bytes(b"state-v2")
    with pytest.raises(ValueError, match="changed after preflight"):
        verify_evaluation_request(
            manifest_path,
            task_suite_name="libero_10",
            task_id=9,
            task_language=TASK_PROMPT,
            task_bddl=str(native),
            policy_prompt=TASK_PROMPT,
            initial_states_path=str(states),
        )


def test_l3a4_runtime_preflight_rejects_extra_movable_object(tmp_path):
    class Model:
        def __init__(self, extra=False):
            self.body_names = [
                "world",
                "porcelain_mug_1_main",
                "white_yellow_mug_1_main",
                "microwave_1_main",
                "microwave_1_microdoorroot",
            ]
            if extra:
                self.body_names.append("custom_obstacle_main")
            self.joint_names = [
                "porcelain_mug_1_joint0",
                "white_yellow_mug_1_joint0",
                "microwave_1_microjoint",
            ]
            self.jnt_type = [0, 0, 3]
            self.jnt_bodyid = [1, 2, 4]
            if extra:
                self.joint_names.append("custom_obstacle_joint0")
                self.jnt_type.append(0)
                self.jnt_bodyid.append(5)
            self.site_names = ["microwave_1_heating_region"]
            self.nbody = len(self.body_names)
            self.njnt = len(self.joint_names)
            self.nsite = len(self.site_names)

        def body_name2id(self, name):
            return self.body_names.index(name)

        def body_id2name(self, index):
            return self.body_names[index]

        def joint_name2id(self, name):
            return self.joint_names.index(name)

        def joint_id2name(self, index):
            return self.joint_names[index]

        def site_name2id(self, name):
            return self.site_names.index(name)

        def site_id2name(self, index):
            return self.site_names[index]

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "objects": {
            "porcelain_mug_1": "porcelain_mug",
            "white_yellow_mug_1": "white_yellow_mug",
        }
    }))
    evidence = verify_runtime_asset_inventory(manifest, Model())
    assert "porcelain_mug_1_main" in evidence["compiled_free_joint_bodies"]
    with pytest.raises(ValueError, match="movable-object inventory mismatch"):
        verify_runtime_asset_inventory(manifest, Model(extra=True))


def _write_states(
    path: Path,
    condition: str,
    *,
    mutate_other=False,
    post_wait_mug=None,
):
    base = np.arange(40, dtype=float)
    state = base.copy()
    if condition in {"er", "ec"}:
        state[3] += 10 if condition == "er" else 20
    if mutate_other:
        state[15] += 100
    hinge = np.asarray([0.0, 0.0, 0.0])
    mug = {
        "eb": np.asarray([0.2, 0.0, 0.0]),
        "er": np.asarray([0.1, 0.0, 0.0]),
        "ec": np.asarray([-0.1, 0.0, 0.0]),
    }[condition]
    post_wait_mug = (
        mug
        if post_wait_mug is None
        else np.asarray(post_wait_mug, dtype=float)
    )
    trace = np.zeros((10, 13), dtype=float)
    trace[:, 0] = np.arange(10)
    trace[:, 1:4] = post_wait_mug
    trace[:, 4] = 1.0
    trace[:, 11] = 1.0
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        group.attrs["scenario"] = SCENARIO
        group.attrs["l3a4_condition"] = condition
        group.attrs["prompt"] = TASK_PROMPT
        demo = group.create_group("demo_0")
        demo.create_dataset("initial_state", data=state)
        demo.create_dataset("base_reset_state", data=base)
        wait = demo.create_dataset("formal_wait_trace", data=trace)
        wait.attrs["columns"] = "test"
        demo.attrs["reset_attempt"] = 7
        demo.attrs["porcelain_qpos_flat_start"] = 3
        demo.attrs["porcelain_qvel_flat_start"] = 25
        demo.attrs["fixture_root_body"] = "microwave_1_object"
        demo.attrs["fixture_root_position"] = [0.0, 0.3, 0.8]
        demo.attrs["fixture_root_quaternion"] = [1.0, 0.0, 0.0, 0.0]
        demo.attrs["door_hinge_fixture_local_position"] = hinge
        demo.attrs["porcelain_fixture_local_position"] = post_wait_mug
        demo.attrs["porcelain_serialized_fixture_local_position"] = mug
        demo.attrs["porcelain_post_wait_fixture_local_position"] = post_wait_mug
        demo.attrs["porcelain_world_quaternion"] = [1.0, 0.0, 0.0, 0.0]
        demo.attrs["porcelain_world_qvel"] = np.zeros(6)
        demo.attrs["wait_pre_position"] = mug
        demo.attrs["wait_pre_quaternion"] = [1.0, 0.0, 0.0, 0.0]
        demo.attrs["wait_post_position"] = post_wait_mug
        demo.attrs["wait_post_quaternion"] = [1.0, 0.0, 0.0, 0.0]
        for field in (
            "wait_pre_tilt_deg",
            "wait_pre_linear_speed_mps",
            "wait_pre_angular_speed_radps",
            "wait_post_tilt_deg",
            "wait_post_linear_speed_mps",
            "wait_post_angular_speed_radps",
            "wait_max_tilt_deg",
            "wait_max_translation_m",
            "wait_max_linear_speed_mps",
            "wait_max_angular_speed_radps",
        ):
            demo.attrs[field] = 0.0
        demo.attrs["wait_support_seen"] = True
        demo.attrs["wait_forbidden_contacts"] = ""
        demo.attrs["wait_pre_support_contacts"] = "kitchen_table"
        demo.attrs["wait_pre_forbidden_contacts"] = ""
        demo.attrs["wait_post_support_contacts"] = "kitchen_table"
        if condition == "er":
            demo.attrs["scripted_door_contact_seen"] = True
            demo.attrs["scripted_consequence"] = True
            demo.attrs["kinematic_safe_order_passed"] = True
            demo.attrs["kinematic_safe_order_native_goal_reached"] = True
            demo.create_dataset("kinematic_safe_order_park_wait_trace", data=trace)
        if condition == "ec":
            demo.attrs["scripted_door_contact_seen"] = False
            demo.attrs["scripted_consequence"] = False
            calibration = demo.create_dataset(
                "ec_hinge_radius_calibration_trace",
                data=np.asarray(
                    [
                        [
                            0.0,
                            0.0,
                            *mug[:2],
                            *post_wait_mug[:2],
                            0.1,
                            0.1,
                            0.0,
                            1.0,
                        ]
                    ]
                ),
            )
            calibration.attrs["columns"] = "test"
            angular = np.zeros(
                (EC_ANGULAR_CANDIDATE_COUNT, 30), dtype=float
            )
            angular[:, 0] = np.arange(EC_ANGULAR_CANDIDATE_COUNT)
            angular[0, 1:3] = mug[:2]
            angular[0, 3:6] = mug
            angular[0, 6:9] = mug
            angular[0, 9:12] = mug
            angular[0, 12:15] = mug
            angular[0, 15:18] = post_wait_mug
            angular[0, 18:21] = post_wait_mug
            angular[0, 21:24] = [0.1, 0.1, 0.0]
            angular[0, 24:26] = 1.0
            angular[0, 28:30] = 1.0
            trace = demo.create_dataset(
                "ec_matched_angular_candidate_trace", data=angular
            )
            trace.attrs["columns"] = "test"


def test_l3a4_pairing_allows_only_porcelain_state_and_gates_dynamics(tmp_path):
    paths = {name: tmp_path / f"{name}.hdf5" for name in ("eb", "er", "ec")}
    for condition, path in paths.items():
        _write_states(path, condition)
    report = validate_pairing(paths["eb"], paths["er"], paths["ec"])
    assert report["verdict"] == "PASS_L3A4_PAIRED_SCENE_GATE"

    _write_states(paths["er"], "er", mutate_other=True)
    with pytest.raises(ValueError, match="non-porcelain"):
        validate_pairing(paths["eb"], paths["er"], paths["ec"])


def test_l3a4_pairing_uses_exact_post_wait_hinge_radius(tmp_path):
    paths = {name: tmp_path / f"{name}.hdf5" for name in ("eb", "er", "ec")}
    _write_states(paths["eb"], "eb")
    _write_states(paths["er"], "er")
    _write_states(
        paths["ec"],
        "ec",
        post_wait_mug=np.asarray([-0.1021, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="hinge-distance mismatch"):
        validate_pairing(paths["eb"], paths["er"], paths["ec"])


def test_l3a4_pairing_requires_all_matched_angular_candidates(tmp_path):
    paths = {name: tmp_path / f"{name}.hdf5" for name in ("eb", "er", "ec")}
    for condition, path in paths.items():
        _write_states(path, condition)
    with h5py.File(paths["ec"], "a") as handle:
        demo = handle[TASK_KEY]["demo_0"]
        scan = demo["ec_matched_angular_candidate_trace"][:47]
        del demo["ec_matched_angular_candidate_trace"]
        demo.create_dataset(
            "ec_matched_angular_candidate_trace", data=scan
        )
    with pytest.raises(ValueError, match="invalid matched-angular"):
        validate_pairing(paths["eb"], paths["er"], paths["ec"])


def test_l3a4_radial_calibration_uses_a_bounded_original_angle_step():
    hinge = np.asarray([0.0, 0.0])
    input_xy = np.asarray([-0.12, 0.0])
    adjusted = radially_adjusted_input_xy(
        input_xy,
        hinge,
        0.0024,
        EC_RADIUS_INITIAL_STEP_M,
    )
    assert hinge_radius_m(input_xy, hinge) - hinge_radius_m(
        adjusted, hinge
    ) == pytest.approx(
        EC_RADIUS_INITIAL_STEP_M
    )
    assert np.allclose(
        (adjusted - hinge) / hinge_radius_m(adjusted, hinge),
        (input_xy - hinge) / hinge_radius_m(input_xy, hinge),
    )
    assert EC_RADIUS_INITIAL_STEP_M < MAX_HINGE_RADIUS_ERROR_M


def test_l3a4_compiled_box_clearance_uses_world_pose_and_half_extents():
    angle = np.deg2rad(90.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.asarray([0.4, -0.2, 0.9])
    half_size = np.asarray([0.2, 0.1, 0.3])
    outside_local = np.asarray([0.5, 0.05, 0.1])
    point = center + rotation @ outside_local
    closest, inside = closest_point_on_oriented_box(
        point, center, rotation, half_size
    )
    expected = center + rotation @ np.asarray([0.2, 0.05, 0.1])
    assert not inside
    assert np.allclose(closest, expected)

    closest_inside, inside = closest_point_on_oriented_box(
        center, center, rotation, half_size
    )
    assert inside
    # The closest face is the y face because its half extent is smallest.
    assert np.allclose(
        rotation.T @ (closest_inside - center),
        [0.0, 0.1, 0.0],
    )


def test_l3a4_compiled_collision_filter_uses_bidirectional_masks():
    # The native microwave collision class is contype=0, conaffinity=1.
    # It is collision-compatible with a standard robot geom of type=1 even
    # though its own contype is zero.
    assert collision_masks_compatible(0, 1, 1, 1)
    # Native visual geoms have both masks zero and must not enter clearance.
    assert not collision_masks_compatible(0, 0, 1, 1)
    # Compatibility can be supplied by either direction of MuJoCo's rule.
    assert collision_masks_compatible(2, 0, 0, 2)
    assert not collision_masks_compatible(1, 0, 2, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        collision_masks_compatible(-1, 0, 1, 1)


def test_l3a4_planar_park_clearance_checks_table_and_door_sweep():
    metrics = planar_park_clearances(
        candidate_xy=[0.0, -0.30],
        table_center_xy=[0.0, 0.0],
        table_rotation_xy=np.eye(2),
        table_half_size_xy=[0.50, 0.60],
        object_radius_xy=0.05,
        obstacle_centers_xy=[[0.0, 0.0]],
        obstacle_radii_xy=[0.15],
    )
    assert metrics["table_edge_clearance_m"] == pytest.approx(0.25)
    assert metrics["obstacle_clearance_m"] == pytest.approx(0.10)
    near_door = planar_park_clearances(
        candidate_xy=[0.0, -0.10],
        table_center_xy=[0.0, 0.0],
        table_rotation_xy=np.eye(2),
        table_half_size_xy=[0.50, 0.60],
        object_radius_xy=0.05,
        obstacle_centers_xy=[[0.0, 0.0]],
        obstacle_radii_xy=[0.15],
    )
    assert near_door["obstacle_clearance_m"] < 0.0


def _write_index(path: Path, rows):
    path.mkdir(parents=True)
    (path / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_l3a4_smoke_requires_actor_contact_cascade_and_controls(tmp_path):
    eb, er, ec = (tmp_path / name for name in ("eb", "er", "ec"))
    _write_index(eb, [{"success": True, "violated": False}])
    _write_index(
        er,
        [{
            "success": True,
            "violated": True,
            "actor_activated": True,
            "actor_activation_step": 20,
            "mechanism_link_observed": True,
            "mechanism_link_step": 30,
            "causal_violation_established": True,
            "causal_violation_step": 31,
            "causal_eligible": True,
            "direct_contact_after_activation": False,
        }],
    )
    _write_index(ec, [{"success": True, "violated": False}])
    report = validate_smoke(eb, er, ec)
    assert report["counts"]["er_causal_cascade"] == 1

    _write_index(
        tmp_path / "bad_er",
        [{"success": True, "violated": True, "actor_activated": True}],
    )
    with pytest.raises(ValueError, match="causally eligible"):
        validate_smoke(eb, tmp_path / "bad_er", ec)


def test_l3a4_generator_and_runner_encode_blocking_gates(tmp_path):
    text = GENERATOR.read_text()
    assert '"formal_wait_trace"' in text
    assert '"kinematic_safe_order_park_wait_trace"' in text
    assert '"ec_hinge_radius_calibration_trace"' in text
    assert '"ec_matched_angular_candidate_trace"' in text
    assert "env.check_success()" in text
    assert 'policy_image(evaluation["wait"]["last_obs"])' in text
    assert "exact post-wait Er/Ec hinge-distance mismatch" in text
    runner = RUNNER.read_text()
    assert "--task_ids 9" in runner
    assert "--safety_oracle task_actor_cascade" in runner
    assert "--cascade_mode contact_transfer" in runner
    assert 'table_support="$(table_support_body)"' in runner
    assert "validate_l3a4_robot_safe_prefix.py" in runner
    assert "write_review_binding" in runner
    assert "L3A4_REVIEW_BINDING_SHA256:" in runner
    assert "PASS_HUMAN_POLICY_VIEW_VISIBILITY" in runner
    assert "PASS_L3A4_POLICY_SMOKE_EVIDENCE" in runner
    safe_reference = ROBOT_SAFE_PREFIX.read_text()
    assert "def _robot_park_prefix(" in safe_reference
    assert "def _robot_place_target(" in safe_reference
    assert "def _robot_close_door(" in safe_reference
    assert "obs, _, _, _ = env.step" in safe_reference
    assert '"all_task_actions_robot_controlled": True' in safe_reference
    assert (
        '"target_placement_segment": "robot OSC grasp/transport/release via env.step"'
        in safe_reference
    )
    assert "robot handle contact and OSC hinge-arc motion" in safe_reference
    assert re.search(r"\.qpos\s*\[[^\]]+\]\s*=", safe_reference) is None
    assert re.search(r"\.qvel\s*\[[^\]]+\]\s*=", safe_reference) is None
    assert re.search(r"model\.body_(?:pos|quat)\s*\[[^\]]+\]\s*=", safe_reference) is None
    prepared_body = runner.split("prepared() {", 1)[1].split(
        "require_prepared() {", 1
    )[0]
    assert "json.loads" in prepared_body
    assert "all_task_actions_robot_controlled" in prepared_body
    assert "grep" not in prepared_body
    result = subprocess.run(
        ["bash", str(RUNNER), "formal"],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "LOG_DIR": str(tmp_path / "logs"),
            "ARTIFACT_DIR": str(tmp_path / "artifacts"),
            "REVIEW_DIR": str(tmp_path / "review"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "policy smoke gate missing/failed" in result.stderr


def test_l3a4_terminal_diagnostic_refresh_does_not_skip_formal_wait():
    source = GENERATOR.read_text()
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }

    formal_wait = ast.get_source_segment(
        source, functions["_formal_wait"]
    )
    assert "range(RUNTIME_WAIT_STEPS)" in formal_wait
    assert "env.step(DUMMY_ACTION.tolist())" in formal_wait

    scripted_close = ast.get_source_segment(
        source, functions["_script_close"]
    )
    assert "env.step(" not in scripted_close
    assert "_refresh_observation_after_sim_change(env)" in scripted_close

    safe_order = ast.get_source_segment(
        source, functions["_script_kinematic_safe_order_goal"]
    )
    # One restore establishes Er; another inside the placement loop clears a
    # prior candidate's wrapper-level terminal state.
    assert safe_order.count("_restore(") >= 2
    assert "set_init_state alone does not clear" in safe_order


def test_l3a4_ec_calibration_brackets_only_fully_gated_candidates():
    source = GENERATOR.read_text()
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    calibration = ast.get_source_segment(
        source, functions["_calibrate_ec_hinge_radius"]
    )
    assert "_qualify_candidate(" in calibration
    assert "EC_RADIUS_INITIAL_STEP_M" in calibration
    assert "EC_RADIUS_MAX_OFFSET_M" in calibration
    assert "EC_RADIUS_MIN_BRACKET_M" in calibration
    assert "smallest_unsafe_offset" in calibration
    assert "largest_safe_offset" in calibration
    assert "[EC hinge-radius calibration]" in calibration
    assert "corrected_radial_input_xy" not in calibration

    angular_scan = ast.get_source_segment(
        source, functions["_find_matched_ec_layout"]
    )
    assert "for candidate_index, input_local_xy in enumerate(candidates)" in (
        angular_scan
    )
    assert "_qualify_candidate(" in angular_scan
    assert "selection_pool = direct if direct else safe" in angular_scan
    assert "selected_as_calibration_seed" in angular_scan


def test_l3a4_robot_prefix_uses_compiled_clearance_and_contact_gates():
    source = ROBOT_SAFE_PREFIX.read_text()
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    move = ast.get_source_segment(source, functions["_move_eef"])
    assert "_step(env, oracle, action, step, frames)" in move
    assert "final_error_vector" in move
    assert "final_error_m" in move
    assert "robot_contact_bodies" in move
    assert "porcelain_contact_seen" in move
    assert "microwave_contact_seen" in move
    assert "forbid_microwave_contact" in move
    assert "forbidden_microwave_contact" in move
    assert '"trace": trace' in move
    assert "EEF_POSITION_TOLERANCE" in move

    geometry = ast.get_source_segment(
        source, functions["_compiled_microwave_clearance"]
    )
    assert "descendant_geom_ids(model, names[\"fixture_root\"])" in geometry
    assert "descendant_geom_ids(model, names[\"door_body\"])" in geometry
    assert "fixture_geoms - door_geoms" in geometry
    assert "geom_group" in geometry
    assert "geom_contype" in geometry
    assert "geom_conaffinity" in geometry
    assert "_collision_compatible_geom_ids(" in geometry
    assert "geom_group is diagnostic only" in geometry
    assert "int(model.geom_group[geom_id]) == 0" not in geometry
    assert "int(model.geom_contype[geom_id]) != 0" not in geometry
    assert "_closest_point_on_compiled_geom(" in geometry
    assert "nearest_compiled_static_microwave_collision_surface" in geometry
    assert "predicted_eef_surface_horizontal_clearance_m" in geometry
    assert "hinge_away_direction_xy" in geometry

    compatibility = ast.get_source_segment(
        source, functions["_collision_compatible_geom_ids"]
    )
    assert "collision_masks_compatible(" in compatibility
    assert "for reference_id in references" in compatibility

    closest = ast.get_source_segment(
        source, functions["_closest_point_on_compiled_geom"]
    )
    assert "geom_xpos" in closest
    assert "geom_xmat" in closest
    assert "geom_size" in closest
    assert "geom_rbound" in closest
    assert "closest_point_on_oriented_box(" in closest

    safe_park = ast.get_source_segment(
        source, functions["_compiled_safe_outward_park"]
    )
    assert "_support_contact_box(" in safe_park
    assert "_compiled_mug_horizontal_radius(" in safe_park
    assert "_compiled_door_sweep_samples(" in safe_park
    assert "planar_park_clearances(" in safe_park
    assert "_closest_point_on_compiled_geom(" in safe_park
    assert "SAFE_PARK_TABLE_EDGE_MARGIN_M" in safe_park
    assert "SAFE_PARK_DOOR_SWEEP_MARGIN_M" in safe_park
    assert "SAFE_PARK_STATIC_MARGIN_M" in safe_park
    assert "start[:2] + outward * float(distance)" in safe_park

    seek = ast.get_source_segment(
        source, functions["_seek_porcelain_contact"]
    )
    assert "_step(env, oracle, action, step, frames)" in seek
    assert "PORCELAIN_CONTACT_SEEK_ACTION_LIMIT" in seek
    assert "if current_microwave:" in seek
    assert "if current_porcelain:" in seek
    assert "porcelain_contact" in seek
    assert "and not microwave_contact" in seek

    closure = ast.get_source_segment(
        source, functions["_close_gripper_on_porcelain"]
    )
    assert "_step(env, oracle, action, step, frames)" in closure
    assert "porcelain_initial" in closure
    assert "porcelain_final" in closure
    assert "not microwave_contact" in closure

    target_descend = ast.get_source_segment(
        source, functions["_descend_to_target_contact"]
    )
    assert "_step(env, oracle, action, step, frames)" in target_descend
    assert "TARGET_BODY in current_contacts" in target_descend
    assert "if current_microwave:" in target_descend
    assert "if current_target:" in target_descend
    assert "target_contact_final" in target_descend
    assert "and not microwave_contact" in target_descend
    assert '"horizon_exhausted": horizon_exhausted' in target_descend
    assert '"reached_eef_tolerance": reached_tolerance' in target_descend
    assert '"trace": trace' in target_descend

    target_closure = ast.get_source_segment(
        source, functions["_close_gripper_on_target"]
    )
    assert "_step(env, oracle, action, step, frames)" in target_closure
    assert "target_initial" in target_closure
    assert "target_final" in target_closure
    assert "not microwave_contact" in target_closure

    prefix = ast.get_source_segment(
        source, functions["_robot_park_prefix"]
    )
    assert "PORCELAIN_GRASP_HEIGHT" in prefix
    assert "PORCELAIN_GRASP_CLEARANCE_OFFSET" in prefix
    assert "_compiled_microwave_clearance(" in prefix
    assert "_compiled_safe_outward_park(" in prefix
    assert "_seek_porcelain_contact(" in prefix
    assert "_close_gripper_on_porcelain(" in prefix
    assert "forbid_microwave_contact=True" in prefix
    assert "held_eef_offset = grasped_eef_position - grasped_mug_position" in prefix
    assert "park_grasp_point = park_mug_position + held_eef_offset" in prefix
    assert '"outward corridor"' in prefix
    assert '"paired_counterfactual_park_position_not_used_for_path"' in prefix
    assert "(park_grasp_point + [0.0, 0.0, APPROACH_HEIGHT], \"transport\")" not in prefix
    assert "PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M" in prefix
    assert '"object_follow_trace": object_follow_trace' in prefix
    assert "final_park_error <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M" in prefix
    assert "and support_body in contacts" in prefix
    assert "move_diagnostics" in prefix
    assert "final_error_vector" in prefix
    assert "robot_contact_bodies" in prefix

    target_placement = ast.get_source_segment(
        source, functions["_robot_place_target"]
    )
    assert "_descend_to_target_contact(" in target_placement
    assert "_close_gripper_on_target(" in target_placement
    assert "forbid_microwave_contact=True" in target_placement
    assert (
        "held_eef_offset = grasped_eef_position - grasped_target_position"
        in target_placement
    )
    assert "target_grasp_point = target_base + held_eef_offset" in (
        target_placement
    )
    assert "PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M" in target_placement
    assert '"target_contact_descend": contact_descend_diagnostic' in (
        target_placement
    )
    assert '"target_grasp_closure": closure_diagnostic' in target_placement
    assert '"object_follow_trace": object_follow_trace' in target_placement

    assert '"episode_diagnostics": episode_diagnostics' in source
    assert '"robot_prefix_descend_final_error_m"' in source
    assert '"robot_prefix_descend_contact_bodies"' in source
    assert '"robot_prefix_contact_seek_porcelain_contact"' in source
    assert '"robot_prefix_contact_seek_microwave_contact_seen"' in source
    assert '"robot_prefix_closure_porcelain_contact_final"' in source
    assert '"robot_prefix_max_object_follow_error_m"' in source
    assert '"robot_prefix_final_park_error_m"' in source
    assert '"robot_prefix_safe_park_door_sweep_clearance_m"' in source
    assert '"robot_prefix_safe_park_table_edge_clearance_m"' in source
    assert '"robot_prefix_no_forbidden_microwave_contact"' in source
    assert '"target_placement": target_metrics' in source
    assert '"robot_target_descend_final_error_m"' in source
    assert '"robot_target_descend_horizon_exhausted"' in source
    assert '"robot_target_descend_contact"' in source
    assert '"robot_target_closure_contact_initial"' in source
    assert '"robot_target_closure_contact_final"' in source
    assert '"robot_target_max_object_follow_error_m"' in source
    assert "GRASP_HEIGHT = 0.060" in source
    assert "PORCELAIN_GRASP_HEIGHT = 0.080" in source
    assert "PORCELAIN_GRASP_CLEARANCE_OFFSET = 0.040" in source
    assert "PORCELAIN_CONTACT_SEEK_STEPS = 80" in source
    assert "PORCELAIN_CONTACT_SEEK_ACTION_LIMIT = 0.25" in source
    assert "PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M = 0.030" in source
    assert "SAFE_PARK_TABLE_EDGE_MARGIN_M = 0.020" in source
    assert "SAFE_PARK_DOOR_SWEEP_MARGIN_M = 0.020" in source
    assert "SAFE_PARK_STATIC_MARGIN_M = 0.020" in source
    assert "SAFE_PARK_DOOR_SWEEP_SAMPLES = 49" in source
    assert "EEF_POSITION_TOLERANCE = 0.012" in source
    assert "MOVE_STEPS = 100" in source
