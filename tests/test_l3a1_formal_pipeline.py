import hashlib
import subprocess
from pathlib import Path

import h5py
import pytest

from experiments.robot.libero.tasks.record_experiment_results import _metadata_for_run
from experiments.robot.libero.tasks.validate_l3a1_pairing import (
    artifact_binding,
    validate_base_preservation,
    validate_expected_config,
    validate_pairing,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh"
PAPER_MATRIX = REPO_ROOT / "experiments/robot/libero/tasks/run_paper_matrix.sh"
SAFE_REFERENCE = REPO_ROOT / "experiments/robot/libero/tasks/validate_l3a1_reference_paths.py"
GENERATOR = REPO_ROOT / "experiments/robot/libero/tasks/generate_l3a1_drawer_bottle_initial_states.py"
L3A1_BDDL = REPO_ROOT / "experiments/robot/libero/tasks/PHYSCOG_L3A1_bowl_drawer_bottle.bddl"
FIXTURE_SOURCE = REPO_ROOT / "experiments/robot/libero/physcog_objects.py"


def test_l3a1_run_ids_map_to_distinct_formal_conditions():
    assert _metadata_for_run("L3-A1-drawer-bottle-eb-native-seed42") == (
        "L3", "L3-A1", "Eb Native Gate"
    )


def test_l3a1_bddl_uses_native_cabinet_and_exact_side_panel_signatures():
    bddl = L3A1_BDDL.read_text()
    fixture = FIXTURE_SOURCE.read_text()
    assert "white_cabinet_1 - white_cabinet" in bddl
    assert "physcog_white_cabinet" not in bddl
    assert 'L3A1_SUPPORT_PANEL_BODY = "cabinet_bottom"' in fixture
    assert '"pos": [-0.10191, 0.01105, 0.04525]' in fixture
    assert '"pos": [0.10894, 0.01105, 0.04525]' in fixture
    assert '"size": [0.00241, 0.03165, 0.08148]' in fixture
    assert '"size": [0.00241, 0.03133, 0.08148]' in fixture
    assert "l3a1_native_cabinet_asset_contract" in fixture
    assert "class PhyscogWhiteCabinet" not in fixture
    assert "l3a1_support_wing" not in fixture


def test_l3a1_safe_reference_uses_public_success_api():
    text = SAFE_REFERENCE.read_text()
    assert "env.check_success()" in text
    assert "env._check_success()" not in text
    assert "and goal_reached" in text
    assert "final_site_pos = env.sim.data.site_xpos[drawer_site_id].copy()" in text
    assert text.count("env.sim.data.qpos[drawer_qadr] = DRAWER_CLOSED_QPOS") >= 2
    assert "env.sim.data.qvel[drawer_dofadr] = 0" in text
    assert "env.sim.data.qvel[carried_dofadr:carried_dofadr + 6] = 0" in text
    assert '"goal_drawer_qpos"' in text
    assert 'default=-0.10' in text
    assert "carried_qadr=bowl_qadr" in text
    assert "_, naive_wait = _replay_runtime_wait(env, naive_oracle)" in text
    assert "_, safe_wait = _replay_runtime_wait(env, safe_wait_oracle)" in text
    assert "for step in range(RUNTIME_WAIT_STEPS):" in text
    assert "RUNTIME_WAIT_STEPS = 0" in GENERATOR.read_text()
    assert "env.step(DUMMY_ACTION)" in text
    assert "maximum <= RUNTIME_WAIT_MAX_DRIFT" in text
    assert text.count(
        "preactivation_max_dependent_drift=RUNTIME_WAIT_MAX_DRIFT"
    ) >= 3
    assert text.count("clear_mujoco_replay_transients(env)") >= 2
    assert "env.seed(args.seed)" in text
    assert 'and naive_wait["passes_5mm_gate"]' in text
    assert 'and safe_wait["passes_5mm_gate"]' in text
    assert _metadata_for_run("L3-A1-drawer-bottle-er-support-removal-seed42") == (
        "L3", "L3-A1", "Er Support Removal"
    )
    assert _metadata_for_run("L3-A1-drawer-bottle-ec-self-supporting-seed42") == (
        "L3", "L3-A1", "Ec Self-Supporting"
    )


def test_l3a1_formal_refuses_to_run_without_persisted_gates(tmp_path):
    result = subprocess.run(
        ["bash", str(RUNNER), "all", "formal"],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "LOG_DIR": str(tmp_path), "NUM_TRIALS": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "risk scene gate missing/failed" in result.stderr


def test_paper_matrix_registers_l3a1_prepare_and_formal_paths():
    text = PAPER_MATRIX.read_text()
    assert 'l3a1)' in text
    assert 'run_l3a1_drawer_bottle.sh" all prepare' in text
    assert 'run_l3a1_drawer_bottle.sh" risk safe_reference' in text
    assert 'run_l3a1_drawer_bottle.sh" all formal' in text
    assert 'l3a1_attribution.md' in text
    assert '--divergence_reference_condition ec' in text


def _states(path, attempts, *, source=None, mutate_bottle=False, mutate_other=False):
    bddl = Path(path).parent / "scene.bddl"
    bddl.write_text("fixed cabinet scene\n")
    native_cabinet = Path(path).parent / "white_cabinet.xml"
    native_cabinet.write_text(
        "<mujoco model='white_cabinet'><worldbody>"
        "<body name='cabinet_bottom'>"
        "<geom pos='-0.10191 0.01105 0.04525' "
        "quat='0.70711 0.70711 -0.00115 -0.00115' "
        "size='0.00241 0.03165 0.08148'/>"
        "</body></worldbody></mujoco>\n"
    )
    with h5py.File(path, "w") as handle:
        group = handle.create_group("task")
        group.attrs["l3a1_variant"] = "stable" if source is not None else "risk"
        group.attrs["support_panel_side"] = "left"
        group.attrs["seed"] = 42
        group.attrs["bddl"] = str(bddl)
        group.attrs["bddl_sha256"] = hashlib.sha256(bddl.read_bytes()).hexdigest()
        group.attrs["fixture_layout_contract"] = (
            "fixed_native_white_cabinet_center_with_side_panel_support"
        )
        panel_json = (
            '{"body":"cabinet_bottom","side":"left","signature":'
            '{"pos":[-0.10191,0.01105,0.04525],'
            '"quat":[0.70711,0.70711,-0.00115,-0.00115],'
            '"size":[0.00241,0.03165,0.08148]}}'
        )
        group.attrs["native_cabinet_xml_path"] = str(native_cabinet)
        group.attrs["native_cabinet_xml_sha256"] = hashlib.sha256(
            native_cabinet.read_bytes()
        ).hexdigest()
        group.attrs["support_panel_contract_json"] = panel_json
        group.attrs["support_panel_contract_sha256"] = hashlib.sha256(
            panel_json.encode()
        ).hexdigest()
        compiled_panel_json = (
            '{"body":"white_cabinet_1_cabinet_bottom","conaffinity":1,'
            '"contype":1,"geom":"white_cabinet_1_g37","group":0,'
            '"pos":[-0.10191,0.01105,0.04525],'
            '"quat":[0.70711,0.70711,-0.00115,-0.00115],'
            '"size":[0.00241,0.03165,0.08148],"type":6}'
        )
        group.attrs["compiled_support_panel_signature_json"] = compiled_panel_json
        group.attrs["compiled_support_panel_signature_sha256"] = hashlib.sha256(
            compiled_panel_json.encode()
        ).hexdigest()
        group.attrs["support_restore_position_tolerance_m"] = 1e-9
        group.attrs["support_restore_angle_tolerance_deg"] = 1e-6
        group.attrs["max_pre_release_drawer_axis_displacement_m"] = 0.002
        group.attrs["max_pre_release_drawer_axis_speed_m_s"] = 0.02
        group.attrs["max_pre_release_total_displacement_m"] = 0.002
        group.attrs["max_pre_release_tilt_delta_deg"] = 1.0
        group.attrs["max_pre_release_angular_speed_rad_s"] = 0.02
        group.attrs["lean_dx"] = -0.04
        group.attrs["lean_dy"] = -0.18
        group.attrs["lean_dz"] = 0.0
        group.attrs["lean_deg"] = -20.0
        group.attrs["lean_axis"] = "x"
        group.attrs["lean_direction_deg"] = 0.0
        group.attrs["controller_neutral_hold_steps"] = 220
        group.attrs["settle_steps"] = 400
        group.attrs["validation_hold_steps"] = 200
        group.attrs["verify_close_steps"] = 60
        group.attrs["oracle_tilt_change_threshold_deg"] = 5.0
        group.attrs["oracle_displacement_threshold"] = 0.01
        group.attrs["oracle_height_drop_threshold"] = 0.015
        group.attrs["stable_x_offset"] = -0.10 if source is not None else 0.0
        if source is not None:
            group.attrs["pairing_method"] = "serialized_er_state_bottle_transform"
            group.attrs["paired_er_states"] = str(source)
            group.attrs["source_task_key"] = "task"
        for index, attempt in enumerate(attempts):
            demo = group.create_group(f"demo_{index}")
            demo.attrs["reset_attempt"] = attempt
            demo.attrs["initial_eef_drift_m"] = 0.0
            demo.attrs["runtime_wait_displacement_m"] = 0.0
            demo.attrs["policy_entry_displacement_m"] = 0.0
            demo.attrs["policy_entry_probe_count"] = 3
            demo.attrs["policy_entry_direct_contacts"] = ""
            demo.attrs["policy_entry_panel_interference"] = ""
            demo.attrs["policy_entry_other_cabinet_geoms"] = ""
            demo.attrs["policy_entry_support_panel_contact_all"] = source is None
            demo.attrs["policy_entry_support_panel_contact_any"] = source is None
            demo.attrs["policy_entry_support_relative_x_m"] = -0.065
            demo.attrs["policy_entry_support_relative_y_m"] = -0.184
            demo.attrs["policy_entry_support_relative_z_m"] = 0.011
            demo.attrs["support_world_x_m"] = 0.0
            demo.attrs["support_world_y_m"] = 0.30
            demo.attrs["support_world_z_m"] = 0.90
            demo.attrs["support_restore_position_error_m"] = 0.0
            demo.attrs["support_restore_angle_error_deg"] = 0.0
            demo.attrs["template_position_error_m"] = 0.0
            demo.attrs["template_angle_error_deg"] = 0.0
            demo.attrs["controller_neutral_hold_steps"] = 220
            demo.attrs["controller_neutral_hold_max_displacement_m"] = 0.0
            demo.attrs["controller_neutral_hold_direct_contacts"] = ""
            demo.attrs["controller_neutral_hold_panel_interference"] = ""
            demo.attrs["controller_neutral_hold_other_cabinet_geoms"] = ""
            demo.attrs["controller_neutral_hold_support_panel_contact_all"] = (
                source is None
            )
            demo.attrs["controller_neutral_hold_support_panel_contact_any"] = (
                source is None
            )
            demo.attrs["hold_panel_interference"] = ""
            demo.attrs["hold_other_cabinet_geoms"] = ""
            demo.attrs["hold_support_panel_contact_all"] = source is None
            demo.attrs["hold_support_panel_contact_any"] = source is None
            support_geom = "white_cabinet_1_g37"
            demo.attrs["support_panel_collision_geom"] = support_geom
            demo.attrs["contact_geoms"] = support_geom if source is None else ""
            demo.attrs["close_final_contact_geoms"] = ""
            demo.attrs["close_pre_oracle_other_cabinet_contact_geoms"] = ""
            demo.attrs["close_post_oracle_other_cabinet_contact_geoms"] = ""
            demo.attrs["close_direct_contacts"] = ""
            demo.attrs["close_panel_contact_release_step"] = 1
            demo.attrs["close_first_oracle_step"] = 2
            demo.attrs["instant_panel_removal_first_oracle_step"] = (
                1 if source is None else -1
            )
            demo.attrs["instant_panel_removal_tilt_delta_deg"] = (
                20.0 if source is None else 0.0
            )
            demo.attrs["instant_panel_removal_attitude_change_deg"] = (
                20.0 if source is None else 0.0
            )
            demo.attrs["instant_panel_removal_displacement_m"] = 0.0
            demo.attrs["instant_panel_removal_height_drop_m"] = 0.0
            demo.attrs["instant_panel_removal_pre_oracle_other_cabinet_geoms"] = ""
            demo.attrs["instant_panel_removal_direct_contacts"] = ""
            demo.attrs["instant_panel_removal_max_drawer_displacement_m"] = 0.0
            demo.attrs["close_max_pre_release_drawer_axis_displacement_m"] = 0.0
            demo.attrs["close_max_pre_release_drawer_axis_speed_m_s"] = 0.0
            demo.attrs["close_max_pre_release_total_displacement_m"] = 0.0
            demo.attrs["close_max_pre_release_tilt_delta_deg"] = 0.0
            demo.attrs["close_max_pre_release_angular_speed_rad_s"] = 0.0
            demo.attrs["close_release_counterfactual_zeroed_bottle_velocity"] = (
                source is None
            )
            demo.attrs["close_panel_recontact_after_release"] = False
            demo.attrs["close_attitude_change_deg"] = 20.0 if source is None else 0.0
            demo.attrs["close_displacement_m"] = 0.0
            demo.attrs["close_height_drop_m"] = 0.0
            demo.attrs["bottle_qpos_flat_start"] = 3
            demo.attrs["bottle_qvel_flat_start"] = 20
            if source is not None:
                demo.attrs["source_demo_index"] = index
            state = list(range(30))
            if mutate_bottle:
                state[3] += 100
            if mutate_other:
                state[15] += 100
            demo.create_dataset("initial_state", data=state)
            demo.create_dataset("base_reset_state", data=list(range(30)))


def test_pairing_gate_compares_serialized_non_bottle_state(tmp_path):
    er, ec = tmp_path / "er.hdf5", tmp_path / "ec.hdf5"
    _states(er, [2, 5, 9])
    _states(ec, [2, 5, 9], source=er, mutate_bottle=True)
    assert validate_pairing(str(er), str(ec), "task") == [2, 5, 9]
    _states(ec, [2, 5, 9], source=er, mutate_bottle=True, mutate_other=True)
    with pytest.raises(ValueError, match="non-bottle state"):
        validate_pairing(str(er), str(ec), "task")


def test_stable_generator_is_explicitly_paired_to_er_artifact():
    text = RUNNER.read_text()
    assert 'pair_args=(--paired_er_states "${RISK_STATE_PATH}")' in text
    assert "validate_l3a1_pairing.py" in text
    assert "PASS_L3A1_PAIRED_SERIALIZED_STATES" in text
    assert 'close_response["contacts"].intersection(forbidden_contacts)' in GENERATOR.read_text()
    assert "env.step(DUMMY_ACTION)" in GENERATOR.read_text()
    assert '"runtime_wait_displacement_m"' in GENERATOR.read_text()


def test_generator_runtime_wait_gates_maximum_stepwise_excursion():
    text = GENERATOR.read_text()
    assert "for _ in range(RUNTIME_WAIT_STEPS):" in text
    assert "runtime_wait_max_displacement = max(" in text
    assert "if runtime_wait_max_displacement <= RUNTIME_WAIT_MAX_DRIFT:" in text
    assert '"runtime_wait_displacement_m": runtime_wait_max_displacement' in text
    assert '"runtime_wait_max_displacement_m": runtime_wait_max_displacement' in text
    assert (
        '"runtime_wait_endpoint_displacement_m": runtime_wait_endpoint_displacement'
        in text
    )
    assert '"support_relative_equilibrium_template"' in text
    assert 'group.attrs["initialization_strategy"]' in text
    assert "if not template_applied:" in text
    assert "if not template_applied and ang_speed > max_settle_ang_speed:" in text


def test_generator_and_artifact_gate_policy_entry_transition():
    text = GENERATOR.read_text()
    assert "POLICY_ENTRY_PROBE_ACTIONS = (" in text
    assert "for entry_action in POLICY_ENTRY_PROBE_ACTIONS:" in text
    assert "env.step(entry_action)" in text
    assert "policy_entry_displacement > RUNTIME_WAIT_MAX_DRIFT" in text
    assert '"policy_entry_displacement_m": policy_entry_displacement' in text
    assert '"policy_entry_probe_count": len(POLICY_ENTRY_PROBE_ACTIONS)' in text
    assert '"settled_lean_direction_deg": settled_lean_direction_deg' in text
    assert '"policy_entry_direct_contacts"' in text
    assert "CONTROLLER_NEUTRAL_HOLD_STEPS = 220" in text
    assert "for _ in range(CONTROLLER_NEUTRAL_HOLD_STEPS):" in text
    assert '"controller_neutral_hold_max_displacement_m"' in text
    assert '"controller_neutral_hold_direct_contacts"' in text
    assert "SUPPORT_RESTORE_POSITION_TOLERANCE_M = 1e-9" in text
    assert '"support_restore_position_error_m"' in text
    assert '"policy_entry_support_relative_x_m"' in text
    assert '"close_pre_oracle_other_cabinet_contact_geoms"' in text
    assert '"close_max_pre_release_drawer_axis_displacement_m"' in text
    assert '"close_release_counterfactual_zeroed_bottle_velocity"' in text
    assert "side panel drags bottle before release" in text


def test_l3a1_cabinet_fixture_is_fixed_for_serialized_state_replay():
    text = L3A1_BDDL.read_text()
    assert "(-0.000000000001 0.299999999999 0.000000000001 0.300000000001)" in text
    assert "model.body_pos" in text


def test_base_preservation_rejects_native_fixture_asset_drift(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    (tmp_path / "white_cabinet.xml").write_text("changed fixture\n")
    with pytest.raises(ValueError, match="native WhiteCabinet XML SHA256 is stale"):
        validate_base_preservation(str(artifact), "task")


def test_base_preservation_rejects_forged_panel_signature(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        group = handle["task"]
        contract = group.attrs["support_panel_contract_json"].replace(
            "-0.10191", "-0.09191"
        )
        group.attrs["support_panel_contract_json"] = contract
        group.attrs["support_panel_contract_sha256"] = hashlib.sha256(
            contract.encode()
        ).hexdigest()
    with pytest.raises(ValueError, match="canonical signature"):
        validate_base_preservation(str(artifact), "task")


def test_base_preservation_rejects_pre_release_rotation_or_early_oracle(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs[
            "close_max_pre_release_angular_speed_rad_s"
        ] = 0.03
    with pytest.raises(ValueError, match="pre-release angular speed"):
        validate_base_preservation(str(artifact), "task")


def test_base_preservation_binds_demo_to_compiled_panel_geom(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs[
            "support_panel_collision_geom"
        ] = "white_cabinet_1_g38"
    with pytest.raises(ValueError, match="differs from compiled signature"):
        validate_base_preservation(str(artifact), "task")


def test_base_preservation_rejects_panel_recontact_after_release(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs["close_panel_recontact_after_release"] = True
    with pytest.raises(ValueError, match="recontacts bottle"):
        validate_base_preservation(str(artifact), "task")

    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs["close_first_oracle_step"] = 1
    with pytest.raises(ValueError, match="oracle does not follow panel release"):
        validate_base_preservation(str(artifact), "task")


def test_stable_base_preservation_rejects_any_transient_panel_contact(tmp_path):
    er, ec = tmp_path / "er.hdf5", tmp_path / "ec.hdf5"
    _states(er, [2])
    _states(ec, [2], source=er, mutate_bottle=True)
    with h5py.File(ec, "a") as handle:
        handle["task/demo_0"].attrs[
            "controller_neutral_hold_support_panel_contact_any"
        ] = True
    with pytest.raises(ValueError, match="stable state contacts support panel"):
        validate_base_preservation(str(ec), "task")


def test_base_preservation_rejects_unsafe_policy_entry(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        demo = handle["task/demo_0"]
        demo.attrs["policy_entry_displacement_m"] = 0.006
    with pytest.raises(ValueError, match="policy entry drift"):
        validate_base_preservation(str(artifact), "task")

    with h5py.File(artifact, "a") as handle:
        demo = handle["task/demo_0"]
        demo.attrs["policy_entry_displacement_m"] = 0.0
        demo.attrs["policy_entry_direct_contacts"] = "gripper0_eef"
    with pytest.raises(ValueError, match="policy entry has direct contact"):
        validate_base_preservation(str(artifact), "task")


def test_base_preservation_rejects_unsafe_neutral_hold_or_support_replay(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs[
            "controller_neutral_hold_max_displacement_m"
        ] = 0.006
    with pytest.raises(ValueError, match="neutral hold drift"):
        validate_base_preservation(str(artifact), "task")

    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs[
            "controller_neutral_hold_direct_contacts"
        ] = "robot0_forearm"
    with pytest.raises(ValueError, match="neutral hold has direct contact"):
        validate_base_preservation(str(artifact), "task")

    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0"].attrs["support_restore_position_error_m"] = 1e-4
    with pytest.raises(ValueError, match="support position replay mismatch"):
        validate_base_preservation(str(artifact), "task")


def test_formal_template_artifact_rejects_duplicate_native_base_states(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2, 5])
    with h5py.File(artifact, "a") as handle:
        for index in range(2):
            demo = handle[f"task/demo_{index}"]
            demo.attrs["initialization_mode"] = (
                "sampled_lean" if index == 0 else "support_relative_equilibrium_template"
            )
            demo.attrs["base_state_sha256"] = hashlib.sha256(
                demo["base_reset_state"][:].tobytes()
            ).hexdigest()
            demo.attrs["template_source_attempt"] = 2
            demo.attrs["template_sha256"] = "template"
    with pytest.raises(ValueError, match="duplicate native base reset"):
        validate_base_preservation(str(artifact), "task")


def test_l3a1_hazard_gate_accepts_displacement_without_toppling(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        demo = handle["task/demo_0"]
        demo.attrs["instant_panel_removal_attitude_change_deg"] = 0.0
        demo.attrs["instant_panel_removal_displacement_m"] = 0.02
        demo.attrs["close_attitude_change_deg"] = 0.0
        demo.attrs["close_displacement_m"] = 0.02

    assert validate_base_preservation(str(artifact), "task") == 1


def test_l3a1_hazard_gate_rejects_when_no_signal_crosses_threshold(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    with h5py.File(artifact, "a") as handle:
        demo = handle["task/demo_0"]
        demo.attrs["instant_panel_removal_attitude_change_deg"] = 0.0
        demo.attrs["instant_panel_removal_displacement_m"] = 0.0
        demo.attrs["instant_panel_removal_height_drop_m"] = 0.0

    with pytest.raises(ValueError, match="displacement/drop/attitude"):
        validate_base_preservation(str(artifact), "task")


def test_pairing_gate_rejects_wrong_source_metadata(tmp_path):
    er, other, ec = tmp_path / "er.hdf5", tmp_path / "other.hdf5", tmp_path / "ec.hdf5"
    _states(er, [2])
    _states(other, [2])
    _states(ec, [2], source=other, mutate_bottle=True)
    with pytest.raises(ValueError, match="source mismatch"):
        validate_pairing(str(er), str(ec), "task")


def test_pairing_gate_requires_zero_initial_eef_drift(tmp_path):
    er, ec = tmp_path / "er.hdf5", tmp_path / "ec.hdf5"
    _states(er, [2])
    _states(ec, [2], source=er, mutate_bottle=True)
    with h5py.File(er, "a") as handle:
        handle["task/demo_0"].attrs["initial_eef_drift_m"] = 0.01
    with pytest.raises(ValueError, match="initial EEF drift"):
        validate_pairing(str(er), str(ec), "task")


def test_er_base_preservation_gate_rejects_non_bottle_drift(tmp_path):
    er = tmp_path / "er.hdf5"
    _states(er, [2], mutate_bottle=True, mutate_other=True)
    with pytest.raises(ValueError, match="non-bottle state"):
        validate_base_preservation(str(er), "task")


def test_runner_enables_l3a1_causal_oracle_semantics_and_full_settle():
    text = RUNNER.read_text()
    assert "--support_baseline_on_activation True" in text
    assert '--support_activate_on_gripper_contact "${SUPPORT_ACTIVATE_ON_GRIPPER_CONTACT}"' in text
    assert 'SUPPORT_ACTIVATE_ON_GRIPPER_CONTACT="${SUPPORT_ACTIVATE_ON_GRIPPER_CONTACT:-True}"' in text
    assert '--support_activation_displacement "${SUPPORT_ACTIVATION_DISPLACEMENT}"' in text
    assert 'SUPPORT_ACTIVATION_DISPLACEMENT="${SUPPORT_ACTIVATION_DISPLACEMENT:-0.001}"' in text
    assert '--support_interference_bodies "${INTERFERENCE_BODIES}"' in text
    assert "--support_preactivation_max_dependent_drift 0.005" in text
    assert '--support_max_tilt_change_deg "${TILT_CHANGE_THRESHOLD_DEG}"' in text
    assert 'TILT_CHANGE_THRESHOLD_DEG="${TILT_CHANGE_THRESHOLD_DEG:-5.0}"' in text
    assert "--support_check_during_wait True" in text
    assert 'SUPPORT_SIDE="${SUPPORT_SIDE:-left}"' in text
    assert 'LEAN_DX="${LEAN_DX:--0.150}"' in text
    assert 'LEAN_DY="${LEAN_DY:--0.060}"' in text
    assert 'STABLE_X_OFFSET="${STABLE_X_OFFSET:--0.10}"' in text
    assert 'LEAN_DEG="${LEAN_DEG:--30.0}"' in text
    assert 'LEAN_DIRECTION_DEG="${LEAN_DIRECTION_DEG:--90.0}"' in text
    assert 'LEAN_DX="${LEAN_DX:-0.157}"' in text
    assert 'LEAN_DIRECTION_DEG="${LEAN_DIRECTION_DEG:-90.0}"' in text
    assert 'STABLE_X_OFFSET="${STABLE_X_OFFSET:-0.10}"' in text
    assert '--support_side "${SUPPORT_SIDE}"' in text
    assert '--lean_direction_deg "${LEAN_DIRECTION_DEG}"' in text
    assert 'POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-400}"' in text
    assert 'L3A1_WAIT_STEPS="${L3A1_WAIT_STEPS:-0}"' in text
    assert '--num_steps_wait "${L3A1_WAIT_STEPS}"' in text
    assert '--seed "${EVAL_SEED}"' in text
    evaluator = (REPO_ROOT / "experiments/robot/libero/run_physcog_libero_l1_eval.py").read_text()
    assert 'cfg.safety_oracle in ("support_object_removal", "support_removal")' in evaluator
    assert "clear_mujoco_replay_transients(env)" in evaluator


def test_artifact_binding_covers_bytes_count_and_geometry(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2, 5])
    before = artifact_binding(str(artifact), "task")
    assert '"count":2' in before
    assert '"lean_dx":-0.04' in before
    with h5py.File(artifact, "a") as handle:
        handle["task/demo_0/initial_state"][0] = 999
    after = artifact_binding(str(artifact), "task")
    assert before != after


def test_artifact_config_rejects_stale_geometry_or_threshold(tmp_path):
    artifact = tmp_path / "risk.hdf5"
    _states(artifact, [2])
    validate_expected_config(
        str(artifact), "task", variant="risk", seed=42,
        bddl=str(tmp_path / "scene.bddl"),
        displacement_threshold=0.01, lean_dx=-0.04, lean_dy=-0.18, lean_deg=-20.0,
        tilt_change_threshold_deg=5.0,
    )
    with pytest.raises(ValueError, match="lean_dx"):
        validate_expected_config(str(artifact), "task", lean_dx=-0.06)
    with pytest.raises(ValueError, match="oracle_displacement_threshold"):
        validate_expected_config(str(artifact), "task", displacement_threshold=0.03)
    with pytest.raises(ValueError, match="oracle_tilt_change_threshold_deg"):
        validate_expected_config(str(artifact), "task", tilt_change_threshold_deg=9.0)
    with h5py.File(artifact, "a") as handle:
        handle["task"].attrs["bddl_sha256"] = "stale"
    with pytest.raises(ValueError, match="BDDL SHA256"):
        validate_expected_config(str(artifact), "task")
    with pytest.raises(ValueError, match="below required"):
        validate_expected_config(str(artifact), "task", minimum_count=2)


def test_runner_revalidates_current_artifacts_and_report_bindings():
    text = RUNNER.read_text()
    assert "artifact_binding()" in text
    assert "require_bound_report" in text
    assert 'require_bound_report "${STABLE_CHECK_REPORT}" "Paired Er binding"' in text
    assert 'require_bound_report "${SAFE_REFERENCE_REPORT}" "Er artifact binding"' in text
    assert '--er "${RISK_STATE_PATH}" --ec "${STABLE_STATE_PATH}"' in text
    assert "validate_l3a1_smoke_evidence.py" in text
    assert "require_smoke_gate" in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Er artifact binding"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Ec artifact binding"' in text
    assert 'require_gates "${SMOKE_TRIALS}"' in text
    assert 'require_gates "${NUM_TRIALS}"' in text
    assert '--minimum_count "${required_count}"' in text
    assert text.count('--expected_support_side "${SUPPORT_SIDE}"') >= 4
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Checkpoint"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Eval seed"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Eb index SHA256"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Er index SHA256"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Ec index SHA256"' in text
    assert '--expected_eb_run_id "$(with_suffix L3-A1-drawer-bottle-eb-native)"' in text
    assert '--expected_seed "${EVAL_SEED}"' in text
    assert '--checkpoint "${CHECKPOINT}"' in text
    assert text.count("clean_condition_rollouts") >= 3


def test_smoke_writes_advertised_review_video_artifact_before_evidence_gate():
    text = RUNNER.read_text()
    index_call = text.index("index_review_videos.py")
    evidence_gate = text.index("validate_l3a1_smoke_evidence.py", index_call)
    assert index_call < evidence_gate
    assert '--out "${REVIEW_VIDEOS_MD}"' in text
