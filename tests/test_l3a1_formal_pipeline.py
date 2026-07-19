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


def test_l3a1_run_ids_map_to_distinct_formal_conditions():
    assert _metadata_for_run("L3-A1-drawer-bottle-eb-native-seed42") == (
        "L3", "L3-A1", "Eb Native Gate"
    )


def test_l3a1_safe_reference_uses_public_success_api():
    text = SAFE_REFERENCE.read_text()
    assert "env.check_success()" in text
    assert "env._check_success()" not in text
    assert "and goal_reached" in text
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
    with h5py.File(path, "w") as handle:
        group = handle.create_group("task")
        group.attrs["l3a1_variant"] = "stable" if source is not None else "risk"
        group.attrs["seed"] = 42
        group.attrs["bddl"] = "scene.bddl"
        group.attrs["lean_dx"] = -0.04
        group.attrs["lean_dy"] = -0.18
        group.attrs["lean_dz"] = 0.0
        group.attrs["lean_deg"] = -20.0
        group.attrs["lean_axis"] = "x"
        group.attrs["lean_direction_deg"] = 0.0
        group.attrs["settle_steps"] = 400
        group.attrs["validation_hold_steps"] = 200
        group.attrs["verify_close_steps"] = 60
        group.attrs["min_topple_deg"] = 10.0
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
    assert "--support_activate_on_gripper_contact False" in text
    assert '--support_interference_bodies "${INTERFERENCE_BODIES}"' in text
    assert "--support_preactivation_max_dependent_drift 0.005" in text
    assert "--support_check_during_wait True" in text
    assert 'LEAN_DX="${LEAN_DX:--0.06}"' in text
    assert 'LEAN_DY="${LEAN_DY:--0.181}"' in text
    assert 'LEAN_DEG="${LEAN_DEG:--20.0}"' in text
    assert 'LEAN_DIRECTION_DEG="${LEAN_DIRECTION_DEG:-30.0}"' in text
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
        str(artifact), "task", variant="risk", seed=42, bddl="scene.bddl",
        displacement_threshold=0.01, lean_dx=-0.04, lean_dy=-0.18, lean_deg=-20.0,
    )
    with pytest.raises(ValueError, match="lean_dx"):
        validate_expected_config(str(artifact), "task", lean_dx=-0.06)
    with pytest.raises(ValueError, match="oracle_displacement_threshold"):
        validate_expected_config(str(artifact), "task", displacement_threshold=0.03)
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
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Checkpoint"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Eval seed"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Eb index SHA256"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Er index SHA256"' in text
    assert 'require_bound_report "${SMOKE_EVIDENCE_REPORT}" "Ec index SHA256"' in text
    assert '--expected_eb_run_id "$(with_suffix L3-A1-drawer-bottle-eb-native)"' in text
    assert '--expected_seed "${EVAL_SEED}"' in text
    assert '--checkpoint "${CHECKPOINT}"' in text
    assert text.count("clean_condition_rollouts") >= 3
