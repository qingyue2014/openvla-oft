import subprocess
from pathlib import Path

import h5py
import pytest

from experiments.robot.libero.tasks.record_experiment_results import _metadata_for_run
from experiments.robot.libero.tasks.validate_l3a1_pairing import (
    validate_base_preservation,
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
        if source is not None:
            group.attrs["pairing_method"] = "serialized_er_state_bottle_transform"
            group.attrs["paired_er_states"] = str(source)
            group.attrs["source_task_key"] = "task"
        for index, attempt in enumerate(attempts):
            demo = group.create_group(f"demo_{index}")
            demo.attrs["reset_attempt"] = attempt
            demo.attrs["initial_eef_drift_m"] = 0.0
            demo.attrs["runtime_wait_displacement_m"] = 0.0
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
    assert 'LEAN_DX="${LEAN_DX:--0.04}"' in text
    assert 'POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-400}"' in text
