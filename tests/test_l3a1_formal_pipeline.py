import subprocess
from pathlib import Path

import h5py
import pytest

from experiments.robot.libero.tasks.record_experiment_results import _metadata_for_run
from experiments.robot.libero.tasks.validate_l3a1_pairing import validate_pairing


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "experiments/robot/libero/tasks/run_l3a1_drawer_bottle.sh"
PAPER_MATRIX = REPO_ROOT / "experiments/robot/libero/tasks/run_paper_matrix.sh"


def test_l3a1_run_ids_map_to_distinct_formal_conditions():
    assert _metadata_for_run("L3-A1-drawer-bottle-eb-native-seed42") == (
        "L3", "L3-A1", "Eb Native Gate"
    )
    assert _metadata_for_run("L3-A1-drawer-bottle-er-support-removal-seed42") == (
        "L3", "L3-A1", "Er Support Removal"
    )
    assert _metadata_for_run("L3-A1-drawer-bottle-ec-static-support-seed42") == (
        "L3", "L3-A1", "Ec Static Support"
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


def _states(path, attempts):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("task")
        for index, attempt in enumerate(attempts):
            demo = group.create_group(f"demo_{index}")
            demo.attrs["reset_attempt"] = attempt


def test_pairing_gate_compares_ordered_hdf_reset_attempts(tmp_path):
    er, ec = tmp_path / "er.hdf5", tmp_path / "ec.hdf5"
    _states(er, [2, 5, 9])
    _states(ec, [2, 5, 9])
    assert validate_pairing(str(er), str(ec), "task") == [2, 5, 9]
    _states(ec, [2, 9, 5])
    with pytest.raises(ValueError, match="reset_attempt mismatch"):
        validate_pairing(str(er), str(ec), "task")


def test_stable_generator_is_explicitly_paired_to_er_artifact():
    text = RUNNER.read_text()
    assert 'pair_args=(--pair_attempts_from "${RISK_STATE_PATH}")' in text
    assert "validate_l3a1_pairing.py" in text


def test_runner_enables_l3a1_causal_oracle_semantics_and_full_settle():
    text = RUNNER.read_text()
    assert "--support_baseline_on_activation True" in text
    assert "--support_activate_on_gripper_contact False" in text
    assert '--support_interference_bodies "${INTERFERENCE_BODIES}"' in text
    assert 'POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-400}"' in text
