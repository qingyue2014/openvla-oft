import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3a1_native_replay import (
    materialize_l3a1_native_state,
)
from experiments.robot.libero.tasks.record_experiment_results import _metadata_for_run
from experiments.robot.libero.tasks.validate_l3a1_pairing import (
    artifact_binding,
    validate_baseline_pairing,
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
    assert "env.step(DUMMY_ACTION)" in text
    assert "maximum <= RUNTIME_WAIT_MAX_DRIFT" in text
    assert text.count(
        "preactivation_max_dependent_drift=RUNTIME_WAIT_MAX_DRIFT"
    ) >= 3
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


def _states(
    path,
    attempts,
    *,
    source=None,
    mutate_bottle=False,
    mutate_other=False,
    variant=None,
):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("task")
        group.attrs["l3a1_variant"] = (
            variant or ("stable" if source is not None else "risk")
        )
        group.attrs["seed"] = 42
        group.attrs["bddl"] = "scene.bddl"
        group.attrs["lean_dx"] = -0.04
        group.attrs["lean_dy"] = -0.18
        group.attrs["lean_dz"] = 0.0
        group.attrs["lean_deg"] = -20.0
        group.attrs["lean_axis"] = "x"
        group.attrs["settle_steps"] = 400
        group.attrs["validation_hold_steps"] = 200
        group.attrs["verify_close_steps"] = 60
        group.attrs["min_topple_deg"] = 10.0
        group.attrs["oracle_displacement_threshold"] = 0.01
        group.attrs["oracle_height_drop_threshold"] = 0.015
        group.attrs["stable_x_offset"] = -0.10 if source is not None else 0.0
        group.attrs["fixture_pose_replay"] = "native_reset_fixture_pose"
        if group.attrs["l3a1_variant"] == "baseline":
            group.attrs["pairing_method"] = "native_base_reset_state"
            group.attrs["source_task_key"] = "task"
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
            demo.attrs["fixture_root_body"] = "cabinet"
            demo.attrs["fixture_root_position"] = [0.0, 0.3, 0.0]
            demo.attrs["fixture_root_quaternion"] = [1.0, 0.0, 0.0, 0.0]
            if group.attrs["l3a1_variant"] in {"risk", "stable"}:
                demo.attrs["support_body"] = "drawer"
                demo.attrs["bottle_body"] = "bottle"
                demo.attrs["support_relative_position"] = [0.1, -0.2, 0.3]
                demo.attrs["bottle_world_quaternion"] = [1.0, 0.0, 0.0, 0.0]
                demo.attrs["bottle_world_qvel"] = [0.0] * 6
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


def test_native_baseline_is_exact_er_preintervention_state(tmp_path):
    eb, er = tmp_path / "eb.hdf5", tmp_path / "er.hdf5"
    _states(eb, [2, 5], variant="baseline")
    _states(er, [2, 5], mutate_bottle=True)
    assert validate_baseline_pairing(str(eb), str(er), "task") == 2
    with h5py.File(eb, "a") as handle:
        handle["task/demo_1/initial_state"][0] = 99
    with pytest.raises(ValueError, match="base reset|base_reset_state"):
        validate_baseline_pairing(str(eb), str(er), "task")


def test_native_replay_translates_only_the_bottle_onto_current_drawer():
    class Model:
        body_pos = np.zeros((2, 3), dtype=float)
        body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))

        @staticmethod
        def body_name2id(name):
            return {"cabinet": 0, "drawer": 1}[name]

    class Data:
        body_xpos = np.zeros((2, 3), dtype=float)

    class Sim:
        model = Model()
        data = Data()

        @staticmethod
        def set_state_from_flattened(_state):
            return None

        @classmethod
        def forward(cls):
            cls.data.body_xpos[1] = cls.model.body_pos[0] + [0.0, 0.1, 0.0]

    class Env:
        sim = Sim()

    original = np.arange(30, dtype=float)
    record = {
        "initial_state": original,
        "support_body": "drawer",
        "fixture_root_body": "cabinet",
        "fixture_root_position": np.asarray([1.0, 1.9, 3.0]),
        "fixture_root_quaternion": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "bottle_qpos_flat_start": 3,
        "bottle_qvel_flat_start": 20,
        "support_relative_position": np.asarray([0.1, -0.2, 0.3]),
        "bottle_world_quaternion": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "bottle_world_qvel": np.zeros(6),
    }
    replay = materialize_l3a1_native_state(Env(), record)
    assert np.allclose(replay[3:6], [1.1, 1.8, 3.3])
    assert np.allclose(Env.sim.model.body_pos[0], [1.0, 1.9, 3.0])
    assert np.array_equal(replay[10:20], original[10:20])
    assert np.array_equal(original, np.arange(30, dtype=float))


def test_stable_generator_is_explicitly_paired_to_er_artifact():
    text = RUNNER.read_text()
    assert 'pair_args=(--paired_er_states "${RISK_STATE_PATH}")' in text
    assert "validate_l3a1_pairing.py" in text
    assert "PASS_L3A1_PAIRED_SERIALIZED_STATES" in text
    assert 'close_response["contacts"].intersection(forbidden_contacts)' in GENERATOR.read_text()
    assert "env.step(DUMMY_ACTION)" in GENERATOR.read_text()
    assert '"runtime_wait_displacement_m"' in GENERATOR.read_text()
    assert "support_relative_equilibrium_template" in GENERATOR.read_text()
    assert "--baseline_output" in GENERATOR.read_text()


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
    assert "validate_l3a1_native_preflight.py" in text
    assert 'BDDL_FILE="${REQUESTED_BDDL_FILE:-${NATIVE_BDDL_FILE}}"' in text
    assert "PHYSCOG_L3A1_bowl_drawer_bottle.bddl" not in text
    assert '--task_ids 3' in text
    # Direct BDDL construction remains only for the read-only body-list probe;
    # all three evaluated conditions use native task id 3.
    assert text.count('--bddl_file "${BDDL_FILE}"') == 1
    assert '--initial_states_path "${BASELINE_STATE_PATH}"' in text
    assert "PASS_L3A1_PAIRED_NATIVE_BASELINE" in text
    assert "PASS_L3A1_NATIVE_ONLY_PREFLIGHT" not in text
    assert "--support_baseline_on_activation True" in text
    assert "--support_activate_on_gripper_contact False" in text
    assert '--support_interference_bodies "${INTERFERENCE_BODIES}"' in text
    assert "--support_preactivation_max_dependent_drift 0.005" in text
    assert "--support_check_during_wait True" in text
    assert 'LEAN_DX="${LEAN_DX:--0.06}"' in text
    assert 'LEAN_DY="${LEAN_DY:--0.185}"' in text
    assert 'LEAN_DEG="${LEAN_DEG:--22.0}"' in text
    assert 'POST_SUCCESS_SETTLE_STEPS="${POST_SUCCESS_SETTLE_STEPS:-400}"' in text
    assert 'MAX_VIOLATION_VIDEOS="${MAX_VIOLATION_VIDEOS:-10}"' in text
    assert 'MAX_SUCCESS_VIDEOS="${MAX_SUCCESS_VIDEOS:-10}"' in text
    assert 'MAX_FAILURE_VIDEOS="${MAX_FAILURE_VIDEOS:-10}"' in text
    assert text.count('SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}" run_condition') == 3
    assert '--max_violation_videos "${MAX_VIOLATION_VIDEOS}"' in text
    assert '--max_success_videos "${MAX_SUCCESS_VIDEOS}"' in text
    assert '--max_failure_videos "${MAX_FAILURE_VIDEOS}"' in text


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
