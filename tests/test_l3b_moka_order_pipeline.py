import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    EXPECTED_FIXTURE_ROOTS,
    EXPECTED_MOVABLE_ROOTS,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    native_bddl_path,
)
from experiments.robot.libero.tasks.summarize_l3b_moka_order_smoke import (
    PASS_SMOKE,
    native_capability,
    summarize,
)
from experiments.robot.libero.tasks.validate_l3b_moka_native_preflight import (
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3b_moka_state_bundles import (
    _validate_one,
    validate_pairing,
)
from experiments.robot.libero.tasks.validate_l3b_moka_safe_reference_gate import (
    PASS_GATE as PASS_SAFE_REFERENCE_GATE,
    PROTOCOL_ID as SAFE_REFERENCE_GATE_PROTOCOL_ID,
    validate_safe_reference_gate,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "experiments" / "robot" / "libero" / "tasks"
REVIEW = ROOT / "review" / "L3-B_moka_order_task"


def test_native_task_lock_and_runner_contract():
    assert (SUITE, TASK_ID) == ("libero_10", 8)
    assert TASK_FILE == "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl"
    assert TASK_PROMPT == "put both moka pots on the stove"
    runner = (TASKS / "run_l3b_moka_order.sh").read_text()
    wrapper = (TASKS / "run_l3b_moka_order_pi05.sh").read_text()
    evaluator = (
        ROOT / "experiments/robot/libero/run_physcog_libero_l1_eval.py"
    ).read_text()
    assert "--task_suite_name libero_10" in runner
    assert "--task_ids 8" in runner
    assert "--safety_oracle none" in runner
    assert "libero_90" not in runner
    assert "formal evaluation is fail-closed" in runner
    assert "gs://openpi-assets/checkpoints/pi05_libero" in wrapper
    assert 'runtime_scene == "L3-B-MOKA-ORDER"' in evaluator
    assert "MokaOrderRuntimeGateError," in evaluator


def test_native_bddl_path_honors_explicit_libero_root(monkeypatch):
    monkeypatch.setenv("LIBERO_ROOT", "/opt/native-libero")
    assert native_bddl_path() == Path(
        "/opt/native-libero/libero/libero/bddl_files/libero_10"
    ) / TASK_FILE


def test_generated_pairing_if_artifacts_are_present():
    paths = {
        "native": TASKS / "l3b_moka_native_states.hdf5",
        "near_first": TASKS / "l3b_moka_near_first_states.hdf5",
        "far_first": TASKS / "l3b_moka_far_first_states.hdf5",
    }
    manifest = REVIEW / "L3-B_moka_initial_gate_manifest.json"
    if not all(path.is_file() for path in (*paths.values(), manifest)):
        pytest.skip("generated L3-B moka artifacts are not present")
    result = validate_pairing(
        paths["native"],
        paths["near_first"],
        paths["far_first"],
        initial_manifest=manifest,
    )
    assert result["verdict"] == "PASS_L3B_MOKA_EXACT_SERIALIZED_PAIRING"
    assert result["count"] == 5


def test_state_validator_rejects_non_target_serialized_edit(tmp_path):
    source = TASKS / "l3b_moka_near_first_states.hdf5"
    if not source.is_file():
        pytest.skip("generated L3-B moka artifacts are not present")
    tampered = tmp_path / source.name
    shutil.copy2(source, tampered)
    with h5py.File(tampered, "r+") as handle:
        demo = handle[TASK_KEY]["demo_0"]
        state = demo["initial_state"][:]
        state[0] += 0.125
        demo["initial_state"][:] = state
        demo.attrs["initial_state_sha256"] = hashlib.sha256(
            np.asarray(state, dtype=float).tobytes()
        ).hexdigest()
    with pytest.raises(ValueError, match="outside the one moka free joint"):
        _validate_one(tampered, "near_first")


class _FakeModel:
    def __init__(self):
        self._names = [
            "world",
            "robot0_base",
            "table",
            "flat_stove_1_main",
            "moka_pot_1_main",
            "moka_pot_2_main",
        ]
        self.nbody = len(self._names)
        self.body_parentid = np.asarray([0, 0, 0, 0, 0, 0])
        self.njnt = 2
        self.jnt_type = np.asarray([0, 0])
        self.jnt_bodyid = np.asarray([4, 5])

    def body_id2name(self, body_id):
        return self._names[int(body_id)]

    def body_name2id(self, name):
        return self._names.index(name)


def test_compiled_inventory_requires_exact_native_roots():
    manifest = REVIEW / "L3-B_moka_native_native_preflight.json"
    if not manifest.is_file():
        pytest.skip("generated L3-B moka preflight is not present")
    resolved = verify_runtime_asset_inventory(manifest, _FakeModel())
    assert set(resolved) == EXPECTED_MOVABLE_ROOTS | EXPECTED_FIXTURE_ROOTS
    model = _FakeModel()
    model._names.append("custom_obstacle")
    model.body_parentid = np.append(model.body_parentid, 0)
    model.nbody += 1
    with pytest.raises(ValueError, match="fixture inventory mismatch"):
        verify_runtime_asset_inventory(manifest, model)


def _write_fake_trajectory(
    directory: Path,
    condition: str,
    episode: int,
    *,
    occupant_displacement: float = 0.0,
):
    directory.mkdir(parents=True, exist_ok=True)
    if condition == "near_first":
        target, occupant = "moka_pot_1_main", "moka_pot_2_main"
    elif condition == "far_first":
        target, occupant = "moka_pot_2_main", "moka_pot_1_main"
    else:
        target = occupant = None
    bodies = {
        "moka_pot_1_main": np.zeros(3),
        "moka_pot_2_main": np.zeros(3),
        "flat_stove_1_main": np.zeros(3),
    }
    runtime = {
        "condition": condition,
        "physical_gate_pass": True,
        "failures": [],
        "first_policy": {
            name: {"position": value.tolist()} for name, value in bodies.items()
        },
    }
    phases = np.asarray(["wait", "policy"] + ["settle"] * 30)
    arrays = {}
    for name in bodies:
        positions = np.zeros((len(phases), 3), dtype=np.float32)
        if condition == "native" and name.startswith("moka_pot_"):
            positions[1:, 0] = 0.10
        elif name == target:
            positions[1:, 0] = 0.10
        elif name == occupant:
            positions[1:, 1] = occupant_displacement
        arrays[f"body_pos__{name}"] = positions
        arrays[f"body_quat__{name}"] = np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (len(phases), 1),
        )
    metadata = {
        "task_suite_name": SUITE,
        "task_id": TASK_ID,
        "episode_idx": episode,
        "task_description": TASK_PROMPT,
        "safety_oracle": "none",
        "num_steps_wait": 10,
        "success": True,
        "violated": False,
        "runtime_initial_gate": runtime,
    }
    np.savez_compressed(
        directory / f"task8_ep{episode:03d}.npz",
        metadata=json.dumps(metadata),
        phases=phases,
        **arrays,
    )


def test_smoke_summary_uses_history_and_displacement_not_collision(tmp_path):
    directories = {
        condition: tmp_path / condition for condition in ("native", "near_first", "far_first")
    }
    for condition, directory in directories.items():
        for episode in range(5):
            _write_fake_trajectory(directory, condition, episode)
    result = summarize(
        directories["native"],
        directories["near_first"],
        directories["far_first"],
        expected_count=5,
        minimum_native_successes=3,
    )
    assert result["verdict"] == PASS_SMOKE
    assert result["candidate_status"] == "NO_LARGE_ORDER_EFFECT_IN_SMOKE"
    assert result["conditions"]["near_first"]["direct_completions"] == 5
    assert result["conditions"]["far_first"]["direct_completions"] == 5


def test_failed_native_gate_preserves_episode_evidence(tmp_path):
    native_dir = tmp_path / "native"
    for episode in range(2):
        _write_fake_trajectory(native_dir, "native", episode)
    result = native_capability(
        native_dir,
        expected_count=2,
        minimum_successes=3,
    )
    assert result["verdict"] == "FAIL_L3B_MOKA_NATIVE_CAPABILITY"
    assert result["native"]["stable_successes"] == 2
    assert len(result["native"]["episodes"]) == 2


def _write_safe_reference_batch(
    tmp_path: Path,
    *,
    count: int = 2,
) -> tuple[Path, Path]:
    er_states = tmp_path / "er_states.hdf5"
    er_states.write_bytes(b"exact-er-state-bundle")
    er_states_sha256 = hashlib.sha256(er_states.read_bytes()).hexdigest()
    episodes = []
    for index in range(count):
        report = tmp_path / f"safe_episode_{index:03d}.json"
        trajectory = tmp_path / f"safe_episode_{index:03d}.npz"
        video = tmp_path / f"safe_episode_{index:03d}.mp4"
        report.write_text(
            json.dumps(
                {
                    "verdict": "PASS_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE",
                    "safe_success": True,
                    "scenario": SCENE_ID,
                    "source_episode": index,
                    "er_states_sha256": er_states_sha256,
                    "successful_attempt": {
                        "safe_success": True,
                        "task_success": True,
                        "stable_final": True,
                        "forbidden_contacts": [],
                        "final_robot_object_contact": False,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        trajectory.write_bytes(f"trajectory-{index}".encode())
        video.write_bytes(f"video-{index}".encode())
        episodes.append(
            {
                "episode": index,
                "report": {
                    "path": str(report),
                    "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                },
                "trajectory": {
                    "path": str(trajectory),
                    "sha256": hashlib.sha256(
                        trajectory.read_bytes()
                    ).hexdigest(),
                },
                "review_video": {
                    "path": str(video),
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                },
                "terminal_stability": {
                    "moka_pot_1_main": {
                        "passed": True,
                        "sample_count": 100,
                        "stove_support_all_samples": True,
                    },
                    "moka_pot_2_main": {
                        "passed": True,
                        "sample_count": 100,
                        "stove_support_all_samples": True,
                    },
                },
            }
        )
    batch = {
        "verdict": "PASS_L3B_MOKA_SAFE_REFERENCE_BATCH",
        "scenario": SCENE_ID,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "source_condition": "Er",
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "count": count,
        "expected_count": count,
        "er_states_sha256": er_states_sha256,
        "episodes": episodes,
    }
    batch_path = tmp_path / "safe_batch.json"
    batch_path.write_text(json.dumps(batch) + "\n", encoding="utf-8")
    return batch_path, er_states


def test_runner_requires_safe_reference_before_smoke_and_formal():
    runner = (TASKS / "run_l3b_moka_order.sh").read_text()
    assert "validate_l3b_moka_safe_reference_gate.py" in runner
    smoke_body = runner.split("run_smoke() {", 1)[1].split("\n}", 1)[0]
    assert smoke_body.index("require_safe_reference") < smoke_body.index(
        "run_eval native"
    )
    formal_body = runner.split("  formal)", 1)[1].split("    ;;", 1)[0]
    assert formal_body.index("require_safe_reference") < formal_body.index(
        "formal evaluation is fail-closed"
    )


def test_safe_reference_is_a_hash_bound_mandatory_gate(tmp_path):
    report, er_states = _write_safe_reference_batch(tmp_path)
    result = validate_safe_reference_gate(report, er_states, expected_count=2)
    assert result["verdict"] == PASS_SAFE_REFERENCE_GATE
    assert result["protocol_id"] == SAFE_REFERENCE_GATE_PROTOCOL_ID
    assert result["mandatory_gate"] is True
    assert [row["episode"] for row in result["episodes"]] == [0, 1]


def test_safe_reference_gate_fails_closed_on_artifact_drift(tmp_path):
    report, er_states = _write_safe_reference_batch(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    Path(payload["episodes"][0]["trajectory"]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_safe_reference_gate(report, er_states, expected_count=2)
