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
    bind_preregistration,
    native_capability,
    summarize,
)
from experiments.robot.libero.tasks.validate_l3b_moka_native20_prereg import (
    ARTIFACTS,
    MINIMUM_STABLE_SUCCESSES,
    POOL_COUNT,
    POOL_INDICES,
    PREREGISTRATION_ID,
    validate_spec,
)
from experiments.robot.libero.tasks.validate_l3b_moka_native_preflight import (
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3b_moka_state_bundles import (
    _validate_one,
    validate_pairing,
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
    assert "except MokaOrderRuntimeGateError:" in evaluator


def test_native20_preregistration_and_dedicated_runner_are_locked():
    prereg_path = TASKS / "l3b_moka_native20_prereg.json"
    result = validate_spec(prereg_path)
    assert result["preregistration_id"] == PREREGISTRATION_ID
    assert result["pool_indices"] == POOL_INDICES
    assert POOL_COUNT == 20
    assert MINIMUM_STABLE_SUCCESSES == 12
    wrapper = (TASKS / "run_l3b_moka_native20.sh").read_text()
    for contract in (
        "export NUM_STATES=20",
        "export SMOKE_TRIALS=20",
        "export MIN_NATIVE_SUCCESSES=12",
        "export SCENE_SEED=42",
        "export EVAL_SEED=42",
        "native_capability)",
    ):
        assert contract in wrapper
    assert " smoke)" not in wrapper
    assert ARTIFACTS["review_root"] in wrapper


def test_capability_report_binds_preregistration(tmp_path):
    result = {
        "scenario": SCENE_ID,
        "verdict": "FAIL_L3B_MOKA_NATIVE_CAPABILITY",
    }
    prereg_path = TASKS / "l3b_moka_native20_prereg.json"
    bind_preregistration(
        result,
        prereg_path,
        expected_count=POOL_COUNT,
        minimum_successes=MINIMUM_STABLE_SUCCESSES,
    )
    assert result["preregistration"]["preregistration_id"] == (
        PREREGISTRATION_ID
    )
    assert len(result["preregistration"]["sha256"]) == 64
    with pytest.raises(ValueError, match="locked preregistration"):
        bind_preregistration(
            result,
            prereg_path,
            expected_count=5,
            minimum_successes=3,
        )


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
