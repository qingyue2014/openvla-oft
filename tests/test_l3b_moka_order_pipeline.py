import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v2 as imageio
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITION_INTERVENTION_BODY,
    CONDITION_LABEL,
    CONDITION_REMAINING_BODY,
    CONDITION_SLOT,
    DESIGN_VERSION,
    EXPECTED_FIXTURE_ROOTS,
    EXPECTED_MOVABLE_ROOTS,
    SCENE_ID,
    SLOT_SEPARATION_M,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    native_bddl_path,
    sha256_path,
)
from experiments.robot.libero.tasks.summarize_l3b_moka_safe_references import (
    VERDICT as PASS_SAFE_BATCH,
    summarize as summarize_safe_references,
)
from experiments.robot.libero.tasks.validate_l3b_moka_safe_reference import (
    GRASP_POSE_WAYPOINTS,
    GRASP_REFERENCE_PROVENANCE,
    ORIENTATION_CLEARANCE_DISTANCE_M,
    ORIENTATION_CLEARANCE_HEIGHT_M,
    SAFE_CONTROLLER_VERSION,
)
from experiments.robot.libero.tasks.summarize_l3b_moka_order_smoke import (
    PASS_CONTROL,
    PASS_SMOKE,
    bind_preregistration,
    control_capability,
    native_capability,
    summarize,
)
from experiments.robot.libero.tasks.probe_l3b_moka_order_access import (
    _terminal_stability,
)
from experiments.robot.libero.tasks.validate_l3b_moka_native20_prereg import (
    ARTIFACTS,
    MINIMUM_STABLE_SUCCESSES,
    POOL_COUNT,
    POOL_INDICES,
    PREREGISTRATION_ID,
    validate_spec,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v2_pool import (
    OFFICIAL_STATE_INDICES as V2_OFFICIAL_STATE_INDICES,
    PREREGISTRATION_ID as V2_POOL_PREREGISTRATION_ID,
    validate_spec as validate_v2_pool_spec,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v3_design import (
    OFFICIAL_STATE_INDICES as V3_OFFICIAL_STATE_INDICES,
    PREREGISTRATION_ID as V3_DESIGN_PREREGISTRATION_ID,
    validate_spec as validate_v3_design_spec,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v4_design import (
    OFFICIAL_STATE_INDICES as V4_OFFICIAL_STATE_INDICES,
    PREREGISTRATION_ID as V4_DESIGN_PREREGISTRATION_ID,
    validate_spec as validate_v4_design_spec,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v5_design import (
    LANDING_AXIS_LOCAL_XY,
    OFFICIAL_STATE_INDICES as V5_OFFICIAL_STATE_INDICES,
    PREREGISTRATION_ID as V5_DESIGN_PREREGISTRATION_ID,
    validate_spec as validate_v5_design_spec,
)
from experiments.robot.libero.tasks.validate_l3b_moka_native_preflight import (
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3b_moka_runtime_replay import (
    MAX_MEAN_ABSOLUTE_PIXEL_ERROR,
    MAX_P99_ABSOLUTE_PIXEL_ERROR,
    _compare_images,
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
    assert DESIGN_VERSION == 5
    assert CONDITION_LABEL == {
        "native": "Eb",
        "near_first": "Er",
        "far_first": "Ec",
    }
    assert CONDITION_SLOT["near_first"] == "near"
    assert CONDITION_SLOT["far_first"] == "far"
    assert CONDITION_INTERVENTION_BODY["near_first"] == (
        CONDITION_INTERVENTION_BODY["far_first"]
    ) == "moka_pot_1_main"
    assert CONDITION_REMAINING_BODY["near_first"] == (
        CONDITION_REMAINING_BODY["far_first"]
    ) == "moka_pot_2_main"
    assert SLOT_SEPARATION_M == 0.145
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
    assert "run_safe_reference" in runner
    assert "summarize_l3b_moka_safe_references.py" in runner
    assert "episode < NUM_STATES" in runner
    assert "--design-preregistration" in runner
    assert "run_ec_capability" in runner
    assert "--control-only" in runner
    assert "gs://openpi-assets/checkpoints/pi05_libero" in wrapper
    assert 'runtime_scene == "L3-B-MOKA-ORDER"' in evaluator
    assert "except MokaOrderRuntimeGateError:" in evaluator


def test_v2_pool_is_locked_to_prior_native_stable_successes():
    path = TASKS / "l3b_moka_v2_pool_prereg.json"
    result = validate_v2_pool_spec(path)
    assert result["preregistration_id"] == V2_POOL_PREREGISTRATION_ID
    assert result["official_state_indices"] == [3, 5, 7, 17, 18]
    assert result["official_state_indices"] == V2_OFFICIAL_STATE_INDICES
    assert result["claim_scope"] == (
        "conditional_on_prior_native_pi05_stable_success"
    )
    assert len(result["sha256"]) == 64


def test_v3_design_locks_role_swap_before_rerun():
    path = TASKS / "l3b_moka_v3_design_prereg.json"
    result = validate_v3_design_spec(path)
    assert result["preregistration_id"] == V3_DESIGN_PREREGISTRATION_ID
    assert result["official_state_indices"] == [3, 5, 7, 17, 18]
    assert result["official_state_indices"] == V3_OFFICIAL_STATE_INDICES
    assert result["condition_roles"]["near_first"] == {
        "preplaced_body": "moka_pot_1_main",
        "remaining_body": "moka_pot_2_main",
        "slot": "near",
    }
    assert len(result["failed_ec_report_sha256"]) == 64


def test_v4_design_locks_wider_native_slots_before_rerun():
    path = TASKS / "l3b_moka_v4_design_prereg.json"
    result = validate_v4_design_spec(path)
    assert result["preregistration_id"] == V4_DESIGN_PREREGISTRATION_ID
    assert result["official_state_indices"] == [3, 5, 7, 17, 18]
    assert result["official_state_indices"] == V4_OFFICIAL_STATE_INDICES
    assert result["slot_separation_m"] == 0.145
    assert len(result["failed_ec_report_sha256"]) == 64


def test_v5_design_locks_native_landing_axis_before_rerun():
    path = TASKS / "l3b_moka_v5_design_prereg.json"
    result = validate_v5_design_spec(path)
    assert result["preregistration_id"] == V5_DESIGN_PREREGISTRATION_ID
    assert result["official_state_indices"] == [3, 5, 7, 17, 18]
    assert result["official_state_indices"] == V5_OFFICIAL_STATE_INDICES
    assert result["slot_separation_m"] == 0.145
    assert result["landing_axis_local_xy"] == LANDING_AXIS_LOCAL_XY
    assert result["condition_roles"]["near_first"]["slot"] == (
        "at_default_landing"
    )


def test_safe_grasp_reference_is_compact_and_provenance_bound():
    assert len(GRASP_POSE_WAYPOINTS) == 9
    assert [item["source_step"] for item in GRASP_POSE_WAYPOINTS] == (
        GRASP_REFERENCE_PROVENANCE["source_steps"]
    )
    assert all(
        len(item["offset_xyz"]) == 3
        and len(item["quaternion_xyzw"]) == 4
        and item["gripper"] in (-1.0, 1.0)
        for item in GRASP_POSE_WAYPOINTS
    )
    assert [item["gripper"] for item in GRASP_POSE_WAYPOINTS].count(1.0) == 3
    assert len(GRASP_REFERENCE_PROVENANCE["source_trajectory_sha256"]) == 64
    assert GRASP_REFERENCE_PROVENANCE["source_body"] == "moka_pot_1_main"
    assert GRASP_REFERENCE_PROVENANCE["runtime_body"] == "moka_pot_2_main"


def test_safe_terminal_gate_checks_the_entire_settle_window():
    sample_count = 100
    quaternions = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0]), (sample_count, 1)
    )
    angle = np.deg2rad(2.0)
    quaternions[0] = [
        np.cos(angle / 2.0),
        np.sin(angle / 2.0),
        0.0,
        0.0,
    ]
    measurements = [
        {
            "linear_speed_mps": 0.0,
            "angular_speed_radps": 0.0,
            "contacts": ["flat_stove_1_burner"],
        }
        for _ in range(sample_count)
    ]
    rollout = SimpleNamespace(
        recorder=SimpleNamespace(
            body_pos={"pot": np.zeros((sample_count, 3))},
            body_quat={"pot": quaternions},
            phases=["settle"] * sample_count,
        ),
        body_measurements={"pot": measurements},
    )
    result = _terminal_stability(rollout, "pot")
    assert result["sample_count"] == sample_count
    assert result["max_tilt_deg"] == pytest.approx(2.0)
    assert not result["passed"]


def test_safe_batch_summary_binds_every_episode_artifact(tmp_path):
    reports = []
    terminal = {
        body: {
            "passed": True,
            "sample_count": 100,
            "stove_support_all_samples": True,
        }
        for body in ("moka_pot_1_main", "moka_pot_2_main")
    }
    for episode in range(2):
        trajectory = tmp_path / f"safe_{episode}.npz"
        video = tmp_path / f"safe_{episode}.mp4"
        trajectory.write_bytes(f"trajectory-{episode}".encode())
        video.write_bytes(f"video-{episode}".encode())
        report = {
            "verdict": "PASS_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE",
            "safe_success": True,
            "scenario": SCENE_ID,
            "design_version": DESIGN_VERSION,
            "native_suite": SUITE,
            "native_task_id": TASK_ID,
            "native_prompt": TASK_PROMPT,
            "safe_controller_version": SAFE_CONTROLLER_VERSION,
            "orientation_clearance": {
                "height_m": ORIENTATION_CLEARANCE_HEIGHT_M,
                "distance_m": ORIENTATION_CLEARANCE_DISTANCE_M,
            },
            "source_episode": episode,
            "er_states_sha256": "a" * 64,
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256_path(trajectory),
            "review_video": str(video),
            "review_video_sha256": sha256_path(video),
            "successful_attempt": {
                "safe_success": True,
                "task_success": True,
                "stable_final": True,
                "forbidden_contacts": [],
                "final_robot_object_contact": False,
                "grasp_lift_m": 0.11,
                "preplaced_body_displacement_m": 0.0,
                "target_xy_error_m": 0.02,
                "terminal_stability": terminal,
            },
        }
        report_path = tmp_path / f"safe_{episode}.json"
        report_path.write_text(json.dumps(report))
        reports.append(report_path)
    result = summarize_safe_references(reports, expected_count=2)
    assert result["verdict"] == PASS_SAFE_BATCH
    assert result["count"] == 2
    assert result["minimum_grasp_lift_m"] == pytest.approx(0.11)
    assert result["safe_controller_version"] == SAFE_CONTROLLER_VERSION
    assert result["orientation_clearance_height_m"] == pytest.approx(
        ORIENTATION_CLEARANCE_HEIGHT_M
    )
    assert result["orientation_clearance_distance_m"] == pytest.approx(
        ORIENTATION_CLEARANCE_DISTANCE_M
    )


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
    assert "native20_v1 screen is frozen" in wrapper
    assert "source commit 6883655" in wrapper


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


def test_render_replay_tolerates_only_small_antialiasing_tail(tmp_path):
    assert MAX_MEAN_ABSOLUTE_PIXEL_ERROR == 1.0
    assert MAX_P99_ABSOLUTE_PIXEL_ERROR == 8.0
    expected = np.zeros((100, 100, 3), dtype=np.uint8)
    expected_path = tmp_path / "expected.png"
    imageio.imwrite(expected_path, expected)

    small_edge_tail = expected.copy()
    small_edge_tail[:2, :, :] = 6
    result = _compare_images(
        {"agentview_raw_256": small_edge_tail},
        {"agentview_raw_256": str(expected_path)},
    )
    assert result["agentview_raw_256"]["p99_absolute_pixel_error"] == 6.0

    excessive_tail = expected.copy()
    excessive_tail[:2, :, :] = 9
    with pytest.raises(ValueError, match="replay mismatch"):
        _compare_images(
            {"agentview_raw_256": excessive_tail},
            {"agentview_raw_256": str(expected_path)},
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
    assert result["official_native_state_indices"] == [3, 5, 7, 17, 18]


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
        target, occupant = "moka_pot_2_main", "moka_pot_1_main"
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


def test_ec_is_the_matched_capability_control(tmp_path):
    ec_dir = tmp_path / "far_first"
    for episode in range(3):
        _write_fake_trajectory(ec_dir, "far_first", episode)
    result = control_capability(
        ec_dir,
        expected_count=3,
        minimum_successes=2,
    )
    assert result["verdict"] == PASS_CONTROL
    assert result["scene"] == "Ec"
    assert result["control"]["stable_successes"] == 3
