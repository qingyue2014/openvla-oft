import json
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3a4_microwave_common import (
    SCENARIO,
    TASK_FILE,
    TASK_KEY,
    TASK_PROMPT,
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


def _write_states(path: Path, condition: str, *, mutate_other=False):
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
    trace = np.zeros((10, 13), dtype=float)
    trace[:, 0] = np.arange(10)
    trace[:, 1:4] = mug
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
        demo.attrs["porcelain_fixture_local_position"] = mug
        demo.attrs["porcelain_world_quaternion"] = [1.0, 0.0, 0.0, 0.0]
        demo.attrs["porcelain_world_qvel"] = np.zeros(6)
        demo.attrs["wait_pre_position"] = mug
        demo.attrs["wait_pre_quaternion"] = [1.0, 0.0, 0.0, 0.0]
        demo.attrs["wait_post_position"] = mug
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


def test_l3a4_pairing_allows_only_porcelain_state_and_gates_dynamics(tmp_path):
    paths = {name: tmp_path / f"{name}.hdf5" for name in ("eb", "er", "ec")}
    for condition, path in paths.items():
        _write_states(path, condition)
    report = validate_pairing(paths["eb"], paths["er"], paths["ec"])
    assert report["verdict"] == "PASS_L3A4_PAIRED_SCENE_GATE"

    _write_states(paths["er"], "er", mutate_other=True)
    with pytest.raises(ValueError, match="non-porcelain"):
        validate_pairing(paths["eb"], paths["er"], paths["ec"])


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
    assert "env.check_success()" in text
    assert "policy_image(exact_wait" in text
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
