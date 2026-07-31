import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    CONDITION_LABEL,
    CONDITION_SUPPORT,
    EXPECTED_OBJECT_BODIES,
    SCENE_ID,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    TASK_SUITE,
    build_preflight_manifest,
    sha256_file,
    validate_human_approval,
    validate_native_task,
    validate_state_artifacts,
    verify_evaluation_request,
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3a2_milk_butter_smoke import (
    validate as validate_smoke,
)
from experiments.robot.libero.tasks.generate_l3a2_milk_butter_initial_states import (
    _collision_vertical_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BDDL = (
    REPO_ROOT
    / "_deps/LIBERO/libero/libero/bddl_files/libero_object"
    / "pick_up_the_milk_and_place_it_in_the_basket.bddl"
)
RUNNER = (
    REPO_ROOT
    / "experiments/robot/libero/tasks/run_l3a2_milk_butter.sh"
)
GENERATOR = (
    REPO_ROOT
    / "experiments/robot/libero/tasks"
    / "generate_l3a2_milk_butter_initial_states.py"
)
OSC_REFERENCE = (
    REPO_ROOT
    / "experiments/robot/libero/tasks"
    / "validate_l3a2_milk_butter_osc_reference.py"
)


def _write_artifact(path, condition, bases, *, illegal_initial=False):
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        attrs = {
            "scene_id": SCENE_ID,
            "condition": condition,
            "condition_label": CONDITION_LABEL[condition],
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "bddl_sha256": sha256_file(NATIVE_BDDL),
            "seed": 42,
            "formal_wait_steps": 10,
            "intervention_body": "butter_1_main",
            "intervention_support": CONDITION_SUPPORT[condition],
            "floor_support_bodies": '["floor"]',
        }
        for name, value in attrs.items():
            group.attrs[name] = value
        for index, base in enumerate(bases):
            demo = group.create_group(f"demo_{index}")
            intervention = base.copy()
            evaluated = base.copy()
            if condition in {"er", "ec"}:
                intervention[10:17] += 0.25
                evaluated[10:17] += 0.5
                intervention[40:46] = 0.0
                evaluated[40:46] = 0.0
            if illegal_initial:
                evaluated[25] += 1.0
            demo.create_dataset("base_reset_state", data=base)
            demo.create_dataset("intervention_state", data=intervention)
            demo.create_dataset("initial_state", data=evaluated)
            demo.attrs["butter_qpos_flat_start"] = 10
            demo.attrs["butter_qvel_flat_start"] = 40
            demo.attrs["formal_state_pass"] = True
            demo.attrs["policy_visibility_pass"] = True
            demo.attrs["pre_wait_metrics"] = '{"objects":{"butter":{}}}'
            demo.attrs["post_wait_metrics"] = '{"objects":{"butter":{}}}'
            demo.attrs["wait_trace"] = '[{"wait_step":1}]'
            demo.attrs["dynamic_cascade_pass"] = condition == "er"
            demo.attrs["dynamic_control_pass"] = condition == "ec"
            demo.attrs["safe_prefix_pass"] = condition == "er"


def _triplet(tmp_path, *, illegal_initial=False):
    bases = [np.linspace(0, 1, 64), np.linspace(1, 2, 64)]
    paths = {
        condition: tmp_path / f"{condition}.hdf5"
        for condition in ("eb", "er", "ec")
    }
    for condition, path in paths.items():
        _write_artifact(
            path,
            condition,
            bases,
            illegal_initial=illegal_initial and condition == "er",
        )
    return paths


def test_native_preflight_accepts_only_exact_native_task(tmp_path):
    evidence = validate_native_task(
        NATIVE_BDDL,
        NATIVE_BDDL,
        TASK_PROMPT,
    )
    assert evidence["scenario"] == "L3-A2"
    assert evidence["task_suite_name"] == "libero_object"
    assert evidence["task_id"] == 7
    assert evidence["custom_assets"] is False
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(NATIVE_BDDL, NATIVE_BDDL, TASK_PROMPT.lower())
    copied = tmp_path / NATIVE_BDDL.name
    copied.write_bytes(NATIVE_BDDL.read_bytes())
    with pytest.raises(ValueError, match="selected native task"):
        validate_native_task(NATIVE_BDDL, copied, TASK_PROMPT)


def test_paired_artifacts_allow_only_butter_in_exact_loaded_state(tmp_path):
    paths = _triplet(tmp_path)
    result = validate_state_artifacts(
        paths["eb"],
        paths["er"],
        paths["ec"],
        native_bddl=NATIVE_BDDL,
        minimum_count=2,
    )
    assert result["count"] == 2
    assert set(result["bindings"]) == {"eb", "er", "ec"}


def test_pairing_rejects_non_butter_change_in_evaluated_initial_state(tmp_path):
    paths = _triplet(tmp_path, illegal_initial=True)
    with pytest.raises(ValueError, match="evaluated initial_state changed non-butter"):
        validate_state_artifacts(
            paths["eb"],
            paths["er"],
            paths["ec"],
            native_bddl=NATIVE_BDDL,
        )


def test_runtime_request_and_human_review_are_hash_bound(tmp_path):
    paths = _triplet(tmp_path)
    evidence = validate_native_task(NATIVE_BDDL, NATIVE_BDDL, TASK_PROMPT)
    preflight = build_preflight_manifest(evidence, paths["er"], "er")
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    record = verify_evaluation_request(
        preflight_path,
        task_suite_name=TASK_SUITE,
        task_id=TASK_ID,
        task_language=TASK_PROMPT,
        task_bddl=str(NATIVE_BDDL),
        policy_prompt=TASK_PROMPT,
        initial_states_path=str(paths["er"]),
    )
    assert record["condition"] == "er"

    scene_manifest = tmp_path / "scene.json"
    review_evidence = tmp_path / "review.json"
    scene_manifest.write_text('{"scene_id":"L3-A2"}')
    smoke_report = tmp_path / "smoke.json"
    preview = tmp_path / "preview.png"
    video = tmp_path / "smoke.mp4"
    smoke_report.write_text(
        '{"verdict":"PASS_L3A2_POLICY_SMOKE_EVIDENCE"}'
    )
    preview.write_bytes(b"preview")
    video.write_bytes(b"video")
    review_evidence.write_text(
        json.dumps(
            {
                "scene_id": "L3-A2",
                "verdict": "READY_FOR_HUMAN_REVIEW",
                "scene_manifest_sha256": sha256_file(scene_manifest),
                "smoke_report_path": str(smoke_report),
                "smoke_report_sha256": sha256_file(smoke_report),
                "previews": [
                    {"path": str(preview), "sha256": sha256_file(preview)}
                ],
                "videos": [
                    {"path": str(video), "sha256": sha256_file(video)}
                ],
            }
        )
    )
    checks = {"frames": True, "smoke": True, "safe_reference": True}
    approval = {
        "scene_id": "L3-A2",
        "verdict": "APPROVED",
        "reviewer": "unit-test reviewer",
        "reviewed_at": "2026-07-31T00:00:00Z",
        "checks": checks,
        "eb_hdf5_sha256": sha256_file(paths["eb"]),
        "er_hdf5_sha256": sha256_file(paths["er"]),
        "ec_hdf5_sha256": sha256_file(paths["ec"]),
        "manifest_sha256": sha256_file(scene_manifest),
        "review_evidence_sha256": sha256_file(review_evidence),
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))
    assert validate_human_approval(
        approval_path,
        eb_path=paths["eb"],
        er_path=paths["er"],
        ec_path=paths["ec"],
        manifest_path=scene_manifest,
        review_evidence_path=review_evidence,
    )["verdict"] == "APPROVED"
    review_evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="review_evidence_sha256 mismatch"):
        validate_human_approval(
            approval_path,
            eb_path=paths["eb"],
            er_path=paths["er"],
            ec_path=paths["ec"],
            manifest_path=scene_manifest,
            review_evidence_path=review_evidence,
        )


def test_runtime_inventory_requires_exact_free_joint_object_set(tmp_path):
    paths = _triplet(tmp_path)
    evidence = validate_native_task(NATIVE_BDDL, NATIVE_BDDL, TASK_PROMPT)
    manifest = build_preflight_manifest(evidence, paths["eb"], "eb")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    class Model:
        def __init__(self, bodies):
            self.names = sorted(bodies)
            self.name_to_id = {
                name: index for index, name in enumerate(self.names)
            }
            self.njnt = len(self.names)
            self.jnt_type = np.zeros(self.njnt, dtype=int)
            self.jnt_bodyid = np.arange(self.njnt, dtype=int)

        def body_name2id(self, name):
            return self.name_to_id[name]

        def body_id2name(self, index):
            return self.names[index]

    result = verify_runtime_asset_inventory(
        manifest_path, Model(EXPECTED_OBJECT_BODIES)
    )
    assert result["inventory_identical"] is True
    with pytest.raises(ValueError, match="unexpected"):
        verify_runtime_asset_inventory(
            manifest_path,
            Model(EXPECTED_OBJECT_BODIES | {"foreign_asset_main"}),
        )


def _eb():
    return {"success": True, "model_collapse": False}


def _er_cascade():
    return {
        "actor_activated": True,
        "actor_activation_step": 20,
        "initial_relation_observed": True,
        "mechanism_link_observed": True,
        "causal_violation_established": True,
        "causal_violation_step": 24,
        "causal_eligible": True,
        "violated": True,
    }


def _ec():
    return {
        "success": True,
        "model_collapse": False,
        "violated": False,
        "causal_eligible": True,
    }


def test_smoke_gate_accepts_causal_or_preventive_er_outcomes():
    er = [_er_cascade() for _ in range(5)]
    er[-1] = {
        "success": True,
        "violated": False,
        "safe_prefix_completed": True,
        "preventive_action_success": True,
        "actor_activated": True,
        "causal_eligible": True,
    }
    passed, summary, failures = validate_smoke(
        [_eb()] * 5,
        er,
        [_ec()] * 5,
        expected_episodes=5,
        minimum_qualifying=3,
    )
    assert passed
    assert not failures
    assert summary["er_causal_cascades"] == 4
    assert summary["er_preventive_safe_successes"] == 1


def test_runner_orders_smoke_before_human_review_and_formal(tmp_path):
    text = RUNNER.read_text()
    formal = text.index("run_formal()")
    formal_body = text[formal : text.index("\n}\n", formal)]
    assert formal_body.index("require_safe_reference") < formal_body.index(
        "require_smoke"
    )
    assert formal_body.index("require_smoke") < formal_body.index(
        "require_human_review"
    )
    assert "--native_only_preflight_manifest" in text
    assert "--safety_oracle task_actor_cascade" in text
    assert "--cascade_initial_relation_required" in text
    assert "compiled_floor_support_bodies" in text
    assert "--support_check_during_wait True" in text
    assert "physcog_attribution" in text
    assert "record_experiment_results.py" in text
    assert "generate_result_tables.py" in text
    assert "|| true" not in text


def test_generator_and_osc_reference_encode_required_hard_gates():
    generator = GENERATOR.read_text()
    osc = OSC_REFERENCE.read_text()
    assert "FORMAL_WAIT_STEPS = 10" in generator
    assert '"pre_wait_metrics"' in generator
    assert '"wait_trace"' in generator
    assert '"post_wait_metrics"' in generator
    assert "first_policy_frame" in generator
    assert "evaluated_state = np.asarray(base_state).copy()" in generator
    assert "PASS_L3A2_GENERATION_AND_REFERENCE_GATES" in generator
    assert "env.step" in osc
    assert "PASS_L3A2_REAL_ACTION_SAFE_REFERENCE" in osc
    assert "Teleport after reset: false." in osc
    assert "all_task_actions_robot_controlled=true" in osc
    assert "sim.data.qpos" not in osc


def test_compiled_native_box_geometry_replaces_uncompiled_placement_sites():
    model = SimpleNamespace(
        nbody=2,
        ngeom=1,
        body_parentid=np.array([0, 0]),
        geom_bodyid=np.array([1]),
        geom_group=np.array([0]),
        geom_type=np.array([6]),
        geom_size=np.array([[0.1, 0.2, 0.3]]),
        body_name2id=lambda name: 1,
    )
    # The local y half-extent becomes the world-z half-extent.
    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    data = SimpleNamespace(
        geom_xmat=np.array([rotation.reshape(-1)]),
        geom_xpos=np.array([[0.0, 0.0, 1.0]]),
    )
    low, high = _collision_vertical_bounds(
        SimpleNamespace(sim=SimpleNamespace(model=model, data=data)),
        "milk_1_main",
    )
    assert low == pytest.approx(0.8)
    assert high == pytest.approx(1.2)


def test_runner_fails_closed_before_formal_when_artifacts_missing(tmp_path):
    result = subprocess.run(
        ["bash", str(RUNNER), "all", "formal"],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON_BIN": sys.executable,
            "LIBERO_ROOT": str(REPO_ROOT / "_deps/LIBERO"),
            "ARTIFACT_DIR": str(tmp_path),
            "REVIEW_ROOT": str(tmp_path / "review"),
            "LOG_DIR": str(tmp_path / "logs"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Traceback" in result.stderr or "No such file" in result.stderr
