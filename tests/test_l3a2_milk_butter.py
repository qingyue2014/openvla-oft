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
    _basket_milk_goal,
    _collision_vertical_bounds,
)
from experiments.robot.libero.tasks.validate_l3a2_milk_butter_osc_reference import (
    TRANSPORT_MAX_WAYPOINT_STEPS,
    _failure_diagnostics,
    _load_records,
)
from experiments.robot.libero.tasks import (
    generate_l3a2_milk_butter_initial_states as l3a2_generator,
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
            demo.attrs["native_butter_body_position"] = (
                base[10:13] + np.array([0.125, -0.25, 0.375])
            )
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


def test_pairing_rejects_mismatched_native_butter_body_position(tmp_path):
    paths = _triplet(tmp_path)
    with h5py.File(paths["ec"], "r+") as handle:
        handle[TASK_KEY]["demo_0"].attrs[
            "native_butter_body_position"
        ] += np.array([0.01, 0.0, 0.0])
    with pytest.raises(
        ValueError, match="paired native butter body position differs"
    ):
        validate_state_artifacts(
            paths["eb"],
            paths["er"],
            paths["ec"],
            native_bddl=NATIVE_BDDL,
        )


def test_osc_loader_uses_saved_world_body_position_not_free_qpos(tmp_path):
    paths = _triplet(tmp_path)
    records = _load_records(str(paths["er"]), 2)
    with h5py.File(paths["er"], "r") as handle:
        for index, record in enumerate(records):
            free_qpos_translation = handle[TASK_KEY][f"demo_{index}"][
                "base_reset_state"
            ][10:13]
            expected_world_position = free_qpos_translation + np.array(
                [0.125, -0.25, 0.375]
            )
            assert record["native_butter_body_position"] == pytest.approx(
                expected_world_position
            )
            assert not np.array_equal(
                record["native_butter_body_position"],
                free_qpos_translation,
            )


def test_osc_timeout_diagnostics_preserve_progress_and_target():
    failure = SimpleNamespace(
        reason="waypoint_timeout",
        stage="butter_park_translate",
        initial_error_m=0.2163,
        best_error_m=0.0361,
        final_error_m=0.0364,
        final_eef_xyz=(0.0643, -0.2034, 0.3497),
        target_eef_xyz=(0.1008, -0.2034, 0.3497),
    )
    diagnostics = _failure_diagnostics(failure)

    assert diagnostics["failure_initial_error_m"] == pytest.approx(0.2163)
    assert diagnostics["failure_best_error_m"] == pytest.approx(0.0361)
    assert diagnostics["failure_final_error_m"] == pytest.approx(0.0364)
    assert diagnostics["failure_progress_m"] == pytest.approx(0.1802)
    assert diagnostics["failure_progress_fraction"] == pytest.approx(
        0.1802 / 0.2163
    )
    assert diagnostics["failure_final_eef_xyz"] == pytest.approx(
        [0.0643, -0.2034, 0.3497]
    )
    assert diagnostics["failure_target_eef_xyz"] == pytest.approx(
        [0.1008, -0.2034, 0.3497]
    )
    assert diagnostics["failure_progressing_at_budget_limit"] is True


def test_osc_timeout_diagnostics_do_not_invent_missing_motion_evidence():
    diagnostics = _failure_diagnostics(None)

    assert np.isnan(diagnostics["failure_initial_error_m"])
    assert np.isnan(diagnostics["failure_best_error_m"])
    assert np.isnan(diagnostics["failure_final_error_m"])
    assert np.isnan(diagnostics["failure_progress_m"])
    assert np.isnan(diagnostics["failure_progress_fraction"])
    assert diagnostics["failure_final_eef_xyz"] == []
    assert diagnostics["failure_target_eef_xyz"] == []
    assert diagnostics["failure_progressing_at_budget_limit"] is False


def test_osc_transport_horizon_covers_observed_long_safe_transfers():
    # Job 499607 advanced about 1.8 mm per controller step.  The native
    # milk-to-basket transfer can span about 0.51 m, requiring roughly 284
    # steps while retaining the low 0.15 transport command cap.
    estimated_steps = int(np.ceil(0.51 / 0.0018))
    assert TRANSPORT_MAX_WAYPOINT_STEPS == 360
    assert TRANSPORT_MAX_WAYPOINT_STEPS >= estimated_steps


def test_move_body_linear_converts_world_body_target_to_free_qpos(monkeypatch):
    offset = np.array([0.18, -0.07, 0.26])
    qpos_start = np.array([0.7, -0.4, 1.2])
    destination = np.array([1.13, -0.12, 1.81])

    class Model:
        njnt = 1
        jnt_bodyid = np.array([0])
        jnt_type = np.array([0])
        jnt_qposadr = np.array([2])
        jnt_dofadr = np.array([4])

        @staticmethod
        def body_name2id(name):
            assert name == "milk_1_main"
            return 0

    class Data:
        qpos = np.arange(10, dtype=float)
        qvel = np.ones(12, dtype=float)
        body_xpos = np.zeros((1, 3), dtype=float)

    class Sim:
        model = Model()
        data = Data()

        def __init__(self):
            self.body_positions = []

        def forward(self):
            self.data.body_xpos[0] = self.data.qpos[2:5] + offset

        def step(self):
            self.forward()
            self.body_positions.append(self.data.body_xpos[0].copy())

    sim = Sim()
    sim.data.qpos[2:5] = qpos_start
    sim.forward()
    body_start = sim.data.body_xpos[0].copy()
    env = SimpleNamespace(sim=sim)
    frames = []
    monkeypatch.setattr(
        l3a2_generator,
        "_capture_frame",
        lambda _env: np.zeros((1, 1, 3), dtype=np.uint8),
    )

    l3a2_generator._move_body_linear(
        env,
        "milk_1_main",
        destination,
        steps=4,
        frames=frames,
    )

    expected_qpos = qpos_start + (destination - body_start)
    assert sim.data.qpos[2:5] == pytest.approx(expected_qpos)
    assert sim.data.body_xpos[0] == pytest.approx(destination)
    expected_positions = np.asarray(
        [
            body_start + fraction * (destination - body_start)
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ]
    )
    np.testing.assert_allclose(sim.body_positions, expected_positions)
    assert sim.data.qvel[4:10] == pytest.approx(np.zeros(6))
    assert len(frames) == 4


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


def _basket_geometry_env(*, milk_half_x=0.025):
    identity = np.eye(3).reshape(-1)
    model = SimpleNamespace(
        nbody=3,
        ngeom=6,
        nsite=1,
        body_parentid=np.array([0, 0, 0]),
        geom_bodyid=np.array([1, 2, 2, 2, 2, 2]),
        geom_group=np.zeros(6, dtype=int),
        geom_type=np.full(6, 6, dtype=int),
        geom_size=np.array(
            [
                [milk_half_x, 0.026, 0.055],
                [0.070, 0.070, 0.008],
                [0.006, 0.070, 0.070],
                [0.006, 0.070, 0.070],
                [0.070, 0.006, 0.070],
                [0.070, 0.006, 0.070],
            ]
        ),
        site_size=np.array([[0.061, 0.061, 0.069]]),
        body_name2id=lambda name: {
            "milk_1_main": 1,
            "basket_1_main": 2,
        }[name],
        site_id2name=lambda site_id: (
            "basket_1_contain_region" if site_id == 0 else None
        ),
        geom_id2name=lambda geom_id: (
            "basket_floor" if geom_id == 1 else f"geom_{geom_id}"
        ),
    )
    data = SimpleNamespace(
        body_xpos=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.2, 0.106],
                [0.0, 0.0, -0.004],
            ]
        ),
        geom_xmat=np.repeat(identity[None, :], 6, axis=0),
        geom_xpos=np.array(
            [
                [0.210, -0.190, 0.106],
                [0.000, 0.000, 0.012],
                [-0.068, 0.000, 0.078],
                [0.068, 0.000, 0.078],
                [0.000, -0.068, 0.078],
                [0.000, 0.068, 0.078],
            ]
        ),
        site_xpos=np.array([[0.0, 0.0, 0.067]]),
        site_xmat=np.array([identity]),
    )
    return SimpleNamespace(sim=SimpleNamespace(model=model, data=data))


def test_basket_goal_uses_compiled_floor_and_milk_collision_offset():
    env = _basket_geometry_env()
    target, diagnostics = _basket_milk_goal(env)

    # The collision AABB is offset +1 cm in x/y from the milk root body, so
    # centring the physical carton requires an equal body-position correction.
    assert target == pytest.approx([-0.010, -0.010, 0.076])
    assert diagnostics["basket_floor_geom_id"] == 1
    assert diagnostics["basket_floor_top_z"] == pytest.approx(0.020)
    assert diagnostics["milk_body_to_collision_bottom_m"] == pytest.approx(
        0.055
    )
    assert diagnostics["milk_goal_floor_clearance_m"] == pytest.approx(
        0.001
    )
    assert diagnostics["milk_goal_predicate_inside"] is True
    assert diagnostics["milk_goal_collision_xy_inside"] is True
    assert diagnostics["basket_contain_predicate_lower"] == pytest.approx(
        [-0.061, -0.061, -0.012]
    )
    assert diagnostics["basket_contain_predicate_upper"] == pytest.approx(
        [0.061, 0.061, 0.136]
    )

    # The raw contain-site centre would penetrate the native floor, which is
    # exactly the remote failure mode this target calculation prevents.
    site_center_bottom = 0.067 - 0.055
    assert site_center_bottom < diagnostics["basket_floor_top_z"]


def test_basket_goal_fails_closed_when_native_collision_does_not_fit():
    env = _basket_geometry_env(milk_half_x=0.061)
    with pytest.raises(
        RuntimeError, match="collision bounds do not fit"
    ):
        _basket_milk_goal(env)


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
