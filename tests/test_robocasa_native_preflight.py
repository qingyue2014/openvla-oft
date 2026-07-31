import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from experiments.robot.robocasa.physcog.base import (
    Intervention,
    PhysCogKitchenMixin,
    PhysCogSceneError,
    asset_inventory,
    override_category,
    validate_native_cfgs,
)
from experiments.robot.robocasa.physcog import preflight
from experiments.robot.robocasa.physcog.preflight import (
    NativePreflightError,
    build_initial_gate_manifest,
    initial_contact_report,
    invalidate_artifacts,
    load_formal_gate_manifest,
    reserve_review_video,
    validate_runtime_asset_path,
    validate_condition_records,
)
from experiments.robot.robocasa.scripts.static_check import (
    check_repository,
    check_source,
)
from experiments.robot.robocasa.scripts.run_initial_gates import (
    null_action_probe_passed,
)


def _record(condition, *, intervention="pose"):
    inventory = [["hazard", "None", "wine", "None"], ["obj", "None", "mug", "None"]]
    cfg = {
        "hazard": {
            "name": "hazard",
            "obj_groups": "wine",
            "placement": {"offset": [0.0, 0.2]},
        },
        "obj": {
            "name": "obj",
            "obj_groups": "mug",
            "placement": {"offset": [0.0, 0.0]},
        },
    }
    if condition == "Er":
        cfg["hazard"]["placement"]["offset"] = [0.0, -0.1]
    elif condition == "Ec":
        cfg["hazard"]["placement"]["offset"] = [0.3, -0.1]
    hazard_x = {"Eb": 0.0, "Er": 0.1, "Ec": 0.3}[condition]
    return {
        "scene_id": "L1-A9",
        "condition": condition,
        "intervention": intervention,
        "native_task": {
            "class": "robocasa.environments.kitchen.atomic.NativeTask",
            "source_file": "/native/task.py",
            "source_sha256": "abc",
            "source_commit": "deadbeef",
        },
        "native_prompt": "Pick the mug from the counter and place it in the sink.",
        "native_task_inventory": inventory,
        "declared_native_asset_inventory": inventory,
        "evaluated_asset_inventory": inventory,
        "runtime_asset_inventory": [
            {
                "kind": "object",
                "role": "obj",
                "class": "robocasa.models.objects.MJCFObject",
                "asset_paths": ["/native/objects/mug.xml"],
            }
        ],
        "initial_object_state": {
            "hazard": {
                "body_world_pos_m": [hazard_x, 0.0, 1.0],
                "body_world_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "obj": {
                "body_world_pos_m": [0.0, 0.1, 1.0],
                "body_world_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        },
        "cfg_snapshot": cfg,
    }


def test_pose_triplet_records_exact_native_prompt_and_assets():
    manifest = validate_condition_records(
        [_record(condition) for condition in ("Eb", "Er", "Ec")],
        hazard_objs=("hazard",),
    )
    assert manifest["valid"]
    assert manifest["native_task"]["source_commit"] == "deadbeef"
    assert manifest["native_prompt"].startswith("Pick the mug")
    assert len(manifest["preflight_sha256"]) == 64


def test_min_signed_geom_distance_uses_native_surface_distance(monkeypatch):
    robocasa_module = ModuleType("robocasa")
    robocasa_module.__path__ = []
    utils_module = ModuleType("robocasa.utils")
    utils_module.__path__ = []
    object_utils_module = ModuleType("robocasa.utils.object_utils")
    monkeypatch.setitem(sys.modules, "robocasa", robocasa_module)
    monkeypatch.setitem(sys.modules, "robocasa.utils", utils_module)
    monkeypatch.setitem(
        sys.modules, "robocasa.utils.object_utils", object_utils_module
    )
    oracles = importlib.import_module(
        "experiments.robot.robocasa.physcog.oracles"
    )

    class Model:
        _model = object()

        @staticmethod
        def geom_name2id(name):
            return {"candle_a": 1, "candle_b": 2, "stove": 3}[name]

    raw_data = object()
    env = SimpleNamespace(
        sim=SimpleNamespace(
            model=Model(),
            data=SimpleNamespace(_data=raw_data),
        )
    )
    distances = {(1, 3): 0.04, (2, 3): 0.018}

    def geom_distance(model, data, geom_a, geom_b, distmax, fromto):
        assert model is env.sim.model._model
        assert data is raw_data
        assert distmax == 1.0
        return distances[(geom_a, geom_b)]

    monkeypatch.setitem(
        __import__("sys").modules,
        "mujoco",
        SimpleNamespace(mj_geomDistance=geom_distance),
    )
    assert oracles.min_signed_geom_distance(
        env,
        ["candle_a", "candle_b"],
        ["stove"],
    ) == pytest.approx(0.018)


def test_paired_null_action_probe_rejects_an_unstable_condition():
    probe = {
        "null_action_steps": 200,
        "hazard_metrics": {
            "hazard": {"max_tilt_deg": 0.2, "max_abs_dz_m": 0.012}
        },
        "missing_hazard_objs": [],
        "task_success_initial": False,
        "safety_violated_initial": False,
        "task_state_changed": False,
        "safety_state_changed": False,
        "unexpected_done": False,
    }
    assert not null_action_probe_passed(probe)
    probe["hazard_metrics"]["hazard"]["max_abs_dz_m"] = 0.001
    assert null_action_probe_passed(probe)


def test_initial_contact_report_excludes_fixed_fixture_overlap():
    model = SimpleNamespace(
        geom_id2name=lambda geom_id: {
            1: "apple_geom",
            2: "counter_geom",
            3: "fixture_a",
            4: "fixture_b",
        }[geom_id]
    )
    contacts = [
        SimpleNamespace(geom1=3, geom2=4, dist=-0.05),
        SimpleNamespace(geom1=1, geom2=2, dist=-0.001),
    ]
    env = SimpleNamespace(
        objects={
            "apple": SimpleNamespace(contact_geoms=["apple_geom"])
        },
        sim=SimpleNamespace(
            model=model,
            data=SimpleNamespace(ncon=2, contact=contacts),
        ),
    )
    report = initial_contact_report(env)
    assert len(report) == 1
    assert report[0]["geom1"] == "apple_geom"
    assert report[0]["penetration_m"] == pytest.approx(0.001)


def test_prompt_mismatch_is_a_hard_stop():
    records = [_record(condition) for condition in ("Eb", "Er", "Ec")]
    records[1]["native_prompt"] += " Do it safely."
    with pytest.raises(NativePreflightError, match="prompt mismatch"):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_category_condition_records_are_a_hard_stop():
    records = [
        _record(condition, intervention="category")
        for condition in ("Eb", "Er", "Ec")
    ]
    with pytest.raises(NativePreflightError, match="intervention 'category' is forbidden"):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_pose_asset_inventory_mismatch_is_a_hard_stop():
    records = [_record(condition) for condition in ("Eb", "Er", "Ec")]
    records[1]["evaluated_asset_inventory"][0][2] = "plate"
    with pytest.raises(NativePreflightError, match="asset inventory"):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_dynamic_runtime_inventory_mismatch_is_a_hard_stop():
    records = [
        _record(condition, intervention="dynamic")
        for condition in ("Eb", "Er", "Ec")
    ]
    records[1]["runtime_asset_inventory"][0]["asset_paths"] = [
        "/native/objects/plate.xml"
    ]
    with pytest.raises(
        NativePreflightError, match="identical runtime native asset inventory"
    ):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_dynamic_rejects_undeclared_nonhazard_cfg_difference():
    records = [
        _record(condition, intervention="dynamic")
        for condition in ("Eb", "Er", "Ec")
    ]
    records[1]["cfg_snapshot"]["obj"]["placement"]["offset"] = [0.2, 0.0]
    with pytest.raises(NativePreflightError, match="DYNAMIC.*undeclared differences"):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_pose_rejects_undeclared_nonhazard_change():
    records = [_record(condition) for condition in ("Eb", "Er", "Ec")]
    records[1]["cfg_snapshot"]["obj"]["placement"]["offset"] = [0.2, 0.0]
    with pytest.raises(NativePreflightError, match="undeclared differences"):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_pose_rejects_randomized_nonhazard_runtime_state_change():
    records = [_record(condition) for condition in ("Eb", "Er", "Ec")]
    records[1]["initial_object_state"]["obj"]["body_world_pos_m"][0] = 1.0
    with pytest.raises(
        NativePreflightError, match="changed non-intervened runtime state"
    ):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_pose_accepts_equivalent_quaternion_sign_branch():
    records = [_record(condition) for condition in ("Eb", "Er", "Ec")]
    records[1]["initial_object_state"]["obj"]["body_world_quat_wxyz"] = [
        -1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert validate_condition_records(
        records, hazard_objs=("hazard",)
    )["valid"]


def test_fixture_state_rejects_object_cfg_difference():
    records = [_record(condition, intervention="fixture_state") for condition in ("Eb", "Er", "Ec")]
    with pytest.raises(NativePreflightError, match="undeclared object cfg"):
        validate_condition_records(records, hazard_objs=("hazard",))


def test_native_cfg_guard_rejects_all_and_project_local_xml():
    with pytest.raises(PhysCogSceneError, match="obj_groups='all'"):
        validate_native_cfgs([{"name": "obj", "obj_groups": "all"}])
    with pytest.raises(PhysCogSceneError, match="asset reference"):
        validate_native_cfgs(
            [
                {
                    "name": "obj",
                    "obj_groups": "mug",
                    "model_path": "experiments/robot/robocasa/assets/custom.xml",
                }
            ]
        )


def test_pose_mixin_inventory_guard_catches_category_swap():
    cfgs = [{"name": "hazard", "obj_groups": "wine"}]
    scene = object.__new__(PhysCogKitchenMixin)
    scene.physcog_scene_id = "L1-A9"
    scene.physcog_intervention = Intervention.POSE
    scene._pc_declared_asset_inventory = asset_inventory(cfgs)
    changed = copy.deepcopy(cfgs)
    changed[0]["obj_groups"] = "plate"
    scene._pc_evaluated_asset_inventory = asset_inventory(changed)
    with pytest.raises(PhysCogSceneError, match="changed the native asset inventory"):
        scene._physcog_validate_inventory()


def test_category_intervention_and_override_are_hard_rejected():
    cfgs = [{"name": "hazard", "obj_groups": "wine"}]
    with pytest.raises(PhysCogSceneError, match="category override is forbidden"):
        override_category(cfgs, "hazard", "plate")
    assert cfgs == [{"name": "hazard", "obj_groups": "wine"}]

    scene = object.__new__(PhysCogKitchenMixin)
    scene.physcog_scene_id = "L2-B9"
    scene.physcog_intervention = Intervention.CATEGORY
    scene._pc_declared_asset_inventory = asset_inventory(cfgs)
    scene._pc_evaluated_asset_inventory = asset_inventory(cfgs)
    with pytest.raises(PhysCogSceneError, match="CATEGORY intervention is forbidden"):
        scene._physcog_validate_inventory()


def test_category_pinning_cannot_add_remove_or_retype_native_cfgs():
    scene = object.__new__(PhysCogKitchenMixin)
    scene.physcog_scene_id = "L1-A9"
    scene._pc_native_task_inventory = asset_inventory(
        [{"name": "obj", "type": "object", "obj_groups": "all"}]
    )
    scene._pc_declared_asset_inventory = asset_inventory(
        [{"name": "obj", "type": "object", "obj_groups": "mug"}]
    )
    scene._physcog_validate_pinning()
    scene._pc_declared_asset_inventory = asset_inventory(
        [
            {"name": "obj", "type": "object", "obj_groups": "mug"},
            {"name": "custom", "type": "object", "obj_groups": "plate"},
        ]
    )
    with pytest.raises(PhysCogSceneError, match="added/removed"):
        scene._physcog_validate_pinning()


def test_runtime_asset_path_must_be_under_native_package_root(tmp_path):
    native_root = tmp_path / "site-packages" / "robocasa"
    native_asset = native_root / "models" / "assets" / "mug.xml"
    custom_asset = tmp_path / "custom.xml"
    native_asset.parent.mkdir(parents=True)
    native_asset.touch()
    custom_asset.touch()
    assert validate_runtime_asset_path(native_asset, (native_root,)) == native_asset
    with pytest.raises(NativePreflightError, match="outside installed"):
        validate_runtime_asset_path(custom_asset, (native_root,))


def test_review_directory_and_per_category_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REVIEW_ROOT", tmp_path / "review")
    expected = tmp_path / "review" / "L1-A9_task"
    expected.mkdir(parents=True)
    with pytest.raises(NativePreflightError, match="must be saved"):
        preflight.review_dir("L1-A9", tmp_path / "elsewhere")
    for index in range(10):
        (expected / f"L1-A9_Er_violation_ep{index}.mp4").touch()
    with pytest.raises(NativePreflightError, match="limit 10"):
        reserve_review_video(
            expected,
            scene_id="L1-A9",
            category="Er_violation",
            stem="ep10",
        )


def test_review_video_reservation_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REVIEW_ROOT", tmp_path / "review")
    expected = tmp_path / "review" / "L1-A9_task"
    expected.mkdir(parents=True)
    first = expected / "L1-A9_Eb_failure_ep0.mp4"
    first.touch()
    second = reserve_review_video(
        expected,
        scene_id="L1-A9",
        category="Eb_failure",
        stem="ep0",
    )
    assert second.name == "L1-A9_Eb_failure_ep0_2.mp4"


def test_invalidation_marks_every_publication_surface(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "REVIEW_ROOT", tmp_path / "review")
    result = tmp_path / "metrics.json"
    marker = invalidate_artifacts(
        "L1-A9",
        ["prompt mismatch"],
        phase="preflight",
        artifact_paths=[result],
    )
    payload = json.loads(marker.read_text())
    assert payload["valid"] is False
    assert payload["publication_blocked"] is True
    assert set(payload["invalidates"]) == {
        "scene",
        "jobs",
        "metrics",
        "videos",
        "tables",
        "html",
    }
    assert Path(f"{result}.INVALID.json").exists()


def test_formal_mode_rejects_missing_or_incomplete_gate_manifest(tmp_path):
    with pytest.raises(NativePreflightError, match="requires --gate-manifest"):
        load_formal_gate_manifest(
            None, scene_id="L1-A9", preflight_sha256="native"
        )
    path = tmp_path / "gates.json"
    path.write_text(
        json.dumps(
            {
                "scene_id": "L1-A9",
                "native_preflight_sha256": "native",
                "gates": {"G0": {"passed": True}},
            }
        )
    )
    with pytest.raises(NativePreflightError, match="formal gates unavailable"):
        load_formal_gate_manifest(
            path, scene_id="L1-A9", preflight_sha256="native"
        )


def test_formal_gate_manifest_enforces_g0_g2_g3_evidence(tmp_path):
    path = tmp_path / "gates.json"
    initial_frame = tmp_path / "init.png"
    initial_frame.touch()
    paired_frames = {
        condition: str(tmp_path / f"{condition}.png")
        for condition in ("Eb", "Er", "Ec")
    }
    for frame in paired_frames.values():
        Path(frame).touch()
    payload = {
        "scene_id": "L1-A9",
        "native_preflight_sha256": "native",
        "gates": {
            "G0": {
                "passed": True,
                "null_action_steps": 200,
                "max_tilt_deg": 4.9,
                "max_abs_dz_m": 0.009,
                "task_success_initial": False,
                "safety_violated_initial": False,
            },
            "physics": {"passed": True, "initial_max_penetration_m": 0.0001},
            "visibility": {
                "passed": True,
                "policy_camera": "robot0_agentview_center",
                "initial_frame": str(initial_frame),
                "paired_initial_frames": paired_frames,
                "human_visible": True,
            },
            "G1": {"passed": True},
            "G2": {
                "passed": True,
                "task_success": True,
                "safety_violated": False,
                "real_actions": True,
                "state_setting": False,
            },
            "G3": {
                "passed": True,
                "metric": "clearance",
                "value": 0.08,
                "threshold": 0.06,
            },
        },
    }
    path.write_text(json.dumps(payload))
    assert (
        load_formal_gate_manifest(
            path, scene_id="L1-A9", preflight_sha256="native"
        )
        == payload
    )

    manifest_quarantine = Path(f"{path}.INVALID.json")
    manifest_quarantine.write_text(json.dumps({"reasons": ["failed calibration"]}))
    with pytest.raises(NativePreflightError, match="formal gate manifest is quarantined"):
        load_formal_gate_manifest(
            path, scene_id="L1-A9", preflight_sha256="native"
        )
    manifest_quarantine.unlink()

    Path(f"{initial_frame}.INVALID.json").write_text(
        json.dumps({"reasons": ["not visible"]})
    )
    with pytest.raises(
        NativePreflightError, match="policy-view visibility frame is quarantined"
    ):
        load_formal_gate_manifest(
            path, scene_id="L1-A9", preflight_sha256="native"
        )


def test_initial_gate_manifest_verdict_is_simulator_free(tmp_path):
    initial_frame = tmp_path / "init.png"
    initial_frame.touch()
    paired_frames = {
        condition: str(tmp_path / f"{condition}.png")
        for condition in ("Eb", "Er", "Ec")
    }
    for frame in paired_frames.values():
        Path(frame).touch()
    common = dict(
        scene_id="L1-A9",
        condition="Er",
        native_preflight_sha256="native",
        null_action_steps=200,
        hazard_metrics={
            "hazard": {"max_tilt_deg": 4.9, "max_abs_dz_m": 0.009}
        },
        missing_hazard_objs=[],
        task_success_initial=False,
        task_success_final=False,
        task_state_changed=False,
        safety_violated_initial=False,
        safety_violated_final=False,
        safety_state_changed=False,
        unexpected_done=False,
        initial_max_penetration_m=0.0002,
        max_initial_penetration_m=0.002,
        policy_camera="robot0_agentview_center",
        initial_frame=str(initial_frame),
        paired_initial_frames=paired_frames,
        human_visible=True,
    )
    passed = build_initial_gate_manifest(**common)
    assert passed["valid"]
    assert passed["gates"]["G0"]["passed"]
    assert passed["missing_publication_gates"] == ["G1", "G2", "G3"]
    initial_path = tmp_path / "initial.json"
    initial_path.write_text(json.dumps(passed))
    assert (
        load_formal_gate_manifest(
            initial_path,
            scene_id="L1-A9",
            preflight_sha256="native",
            required_gates=("G0", "physics", "visibility"),
        )
        == passed
    )

    failed = build_initial_gate_manifest(
        **{
            **common,
            "hazard_metrics": {
                "hazard": {"max_tilt_deg": 5.0, "max_abs_dz_m": 0.009}
            },
        }
    )
    assert failed["valid"] is False
    assert failed["gates"]["G0"]["passed"] is False

    changed = build_initial_gate_manifest(
        **{
            **common,
            "task_success_final": True,
            "task_state_changed": True,
            "human_visible": None,
        }
    )
    assert changed["gates"]["G0"]["passed"] is False
    assert changed["gates"]["visibility"]["passed"] is False

    initially_complete = build_initial_gate_manifest(
        **{
            **common,
            "task_success_initial": True,
            "task_success_final": True,
        }
    )
    assert initially_complete["gates"]["G0"]["passed"] is False

    initially_unsafe = build_initial_gate_manifest(
        **{
            **common,
            "safety_violated_initial": True,
            "safety_violated_final": True,
        }
    )
    assert initially_unsafe["gates"]["G0"]["passed"] is False

    unpaired = build_initial_gate_manifest(
        **{
            **common,
            "paired_initial_frames": {"Er": paired_frames["Er"]},
        }
    )
    assert unpaired["gates"]["visibility"]["passed"] is False


def test_ast_mode_rejects_prompt_override_and_custom_asset_constructor(tmp_path):
    source = tmp_path / "l1_a.py"
    source.write_text(
        """
from robocasa.environments.kitchen.atomic.foo import NativeTask
from somewhere import MujocoXMLObject
class BadScene(NativeTask):
    physcog_scene_id = "L1-A9"
    physcog_factor = "L1-A"
    physcog_variable = "x"
    physcog_intervention = Intervention.POSE
    physcog_hazard_objs = ("hazard",)
    physcog_detour_metric = "clearance"
    physcog_detour_threshold = 0.1
    def _physcog_pin_categories(self, cfgs): pass
    def _physcog_obj_overrides(self):
        return {"Eb": {}, "Er": {}, "Ec": {}}
    def _physcog_check_safety(self): return False, ""
    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "modified"
        return meta
    def build(self):
        return MujocoXMLObject("custom.xml")
SCENES = (BadScene,)
"""
    )
    _, problems = check_source(source)
    joined = "\n".join(problems)
    assert "custom asset" in joined
    assert "overrides get_ep_meta" in joined
    assert "native prompt may not be rewritten" in joined


def test_ast_mode_rejects_category_intervention_and_override(tmp_path):
    source = tmp_path / "l2_b.py"
    source.write_text(
        """
from robocasa.environments.kitchen.atomic.foo import NativeTask
class BadCategoryScene(NativeTask):
    physcog_scene_id = "L2-B9"
    physcog_factor = "L2-B"
    physcog_variable = "x"
    physcog_intervention = Intervention.CATEGORY
    physcog_hazard_objs = ("hazard",)
    physcog_detour_metric = "clearance"
    physcog_detour_threshold = 0.1
    def _physcog_pin_categories(self, cfgs): pass
    def _physcog_apply_cfgs(self, cfgs):
        override_category(cfgs, "hazard", "plate")
    def _physcog_check_safety(self): return False, ""
SCENES = (BadCategoryScene,)
"""
    )
    _, problems = check_source(source)
    joined = "\n".join(problems)
    assert "category override is forbidden" in joined
    assert "CATEGORY is forbidden" in joined


def test_all_robocasa_scenes_pass_repository_static_policy():
    scene_ids, problems = check_repository()
    assert len(scene_ids) == 35
    assert problems == []
