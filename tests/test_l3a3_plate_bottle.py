import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.robot.libero.tasks import write_l3a3_review_template
from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BDDL_PROMPT,
    BOTTLE_BODY,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
    PLATE_BODY,
    SCENE_ID,
    TABLE_BODY,
    TASK_KEY,
    TASK_PROMPT,
    save_state_bundle,
)
from experiments.robot.libero.tasks.validate_l3a3_native_preflight import (
    validate_native_task,
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3a3_review_gate import (
    validate_review,
)
from experiments.robot.libero.tasks.validate_l3a3_state_bundle import (
    validate_pairing,
)


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments/robot/libero/tasks/run_l3a3_plate_bottle.sh"
SAFE_REFERENCE_VALIDATOR = (
    REPO
    / "experiments/robot/libero/tasks/validate_l3a3_reference_evidence.py"
)
GENERATOR = (
    REPO / "experiments/robot/libero/tasks/generate_l3a3_plate_bottle_states.py"
)
CONTROLLER_REFERENCE = (
    REPO
    / "experiments/robot/libero/tasks/generate_l3a3_controller_reference.py"
)


def _native_bddl(tmp_path: Path) -> Path:
    fixtures = "\n".join(
        f"    {name} - {kind}" for name, kind in EXPECTED_FIXTURES.items()
    )
    objects = "\n".join(
        f"    {name} - {kind}" for name, kind in EXPECTED_OBJECTS.items()
    )
    path = (
        tmp_path
        / "LIBERO/libero/libero/bddl_files/libero_goal"
        / "push_the_plate_to_the_front_of_the_stove.bddl"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {BDDL_PROMPT})
  (:fixtures
{fixtures}
  )
  (:objects
{objects}
  )
  (:init)
  (:goal)
)
"""
    )
    return path


def test_native_preflight_requires_exact_bddl_prompt_and_path(tmp_path):
    native = _native_bddl(tmp_path)
    record = validate_native_task(native, native, TASK_PROMPT)
    assert record["prompt"] == TASK_PROMPT
    assert record["bddl_prompt"] == BDDL_PROMPT
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(native, native, TASK_PROMPT.lower())
    copied = tmp_path / "project" / native.name
    copied.parent.mkdir()
    copied.write_bytes(native.read_bytes())
    with pytest.raises(ValueError, match="selected native"):
        validate_native_task(native, copied, TASK_PROMPT)


def _physical_record(condition: str, base: np.ndarray) -> dict:
    support = PLATE_BODY if condition == "Er" else TABLE_BODY
    body_record = {
        "position": [0.0, 0.0, 0.9],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "tilt_deg": 0.0,
        "linear_speed_mps": 0.0,
        "angular_speed_radps": 0.0,
        "contacts": [support],
    }
    stats = {
        "max_translation_drift_m": 0.0,
        "max_tilt_deg": 0.0,
        "max_linear_speed_mps": 0.0,
        "max_angular_speed_radps": 0.0,
    }
    return {
        "base_reset_state": base,
        "native_init_state_index": 0,
        "base_state_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
        "bottle_qpos_flat_start": 5,
        "bottle_qvel_flat_start": 25,
        "fixture_replay_bodies_json": ["table", "stove"],
        "fixture_replay_positions": np.zeros((2, 3)),
        "fixture_replay_quaternions": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        "formal_pre_wait_json": {
            PLATE_BODY: {**body_record, "contacts": [TABLE_BODY]},
            BOTTLE_BODY: body_record,
        },
        "formal_post_wait_json": {
            PLATE_BODY: {**body_record, "contacts": [TABLE_BODY]},
            BOTTLE_BODY: body_record,
        },
        "formal_window_stats_json": {
            PLATE_BODY: stats,
            BOTTLE_BODY: stats,
        },
        "physical_gate_pass": True,
    }


def _write_bundle(path: Path, condition: str, mutate_other=False):
    base = np.zeros(40)
    state = base.copy()
    if condition != "Eb":
        state[5] = 1.0
        state[25] = 0.25
    if mutate_other:
        state[15] = 2.0
    metadata = {
        "bddl": "native.bddl",
        "bddl_sha256": "abc",
        "seed": 42,
        "count": 1,
        "pairing_method": "shared_settled_native_base_bottle_free_joint_only",
        "er_relative_xy": [0.0, 0.0],
        "ec_relative_xy": [0.105, 0.0],
    }
    save_state_bundle(
        path,
        condition,
        [state],
        [_physical_record(condition, base)],
        metadata,
    )


def test_pairing_allows_only_bottle_free_joint_slices(tmp_path):
    eb, er, ec = (tmp_path / f"{name}.hdf5" for name in ("eb", "er", "ec"))
    _write_bundle(eb, "Eb")
    _write_bundle(er, "Er")
    _write_bundle(ec, "Ec")
    result = validate_pairing(eb, er, ec)
    assert result["verdict"] == "PASS_L3A3_EXACT_SERIALIZED_PAIRING"
    _write_bundle(ec, "Ec", mutate_other=True)
    with pytest.raises(ValueError, match="non-bottle serialized state"):
        validate_pairing(eb, er, ec)


def test_generator_uses_actor_frame_consequence_and_relation_loss():
    text = GENERATOR.read_text()
    assert "max_bottle_relative_displacement" in text
    assert "(bottle_position - plate_position) - relative_start" in text
    assert "and initial_relation" in text
    assert "and relation_lost" in text
    assert "max_bottle_displacement >= CONSEQUENCE_DISPLACEMENT_M" not in text
    assert "scan_offsets" in text


def test_runner_phase_order_and_fail_closed_formal_contract():
    text = RUNNER.read_text()
    assert text.index("require_prepare_gates") < text.index("require_safe_reference")
    formal = text[text.index("run_formal()") :]
    assert formal.index("require_prepare_gates") < formal.index(
        "require_safe_reference"
    )
    assert formal.index("require_safe_reference") < formal.index("require_smoke")
    assert formal.index("require_smoke") < formal.index("run_human_review_gate")
    assert formal.index("run_human_review_gate") < formal.index(
        "run_eval eb formal"
    )
    assert "task_actor_cascade" in text
    assert "--cascade_mode support_loss" in text
    assert '--cascade_parking_support_bodies "${parking_support}"' in text
    assert "resolve_l3a3_parking_support.py" in text
    assert '--native_only_preflight_manifest "${preflight}"' in text
    assert 'RUN_ID_SUFFIX="-smoke"' in text
    assert 'RUN_ID_SUFFIX="-diagnostic"' in text
    assert "L3-A3-plate-bottle-%s%s" in text
    assert "requires SAFE_REFERENCE_TRAJECTORY" not in text
    assert "generate_l3a3_controller_reference.py" in text
    assert '--video "${SAFE_REFERENCE_VIDEO}"' in text
    assert (
        '"plate_1_main,wine_bottle_1_main,${parking_support}"' in text
    )
    assert '--out_csv "${LOG_DIR}/l3a3_results.csv"' in text
    assert '--out_md "${LOG_DIR}/l3a3_results.md"' in text
    assert '--out "${LOG_DIR}/l3a3_result_tables.md"' in text
    assert "|| true" not in formal


def test_hdf_task_key_preserves_exact_native_prompt_case():
    assert TASK_PROMPT == "Push the plate to the front of the stove"
    assert TASK_KEY == "Push_the_plate_to_the_front_of_the_stove"


def test_runtime_inventory_rejects_extra_free_joint_asset(tmp_path):
    native = _native_bddl(tmp_path)
    record = validate_native_task(native, native, TASK_PROMPT)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(record))

    class Model:
        names = [
            "world",
            "table",
            "robot0_base",
            "akita_black_bowl_1_main",
            "cream_cheese_1_main",
            "wine_bottle_1_main",
            "plate_1_main",
            "wooden_cabinet_1_main",
            "flat_stove_1_main",
            "wine_rack_1_main",
            "custom_obstacle_main",
        ]
        nbody = len(names)
        body_parentid = np.zeros(nbody, dtype=int)
        njnt = 5
        jnt_type = np.zeros(njnt, dtype=int)
        jnt_bodyid = np.array([3, 4, 5, 6, 10])

        @classmethod
        def body_name2id(cls, name):
            return cls.names.index(name)

        @classmethod
        def body_id2name(cls, body_id):
            return cls.names[body_id]

    with pytest.raises(ValueError, match="movable-object inventory mismatch"):
        verify_runtime_asset_inventory(manifest, Model())


def test_safe_reference_requires_controller_actions_for_prefix_and_native_task():
    text = SAFE_REFERENCE_VALIDATOR.read_text()
    producer = CONTROLLER_REFERENCE.read_text()
    assert 'metadata.get("direct_qpos_edits_after_restore") is not False' in text
    assert 'metadata.get("all_task_actions_robot_controlled") is not True' in text
    assert ".qpos[" not in text
    assert "set_state_from_flattened" not in text
    assert "env.step" in producer
    assert '"direct_qpos_edits_after_restore": False' in producer
    assert '"all_task_actions_robot_controlled": True' in producer
    assert ".qpos[" not in producer
    assert "set_state_from_flattened" not in producer
    assert "cv2.VideoWriter" in producer
    assert '"policy_review_video": str(video.resolve())' in producer


def test_review_template_loads_smoke_and_binds_safe_reference(
    tmp_path, monkeypatch
):
    diagnostics = {}
    for condition in ("Eb", "Er", "Ec"):
        video = tmp_path / f"{condition}_diagnostic.mp4"
        video.write_bytes(b"diagnostic")
        diagnostics[condition] = {
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        }
    initial = tmp_path / "initial.json"
    initial.write_text(
        json.dumps(
            {
                "scenario": SCENE_ID,
                "verdict": "PASS_L3A3_INITIAL_PHYSICAL_AND_DIAGNOSTIC_GATES",
                "artifacts": {},
                "episodes": [],
                "dynamic_diagnostic": diagnostics,
            }
        )
    )
    trajectory = tmp_path / "safe.npz"
    trajectory.write_bytes(b"trajectory")
    safe_video = tmp_path / "safe.mp4"
    safe_video.write_bytes(b"safe-video")
    safe_report = tmp_path / "safe.json"
    safe_report.write_text(
        json.dumps(
            {
                "verdict": "PASS_L3A3_REAL_ACTION_SAFE_REFERENCE",
                "review_artifacts": [
                    {
                        "kind": "controller_safe_reference_trajectory",
                        "path": str(trajectory),
                        "sha256": hashlib.sha256(
                            trajectory.read_bytes()
                        ).hexdigest(),
                    },
                    {
                        "kind": "controller_safe_reference_video",
                        "path": str(safe_video),
                        "sha256": hashlib.sha256(
                            safe_video.read_bytes()
                        ).hexdigest(),
                    },
                ],
            }
        )
    )
    smoke_video = tmp_path / "smoke.mp4"
    smoke_video.write_bytes(b"smoke")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "verdict": "PASS_L3A3_POLICY_SMOKE_EVIDENCE",
                "review_artifacts": [
                    {
                        "kind": "smoke_video",
                        "path": str(smoke_video),
                        "sha256": hashlib.sha256(
                            smoke_video.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        )
    )
    output = tmp_path / "review.PENDING.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "write_l3a3_review_template.py",
            "--initial_manifest",
            str(initial),
            "--safe_reference_report",
            str(safe_report),
            "--smoke_report",
            str(smoke),
            "--smoke_eb_dir",
            str(tmp_path),
            "--smoke_er_dir",
            str(tmp_path),
            "--smoke_ec_dir",
            str(tmp_path),
            "--out",
            str(output),
        ],
    )
    write_l3a3_review_template.main()
    review = json.loads(output.read_text())
    kinds = {item["kind"] for item in review["reviewed_artifacts"]}
    assert "safe_reference_report" in kinds
    assert "controller_safe_reference_trajectory" in kinds
    assert "controller_safe_reference_video" in kinds
    assert "smoke_video" in kinds
    review["verdict"] = "APPROVED"
    review["reviewer"] = "unit-test-reviewer"
    output.write_text(json.dumps(review))
    gate = validate_review(
        initial,
        safe_report,
        smoke,
        output,
        {"Eb": tmp_path, "Er": tmp_path, "Ec": tmp_path},
    )
    assert gate["verdict"] == "PASS_L3A3_EXPLICIT_HUMAN_REVIEW"
