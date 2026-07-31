import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l1a4_spatial_pipeline import (
    EC_LURE_XY,
    LURE,
    MAX_EC_NATIVE_CENTER_DISPLACEMENT_M,
    MAX_LAYOUT_POSITION_ERROR_M,
    MAX_RECEPTACLE_TILT_DEG,
    NATIVE_LURE_XY,
    NATIVE_PLATE_XY,
    NATIVE_RAMEKIN_XY,
    NATIVE_TARGET_XY,
    PLATE,
    RAMEKIN,
    RELATION_TRANSFORM_CANDIDATES,
    TARGET,
    _between_metrics,
    _check_receptacle_tilts,
    _decode_segmentation_rgb,
    _relation_positions,
)
from experiments.robot.libero.tasks.validate_l1a4_spatial_native_preflight import (
    BDDL_PROMPT,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
    FORMAL_WAIT_STEPS,
    INTERVENTION_ID,
    PHYSICAL_GATE_VERDICT,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    validate_native_task,
    verify_state_file,
)


def _native_bddl(tmp_path: Path) -> Path:
    path = tmp_path / "bddl_files" / TASK_SUITE / TASK_FILE
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""
(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {BDDL_PROMPT})
  (:fixtures
    main_table - table
    wooden_cabinet_1 - wooden_cabinet
    flat_stove_1 - flat_stove
  )
  (:objects
    akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl
    cookies_1 - cookies
    glazed_rim_porcelain_ramekin_1 - glazed_rim_porcelain_ramekin
    plate_1 - plate
  )
  (:init)
  (:goal (And))
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_l1a4_v5_is_the_only_official_formal_record():
    tasks_dir = Path("experiments/robot/libero/tasks")
    record = json.loads(
        (tasks_dir / "l1a4_official_formal.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["task"] == "L1-A4"
    assert record["canonical_version"] == "v5"
    assert record["status"] == "OFFICIAL_FORMAL"
    assert record["native_task"]["task_suite_name"] == TASK_SUITE
    assert record["native_task"]["task_id"] == TASK_ID
    assert record["native_task"]["task_file"] == TASK_FILE
    assert record["native_task"]["prompt"] == TASK_PROMPT
    assert (
        record["intervention"]["intervention_id"] == INTERVENTION_ID
    )
    assert record["formal_evaluation"]["job_id"] == "499357"
    assert (
        record["review"]["human_verdict"]
        == "PASS_HUMAN_L1A4_V5_FORMAL_VIDEO_REVIEW"
    )
    assert (
        record["legacy"]["libero_90_ordinal_l1a4"]
        == "RETIRED_NONCANONICAL"
    )
    for condition, state in record["intervention"]["state_files"].items():
        path = Path(state["path"])
        assert path == tasks_dir / f"l1a4_spatial_{condition}_states.hdf5"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == state["sha256"]


def test_l1a4_spatial_preflight_accepts_only_native_inventory(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    assert record["task_id"] == TASK_ID
    assert record["fixtures"] == EXPECTED_FIXTURES
    assert record["objects"] == EXPECTED_OBJECTS
    assert record["prompt"] == TASK_PROMPT
    assert record["bddl_prompt"] == BDDL_PROMPT

    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(bddl, bddl, TASK_PROMPT + " safely")


def test_l1a4_spatial_preflight_rejects_inventory_change(tmp_path):
    bddl = _native_bddl(tmp_path)
    text = bddl.read_text(encoding="utf-8").replace(
        "plate_1 - plate",
        "plate_1 - plate\n    wine_bottle_1 - wine_bottle",
    )
    bddl.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="asset inventory mismatch"):
        validate_native_task(bddl, bddl, TASK_PROMPT)


def test_l1a4_spatial_state_file_carries_native_fingerprints(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    state_path = tmp_path / "l1a4_spatial_er.hdf5"
    with h5py.File(state_path, "w") as handle:
        handle.attrs["native_only"] = True
        handle.attrs["task_suite_name"] = TASK_SUITE
        handle.attrs["task_id"] = TASK_ID
        handle.attrs["task_file"] = TASK_FILE
        handle.attrs["native_prompt"] = TASK_PROMPT
        handle.attrs["native_bddl_sha256"] = record["bddl_sha256"]
        handle.attrs["asset_inventory_sha256"] = record[
            "asset_inventory_sha256"
        ]
        handle.attrs["condition"] = "er"
        handle.attrs["intervention_id"] = INTERVENTION_ID
        handle.attrs["physical_gate_verdict"] = PHYSICAL_GATE_VERDICT
        handle.attrs["formal_wait_steps"] = FORMAL_WAIT_STEPS
        handle.attrs["max_receptacle_tilt_deg"] = (
            MAX_RECEPTACLE_TILT_DEG
        )
    verify_state_file(state_path, record)

    with h5py.File(state_path, "r+") as handle:
        handle.attrs["native_prompt"] = "modified prompt"
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_state_file(state_path, record)


def test_l1a4_spatial_state_file_rejects_stale_intervention(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    state_path = tmp_path / "l1a4_spatial_ec.hdf5"
    with h5py.File(state_path, "w") as handle:
        handle.attrs["native_only"] = True
        handle.attrs["task_suite_name"] = TASK_SUITE
        handle.attrs["task_id"] = TASK_ID
        handle.attrs["task_file"] = TASK_FILE
        handle.attrs["native_prompt"] = TASK_PROMPT
        handle.attrs["native_bddl_sha256"] = record["bddl_sha256"]
        handle.attrs["asset_inventory_sha256"] = record[
            "asset_inventory_sha256"
        ]
        handle.attrs["condition"] = "ec"
        handle.attrs["intervention_id"] = "retired_fixed_layout"
        handle.attrs["physical_gate_verdict"] = PHYSICAL_GATE_VERDICT
        handle.attrs["formal_wait_steps"] = FORMAL_WAIT_STEPS
        handle.attrs["max_receptacle_tilt_deg"] = (
            MAX_RECEPTACLE_TILT_DEG
        )
    with pytest.raises(ValueError, match="intervention_id"):
        verify_state_file(state_path, record)


def test_l1a4_spatial_ec_stays_near_native_distribution():
    assert np.allclose(EC_LURE_XY, NATIVE_LURE_XY)
    native_axis = NATIVE_PLATE_XY - NATIVE_RAMEKIN_XY
    native_target_plate = np.linalg.norm(
        NATIVE_TARGET_XY - NATIVE_PLATE_XY
    )
    native_target_ramekin = np.linalg.norm(
        NATIVE_TARGET_XY - NATIVE_RAMEKIN_XY
    )
    assert len(RELATION_TRANSFORM_CANDIDATES) == 12
    for transform in RELATION_TRANSFORM_CANDIDATES:
        positions = _relation_positions(transform)
        target = positions[TARGET]
        plate = positions[PLATE]
        ramekin = positions[RAMEKIN]
        assert np.linalg.norm(plate - ramekin) == pytest.approx(
            np.linalg.norm(native_axis)
        )
        assert np.linalg.norm(target - plate) == pytest.approx(
            native_target_plate
        )
        assert np.linalg.norm(target - ramekin) == pytest.approx(
            native_target_ramekin
        )
        for body, position in positions.items():
            assert np.linalg.norm(
                position
                - {
                    TARGET: NATIVE_TARGET_XY,
                    PLATE: NATIVE_PLATE_XY,
                    RAMEKIN: NATIVE_RAMEKIN_XY,
                }[body]
            ) < MAX_EC_NATIVE_CENTER_DISPLACEMENT_M
        assert transform[0] > 0
        assert transform[1] > 0
        assert transform[2] < 0
    assert MAX_LAYOUT_POSITION_ERROR_M == pytest.approx(0.02)
    assert MAX_EC_NATIVE_CENTER_DISPLACEMENT_M == pytest.approx(0.17)


def test_l1a4_rejects_in_place_post_wait_tipping():
    assert FORMAL_WAIT_STEPS == 10
    assert MAX_RECEPTACLE_TILT_DEG == pytest.approx(1.0)
    _check_receptacle_tilts(
        "Er",
        "first-policy-frame",
        {TARGET: 0.2, LURE: 0.1, PLATE: 0.3, RAMEKIN: 0.1},
    )
    with pytest.raises(RuntimeError, match="first-policy-frame"):
        _check_receptacle_tilts(
            "Er",
            "first-policy-frame",
            {TARGET: 15.57, LURE: 0.1, PLATE: 0.3, RAMEKIN: 0.1},
        )


def test_l1a4_runner_reuses_physical_gate_for_review_and_evaluation():
    runner = (
        Path("experiments/robot/libero/tasks/run_l1a4_spatial.sh")
        .read_text(encoding="utf-8")
    )
    assert '--num_steps_wait "${FORMAL_WAIT_STEPS}"' in runner
    assert "physical_review)" in runner
    assert 'python "${PIPELINE}" physical_review' in runner
    assert 'REVIEW_DIR="${REVIEW_DIR:-review/L1-A4_task}"' in runner
    assert "require_smoke_review()" in runner
    assert "PASS_HUMAN_L1A4_V5_SMOKE_REVIEW" in runner
    assert "formal)\n    ensure_states\n    require_smoke_review" in runner


class _SegGeom:
    def __init__(self, segid, objtype, objid):
        self.segid = segid
        self.objtype = objtype
        self.objid = objid


class _SegScene:
    ngeom = 2
    geoms = (
        _SegGeom(segid=0, objtype=5, objid=300),
        _SegGeom(segid=1, objtype=5, objid=301),
    )


def test_l1a4_segmentation_decode_promotes_uint8_before_id_math():
    encoded = np.array(
        [[[1, 0, 0], [2, 0, 0], [255, 255, 255]]],
        dtype=np.uint8,
    )
    decoded = _decode_segmentation_rgb(encoded, _SegScene())
    assert decoded.dtype == np.int32
    assert decoded.tolist() == [[[5, 300], [5, 301], [-1, -1]]]


class _Model:
    def __init__(self, positions):
        self._positions = positions

    def body_name2id(self, name):
        return list(self._positions).index(name)


class _Data:
    def __init__(self, positions):
        self.body_xpos = np.asarray(list(positions.values()), dtype=float)


class _Sim:
    def __init__(self, positions):
        self.model = _Model(positions)
        self.data = _Data(positions)


class _Env:
    def __init__(self, positions):
        self.sim = _Sim(positions)


def test_l1a4_spatial_between_relation_is_unique():
    env = _Env(
        {
            TARGET: [-0.04, -0.10, 0.80],
            LURE: [-0.04, 0.20, 0.80],
            PLATE: [0.12, -0.10, 0.80],
            RAMEKIN: [-0.20, -0.10, 0.80],
        }
    )
    target = _between_metrics(env, TARGET)
    lure = _between_metrics(env, LURE)
    assert target["is_between"]
    assert target["projection"] == pytest.approx(0.5)
    assert target["perpendicular_m"] == pytest.approx(0.0)
    assert not lure["is_between"]
    assert lure["midpoint_distance_m"] > 0.1
