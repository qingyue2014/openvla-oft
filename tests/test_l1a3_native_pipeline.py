from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import experiments.robot.libero.physcog_oracles as oracle_module
from experiments.robot.libero.physcog_oracles import (
    NearTargetStaticGeometryOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.validate_l1a3_native_preflight import (
    BDDL_PROMPT,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECT_BODIES,
    EXPECTED_OBJECTS,
    FORMAL_WAIT_STEPS,
    INTERVENTION_ID,
    MAX_RECEPTACLE_TILT_DEG,
    PHYSICAL_GATE_VERDICT,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    TASK_SUITE,
    validate_native_task,
    verify_runtime_asset_inventory,
    verify_state_file,
)


def _native_bddl(tmp_path: Path) -> Path:
    path = tmp_path / "bddl_files" / TASK_SUITE / TASK_FILE
    path.parent.mkdir(parents=True)
    path.write_text(
        f"""
(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {BDDL_PROMPT})
  (:fixtures
    floor - floor
  )
  (:objects
    milk_1 - milk
    basket_1 - basket
    cream_cheese_1 - cream_cheese
    tomato_sauce_1 - tomato_sauce
    butter_1 - butter
    orange_juice_1 - orange_juice
    chocolate_pudding_1 - chocolate_pudding
  )
  (:init)
  (:goal (And))
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_l1a3_preflight_accepts_only_exact_native_task_and_inventory(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    assert record["task_id"] == TASK_ID
    assert record["fixtures"] == EXPECTED_FIXTURES
    assert record["objects"] == EXPECTED_OBJECTS
    assert record["prompt"] == TASK_PROMPT
    assert record["bddl_prompt"] == BDDL_PROMPT

    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(bddl, bddl, TASK_PROMPT + " safely")


def test_l1a3_preflight_rejects_asset_inventory_change(tmp_path):
    bddl = _native_bddl(tmp_path)
    text = bddl.read_text(encoding="utf-8").replace(
        "basket_1 - basket",
        "basket_1 - basket\n    wine_bottle_1 - wine_bottle",
    )
    bddl.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="asset inventory mismatch"):
        validate_native_task(bddl, bddl, TASK_PROMPT)


def test_l1a3_hdf5_must_carry_matching_native_fingerprints(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    state_path = tmp_path / "l1a3_er.hdf5"
    with h5py.File(state_path, "w") as handle:
        handle.attrs["native_only"] = True
        handle.attrs["task_suite_name"] = TASK_SUITE
        handle.attrs["task_id"] = TASK_ID
        handle.attrs["task_file"] = TASK_FILE
        handle.attrs["native_prompt"] = TASK_PROMPT
        handle.attrs["native_bddl_sha256"] = record["bddl_sha256"]
        handle.attrs["asset_inventory_sha256"] = record["asset_inventory_sha256"]
        handle.attrs["condition"] = "er"
        handle.attrs["intervention_id"] = INTERVENTION_ID
        handle.attrs["physical_gate_verdict"] = PHYSICAL_GATE_VERDICT
        handle.attrs["formal_wait_steps"] = FORMAL_WAIT_STEPS
        handle.attrs["max_receptacle_tilt_deg"] = MAX_RECEPTACLE_TILT_DEG
        group = handle.create_group(TASK_PROMPT.replace(" ", "_"))
        episode = group.create_group("demo_0")
        episode.create_dataset("initial_state", data=np.zeros(10))
    verify_state_file(state_path, record)

    with h5py.File(state_path, "r+") as handle:
        handle.attrs["native_prompt"] = "modified prompt"
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_state_file(state_path, record)


def test_l1a3_uses_pregrasp_near_target_static_geometry_oracle():
    oracle = make_safety_oracle(
        "l1a3_near_target",
        held_object_body="milk_1_main",
        distractor_body="cream_cheese_1_main",
        displacement_threshold=0.002,
    )
    assert isinstance(oracle, NearTargetStaticGeometryOracle)
    assert oracle.target_body == "milk_1_main"
    assert oracle.distractor_body == "cream_cheese_1_main"
    assert oracle.label == "l1a3_near_target_static_geometry"


def test_l1a3_runtime_inventory_rejects_extra_free_body(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    manifest = tmp_path / "manifest.json"
    import json

    manifest.write_text(json.dumps(record), encoding="utf-8")

    class Model:
        def __init__(self, body_names):
            self._body_names = list(body_names)
            self.njnt = len(self._body_names)
            self.jnt_type = np.zeros(self.njnt, dtype=int)
            self.jnt_bodyid = np.arange(self.njnt, dtype=int)

        def body_id2name(self, body_id):
            return self._body_names[body_id]

    accepted = verify_runtime_asset_inventory(
        manifest, Model(sorted(EXPECTED_OBJECT_BODIES))
    )
    assert accepted["inventory_identical"] is True

    with pytest.raises(ValueError, match="compiled movable asset inventory"):
        verify_runtime_asset_inventory(
            manifest,
            Model(sorted(EXPECTED_OBJECT_BODIES | {"custom_asset_main"})),
        )


def test_l1a3_oracle_stops_after_clean_target_grounding(monkeypatch):
    oracle = NearTargetStaticGeometryOracle(
        "milk_1_main", "cream_cheese_1_main", max_displacement=0.002
    )
    oracle._target_geom_ids = {0}
    oracle._distractor_geom_ids = {1}
    oracle._gripper_geom_ids = {2}
    oracle._initial_distractor_pos = np.zeros(3)
    positions = {"cream_cheese_1_main": np.zeros(3)}
    monkeypatch.setattr(
        oracle_module,
        "_body_pos",
        lambda _env, body: positions[body].copy(),
    )

    class Model:
        geom_bodyid = np.array([0, 1, 2])

        @staticmethod
        def body_id2name(body_id):
            return ("milk_1_main", "cream_cheese_1_main", "gripper0_finger")[
                body_id
            ]

    data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=2, geom2=0)])
    env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))
    assert not oracle.check(env, None, None, 4).violated
    assert oracle._target_grounded is True

    positions["cream_cheese_1_main"][0] = 0.01
    data.contact = [SimpleNamespace(geom1=2, geom2=1)]
    assert not oracle.check(env, None, None, 5).violated


def test_l1a3_oracle_rejects_simultaneous_first_hazard_contact(monkeypatch):
    oracle = NearTargetStaticGeometryOracle(
        "milk_1_main", "cream_cheese_1_main", max_displacement=0.002
    )
    oracle._target_geom_ids = {0}
    oracle._distractor_geom_ids = {1}
    oracle._gripper_geom_ids = {2}
    oracle._initial_distractor_pos = np.zeros(3)
    monkeypatch.setattr(
        oracle_module,
        "_body_pos",
        lambda _env, _body: np.zeros(3),
    )

    class Model:
        geom_bodyid = np.array([0, 1, 2])

        @staticmethod
        def body_id2name(body_id):
            return ("milk_1_main", "cream_cheese_1_main", "gripper0_finger")[
                body_id
            ]

    contacts = [
        SimpleNamespace(geom1=2, geom2=0),
        SimpleNamespace(geom1=2, geom2=1),
    ]
    env = SimpleNamespace(
        sim=SimpleNamespace(
            model=Model(),
            data=SimpleNamespace(ncon=2, contact=contacts),
        )
    )
    status = oracle.check(env, None, None, 3)
    assert status.violated
    assert "pre-grounding gripper/fingertip contact" in status.reason
