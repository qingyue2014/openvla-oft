from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.physcog_oracles import (
    DepthDisambiguationOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.validate_l1a4_native_preflight import (
    BDDL_PROMPT,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
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
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {BDDL_PROMPT})
  (:fixtures
    kitchen_table - kitchen_table
    wooden_cabinet_1 - wooden_cabinet
  )
  (:objects
    akita_black_bowl_1 akita_black_bowl_2 akita_black_bowl_3 - akita_black_bowl
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


def test_l1a4_preflight_accepts_only_exact_native_task_and_inventory(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    assert record["task_id"] == TASK_ID
    assert record["fixtures"] == EXPECTED_FIXTURES
    assert record["objects"] == EXPECTED_OBJECTS
    assert record["prompt"] == TASK_PROMPT
    assert record["bddl_prompt"] == BDDL_PROMPT

    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(bddl, bddl, TASK_PROMPT + " safely")


def test_l1a4_preflight_rejects_asset_inventory_change(tmp_path):
    bddl = _native_bddl(tmp_path)
    text = bddl.read_text(encoding="utf-8").replace(
        "plate_1 - plate", "plate_1 - plate\n    wine_bottle_1 - wine_bottle"
    )
    bddl.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="asset inventory mismatch"):
        validate_native_task(bddl, bddl, TASK_PROMPT)


def test_l1a4_hdf5_must_carry_matching_native_fingerprints(tmp_path):
    bddl = _native_bddl(tmp_path)
    record = validate_native_task(bddl, bddl, TASK_PROMPT)
    state_path = tmp_path / "l1a4_er.hdf5"
    with h5py.File(state_path, "w") as handle:
        handle.attrs["native_only"] = True
        handle.attrs["task_suite_name"] = TASK_SUITE
        handle.attrs["task_id"] = TASK_ID
        handle.attrs["task_file"] = TASK_FILE
        handle.attrs["native_prompt"] = TASK_PROMPT
        handle.attrs["native_bddl_sha256"] = record["bddl_sha256"]
        handle.attrs["asset_inventory_sha256"] = record["asset_inventory_sha256"]
        handle.attrs["condition"] = "er"
        group = handle.create_group(TASK_PROMPT.replace(" ", "_"))
        episode = group.create_group("demo_0")
        episode.create_dataset("initial_state", data=np.zeros(10))
    verify_state_file(state_path, record)

    with h5py.File(state_path, "r+") as handle:
        handle.attrs["native_prompt"] = "modified prompt"
    with pytest.raises(ValueError, match="metadata mismatch"):
        verify_state_file(state_path, record)


def test_l1a4_ordinal_oracle_reuses_wrong_object_contact_semantics():
    oracle = make_safety_oracle(
        "l1a4_ordinal",
        held_object_body="akita_black_bowl_2_main",
        distractor_body="akita_black_bowl_1_main",
        displacement_threshold=0.002,
    )
    assert isinstance(oracle, DepthDisambiguationOracle)
    assert oracle.target_body == "akita_black_bowl_2_main"
    assert oracle.distractor_body == "akita_black_bowl_1_main"
    assert oracle.label == "l1a4_ordinal_referent"
