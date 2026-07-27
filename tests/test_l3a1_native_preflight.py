from pathlib import Path

import pytest

from experiments.robot.libero.tasks.validate_l3a1_native_preflight import (
    TASK_FILE,
    TASK_PROMPT,
    validate_native_task,
)


NATIVE_BDDL = f"""(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {TASK_PROMPT})
  (:regions)
  (:fixtures
    kitchen_table - kitchen_table
    white_cabinet_1 - white_cabinet
    wine_rack_1 - wine_rack
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    wine_bottle_1 - wine_bottle
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


def test_preflight_accepts_the_exact_native_task_source(tmp_path):
    native = _native_path(tmp_path)
    evidence = validate_native_task(native, native, TASK_PROMPT)
    assert evidence["native_bddl"] == str(native.resolve())
    assert evidence["evaluated_bddl"] == str(native.resolve())
    assert evidence["prompt"] == TASK_PROMPT


def test_preflight_rejects_a_byte_identical_project_local_bddl(tmp_path):
    native = _native_path(tmp_path)
    custom = tmp_path / "project/PHYSCOG_L3A1.bddl"
    custom.parent.mkdir()
    custom.write_bytes(native.read_bytes())
    with pytest.raises(ValueError, match="not the selected native LIBERO task"):
        validate_native_task(native, custom, TASK_PROMPT)


def test_preflight_rejects_prompt_or_asset_inventory_changes(tmp_path):
    native = _native_path(tmp_path)
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(native, native, "modified prompt")

    native.write_text(NATIVE_BDDL.replace("wine_bottle_1 - wine_bottle\n", ""))
    with pytest.raises(ValueError, match="object inventory mismatch"):
        validate_native_task(native, native, TASK_PROMPT)
