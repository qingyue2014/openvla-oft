import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.preflight_l1c1_native import (
    NATIVE_PROMPT,
    validate_state_file,
)


def _context():
    return {
        "native_suite": "libero_spatial",
        "native_task_id": 2,
        "native_prompt": NATIVE_PROMPT,
        "native_bddl": "libero_spatial/native.bddl",
        "native_bddl_sha256": "bddl-sha",
        "native_asset_inventory_sha256": "inventory-sha",
    }


def _write_state_file(path, *, prompt=NATIVE_PROMPT, inventory="inventory-sha"):
    key = prompt.replace(" ", "_")
    with h5py.File(path, "w") as handle:
        group = handle.create_group(key)
        group.attrs["native_suite"] = "libero_spatial"
        group.attrs["native_task_id"] = 2
        group.attrs["native_prompt"] = NATIVE_PROMPT
        group.attrs["native_bddl"] = "libero_spatial/native.bddl"
        group.attrs["native_bddl_sha256"] = "bddl-sha"
        group.attrs["native_asset_inventory_sha256"] = inventory
        group.attrs["custom_assets"] = "[]"
        group.attrs["state_intervention_variant"] = "risk"
        demo = group.create_group("demo_0")
        demo.create_dataset("initial_state", data=np.zeros(3))


def test_validate_state_file_accepts_exact_native_metadata(tmp_path):
    path = tmp_path / "states.hdf5"
    _write_state_file(path)
    result = validate_state_file(path, "Er", _context())
    assert len(result["states"]) == 1
    assert result["record"]["num_states"] == 1
    assert result["record"]["asset_inventory_sha256"] == "inventory-sha"


def test_validate_state_file_rejects_asset_inventory_mismatch(tmp_path):
    path = tmp_path / "states.hdf5"
    _write_state_file(path, inventory="custom-inventory")
    with pytest.raises(RuntimeError, match="asset_inventory"):
        validate_state_file(path, "Er", _context())


def test_validate_state_file_rejects_modified_prompt_group(tmp_path):
    path = tmp_path / "states.hdf5"
    _write_state_file(path, prompt="modified prompt")
    with pytest.raises(RuntimeError, match="prompt group mismatch"):
        validate_state_file(path, "Er", _context())
