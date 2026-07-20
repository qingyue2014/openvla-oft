import json
from types import SimpleNamespace

import pytest

from experiments.robot.libero.tasks import l2a_native_approval as approval


def _fixture(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks"
    preview = task_dir / "l2a_native_preview"
    preview.mkdir(parents=True)
    paths = {
        "BDDL": task_dir / "scene.bddl",
        "PAIRING": task_dir / "pairing.json",
        "EC_STATES": task_dir / "ec.hdf5",
        "ER_STATES": task_dir / "er.hdf5",
        "PREVIEW_DIR": preview,
        "APPROVAL": task_dir / "approval.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(approval, name, path)
    paths["BDDL"].write_text("scene", encoding="utf-8")
    paths["PAIRING"].write_text(json.dumps({"num_states": 1}), encoding="utf-8")
    paths["EC_STATES"].write_bytes(b"ec")
    paths["ER_STATES"].write_bytes(b"er")
    (preview / "episode_000_Ec_stove_off_policy224.png").write_bytes(b"ec-view")
    (preview / "episode_000_Er_stove_on_policy224.png").write_bytes(b"er-view")
    return paths


def test_frozen_approval_detects_any_post_review_change(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    approval.create(SimpleNamespace(reviewer="test", notes="reviewed"))
    approval.verify(SimpleNamespace(expected_states=1))

    paths["ER_STATES"].write_bytes(b"changed")

    with pytest.raises(SystemExit):
        approval.verify(SimpleNamespace(expected_states=1))
