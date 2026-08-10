from pathlib import Path

from experiments.robot.libero.tasks.verify_l1c5_frozen_gate import (
    DEFAULT_MANIFEST,
    EXPECTED_MANIFEST_SHA256,
    _directory_closure,
    _sha256,
    verify,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_l1c5_machine_gate_is_hash_bound_and_complete():
    manifest_path = REPO_ROOT / DEFAULT_MANIFEST
    assert _sha256(manifest_path) == EXPECTED_MANIFEST_SHA256
    verify(REPO_ROOT, manifest_path)


def test_directory_closure_binds_names_and_contents(tmp_path):
    (tmp_path / "a.txt").write_text("same")
    original = _directory_closure(tmp_path)
    (tmp_path / "a.txt").rename(tmp_path / "b.txt")
    renamed = _directory_closure(tmp_path)
    assert original["file_count"] == renamed["file_count"] == 1
    assert original["sha256"] != renamed["sha256"]
