import json
from pathlib import Path

import h5py
import pytest

from experiments.robot.libero.tasks import validate_l3b3_microwave_artifacts as artifacts
from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    PROJECT_TARGET_WORLD_XY,
    SCENE_ID as SOURCE_SCENE_ID,
    TASK_GOAL,
    TASK_KEY,
    TASK_PROMPT,
    sha256_path,
)
from experiments.robot.libero.tasks.physcog_remote_agent import PHASES
from experiments.robot.libero.tasks.summarize_l3b2_moved_cup_pi05_eb import (
    summarize,
)
from experiments.robot.libero.tasks.validate_l3b3_microwave_artifacts import (
    verify_pi05_eb_diagnostic_evaluation_request,
)


def _diagnostic_manifest(tmp_path: Path, states: Path) -> Path:
    record = {
        "scenario": SOURCE_SCENE_ID,
        "design_version": 5,
        "native_suite": "libero_10",
        "native_task_id": 9,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "verdict": "PASS_L3B3_MICROWAVE_PRECONDITION_NATIVE_ONLY_PREFLIGHT",
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "pi05_exact_224_preprocessing_complete": True,
        "pi05_eb_diagnostic_only": True,
        "pi05_formal_authorized": False,
        "human_review_approved": False,
        "formal_authorized": False,
        "native_bddl": {
            "bddl_sha256": "b" * 64,
            "goal_signature_sha256": "g" * 64,
        },
        "native_asset_manifest_sha256": "a" * 64,
        "state_bundles": {
            "native": {"path": str(states), "sha256": sha256_path(states)}
        },
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_pi05_diagnostic_is_hash_bound_to_eb_and_remains_non_formal(
    tmp_path, monkeypatch
):
    states = tmp_path / "eb.hdf5"
    with h5py.File(states, "w") as handle:
        handle.create_group(TASK_KEY).attrs["condition"] = "native"
    manifest = _diagnostic_manifest(tmp_path, states)
    bddl = tmp_path / "native.bddl"
    bddl.write_text("native", encoding="utf-8")
    monkeypatch.setattr(
        artifacts,
        "validate_native_bddl",
        lambda _: {
            "bddl_sha256": "b" * 64,
            "goal_signature_sha256": "g" * 64,
        },
    )
    monkeypatch.setattr(
        artifacts,
        "verify_native_asset_provenance",
        lambda: {"asset_manifest_sha256": "a" * 64},
    )
    result = verify_pi05_eb_diagnostic_evaluation_request(
        manifest,
        task_suite_name="libero_10",
        task_id=9,
        task_language=TASK_PROMPT,
        task_bddl=bddl,
        policy_prompt=TASK_PROMPT,
        initial_states_path=states,
    )
    assert result["formal_authorized"] is False
    assert result["pi05_formal_authorized"] is False

    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["formal_authorized"] = True
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="formal_authorized"):
        verify_pi05_eb_diagnostic_evaluation_request(
            manifest,
            task_suite_name="libero_10",
            task_id=9,
            task_language=TASK_PROMPT,
            task_bddl=bddl,
            policy_prompt=TASK_PROMPT,
            initial_states_path=states,
        )


def test_pi05_diagnostic_summary_requires_exact_runtime_gate(tmp_path):
    rows = []
    for episode in range(3):
        rows.append(
            {
                "model_family": "pi05",
                "task_suite_name": "libero_10",
                "task_id": 9,
                "episode_idx": episode,
                "success": episode == 1,
                "runtime_initial_gate": {
                    "scenario": SOURCE_SCENE_ID,
                    "condition": "native",
                    "physical_gate_pass": True,
                },
                "l3b3_microwave_sequence": {
                    "scenario": SOURCE_SCENE_ID,
                    "condition": "native",
                },
            }
        )
    (tmp_path / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = summarize(tmp_path, expected_count=3)
    assert result["candidate_target_world_xy"] == list(PROJECT_TARGET_WORLD_XY)
    assert result["native_successes"] == 1
    assert result["candidate_should_be_preregistered_as_l3b2"] is True
    assert result["formal_authorized"] is False


def test_remote_phase_is_bounded_and_stores_videos_under_l3b2_review():
    phase = PHASES[("l3b2", "moved_eb_pi05")]
    assert phase.count_env == "SMOKE_TRIALS"
    assert "review/L3-B2_task/moved_cup_pi05_eb_diagnostic" in phase.artifacts
    command = " ".join(phase.command)
    assert "pi05_eb_diagnostic" in command
    assert "formal" not in command
