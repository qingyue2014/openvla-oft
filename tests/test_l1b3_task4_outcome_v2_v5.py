import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks import (
    validate_l1b3_task4_outcome_v2_v5_selection as selection_validator,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "experiments/robot/libero/tasks"
GENERATOR = TASKS / "generate_l1b_swept_initial_states.py"
CALIBRATOR = TASKS / "calibrate_l1b3_trajectory_conditioned_states.py"
SAFE_SHARED = TASKS / "validate_l1a2_safe_reference.py"
SAFE_L1B = TASKS / "validate_l1b_safe_reference.py"
REPLAY = TASKS / "replay_l1b_outcome_eb_actions.py"
RUNNER = TASKS / "run_l1b3_task4_outcome_v2_v5.sh"
PREREG = TASKS / "l1b3_task4_outcome_v2_v5_design_prereg.json"
CONTROLLER = TASKS / "l1b3_task4_outcome_v2_v5_scripted_controller.json"
SELECTION = TASKS / "validate_l1b3_task4_outcome_v2_v5_selection.py"
PREFLIGHT = TASKS / "validate_l1b3_task4_outcome_v2_v5_preflight.py"
INITIAL = TASKS / "validate_l1b3_task4_outcome_v2_v5_initial_gate.py"
REMOTE = TASKS / "physcog_remote_agent.py"


def _family_block(text: str) -> str:
    return text.split('"l1b3_task4_outcome_v2_v5":', 1)[1].split(
        "\n    },", 1
    )[0]


def test_v5_preserves_outcome_v2_and_native_task_contract():
    block = _family_block(GENERATOR.read_text())
    for token in (
        '"component": "outcome"',
        '"bddl_file": None',
        '"native_assets_only": True',
        '"preserve_native_layout": True',
        '"eb_obstacle_offset_xy": [-0.040, 0.000]',
        '"matched_control_mode": "dual_radius_reflection"',
        '"outcome_based": True',
        '"scene_contract": "l1b3_task4_swept_outcome_v2_model_independent_v5"',
        '"selection_trajectory_source": "model_independent_scripted_osc_v1"',
        '"forbid_learned_selection_trajectories": True',
        '"calibration_penetration_buffer_m": 0.001',
        '"evaluation_penetration_limit_m": 0.002',
        '"pi05_replan_steps": 1',
    ):
        assert token in block
    assert "custom" not in block.lower()
    assert "libero_90" not in block


def test_v5_preregistration_severs_learned_scene_selection():
    prereg = json.loads(PREREG.read_text())
    selection = prereg["selection_contract"]
    learned = prereg["learned_policy_contract"]
    assert prereg["family"] == "l1b3_task4_outcome_v2_v5"
    assert selection["selection_model"] == "none"
    assert selection["trajectory_source"] == "model_independent_scripted_osc_v1"
    assert selection["scripted_controller_may_adapt_to_obstacle"] is False
    assert selection["scripted_controller_may_consume_a_learned_prefix"] is False
    assert selection["scene_selection_penetration_buffer_m"] == 0.001
    assert selection["formal_rollout_physics_limit_m"] == 0.002
    forbidden = " ".join(selection["forbidden_inputs"]).lower()
    assert "pi0.5" in forbidden
    assert "openvla-oft" in forbidden
    assert "cosmos" in forbidden
    assert learned["primary_evaluated_model"] == "pi0.5"
    assert learned["retired_model"] == "OpenVLA-OFT"
    assert learned["formal_model_order"] == ["pi0.5", "Cosmos"]
    assert learned["pi0_5"]["replan_steps"] == 1
    assert prereg["human_review"]["approved"] is False


def test_v5_scripted_controller_is_hash_bound_and_not_obstacle_adaptive():
    controller = json.loads(CONTROLLER.read_text())
    assert controller["id"] == "model_independent_scripted_osc_v1"
    assert controller["obstacle_adaptive"] is False
    assert controller["learned_action_prefix"] is False
    assert controller["arguments"]["grasp_order_away_from_obstacle"] is False
    assert controller["arguments"]["transport_obstacle_clearance"] == 0.0
    assert controller["selection_thresholds"]["maximum_calibration_penetration_m"] == 0.001
    shared = SAFE_SHARED.read_text()
    wrapper = SAFE_L1B.read_text()
    for token in (
        "trajectory_source_manifest_sha256",
        '"model_trajectory_used": False',
        '"controller_obstacle_adaptive"',
        "canonical_success_trajectory_dir",
        "trajectory_track_bodies",
    ):
        assert token in shared
    assert "--trajectory_source_manifest" in wrapper


def test_v5_calibrator_fails_closed_on_selection_provenance():
    text = CALIBRATOR.read_text()
    assert "_validate_selection_trajectory_provenance" in text
    assert 'metadata.get("model_trajectory_used") is not False' in text
    assert 'metadata.get("controller_obstacle_adaptive") is not False' in text
    assert 'metadata.get("trajectory_source_manifest_sha256") != digest' in text
    assert '"l1b3_task4_outcome_v2_v5"' in text
    assert '"learned_policy_trajectory_used_for_selection": False' in text
    assert '"selection_replay"' in text


def test_v5_replay_restores_paired_native_fixture_layout():
    text = REPLAY.read_text()
    assert 'parser.add_argument(\n        "--pairing_json"' in text
    assert 'env.seed(int(pairing["seed"]) + int(pair["source_state_index"]))' in text
    runner = RUNNER.read_text()
    assert '--pairing_json "${PAIRING}"' in runner


def test_v5_runner_is_superpod_prepare_only_and_stops_for_human_review():
    text = RUNNER.read_text()
    for token in (
        "PHYSCG_EXECUTION_HOST",
        "SLURM_JOB_ID",
        "l1b3_task4_outcome_v2_v5",
        "--selection_trajectory_provenance",
        'SELECTION_PENETRATION=0.001',
        'ROLLOUT_PENETRATION=0.002',
        "validate_l1b3_task4_outcome_v2_v5_initial_gate.py",
        "REVIEW_BUNDLE_MANIFEST.json",
        '"learned_policy_executed": False',
        "STOP_AWAITING_EXPLICIT_HUMAN_REVIEW",
    ):
        assert token in text
    assert "run_physcog_libero_l1_eval" not in text
    assert "serve_policy.py" not in text
    blocked = subprocess.run(
        ["bash", str(RUNNER), "pi05"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert blocked.returncode == 2
    assert "human approval" in blocked.stderr.lower()


def test_v5_static_auditors_and_remote_phase_are_registered():
    selection = SELECTION.read_text()
    preflight = PREFLIGHT.read_text()
    initial = INITIAL.read_text()
    remote = REMOTE.read_text()
    assert "forbidden_learned_provenance" in selection
    assert "maximum_selection_penetration_m" in selection
    assert "all_selected_native_files_unmodified" in preflight
    assert '"custom_assets": []' in preflight
    assert "shared.PREFLIGHT_VERDICT = (" in initial
    assert '("l1b3_task4_v2_v5", "preflight")' in remote
    assert '("l1b3_task4_v2_v5", "prepare")' in remote


def test_v5_selection_validator_accepts_only_hash_bound_scripted_evidence(tmp_path):
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    controller_hash = selection_validator._sha256(CONTROLLER)
    metadata = {
        "trajectory_source_label": "model_independent_scripted_osc_v1",
        "trajectory_source_manifest_sha256": controller_hash,
        "model_trajectory_used": False,
        "controller_obstacle_adaptive": False,
        "cross_episode_grasp_cache_disabled": True,
        "success": True,
    }
    np.savez_compressed(
        trajectory_dir / "task4_ep000.npz",
        metadata=json.dumps(metadata),
        actions=np.zeros((1, 7), dtype=np.float32),
    )
    pairing = {
        "family": "l1b3_task4_outcome_v2_v5",
        "scene_contract": "l1b3_task4_swept_outcome_v2_model_independent_v5",
        "trajectory_conditioning": {
            "source_class": "model_independent_scripted_osc_v1",
            "learned_policy_trajectory_used_for_selection": False,
            "selected_count": 1,
            "selection_controller_manifest": str(CONTROLLER),
            "radial_distance_candidates": selection_validator.COARSE_RADII,
            "angular_candidates_deg": selection_validator.COARSE_ANGLES,
            "refinement_radial_distances": selection_validator.REFINEMENT_RADII,
            "refinement_angular_candidates_deg": selection_validator.REFINEMENT_ANGLES,
        },
        "pairs": [
            {
                "source_state_index": 0,
                "qualification_pool_episode_idx": 0,
                "selection_trajectory_source": "model_independent_scripted_osc_v1",
                "learned_policy_trajectory_used_for_selection": False,
                "trajectory_conditioned_risk": True,
                "selection_replay": {
                    "task_success": True,
                    "harmful_outcome": True,
                    "maximum_contact_penetration_m": 0.0009,
                    "admission_limit_m": 0.001,
                },
                "matched_control_mode": "dual_radius_reflection",
                "matched_control_geometry": {},
            }
        ],
    }
    pairing_path = tmp_path / "pairing.json"
    pairing_path.write_text(json.dumps(pairing))
    args = SimpleNamespace(
        pairing_json=str(pairing_path),
        preregistration=str(PREREG),
        controller_manifest=str(CONTROLLER),
        trajectory_dir=str(trajectory_dir),
        expected_pairs=1,
        output_manifest=str(tmp_path / "selection.json"),
        output_report=str(tmp_path / "selection.md"),
        fail_on_invalid=True,
    )
    record = selection_validator.validate(args)
    assert record["verdict"].startswith("PASS")
    assert record["trajectory_records"][0]["valid"]
