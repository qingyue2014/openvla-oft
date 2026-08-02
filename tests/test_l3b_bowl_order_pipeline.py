import ast
import json
from pathlib import Path

import pytest

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    CONDITION_LABEL,
    DESIGN_VERSION,
    DRAWER_CLOSED_QPOS,
    MAX_DRAWER_CABINET_PENETRATION_M,
    EXPECTED_INITIAL_PREDICATES,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
)
from experiments.robot.libero.tasks.l3b_bowl_runtime_gate import (
    BowlOrderRuntimeGateError,
    BowlOrderSequenceTracker,
)
from experiments.robot.libero.tasks.validate_l3b_bowl_v1_design import (
    OFFICIAL_STATE_INDICES,
    validate_spec,
)
from experiments.robot.libero.tasks.validate_l3b_bowl_design import (
    V2_ID,
    validate_spec as validate_registered_design,
)
from experiments.robot.libero.tasks.validate_l3b_bowl_human_review import (
    _video_inventory,
)
from experiments.robot.libero.tasks.summarize_l3b_bowl_order import (
    _wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "experiments" / "robot" / "libero" / "tasks"


class _PredicateEnv:
    def __init__(self, close: bool, inside: bool):
        self.env = self
        self.close = close
        self.inside = inside

    def _eval_predicate(self, state):
        if state[0] == "close":
            return self.close
        if state[0] == "in":
            return self.inside
        raise AssertionError(state)


def test_native_task_and_condition_contract():
    assert SCENE_ID == "L3-B-BOWL-ORDER"
    assert DESIGN_VERSION == 2
    assert DRAWER_CLOSED_QPOS == 0.002
    assert MAX_DRAWER_CABINET_PENETRATION_M == 1e-5
    assert (SUITE, TASK_ID) == ("libero_10", 3)
    assert TASK_FILE == (
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_"
        "cabinet_and_close_it.bddl"
    )
    assert TASK_PROMPT == (
        "put the black bowl in the bottom drawer of the cabinet and close it"
    )
    assert CONDITION_LABEL == {
        "native": "Eb",
        "premature_close": "Er",
        "prerequisite_done": "Ec",
    }
    assert EXPECTED_INITIAL_PREDICATES == {
        "native": {"close": False, "in": False},
        "premature_close": {"close": True, "in": False},
        "prerequisite_done": {"close": False, "in": True},
    }


def test_v1_design_locks_fixed_native20_and_event_metric():
    result = validate_spec(TASKS / "l3b_bowl_v1_design_prereg.json")
    assert result["official_state_indices"] == list(range(20))
    assert result["official_state_indices"] == OFFICIAL_STATE_INDICES
    assert result["primary_metric"] == (
        "fraction of Er episodes with full ordered rollback-and-repair trace"
    )
    assert result["collision_oracle"] is False
    assert result["physical_thresholds"]["bowl_max_tilt_deg_throughout"] == 1.0
    assert "er_closed_drawer_target_qpos" not in result["physical_thresholds"]
    assert "drawer_cabinet_self_contact_allowed" not in result["physical_thresholds"]
    assert result["physical_thresholds"]["ec_drawer_joint_locked_during_construction"] is True


def test_v2_design_locks_matched_native50_model_matrix():
    result = validate_registered_design(
        TASKS / "l3b_bowl_v2_design_prereg.json"
    )
    assert result["preregistration_id"] == V2_ID
    assert result["evaluation_version"] == 2
    assert result["official_state_indices"] == list(range(50))
    assert result["count"] == 50
    assert result["model_matrix"] == {
        "pi05": "gs://openpi-assets/checkpoints/pi05_libero",
        "openvla_oft": "moojink/openvla-7b-oft-finetuned-libero-10",
    }


def test_native50_tightens_zero_success_wilson_interval():
    v1 = _wilson_interval(0, 20)
    v2 = _wilson_interval(0, 50)
    assert v2["upper"] < v1["upper"]
    assert v2["upper"] < 0.072


def test_er_tracker_requires_strict_rollback_insert_reclose_order():
    env = _PredicateEnv(True, False)
    tracker = BowlOrderSequenceTracker(env, "premature_close", policy_start_step=10)
    env.close = False
    tracker.observe(20)
    env.inside = True
    tracker.observe(30)
    env.close = True
    tracker.observe(40)
    result = tracker.finalize(task_success=True, final_step=40)
    assert result["rollback_step"] == 20
    assert result["insertion_step"] == 30
    assert result["reclose_step"] == 40
    assert result["full_ordered_repair"] is True
    assert result["failure_stage"] == "full_ordered_repair"
    assert result["defines_task_success"] is False
    assert result["collision_oracle_used"] is False


def test_er_tracker_separates_no_rollback_from_task_failure():
    env = _PredicateEnv(True, False)
    tracker = BowlOrderSequenceTracker(env, "premature_close", policy_start_step=10)
    result = tracker.finalize(task_success=False, final_step=310)
    assert result["rollback_recognized"] is False
    assert result["failure_stage"] == "no_rollback"
    assert result["native_task_success"] is False


def test_tracker_rejects_wrong_condition_predicates():
    env = _PredicateEnv(False, False)
    with pytest.raises(BowlOrderRuntimeGateError):
        BowlOrderSequenceTracker(env, "premature_close", policy_start_step=10)


def test_runner_is_native_only_event_based_and_formal_fail_closed():
    runner = (TASKS / "run_l3b_bowl_order.sh").read_text(encoding="utf-8")
    wrapper = (TASKS / "run_l3b_bowl_order_pi05.sh").read_text(encoding="utf-8")
    evaluator = (
        ROOT / "experiments" / "robot" / "libero" / "run_physcog_libero_l1_eval.py"
    ).read_text(encoding="utf-8")
    assert "--task_suite_name libero_10 --task_ids 3" in runner
    assert "--safety_oracle none" in runner
    assert "libero_90" not in runner
    assert "verify_human_approval" in runner
    assert "formal) run_formal" in runner
    assert '--human-approval "${HUMAN_APPROVAL}"' in runner
    assert '--smoke-report "${SMOKE_REPORT}"' in runner
    assert 'NUM_STATES="${NUM_STATES:-20}"' in runner
    assert 'FORMAL_WAIT_STEPS=10' in runner
    assert 'MAX_VIDEOS_PER_OUTCOME="${MAX_VIDEOS_PER_OUTCOME:-5}"' in runner
    assert "gs://openpi-assets/checkpoints/pi05_libero" in wrapper
    assert 'runtime_scene == "L3-B-BOWL-ORDER"' in evaluator
    assert "BowlOrderSequenceTracker" in evaluator
    assert "l3b_bowl_sequence" in evaluator
    assert "first_policy_image_dir" in evaluator
    assert "safe-witness) run_safe_witness" in runner
    assert "validate_l3b_bowl_safe_witness.py" in runner


def test_executable_safe_witness_is_action_only_and_not_learned():
    path = TASKS / "validate_l3b_bowl_safe_witness.py"
    text = path.read_text(encoding="utf-8")
    assert "materialize_native_scene_state" in text
    assert "BowlOrderRuntimeGate" in text
    assert "BowlOrderSequenceTracker" in text
    assert "TrajectoryRecorder" in text
    assert '"learned_reference_used": False' in text
    assert '"direct_qpos_edits_after_restore": False' in text
    assert "env.step(" in text
    assert "OPEN_TARGET_QPOS = -0.145" in text
    assert "TERMINAL_SETTLE_STEPS = 100" in text
    assert "forbidden robot contact" in text
    assert "GRASP_EEF_OFFSETS" in text
    assert "_stage_bowl_clear_of_drawer" in text
    assert "_return_to_policy_start_pose" in text
    assert '"robot0_eye_in_hand": output_dir' in text

    tree = ast.parse(text)
    forbidden_targets = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                source = ast.get_source_segment(text, target) or ""
                if any(
                    token in source
                    for token in ("sim.data.qpos", "sim.data.qvel", "body_xpos")
                ):
                    forbidden_targets.append(source)
    assert forbidden_targets == []


def test_v2_runner_keeps_models_on_identical_registered_states():
    runner = (TASKS / "run_l3b_bowl_order_v2.sh").read_text(encoding="utf-8")
    assert "NUM_STATES=50" in runner
    assert "FORMAL_EXPECTED_COUNT=50" in runner
    assert "l3b_bowl_v2_design_prereg.json" in runner
    assert "l3b_bowl_v2_eb_states.hdf5" in runner
    assert "pi05_smoke|pi05_formal" in runner
    assert "openvla_oft_smoke|openvla_oft_formal" in runner
    assert "moojink/openvla-7b-oft-finetuned-libero-10" in runner
    assert "prepare|check|safe-witness" in runner


def test_v2r1_runner_is_pi05_only_and_uses_fresh_evidence_paths():
    runner = (TASKS / "run_l3b_bowl_order_v2r1.sh").read_text(encoding="utf-8")
    design = validate_registered_design(TASKS / "l3b_bowl_v2r1_design_prereg.json")
    assert design["preregistration_id"] == "l3b-bowl-order-v2r1-native50-20260802"
    assert design["evaluation_version"] == 3
    assert design["model_matrix"] == {
        "pi05": "gs://openpi-assets/checkpoints/pi05_libero"
    }
    assert design["safe_witness_episode_indices"] == [0, 1, 2, 3, 4]
    assert "NUM_STATES=50" in runner
    assert "SAFE_WITNESS_EPISODES=\"0,1,2,3,4\"" in runner
    assert "L3-B_bowl_order_v2r1_pi05_task" in runner
    assert "pi05_smoke|pi05_formal" in runner
    assert "openvla" not in runner.lower()


def test_v2r2_runner_freezes_cross_platform_qpos_and_pi05_only():
    runner = (TASKS / "run_l3b_bowl_order_v2r2.sh").read_text(encoding="utf-8")
    design = validate_registered_design(TASKS / "l3b_bowl_v2r2_design_prereg.json")
    assert design["preregistration_id"] == "l3b-bowl-order-v2r2-native50-20260802"
    assert design["evaluation_version"] == 4
    assert design["physical_thresholds"]["er_closed_drawer_target_qpos"] == 0.001
    assert design["model_matrix"] == {
        "pi05": "gs://openpi-assets/checkpoints/pi05_libero"
    }
    assert design["official_state_indices"] == list(range(50))
    assert design["safe_witness_episode_indices"] == [0, 1, 2, 3, 4]
    assert "NUM_STATES=50" in runner
    assert "L3-B_bowl_order_v2r2_pi05_task" in runner
    assert "pi05_smoke|pi05_formal" in runner
    assert "openvla" not in runner.lower()


def test_human_approval_inventory_is_stable_after_formal_videos(tmp_path):
    smoke = tmp_path / "smoke" / "native"
    formal = tmp_path / "formal" / "native"
    smoke.mkdir(parents=True)
    formal.mkdir(parents=True)
    for index in range(6):
        (smoke / f"smoke_{index}.mp4").write_bytes(bytes([index]))
    for index in range(12):
        (formal / f"formal_{index}.mp4").write_bytes(bytes([index]))
    inventory = _video_inventory(tmp_path)
    assert len(inventory) == 6
    assert all(item["relative_path"].startswith("smoke/native/") for item in inventory)


def test_spec_states_workload_limit_and_non_memory_claim():
    spec = (TASKS / "L3-B_BOWL_ORDER_SPEC.md").read_text(encoding="utf-8")
    normalized = " ".join(spec.split())
    assert "not a memory-of-past-actions test" in normalized
    assert "workload-asymmetric" in normalized
    assert "`True → False → True`" in normalized
    assert "collision oracle" in normalized
