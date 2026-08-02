import json
from pathlib import Path

import pytest

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    CONDITION_LABEL,
    DESIGN_VERSION,
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
from experiments.robot.libero.tasks.validate_l3b_bowl_human_review import (
    _video_inventory,
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
    assert DESIGN_VERSION == 1
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
    assert result["physical_thresholds"]["ec_drawer_joint_locked_during_construction"] is True


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
