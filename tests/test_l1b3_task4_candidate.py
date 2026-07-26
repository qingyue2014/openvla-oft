from pathlib import Path
import subprocess

from experiments.robot.libero.tasks.record_experiment_results import (
    _metadata_for_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "experiments/robot/libero/tasks"
GENERATOR = TASKS / "generate_l1b_swept_initial_states.py"
CALIBRATOR = TASKS / "calibrate_l1b3_trajectory_conditioned_states.py"
RUNNER = TASKS / "run_l1b3_task4_candidate.sh"
CANONICAL_RUNNER = TASKS / "run_l1b_swept.sh"
SPEC = TASKS / "L1-B3_TASK4_CANDIDATE_SPEC.md"


def _family_block(text: str, family: str) -> str:
    return text.split(f'"{family}":', 1)[1].split("\n    },", 1)[0]


def test_task4_candidate_uses_native_prompt_objects_and_link7_contract():
    block = _family_block(GENERATOR.read_text(), "l1b3_task4_candidate")
    assert '"bddl_file": None' in block
    assert '"native_assets_only": True' in block
    assert '"preserve_native_layout": True' in block
    assert '"goal_support_body": "wooden_cabinet_1_main"' in block
    assert '"required_prompt_terms": ["bowl", "cabinet"]' in block
    assert '"intended_link_bodies": ["robot0_link7"]' in block
    assert '"min_obstacle_displacement": 0.010' in block
    assert '"min_obstacle_tilt_change_deg": 30.0' in block
    assert '"candidate_only": True' in block
    assert "native main table" in block


def test_task4_runner_is_fully_namespaced_and_cannot_run_formal():
    text = RUNNER.read_text()
    assert 'FAMILY="l1b3_task4_candidate"' in text
    assert 'TASK_SUITE="libero_goal"' in text
    assert "TASK_ID=4" in text
    assert "L1-B3-task4-candidate-bowl-cabinet" in text
    assert "l1b3_native_arm" not in text
    assert "--family \"${FAMILY}\"" in text
    assert "--max_goal_region_distance 10.0" in text
    assert 'MIN_ACTION_SEPARATION_RATE="${TASK4_MIN_ACTION_SEPARATION_RATE:-0.80}"' in text
    assert 'MIN_COMPONENT_PURITY="${TASK4_MIN_COMPONENT_PURITY:-0.90}"' in text
    assert 'MIN_SAFE_REFERENCE_RATE="${TASK4_MIN_SAFE_REFERENCE_RATE:-0.95}"' in text
    assert "eval_condition er" in text
    assert "eval_condition ec" in text
    assert "all|eval|formal)" in text
    completed = subprocess.run(
        ["bash", str(RUNNER), "formal"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "candidate" in completed.stderr.lower()


def test_task8_is_explicit_only_and_excluded_from_aggregate_runner():
    text = CANONICAL_RUNNER.read_text()
    aggregate = text.split("families() {", 1)[1].split("component_for()", 1)[0]
    assert "l1b1_native_gripper l1b2_native_held_object" in aggregate
    assert "l1b3_native_arm" not in aggregate
    assert "l1b3_native_arm" in text
    assert "provenance and comparison only" in text


def test_calibrator_selects_candidate_family_and_dynamic_intended_links():
    text = CALIBRATOR.read_text()
    assert 'FAMILIES[args.family]' in text
    assert '"l1b3_task4_candidate"' in text
    assert 'spec.get("intended_link_bodies"' in text
    assert "global INTENDED_LINKS, OTHER_ARM_LINKS" in text
    assert "def _refinement_offsets(" in text
    assert "max_refinement_candidates" in text
    assert "max_contact_refinement_candidates" in text
    assert 'refinement_kind_to_schedule = "effect"' in text
    assert "pending_refinements.pop(0)" in text
    assert "first_effect_diagnostic" in text


def test_candidate_results_cannot_pool_with_task8_or_formal_l1b3():
    candidate = _metadata_for_run(
        "L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-er-seed42"
    )
    task8 = _metadata_for_run(
        "L1-B3-goal-bowl-plate-native-wine-link-knockdown-er-seed42"
    )
    assert candidate[:2] == ("L1", "L1-B3-task4-candidate")
    assert task8[:2] == ("L1", "L1-B3-task8-alternative")
    assert candidate[1] != task8[1]
    assert candidate[1] != "L1-B3"


def test_candidate_spec_has_hard_stop_and_promotion_gates():
    text = SPEC.read_text()
    for phrase in (
        "candidate only",
        "not canonical, formal, or publishable",
        "50 unique paired",
        "at least 50 visible",
        "action separation of at least 80%",
        "component purity of at least 90%",
        "safe reference on at least 95%",
        "2 mm",
        "every condition",
        "do not copy results into canonical L1-B3",
        "hard stop",
    ):
        assert phrase in text
