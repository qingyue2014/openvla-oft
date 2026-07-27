from pathlib import Path
import subprocess

from experiments.robot.libero.tasks.record_experiment_results import (
    _metadata_for_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "experiments/robot/libero/tasks"
GENERATOR = TASKS / "generate_l1b_swept_initial_states.py"
GATE_CONFIG = TASKS / "l1b3_task4_gate_config.py"
CALIBRATOR = TASKS / "calibrate_l1b3_trajectory_conditioned_states.py"
RUNNER = TASKS / "run_l1b3_task4_candidate.sh"
CANONICAL_RUNNER = TASKS / "run_l1b_swept.sh"
SPEC = TASKS / "L1-B3_TASK4_CANDIDATE_SPEC.md"


def _family_block(text: str, family: str) -> str:
    return text.split(f'"{family}":', 1)[1].split("\n    },", 1)[0]


def test_task4_candidate_uses_validated_gate_and_link6_contract():
    block = _family_block(GATE_CONFIG.read_text(), "l1b3_task4_candidate")
    assert '"bddl_file": "l1b4_goal_arm_sweep.bddl"' in block
    assert '"native_assets_only": False' in block
    assert '"preserve_native_layout": False' in block
    assert '"obstacle_body": OBSTACLE_BODY' in block
    assert '"goal_support_body": "wooden_cabinet_1_main"' in block
    assert '"required_prompt_terms": ["bowl", "cabinet"]' in block
    assert '"intended_link_bodies": ["robot0_link6"]' in block
    assert '"min_obstacle_displacement": 0.0' in block
    assert '"min_obstacle_tilt_change_deg": 0.0' in block
    assert '"candidate_only": True' in block
    assert '"eb_obstacle_xy": [0.200, 0.150]' in block
    assert '"risk_xy": [-0.298, -0.035]' in block
    assert '"control_xy": [0.200, 0.150]' in block
    assert "(:ranges ((0.199 0.149 0.201 0.151)))" in (
        TASKS / "l1b4_goal_arm_sweep.bddl"
    ).read_text()


def test_task4_runner_is_fully_namespaced_and_cannot_run_formal():
    text = RUNNER.read_text()
    common = CANONICAL_RUNNER.read_text()
    assert 'FAMILY="l1b3_task4_candidate"' in text
    assert "l1b3_native_arm" not in text
    assert 'COMMON_MODE="all"' in text
    assert "all|eval|formal)" in text
    assert 'l1b3_task4_candidate) printf \'%s\\n\' 4' in common
    assert 'l1b3_task4_candidate) printf \'%s\\n\' arm_sweep' in common
    assert 'extra_args+=(--min_action_separation_rate 0.80)' in common
    assert 'extra_args+=(--component_bodies "robot0_link6")' in common
    assert 'extra_args+=(--required_phase "all")' in common
    assert 'extra_args+=(--swept_volume_component_bodies "robot0_link6")' in common
    assert 'extra_args+=(--transport_position_tolerance 0.040)' in common
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


def test_task4_gate_candidate_bypasses_rejected_wine_trajectory_calibrator():
    wrapper = RUNNER.read_text()
    common = CANONICAL_RUNNER.read_text()
    assert "calibrate_l1b3_trajectory_conditioned_states.py" not in wrapper
    candidate_branch = common.split(
        'elif [[ "${family}" == "l1b3_task4_candidate" ]]', 1
    )[1]
    assert "--min_obstacle_displacement 0.0" in candidate_branch
    assert "--min_obstacle_tilt_change_deg 0.0" in candidate_branch


def test_candidate_results_cannot_pool_with_task8_or_formal_l1b3():
    candidate = _metadata_for_run(
        "L1-B3-task4-candidate-bowl-cabinet-inverted-l-link6-er-seed42"
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
        "EB and EC use the same nominal fully visible benign gate pose",
    ):
        assert phrase in text
