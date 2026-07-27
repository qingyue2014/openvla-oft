import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/robot/libero/tasks/audit_l3a4_goal_task4_native_static.py"
)
SOURCE = SOURCE_PATH.read_text()


def test_source_parses_and_binds_official_goal_task4_contract():
    ast.parse(SOURCE)
    assert 'TASK_SUITE = "libero_goal"' in SOURCE
    assert "TASK_ID = 4" in SOURCE
    assert (
        'TASK_BDDL = "put_the_bowl_on_top_of_the_cabinet.bddl"'
        in SOURCE
    )
    assert 'PROMPT = "put the bowl on top of the cabinet"' in SOURCE
    assert '"policy_prompt_source": "task.language"' in SOURCE
    assert "task.language != PROMPT" in SOURCE
    assert "bddl.parent.name != TASK_SUITE" in SOURCE
    assert '"custom_bddl": False' in SOURCE
    assert '"official_bddl_untouched": True' in SOURCE


def test_internal_bddl_language_typo_is_non_policy_metadata_only():
    assert (
        '"(:language Put the bowl on the top of the drawer)"'
        in SOURCE
    )
    assert '"bddl_internal_language_non_policy_field"' in SOURCE
    assert '"bddl_internal_language_known_typo": True' in SOURCE
    assert '"prompt_override": False' in SOURCE


def test_frozen_grid_is_exactly_36_and_static_only():
    assert "A_RADIAL_OFFSETS_M = (0.000, 0.008, 0.016)" in SOURCE
    assert "B_CLEARANCES_M = (0.002, 0.006, 0.010)" in SOURCE
    assert "if expected_count != 36" in SOURCE
    assert "args.settle_steps != 240 or args.hold_steps != 80" in SOURCE
    assert '"release_run": False' in SOURCE
    assert '"dynamic_run": False' in SOURCE
    assert '"vla_run": False' in SOURCE
    assert '"hdf5_written": False' in SOURCE
    assert '"formal_run": False' in SOURCE


def test_only_native_a_b_serialized_qpos_may_change():
    assert 'for role in ("A", "B")' in SOURCE
    assert "changed <= allowed" in SOURCE
    assert '"non_A_B_state_bytes_identical"' in SOURCE
    assert '"serialized_pose_changes_only": True' in SOURCE
    assert "l1b4_goal_arm_sweep.bddl" not in SOURCE
    assert "l1_b_goal_arm_gate" not in SOURCE


def test_physical_and_policy_view_hard_gates_are_separate():
    for token in (
        '"A_table"',
        '"A_fixture"',
        '"A_plate"',
        '"A_B"',
        '"S_B"',
        '"B_table"',
        '"max_S_A_normal_force_N"',
        '"S_grasp_corridor"',
        '"cabinet_goal_reachability"',
        '"policy_visibility"',
        "_segmentation_geom_ids",
        "POLICY_RESOLUTION = 256",
    ):
        assert token in SOURCE
    assert '"physical_verdict"' in SOURCE
    assert '"visual_verdict"' in SOURCE


def test_evaluator_entry_and_goal_checkpoint_binding_are_exact():
    assert "EVALUATOR_ENV_SEED = 0" in SOURCE
    assert "EVALUATOR_NUM_STEPS_WAIT = 10" in SOURCE
    assert "FUTURE_EVALUATOR_NUM_STEPS_WAIT = 0" in SOURCE
    assert "get_libero_dummy_action(MODEL_FAMILY)" in SOURCE
    assert (
        'POLICY_CHECKPOINT = "moojink/'
        'openvla-7b-oft-finetuned-libero-goal"'
        in SOURCE
    )
    assert 'POLICY_ACTION_HEAD = "action_head--50000_checkpoint.pt"' in SOURCE
    assert '"model_loaded": False' in SOURCE
    assert '"inference_executed": False' in SOURCE


def test_historical_custom_benign_result_is_prior_not_scene_evidence():
    assert (
        '"classification": "custom_benign_task_competence_prior_only"'
        in SOURCE
    )
    assert '"successes": 49' in SOURCE
    assert "It is not reused here" in SOURCE
