from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/robot/libero/tasks/audit_l3a4_spatial_task1_native.py"
).read_text()


def test_exact_task1_contract_and_evaluator_policy_entry_are_bound():
    assert "TASK_ID = 1" in SOURCE
    assert (
        'PROMPT = "pick up the black bowl next to the ramekin '
        'and place it on the plate"'
    ) in SOURCE
    assert "EVALUATOR_NUM_STEPS_WAIT = 10" in SOURCE
    assert "DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]" in SOURCE


def test_bddl_and_runtime_support_semantics_are_both_table():
    assert (
        "(On akita_black_bowl_1 main_table_next_to_ramekin_region)"
        in SOURCE
    )
    assert 'EXPECTED_RUNTIME_SUPPORT_BODY = "table"' in SOURCE
    assert '"bddl_runtime_support_semantics_match": True' in SOURCE
    assert '"support_region_fixture": "main_table"' in SOURCE
