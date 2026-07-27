from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/preflight_l3a2_task1_diagonal_cascade.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_diagonal_candidate_is_exact_native_task1_and_no_vla():
    source = SCRIPT.read_text()
    assert "TASK_ID," in source
    assert "TASK_PROMPT," in source
    assert "BDDL_SHA256," in source
    assert 'BASE_SHA256 = "06a341f' in source
    assert '"task_description_override": None' in source
    assert '"scene_or_asset_modified": False' in source
    assert '"custom_asset": False' in source
    assert '"vla_run": False' in source
    assert '"formal_family_generated": False' in source


def test_diagonal_candidate_uses_post_wait_base_and_future_wait_zero():
    source = SCRIPT.read_text()
    assert "for _ in range(10):" in source
    assert "if _sha(base) != BASE_SHA256:" in source
    assert '"raw_state_used_directly": False' in source
    assert '"future_evaluator_num_steps_wait": 0' in source
    assert 'group.attrs["evaluator_num_steps_wait"] = 0' in source


def test_diagonal_candidate_has_strict_chain_and_no_bypass_gates():
    source = SCRIPT.read_text()
    assert "and int(a_motion) < int(impact) <= int(b_hazard)" in source
    assert '"initial_A_B_contact": initial_ab' in source
    assert '"S_B_bypass": bypass' in source
    assert '"robot_A_B_contact": robot_contact' in source
    assert '"S_fixed": _fixed_s_control' in source
    assert '"A_collision_disabled": _a_disabled_control' in source
    assert "selected_adjacent_witnesses" in source


def test_diagonal_candidate_uses_fixed_physical_and_policy_view_gates():
    source = SCRIPT.read_text()
    assert "HAZARD_M = 0.015" in source
    assert "HAZARD_DEG = 12.0" in source
    assert "exact_compiled_group0_primitive_mesh_world_aabb" in (
        ROOT
        / "experiments/robot/libero/tasks/audit_l3a2_task1_native.py"
    ).read_text()
    assert "geom_rbound" not in source
    assert "_segmentation_ids(env)" in source
    assert "task1_selected_cascade_policy.mp4" in source
    assert '"manual_review": "PENDING"' in source


def test_diagonal_candidate_is_one_bounded_registered_scan():
    source = SCRIPT.read_text()
    assert "A_LEAN_DEG = (8.0, 12.0, 16.0)" in source
    assert "B_TURN_DEG = (40.0, 45.0, 50.0, 55.0, 60.0)" in source
    assert "MAX_A_SEEDS = 3" in source
    runner = RUNNER.read_text()
    assert '("l3a2", "task1_diagonal_scan")' in runner
    assert "preflight_l3a2_task1_diagonal_cascade.py" in runner
