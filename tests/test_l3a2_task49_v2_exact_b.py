from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/robot/libero/tasks/preflight_l3a2_task49_v2_exact_b.py"
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_task49_v2_is_separate_hash_bound_and_b_only_grid():
    source = SCRIPT.read_text()
    assert "SOURCE_JOB = 490084" in source
    assert "A_TEMPLATE_SHA256" in source
    assert "BASE_SHA256" in source
    assert "itertools.product(GRID_M, repeat=2)" in source
    assert "_place_b_exact" in source
    assert "exact_compiled_group0_primitive_mesh_world_aabb" in source
    assert "CLEARANCE_M = 0.0005" in source
    assert "outside_a_b_bit_exact" in source
    assert "original_task49_v1_status" in source


def test_task49_v2_requires_witness_causality_ablations_and_policy_view():
    source = SCRIPT.read_text()
    assert "_neighbors(set(states))" in source
    assert '"S_removal"' in source
    assert '"A_ablation"' in source
    assert '"B_ablation"' in source
    assert "S_B_bypass" in source
    assert "ordered_A_then_B" in source
    assert "agentview_image" in source
    assert '"eb_generated": False' in source
    assert '"vla_run": False' in source


def test_task49_v2_registered():
    source = RUNNER.read_text()
    assert '("l3a2", "task49_v2_exact_b")' in source
    assert "preflight_l3a2_task49_v2_exact_b.py" in source
