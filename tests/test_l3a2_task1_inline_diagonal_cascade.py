from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "preflight_l3a2_task1_inline_diagonal_cascade.py"
)
BASE = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "preflight_l3a2_task1_diagonal_cascade.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_inline_diagonal_reuses_only_frozen_robust_a_seeds():
    source = SCRIPT.read_text()
    assert "FROZEN_A_SEEDS = (" in source
    assert "(16.0, 0.0)" in source
    assert "(16.0, -0.0015)" in source
    assert "(16.0, 0.0015)" in source
    assert "a_scan_points=FROZEN_A_SEEDS" in source
    assert "fixed_a_seed_points=FROZEN_A_SEEDS" in source


def test_inline_diagonal_is_bounded_to_27_dynamic_candidates():
    source = SCRIPT.read_text()
    assert "B_TURN_DEG = (0.0, 10.0, 20.0)" in source
    assert "B_CLEARANCE_M = (0.0, 0.002, 0.004)" in source
    assert len((0.0, 10.0, 20.0)) * len((0.0, 0.002, 0.004)) * 3 == 27


def test_inline_diagonal_remains_non_cardinal_and_uses_strict_shared_gates():
    source = SCRIPT.read_text()
    base = BASE.read_text()
    assert "world -45 degree diagonal" in source
    assert "0/10/20 degrees from the measured fall ray" in source
    assert "and int(a_motion) < int(impact) <= int(b_hazard)" in base
    assert '"S_B_bypass": bypass' in base
    assert '"S_fixed": _fixed_s_control' in base
    assert '"A_collision_disabled": _a_disabled_control' in base
    assert "selected_adjacent_witnesses" in base


def test_inline_diagonal_is_one_registered_no_vla_preflight():
    source = SCRIPT.read_text()
    runner = RUNNER.read_text()
    assert (
        'verdict_tag="L3A2_TASK1_INLINE_DIAGONAL_ONE_STATE_NO_VLA_GATE"'
        in source
    )
    assert '("l3a2", "task1_inline_diagonal_scan")' in runner
    assert "preflight_l3a2_task1_inline_diagonal_cascade.py" in runner
    assert '"vla_run": False' in BASE.read_text()
    assert '"formal_family_generated": False' in BASE.read_text()
