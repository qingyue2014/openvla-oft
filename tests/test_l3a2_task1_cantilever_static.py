from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "preflight_l3a2_task1_cantilever_static.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_cantilever_static_is_exact_native_task1_post_wait_contract():
    source = SCRIPT.read_text()
    assert "TASK_ID," in source
    assert "TASK_PROMPT," in source
    assert "BDDL_SHA256," in source
    assert "for _ in range(10):" in source
    assert "if _sha(base) != BASE_SHA256:" in source
    assert '"task_description_override": None' in source
    assert '"future_evaluator_num_steps_wait": 0' in source
    assert '"scene_or_asset_modified": False' in source
    assert '"custom_asset": False' in source


def test_cantilever_static_grid_is_fixed_to_at_most_36_exact_candidates():
    source = SCRIPT.read_text()
    assert "CANTILEVER_WORLD_DEG = 135.0" in source
    assert "A_CENTER_OFFSET_M = (0.018, 0.026, 0.034)" in source
    assert "A_PITCH_DEG = (-4.0, 0.0, 4.0)" in source
    assert "B_TANGENT_CLEARANCE_M = (0.001, 0.003)" in source
    assert "B_LATERAL_OFFSET_M = (-0.010, 0.010)" in source
    assert '"maximum_AB_candidates": 36' in source
    assert "EXPECTED_A_COLLISION_HALF_SIZES_M" in source
    assert "_collision_vertices(env, A)" in source
    assert "_collision_vertices(env, B)" in source
    assert "geom_rbound" not in source


def test_cantilever_static_requires_load_support_no_bypass_and_80_hold():
    source = SCRIPT.read_text()
    assert "STATIC_HOLD_STEPS = 80" in source
    assert '"S_A": _contact(env, geoms[S], geoms[A])' in source
    assert '"A_table_absent": not _contact(env, geoms[A], table)' in source
    assert '"B_table": _contact(env, geoms[B], table)' in source
    assert '"A_B": _contact(env, geoms[A], geoms[B])' in source
    assert '"S_B": _contact(env, geoms[S], geoms[B])' in source
    assert '"forbid_S_B_bypass": True' in source
    assert "_contact_details(env, geoms[S], geoms[A])" in source


def test_cantilever_static_has_policy_visibility_and_grasp_diagnostics():
    source = SCRIPT.read_text()
    assert "MIN_S_VISIBLE_PIXELS = 100" in source
    assert "_segmentation_ids(env)" in source
    assert "_visible_pixels(env, S, segmentation)" in source
    assert "SIDE_APPROACH_DEG = (0.0, 90.0, 180.0, 270.0)" in source
    assert '"top"' in source
    assert '"open_side_count"' in source
    assert "best_static_policy_agentview.png" in source
    assert '"manual_review": "PENDING"' in source


def test_cantilever_static_never_runs_support_removal_dynamic_or_vla():
    source = SCRIPT.read_text()
    assert "_lift_s" not in source
    assert "_dynamic_gate" not in source
    assert '"hdf5_generated": False' in source
    assert '"support_removal_run": False' in source
    assert '"dynamic_run": False' in source
    assert '"vla_run": False' in source
    assert '"formal_family_generated": False' in source
    assert "h5py" not in source


def test_cantilever_static_is_one_registered_remote_phase():
    runner = RUNNER.read_text()
    assert '("l3a2", "task1_cantilever_static")' in runner
    assert "preflight_l3a2_task1_cantilever_static.py" in runner
