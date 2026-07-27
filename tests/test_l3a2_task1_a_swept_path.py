from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "diagnose_l3a2_task1_a_swept_path.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_a_only_diagnostic_is_exact_native_task1_wait0_and_three_seeds():
    source = SCRIPT.read_text()
    assert "FROZEN_A_SEEDS = (" in source
    assert "(16.0, 0.0)" in source
    assert "(16.0, -0.0015)" in source
    assert "(16.0, 0.0015)" in source
    assert "for _ in range(10):" in source
    assert "if _sha(base) != BASE_SHA256:" in source
    assert '"task_description_override": None' in source
    assert '"future_evaluator_num_steps_wait": 0' in source


def test_a_only_diagnostic_records_pose_vertices_aabb_direction_and_envelope():
    source = SCRIPT.read_text()
    assert '"body_xyz_m"' in source
    assert '"body_quat_wxyz"' in source
    assert '"group0_geometries"' in source
    assert '"geom_name"' in source
    assert '"group0_world_vertices_m"' in source
    assert '"group0_world_aabb"' in source
    assert '"planar_translation_heading_deg"' in source
    assert '"step_translation_xyz_m"' in source
    assert '"step_planar_translation_heading_deg"' in source
    assert '"rotation_from_start_deg"' in source
    assert '"swept_group0_envelope"' in source
    assert '"all_group0_vertices_recorded_each_step": True' in source


def test_a_only_diagnostic_finds_first_static_placeable_future_tangent():
    source = SCRIPT.read_text()
    assert "FUTURE_TANGENCY_GAP_M = 0.001" in source
    assert "PLACEMENT_SETTLE_STEPS = 120" in source
    assert "PLACEMENT_HOLD_STEPS = 120" in source
    assert '"source_trace_step"' in source
    assert '"future_tangency_gap_m"' in source
    assert '"initial_A_B"' in source
    assert '"initial_S_B"' in source
    assert '"diagnostic_only_not_scene_selection": True' in source


def test_a_only_diagnostic_has_no_scene_pass_or_vla_artifacts():
    source = SCRIPT.read_text()
    assert '"scene_verdict": None' in source
    assert '"candidate_selected": False' in source
    assert '"policy_image_or_video_generated": False' in source
    assert '"hdf5_generated": False' in source
    assert '"vla_run": False' in source
    assert '"formal_family_generated": False' in source
    assert "imageio" not in source
    assert "h5py" not in source


def test_a_only_diagnostic_is_one_registered_remote_phase():
    runner = RUNNER.read_text()
    assert '("l3a2", "task1_a_only_swept_diagnostic")' in runner
    assert "diagnose_l3a2_task1_a_swept_path.py" in runner
