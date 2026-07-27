from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/"
    "preflight_l3a2_task2_native_rebind_static.py"
)
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_task2_rebind_uses_exact_native_contract_and_goal():
    source = SCRIPT.read_text()
    assert 'TASK_SUITE = "libero_spatial"' in source
    assert "TASK_ID = 2" in source
    assert "prompt = task.language" in source
    assert "prompt != EXPECTED_TASK_LANGUAGE" in source
    assert "BDDL_SHA256" in source
    assert "BASE_SHA256" in source
    assert '"oracle_defines_task_success": False' in source
    assert '"task_description_override": None' in source
    assert '"S_placement_run": False' in source


def test_task2_rebind_has_frozen_grid_at_most_36():
    source = SCRIPT.read_text()
    assert "WORLD_DIRECTION_DEG = (0.0, 90.0, 180.0)" in source
    assert "A_EDGE_RADIAL_OFFSET_M = (0.010, 0.015, 0.020)" in source
    assert (
        "A_B_COLLISION_SURFACE_GAP_M = (0.002, 0.005, 0.008)"
        in source
    )
    assert "MAXIMUM_CANDIDATES = 27" in source
    assert "len(grid) > 36" in source
    assert "SETTLE_STEPS = 240" in source
    assert "INDEPENDENT_HOLD_STEPS = 80" in source
    assert "SAFE_SECTOR_RADIAL_OFFSET_M = 0.060" in source


def test_task2_rebind_roles_contacts_and_stability_are_strict():
    source = SCRIPT.read_text()
    assert 'S = "akita_black_bowl_1_main"' in source
    assert 'A = "akita_black_bowl_2_main"' in source
    assert 'B = "glazed_rim_porcelain_ramekin_1_main"' in source
    assert 'PLATE = "plate_1_main"' in source
    assert 'COOKIES = "cookies_1_main"' in source
    assert '"S_A": _contact(env, geoms[S], geoms[A])' in source
    assert '"S_B": _contact(env, geoms[S], geoms[B])' in source
    assert '"A_B": _contact(env, geoms[A], geoms[B])' in source
    assert '"A_plate": _contact(env, geoms[A], geoms[PLATE])' in source
    assert '"B_table": _contact(env, geoms[B], table)' in source
    assert "for name in (S, A, B, PLATE)" in source
    assert '"forbid_robot_contacts": True' in source
    assert '"forbid_other_object_contacts": True' in source


def test_task2_rebind_pairing_only_changes_A_B_free_joint_slices():
    source = SCRIPT.read_text()
    assert '"base_plus_native_A_B_free_joint_qpos_qvel_only"' in source
    assert "allowed_by_role = {" in source
    assert "A: _flat_indices(env, A)" in source
    assert "B: _flat_indices(env, B)" in source
    assert '"changed_outside_A_B": outside' in source
    assert '"bit_exact_outside_A_B": not outside' in source
    assert "FUTURE_EVALUATOR_WAIT_STEPS = 0" in source


def test_task2_rebind_exports_exact_policy_preprocess_and_visibility():
    source = SCRIPT.read_text()
    assert "from experiments.robot.openvla_utils import (" in source
    assert "resize_image_for_policy(raw, PROCESSED_IMAGE_SIZE)" in source
    assert "center_crop_image(resized)" in source
    assert "PROCESSED_IMAGE_SIZE = 224" in source
    assert "CENTER_CROP_AREA = 0.9" in source
    assert "_segmentation_ids(env)" in source
    assert "processed_crop_support_pixels" in source
    assert "selected_first_frame_processed_224.png" in source
    assert '"manual_review": "PENDING"' in source
    assert "SAFE_SECTOR_RADIAL_OFFSET_M" in source
    assert "_grasp_space(env" in source


def test_task2_rebind_is_static_only_and_one_remote_phase():
    source = SCRIPT.read_text()
    assert '"dynamic_run": False' in source
    assert '"vla_run": False' in source
    assert '"hdf5_generated": False' in source
    assert '"formal_family_generated": False' in source
    assert "h5py" not in source
    assert "run_physcog_libero" not in source
    runner = RUNNER.read_text()
    assert '("l3a2", "task2_native_rebind_static")' in runner
    assert "preflight_l3a2_task2_native_rebind_static.py" in runner
