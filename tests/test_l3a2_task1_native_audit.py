from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/robot/libero/tasks/audit_l3a2_task1_native.py"
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_task1_contract_is_exact_native_and_vla_free():
    source = SCRIPT.read_text()
    assert "TASK_ID = 1" in source
    assert (
        '"pick up the black bowl next to the ramekin and place it on the plate"'
        in source
    )
    assert 'CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"' in source
    assert '"task_description_override": None' in source
    assert '"scene_or_asset_modified": False' in source
    assert '"vla_run": False' in source


def test_task1_policy_entry_is_post_wait_and_serialized_once():
    source = SCRIPT.read_text()
    assert "states = suite.get_task_init_states(TASK_ID)" in source
    assert "obs = env.set_init_state(states[0])" in source
    assert "for _ in range(POLICY_ENTRY_WAIT_STEPS):" in source
    assert "policy_entry = np.asarray(env.sim.get_state().flatten()).copy()" in source
    assert 'with h5py.File(out / "policy_entry_base.hdf5", "w")' in source
    assert '"raw_state_is_not_policy_entry": True' in source
    assert '"evaluator_num_steps_wait_for_serialized_base": 0' in source
    assert "derive every future Eb/Er/Ec state from this exact base" in source


def test_task1_audit_checks_actual_support_geometry_and_policy_pixels():
    source = SCRIPT.read_text()
    assert '"table",' in source
    assert '"main_table",' not in source
    assert "compiled_bodies = set(env.sim.model.body_names)" in source
    assert "task1 required compiled bodies missing" in source
    assert '"policy_entry_contacts": entry_contacts' in source
    assert '"hold_end_contacts": hold_contacts' in source
    assert "exact_compiled_group0_primitive_mesh_world_aabb" in source
    assert "geom_rbound" not in source
    assert '"group0_physical"' in source
    assert '"group1_visible"' in source
    assert "segmentation_visible_pixels" in source
    assert "HOLD_STEPS = 120" in source


def test_task1_audit_is_registered():
    source = RUNNER.read_text()
    assert '("l3a2", "task1_native_audit")' in source
    assert "audit_l3a2_task1_native.py" in source
