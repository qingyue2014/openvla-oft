from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/robot/libero/tasks/audit_l3a2_task33_native.py"
RUNNER = ROOT / "experiments/robot/libero/tasks/physcog_remote_agent.py"


def test_task33_audit_is_native_read_only_and_vla_free():
    source = SCRIPT.read_text()
    assert "TASK_ID = 33" in source
    assert 'TASK_PROMPT = "close the microwave"' in source
    assert "suite.get_task_init_states(TASK_ID)" in source
    assert "for _ in range(10)" in source
    assert "agentview_image" in source
    assert "camera_heights=256" in source
    assert '"scene_or_asset_modified": False' in source
    assert '"vla_run": False' in source
    assert "np.save" not in source
    assert "set_state_from_flattened(base)" in source


def test_task33_audit_is_registered():
    source = RUNNER.read_text()
    assert '("l3a2", "task33_native_audit")' in source
    assert "audit_l3a2_task33_native.py" in source
