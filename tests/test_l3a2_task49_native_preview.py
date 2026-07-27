import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "experiments/robot/libero/tasks/preview_l3a2_task49_native.py"
)


def test_task49_native_preview_is_exact_and_read_only():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"TASK_ID", "TASK_PROMPT", "BDDL_SHA256"}
    }
    assert constants["TASK_ID"] == 49
    assert constants["TASK_PROMPT"] == (
        "pick up the tomato sauce and put it in the basket"
    )
    assert constants["BDDL_SHA256"] == (
        "cce015229a021baf1124562dd5efbc5bc65195926ecce34254c9da5728690816"
    )
    assert "scene_or_asset_modified" in source
    assert "set_state_from_flattened" not in source
    assert "model.body_pos" not in source
    assert "sim.data.qpos[" not in source
    assert "for _ in range(10)" in source
