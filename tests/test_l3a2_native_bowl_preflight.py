import ast
import hashlib
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks import l3a2_native_bowl_logic as logic


ROOT = Path(__file__).resolve().parents[1]


def test_native_task_binding_is_task23_and_exact_prompt_goal():
    source = (
        ROOT
        / "experiments/robot/libero/tasks/preflight_l3a2_native_bowl.py"
    ).read_text()
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {
            "TASK_ID", "TASK_PROMPT", "TASK_GOAL_CANONICAL",
            "TASK_GOAL_SHA256",
        }
    }
    assert constants["TASK_ID"] == 23
    assert constants["TASK_PROMPT"] == (
        "close the bottom drawer of the cabinet and open the top drawer"
    )
    assert hashlib.sha256(
        constants["TASK_GOAL_CANONICAL"].encode("utf-8")
    ).hexdigest() == constants["TASK_GOAL_SHA256"]


def test_native_preflight_has_no_custom_asset_or_bddl_writer():
    source = (
        ROOT
        / "experiments/robot/libero/tasks/preflight_l3a2_native_bowl.py"
    ).read_text()
    assert "cascade_panel" not in source
    assert "PHYSCOG_L3A2" not in source
    assert "write_bytes" not in source
    assert "write_text" in source  # reports only
    assert "model.body_pos[root_id]" in source
    assert '"condition_specific": False' in source
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "experiments.robot.libero.physcog_objects" not in imports


def test_pairing_mask_allows_only_native_a_and_b_free_joint_slices():
    eb = np.arange(80, dtype=np.float64)
    er = eb.copy()
    ec = eb.copy()
    a_slice = (10, 45)
    b_slice = (20, 55)
    for state in (er, ec):
        state[10:17] += 1
        state[45:51] += 1
        state[20:27] += 2
        state[55:61] += 2
    assert logic.outside_ab_exact((eb, er, ec), a_slice, b_slice)
    er[70] += 1
    assert not logic.outside_ab_exact((eb, er, ec), a_slice, b_slice)


def test_candidate_sweep_is_bounded_and_has_neighbor_spacing():
    response = {
        "support_release_step": 2,
        "timeline": [
            {
                "step": step,
                "component_contact": step < 2,
                "link_displacement_m": max(0.0, (step - 2) * 0.004),
                "link_xyz_m": [0.08 + step * 0.002, 0.02 + step * 0.003, 0.4],
            }
            for step in range(12)
        ],
    }
    candidates = logic.trajectory_candidates(response, 40)
    assert 1 < len(candidates) <= 40
    assert any(
        0 < np.hypot(x1 - x2, y1 - y2) <= 0.0101
        for x1, y1 in candidates
        for x2, y2 in candidates
    )
