import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments/robot/libero/tasks/audit_l3a4_spatial_task1_bowl2_feasibility.py"
)


def test_exact_native_roles_and_no_vla_or_raw_step():
    text = SCRIPT.read_text()
    assert '"A": "akita_black_bowl_2"' in text
    assert '"landmark": "glazed_rim_porcelain_ramekin_1"' in text
    assert "EXPECTED_BDDL_SHA256" in text
    assert "env.sim.step()" not in text
    assert '"vla_run": False' in text


def test_scan_is_fixed_and_bounded_below_96():
    tree = ast.parse(SCRIPT.read_text())
    values = {}
    wanted = {
        "DIRECTIONS", "TILTS_DEG", "SA_OVERLAPS_M", "B_GAPS_M"
    }
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            if node.targets[0].id == "DIRECTIONS":
                values[node.targets[0].id] = len(node.value.keys)
            else:
                values[node.targets[0].id] = len(
                    ast.literal_eval(node.value)
                )
    count = (
        values["DIRECTIONS"]
        * values["TILTS_DEG"]
        * values["SA_OVERLAPS_M"]
        * values["B_GAPS_M"]
    )
    assert count == 72
    assert count <= 96


def test_static_gate_forbids_ab_sb_and_robot_bypass():
    text = SCRIPT.read_text()
    for key in ("S_A", "A_B", "S_B", "robot_A", "robot_B"):
        assert key in text
    assert "signed_SA_directional_clearance_m" in text
    assert "signed_SB_directional_clearance_m" in text
