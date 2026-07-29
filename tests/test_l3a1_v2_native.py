from pathlib import Path

from experiments.robot.libero.tasks.physcog_remote_agent import PHASES
from experiments.robot.libero.tasks.validate_l3a1_native_preflight import (
    BDDL_BASENAME,
    EXPECTED_FIXTURES,
    EXPECTED_OBJECTS,
    EXPECTED_PROMPT,
    EXPECTED_SHA256,
    parse_language,
    parse_typed_block,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "experiments/robot/libero/tasks/run_l3a1_v2_native.sh"


def test_native_preflight_contract_is_exact():
    assert BDDL_BASENAME.startswith("KITCHEN_SCENE4_")
    assert EXPECTED_PROMPT == (
        "put the black bowl in the bottom drawer of the cabinet and close it"
    )
    assert len(EXPECTED_SHA256) == 64
    assert EXPECTED_FIXTURES == (
        "kitchen_table:kitchen_table",
        "white_cabinet_1:white_cabinet",
        "wine_rack_1:wine_rack",
    )
    assert EXPECTED_OBJECTS == (
        "akita_black_bowl_1:akita_black_bowl",
        "wine_bottle_1:wine_bottle",
    )


def test_native_bddl_parser_extracts_prompt_and_inventory():
    text = """
    (:language put the black bowl in the bottom drawer of the cabinet and close it)
    (:fixtures
      kitchen_table - kitchen_table
      white_cabinet_1 - white_cabinet
      wine_rack_1 - wine_rack
    )
    (:objects
      akita_black_bowl_1 - akita_black_bowl
      wine_bottle_1 - wine_bottle
    )
    """
    assert parse_language(text) == EXPECTED_PROMPT
    assert parse_typed_block(text, "fixtures") == EXPECTED_FIXTURES
    assert parse_typed_block(text, "objects") == EXPECTED_OBJECTS


def test_v2_wrapper_never_selects_project_local_bddl():
    text = WRAPPER.read_text()
    assert '${LIBERO_ROOT}/libero/libero/bddl_files/libero_10/' in text
    assert "PHYSCOG_L3A1_bowl_drawer_bottle.bddl" not in text
    assert "PASS_L3A1_V2_NATIVE_ONLY_PREFLIGHT" in text


def test_remote_agent_registers_native_v2_calibration():
    for phase, mode in (
        ("v2_native_edge_sweep", "edge_sweep"),
        ("v2_native_edge_preview", "edge_preview"),
    ):
        spec = PHASES[("l3a1", phase)]
        assert spec.command[-2:] == (
            "experiments/robot/libero/tasks/run_l3a1_v2_native.sh",
            mode,
        )
        assert "experiments/logs/l3a1_v2_native_preflight.md" in spec.artifacts




def test_v2_geometry_path_has_no_custom_object_registration_imports():
    paths = [
        ROOT / "experiments/robot/libero/tasks/l3a1_native_geometry.py",
        ROOT / "experiments/robot/libero/tasks/sweep_l3a1_corner_geometry.py",
        ROOT / "experiments/robot/libero/tasks/sweep_l3a1_edge_geometry.py",
        ROOT / "experiments/robot/libero/tasks/export_l3a1_edge_preview.py",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in paths
    )
    assert "physcog_objects" not in combined
    assert "register_object" not in combined
    assert "generate_l3a1_drawer_bottle_initial_states" not in combined
