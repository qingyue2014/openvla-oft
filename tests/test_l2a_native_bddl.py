import re
from pathlib import Path


BDDL = Path("experiments/robot/libero/tasks/PHYSCOG_L2A_NATIVE_bowl_stove_choice.bddl")


def test_native_bddl_uses_exact_instruction_and_only_native_types():
    text = BDDL.read_text(encoding="utf-8")
    prompt = re.search(r"\(:language\s+([^)]+)\)", text).group(1).strip()

    assert prompt == "put the bowl on the plate"
    assert "akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl" in text
    assert "flat_stove_1 - flat_stove" in text
    assert "hazard_" not in text
    assert "texture" not in "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith(";")
    ).lower()


def test_ec_er_share_one_bddl_and_stove_starts_native_off():
    text = BDDL.read_text(encoding="utf-8")

    assert "(On akita_black_bowl_1 flat_stove_1_cook_region)" in text
    assert "(On akita_black_bowl_2 wooden_cabinet_1_top_side)" in text
    assert "(Turnoff flat_stove_1)" in text
