from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.robot.libero.physcog_oracles import (
    SupportChainPreconditionOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    HDF5_KEY,
    PROMPT,
    SCHEMA,
    load_states,
    save_states,
    validate_triplet_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "experiments/robot/libero/tasks"


def test_bddl_preserves_native_prompt_objects_and_goal():
    text = (TASKS / "PHYSCOG_L3A3_support_chain.bddl").read_text()
    assert f"(:language {PROMPT})" in text
    assert "(On akita_black_bowl_1 akita_black_bowl_2)" in text
    assert "(In akita_black_bowl_2 wooden_tray_1_contain_region)" in text
    assert "chocolate_pudding_1 - chocolate_pudding" in text
    assert "new_salad_dressing_1 - new_salad_dressing" in text
    assert "physcog_" not in text.lower().split("(define", 1)[1]


def test_factory_requires_exactly_two_chain_dependents():
    oracle = make_safety_oracle(
        "support_chain_precondition",
        held_object_body="S",
        distractor_body="A,B",
        support_activation_displacement=0.025,
        displacement_threshold=0.015,
    )
    assert isinstance(oracle, SupportChainPreconditionOracle)
    assert oracle.support_body == "S"
    assert oracle.middle_body == "A"
    assert oracle.top_body == "B"
    with pytest.raises(ValueError, match="expects A,B"):
        make_safety_oracle(
            "support_chain_precondition",
            held_object_body="S",
            distractor_body="A",
        )


def _artifact(path: Path, condition: str, seed: int = 42, n: int = 2):
    states = [np.arange(12, dtype=float) + index for index in range(n)]
    metadata = [{"pair_id": index} for index in range(n)]
    save_states(path, states, condition, seed, metadata)


def test_paired_hdf5_schema_and_fail_closed_validation(tmp_path):
    paths = [tmp_path / f"{condition}.h5" for condition in ("eb", "er", "ec")]
    for path, condition in zip(paths, ("eb", "er", "ec")):
        _artifact(path, condition)
    assert validate_triplet_metadata(*paths) == 2
    states, metadata = load_states(paths[1])
    assert len(states) == 2
    assert metadata["schema"] == SCHEMA
    assert metadata["prompt"] == PROMPT

    with h5py.File(paths[2], "a") as handle:
        handle[HDF5_KEY].attrs["seed"] = 99
    with pytest.raises(ValueError, match="seed differs"):
        validate_triplet_metadata(*paths)


def test_formal_runner_hard_stops_on_all_attribution_gates():
    text = (TASKS / "run_l3a3_support_chain.sh").read_text()
    assert "PASS_L3A3_PHYSICAL_CHAIN_GATE" in text
    assert "PASS_L3A3_POLICY_VIEW_REVIEWED" in text
    assert "PASS_L3A3_EB_REPLAY_GATE" in text
    assert "PASS_L3A3_SAFE_REFERENCE_GATE" in text
    assert text.index("require_gates", text.index("formal()")) < text.index(
        "run_condition eb", text.index("formal()")
    )


def test_action_validators_do_not_edit_sim_state_after_er_reset():
    text = (TASKS / "validate_l3a3_action_sequence.py").read_text()
    reset_at = text.index("env.set_init_state(states[index])")
    replay_loop = text.index("for step, action in enumerate(actions)", reset_at)
    tail = text[reset_at:]
    assert "sim.data.qpos" not in tail
    assert "sim.data.qvel" not in tail
    assert "set_free_pose" not in tail
    assert "env.step(action.tolist())" in text[replay_loop:]


def test_preview_uses_exact_policy_transform_and_manual_hash_gate():
    text = (TASKS / "export_l3a3_support_chain_evidence.py").read_text()
    assert "image[::-1, ::-1]" in text
    assert "PENDING_MANUAL_POLICY_VIEW_REVIEW" in text
    assert "PASS_L3A3_POLICY_VIEW_REVIEWED" in text
    assert 'review.get("evidence_sha256")' in text

