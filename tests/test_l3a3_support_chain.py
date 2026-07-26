from pathlib import Path
import xml.etree.ElementTree as ET

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


def test_bddl_preserves_native_prompt_goal_and_uses_pivot_chain_objects():
    text = (TASKS / "PHYSCOG_L3A3_support_chain.bddl").read_text()
    assert f"(:language {PROMPT})" in text
    assert "(On yellow_book_2 wooden_two_layer_shelf_1_top_side)" in text
    assert (
        "l_three_a_three_support_pad_1 - l_three_a_three_support_pad" in text
    )
    assert "l_three_a_three_top_block_1 - l_three_a_three_top_block" in text
    assert "yellow_book_2 - yellow_book" in text
    assert "black_book_1" not in text
    assert "yellow_book_1" not in text
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


def test_chain_activation_requires_complete_mechanical_path():
    text = (
        ROOT / "experiments/robot/libero/physcog_oracles.py"
    ).read_text()
    block = text[text.index("class SupportChainPreconditionOracle") :]
    assert "self.chain_loaded_at_activation = bool(s_a and a_b)" in block
    assert "bool(s_a or a_b)" not in block


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
    assert "PASS_L3A3_REVIEWED_STATE_BYTES" in text
    assert "missing or mismatched canonical hash-bound reviewed states" in text
    prepare_block = text[
        text.index("prepare_reviewed_states()") : text.index("\npreview()")
    ]
    assert "generate" not in prepare_block
    assert "reviewed_state_bytes_match" in prepare_block


def test_pivot_assets_have_separate_collidable_and_opaque_visual_geoms():
    assets = ROOT / "experiments/robot/libero/assets"
    for relative in (
        "l3a3_support_pad/l3a3_support_pad.xml",
        "l3a3_top_block/l3a3_top_block.xml",
    ):
        root = ET.parse(assets / relative).getroot()
        collision = [g for g in root.iter("geom") if g.get("group") == "0"]
        visual = [g for g in root.iter("geom") if g.get("group") == "1"]
        assert len(collision) == 1
        assert len(visual) == 1
        assert collision[0].get("contype", "1") != "0"
        assert collision[0].get("conaffinity", "1") != "0"
        assert visual[0].get("contype") == "0"
        assert visual[0].get("conaffinity") == "0"
        material_name = visual[0].get("material")
        materials = {
            material.get("name"): material
            for material in root.iter("material")
        }
        rgba = [float(value) for value in materials[material_name].get("rgba").split()]
        assert rgba[-1] == 1.0
    pad = ET.parse(
        assets / "l3a3_support_pad/l3a3_support_pad.xml"
    ).getroot()
    pad_collision = next(
        geom for geom in pad.iter("geom") if geom.get("name") == "pad_collision"
    )
    assert float(pad_collision.get("friction").split()[0]) <= 0.15


def test_pivot_custom_object_classes_are_registered():
    text = (ROOT / "experiments/robot/libero/physcog_objects.py").read_text()
    assert "@register_object\nclass LThreeAThreeSupportPad" in text
    assert 'obj_name="l3a3_support_pad"' in text
    assert "@register_object\nclass LThreeAThreeTopBlock" in text
    assert 'obj_name="l3a3_top_block"' in text
    assert "def assert_l3a3_pivot_objects_registered()" in text
    for script in (
        "generate_l3a3_support_chain_states.py",
        "export_l3a3_support_chain_evidence.py",
        "validate_l3a3_safe_reference.py",
        "validate_l3a3_action_sequence.py",
    ):
        script_text = (TASKS / script).read_text()
        assert "physcog_objects.assert_l3a3_pivot_objects_registered()" in script_text


def test_action_validators_do_not_edit_sim_state_after_er_reset():
    text = (TASKS / "validate_l3a3_action_sequence.py").read_text()
    reset_at = text.index("env.set_init_state(states[index])")
    replay_loop = text.index("for step, action in enumerate(actions)", reset_at)
    tail = text[reset_at:]
    assert "sim.data.qpos" not in tail
    assert "sim.data.qvel" not in tail
    assert "set_free_pose" not in tail
    assert "env.step(action.tolist())" in text[replay_loop:]
    assert "eb_replay requires --eb_states" in text
    assert "metadata.get(\"initial_state_sha256\")" in text
    assert "eb_not_safe_success" in text
    namespace = {}
    exec(
        compile(
            "\n".join(
                line
                for line in text.splitlines()
                if line.startswith("def _episode") or line.startswith("    match =")
                or line.startswith("    return int(")
            ),
            "<episode-parser>",
            "exec",
        ),
        {"re": __import__("re"), "os": __import__("os")},
        namespace,
    )
    assert namespace["_episode"]("eb_expert_ep004.npz") == 4


def test_safe_reference_provides_contact_verified_b_then_a_push_and_video():
    path = TASKS / "validate_l3a3_safe_reference.py"
    text = path.read_text()
    reset_at = text.index("obs = env.set_init_state(er_state)")
    tail = text[reset_at:]
    top_call = text.index("_push_unload(\n            io,\n            TOP_BODY")
    middle_call = text.index("_push_unload(\n                io,\n                MIDDLE_BODY")
    assert top_call < middle_call
    assert middle_call < text.index("_place_target_on_open_top(", middle_call)
    assert "native_S_suffix_only" in text
    assert '"initial_state_sha256": _state_hash(eb_state)' in text
    assert '"--goal_site", default="wooden_two_layer_shelf_1_top_side"' in text
    assert '"--max_waypoint_steps", type=int, default=220' in text
    assert '"--top_push_distance", type=float, default=0.16' in text
    assert '"--middle_push_distance", type=float, default=0.12' in text
    middle_push = text[middle_call : text.index(
        "if failure is None and not oracle.safe_precondition_inserted", middle_call
    )]
    assert "np.array([0.0, -1.0, 0.0])" in middle_push
    push_function = text[text.index("def _push_unload(") : text.index(
        "def _table_stable_unloaded("
    )]
    assert "contact_seen = contact_seen or _gripper_contacts_body" in push_function
    assert "push_contact_not_observed" in text
    assert '"task_push_diagnostic": task_diagnostic' in text
    assert "io.advance(" in text
    assert "env.step" in (
        TASKS / "validate_l3a1_safe_reference.py"
    ).read_text()  # EpisodeIO implementation used by L3-A3
    assert "sim.data.qpos" not in tail
    assert "sim.data.qvel" not in tail
    assert "set_free_pose" not in tail
    assert "_save_video(video_path" in text
    assert "TrajectoryRecorder" in text
    assert '"direct_qpos_edits_after_restore": False' in text
    assert "ec_trajectory_dir" not in text
    runner = (TASKS / "run_l3a3_support_chain.sh").read_text()
    assert "safe_reference_pilot()" in runner
    assert "run_safe_reference_validation 1 1" in runner


def test_generator_requires_second_link_collision_ablation():
    text = (TASKS / "generate_l3a3_support_chain_states.py").read_text()
    assert "_first_link_ablation_gate" in text
    assert "_second_link_ablation_gate" in text
    assert "geom_contype[geom_ids] = 0" in text
    assert "geom_conaffinity[geom_ids] = 0" in text
    assert "hold_ok and removal_ok and first_ablation_ok and ablation_ok" in text
    assert "min_family_acceptance_rate" in text
    assert "_collision_z_bounds" in text


def test_preview_uses_exact_policy_transform_and_manual_hash_gate():
    text = (TASKS / "export_l3a3_support_chain_evidence.py").read_text()
    assert "image[::-1, ::-1]" in text
    assert "PENDING_MANUAL_POLICY_VIEW_REVIEW" in text
    assert "PASS_L3A3_POLICY_VIEW_REVIEWED" in text
    assert 'review.get("evidence_sha256")' in text
    assert "bind_existing_review" in text
    assert 'review.get("reviewed_png_count", 0)' in text
    runner = (TASKS / "run_l3a3_support_chain.sh").read_text()
    assert 'PREVIEW_EPISODES="${PREVIEW_EPISODES:-5}"' in runner
    assert '--episodes "${PREVIEW_EPISODES}"' in runner
