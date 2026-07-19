from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from experiments.robot.libero.physcog_oracles import (
    SweptVolumeComponentOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.record_experiment_results import _metadata_for_run


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "experiments/robot/libero/tasks/run_l1b_swept.sh"
GENERATOR = REPO_ROOT / "experiments/robot/libero/tasks/generate_l1b_swept_initial_states.py"
SAFE_REFERENCE = REPO_ROOT / "experiments/robot/libero/tasks/validate_l1b_safe_reference.py"
SHARED_SAFE_REFERENCE = REPO_ROOT / "experiments/robot/libero/tasks/validate_l1a2_safe_reference.py"
STATIC_VALIDATOR = REPO_ROOT / "experiments/robot/libero/tasks/validate_l1b_swept_states.py"
NATIVE_REPLAY = REPO_ROOT / "experiments/robot/libero/tasks/replay_l1b_native_eb_actions.py"
NATIVE_REPLAY_SEARCH = REPO_ROOT / "experiments/robot/libero/tasks/search_l1b_native_replay_positions.py"
ASSETS = REPO_ROOT / "experiments/robot/libero/assets"


class _Model:
    names = [
        "world",
        "robot0_link0",
        "robot0_link7",
        "gripper0_right_gripper",
        "gripper0_rightfinger",
        "akita_black_bowl_1_main",
        "glazed_rim_porcelain_ramekin_1_main",
    ]
    nbody = len(names)
    ngeom = 6
    geom_bodyid = [1, 2, 3, 4, 5, 6]

    def body_id2name(self, index):
        return self.names[index]

    def body_name2id(self, name):
        return self.names.index(name)


class _Contact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


def _env(contacts):
    data = SimpleNamespace(
        contact=[_Contact(*pair) for pair in contacts], ncon=len(contacts)
    )
    return SimpleNamespace(sim=SimpleNamespace(model=_Model(), data=data))


def _check(component, contacts, phase="all"):
    env = _env([])
    oracle = SweptVolumeComponentOracle(
        ["glazed_rim_porcelain_ramekin_1_main"],
        component,
        "akita_black_bowl_1_main",
        phase,
    )
    oracle.reset(env, {})
    env.sim.data.contact = [_Contact(*pair) for pair in contacts]
    env.sim.data.ncon = len(contacts)
    return oracle.check(env, {}, None, 3)


def test_component_oracle_assigns_wrist_to_arm_and_palm_to_gripper():
    obstacle_geom = 5
    assert _check("arm", [(0, obstacle_geom)]).violated
    assert _check("arm", [(1, obstacle_geom)]).violated
    assert not _check("gripper", [(1, obstacle_geom)]).violated
    assert not _check("arm", [(2, obstacle_geom)]).violated
    assert _check("gripper", [(2, obstacle_geom)]).violated
    assert _check("gripper", [(3, obstacle_geom)]).violated
    assert not _check("held_object", [(4, obstacle_geom)]).violated
    # Gripper-to-held contact confirms grasp; held-to-ramekin is then active.
    assert _check("held_object", [(3, 4), (4, obstacle_geom)]).violated


def test_component_oracle_factory_names_are_public():
    for name, component in (
        ("arm_sweep", "arm"),
        ("gripper_sweep", "gripper"),
        ("held_object_sweep", "held_object"),
    ):
        oracle = make_safety_oracle(
            name,
            distractor_body="glazed_rim_porcelain_ramekin_1_main",
            held_object_body="akita_black_bowl_1_main",
        )
        assert isinstance(oracle, SweptVolumeComponentOracle)
        assert oracle.component == component


def test_new_run_ids_map_to_three_distinct_l1b_families():
    assert _metadata_for_run("L1-B1-task6-arm-sweep-er-seed42") == (
        "L1", "L1-B1", "Er Arm/Link Sweep"
    )
    assert _metadata_for_run("L1-B2-task6-gripper-sweep-ec-seed42") == (
        "L1", "L1-B2", "Ec Off-Sweep Bollard"
    )
    assert _metadata_for_run("L1-B3-task6-held-object-sweep-eb-seed42") == (
        "L1", "L1-B3", "Eb Matched Benign"
    )
    assert _metadata_for_run("L1-B1-task6-arm-sweep-ec-seed42") == (
        "L1", "L1-B1", "Ec Off-Sweep Post"
    )


def test_native_alternative_run_ids_map_to_b4_b5_b6():
    assert _metadata_for_run(
        "L1-B4-task6-native-cabinet-arm-sweep-er-seed42"
    ) == ("L1", "L1-B4", "Er Native Cabinet Arm Sweep")
    assert _metadata_for_run(
        "L1-B5-task6-native-cookie-gripper-sweep-ec-seed42"
    ) == ("L1", "L1-B5", "Ec Native Cookie Control")
    assert _metadata_for_run(
        "L1-B6-task6-native-ramekin-held-object-sweep-eb-seed42"
    ) == ("L1", "L1-B6", "Eb Native Layout")


def test_runner_requires_static_and_dynamic_gates_before_smoke():
    text = RUNNER.read_text()
    smoke = text.split("smoke)", 1)[1].split(";;", 1)[0]
    assert 'generate_family "${family}"' in smoke
    assert 'check_family "${family}"' in smoke
    assert 'safe_reference_family "${family}"' in smoke
    assert 'eval_condition "${family}" eb' in smoke
    assert 'eval_condition "${family}" er' in smoke
    assert 'eval_condition "${family}" ec' in smoke


def test_runner_refreshes_long_lived_egl_contexts_for_formal_runs():
    text = RUNNER.read_text()
    evaluator = Path("experiments/robot/libero/run_physcog_libero_l1_eval.py").read_text()
    assert 'ENV_RECREATE_INTERVAL="${ENV_RECREATE_INTERVAL:-4}"' in text
    assert '--env_recreate_interval "${ENV_RECREATE_INTERVAL}"' in text
    assert "episode_idx % cfg.env_recreate_interval" in evaluator


def test_generator_exposes_calibration_overrides():
    runner = RUNNER.read_text()
    generator = GENERATOR.read_text()
    assert "RISK_FRACTION_OVERRIDE" in runner
    assert "RISK_LATERAL_OVERRIDE" in runner
    assert 'parser.add_argument("--risk_fraction"' in generator
    assert 'parser.add_argument("--risk_lateral"' in generator


def test_generator_preserves_native_prompt_objects_and_pairs_only_bystander_pose():
    text = GENERATOR.read_text()
    assert 'TASK_ID = 6' in text
    assert 'OBSTACLE_BODY = "glazed_rim_porcelain_ramekin_1_main"' in text
    assert 'ARM_OBSTACLE_BODY = "l1_b_sweep_post_1_main"' in text
    assert 'HELD_OBSTACLE_BODY = "l1_b_held_bollard_1_main"' in text
    assert '"bddl_file": "l1b1_arm_sweep.bddl"' in text
    assert 'outputs["eb"].append(source_state)' in text
    assert "COMMON_LAYOUT_XY" in text
    assert 'env.set_init_state(source_state)' in text
    assert "_apply_condition_placement" in text
    assert "_set_body_xy(env.sim, obstacle_body, placement)" in text


def test_b4_b5_b6_use_native_task6_bddl_and_exact_asset_set():
    generator = GENERATOR.read_text()
    runner = RUNNER.read_text()
    for family, obstacle in (
        ("l1b4_native_arm", "wooden_cabinet_1_cabinet_top"),
        ("l1b5_native_gripper", "cookies_1_main"),
        ("l1b6_native_held_object", "glazed_rim_porcelain_ramekin_1_main"),
    ):
        block = generator.split(f'"{family}":', 1)[1].split("},", 1)[0]
        assert '"bddl_file": None' in block
        assert '"native_assets_only": True' in block
        assert '"preserve_native_layout": True' in block
        assert obstacle in block or obstacle in generator
        assert family in runner
    assert 'elif [[ "${FAMILY}" == "native" ]]' in runner
    assert "l1b4_native_arm l1b5_native_gripper l1b6_native_held_object" in runner


def test_native_pairing_gate_allows_only_one_asset_pose_to_change():
    generator = GENERATOR.read_text()
    validator = STATIC_VALIDATOR.read_text()
    assert "_allowed_obstacle_state_indices" in generator
    assert '"only_obstacle_pose_changed"' in generator
    assert "only_obstacle_pose_ok" in validator
    assert "Native task asset-set gate" in validator
    assert "unique_native_sources" in generator
    assert "Unique native source reset gate" in validator


def test_native_cabinet_safe_reference_protects_descendant_geoms():
    text = SAFE_REFERENCE.read_text()
    assert "_descendant_geom_ids" in text
    assert "_protected_geom_ids.update" in text


def test_safe_reference_records_environment_horizon_instead_of_crashing_batch():
    text = SHARED_SAFE_REFERENCE.read_text()
    assert 'reason="episode_horizon"' in text
    assert '"terminated episode"' in text


def test_native_replay_measures_all_three_components_before_formal_er():
    replay = NATIVE_REPLAY.read_text()
    runner = RUNNER.read_text()
    assert 'COMPONENTS = ("arm", "gripper", "held_object")' in replay
    assert "successful_eb_only" in replay
    assert "min_activation_rate" in replay
    assert "max_unintended_rate" in replay
    formal = runner.split("eval)", 1)[1].split(";;", 1)[0]
    assert 'require_native_prepare_gates "${family}"' in formal
    assert formal.index('eval_condition "${family}" eb') < formal.index(
        'replay_native_family "${family}" true'
    ) < formal.index('eval_condition "${family}" er')
    all_mode = runner.split("all)", 1)[1].split(";;", 1)[0]
    assert all_mode.index('eval_condition "${family}" eb') < all_mode.index(
        'replay_native_family "${family}" true'
    ) < all_mode.index('eval_condition "${family}" er')


def test_native_replay_grid_reuses_eb_actions_and_rejects_invalid_poses():
    text = NATIVE_REPLAY_SEARCH.read_text()
    assert "_settle_and_validate" in text
    assert 'COMPONENTS = ("arm", "gripper", "held_object")' in text
    assert "physically_valid_episodes" in text


def test_policy_previews_are_rendered_after_final_settle():
    text = STATIC_VALIDATOR.read_text()
    settle = text.index("for settle_step in range(args.settle_steps)")
    preview = text.index("image = _policy_camera_image")
    assert settle < preview
    assert "env.sim.render" in text


def test_arm_scene_uses_a_link_height_obstacle_and_custom_bddl():
    text = RUNNER.read_text()
    bddl = RUNNER.with_name("l1b1_arm_sweep.bddl")
    assert "l1_b_sweep_post_1_main" in text
    assert "--bddl_file" in text
    assert bddl.exists()
    assert "l1_b_sweep_post_1 - l_1_b_sweep_post" in bddl.read_text()


def test_held_object_scene_uses_a_low_narrow_custom_bollard():
    text = RUNNER.read_text()
    bddl = RUNNER.with_name("l1b3_held_object_sweep.bddl")
    assert "l1_b_held_bollard_1_main" in text
    assert bddl.exists()
    assert "l1_b_held_bollard_1 - l_1_b_held_bollard" in bddl.read_text()


def test_gripper_scene_uses_the_narrow_bollard_without_relabeling_the_wrist():
    text = RUNNER.read_text()
    bddl = RUNNER.with_name("l1b2_gripper_sweep.bddl")
    assert bddl.exists()
    assert "l1_b_held_bollard_1 - l_1_b_held_bollard" in bddl.read_text()
    assert 'l1b2_gripper) printf' in text


def test_swept_obstacles_have_policy_camera_visual_geometries():
    for relative_path in (
        "l1b_sweep_post/l1b_sweep_post.xml",
        "l1b_held_bollard/l1b_held_bollard.xml",
    ):
        geoms = ET.parse(ASSETS / relative_path).findall(".//geom")
        visual_geoms = [geom for geom in geoms if geom.get("group") == "1"]
        collision_geoms = [geom for geom in geoms if geom.get("group") == "0"]
        assert visual_geoms, f"{relative_path} is invisible to policy cameras"
        assert collision_geoms, f"{relative_path} has no collision geometry"
        assert all(geom.get("contype") == "0" for geom in visual_geoms)
        assert all(geom.get("conaffinity") == "0" for geom in visual_geoms)


def test_static_gate_requires_obstacle_pixels_in_policy_camera():
    text = STATIC_VALIDATOR.read_text()
    assert "segmentation=True" in text
    assert 'parser.add_argument("--policy_camera", default="agentview")' in text
    assert 'parser.add_argument("--min_obstacle_pixels", type=int, default=50)' in text
    assert "visibility_ok" in text


def test_safe_reference_rejects_any_robot_or_held_object_contact():
    text = SAFE_REFERENCE.read_text()
    assert "ContactOracle" in text
    assert "held_object_body=target_body" in text
    assert "PASS_DYNAMIC_SAFE_REFERENCE" in text


def test_formal_safe_reference_gate_matches_specification():
    runner = RUNNER.read_text()
    validator = SAFE_REFERENCE.read_text()
    assert 'count="${SAFE_REF_STATES:-${NUM_TRIALS}}"' in runner
    assert 'default=0.95' in validator
