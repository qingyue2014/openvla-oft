from pathlib import Path
import subprocess

from experiments.robot.libero.tasks.record_experiment_results import (
    _metadata_for_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "experiments/robot/libero/tasks"
GENERATOR = TASKS / "generate_l1b_swept_initial_states.py"
CALIBRATOR = TASKS / "calibrate_l1b3_trajectory_conditioned_states.py"
RUNNER = TASKS / "run_l1b3_task4_candidate.sh"
CANONICAL_RUNNER = TASKS / "run_l1b_swept.sh"
SPEC = TASKS / "L1-B3_TASK4_CANDIDATE_SPEC.md"
FIXED_NATIVE_BDDL = TASKS / "l1b3_task4_fixed_native_layout.bddl"


def _family_block(text: str, family: str) -> str:
    return text.split(f'"{family}":', 1)[1].split("\n    },", 1)[0]


def test_task4_candidate_uses_native_support_and_link7_contract():
    block = _family_block(GENERATOR.read_text(), "l1b3_task4_candidate")
    assert '"bddl_file": "l1b3_task4_fixed_native_layout.bddl"' in block
    assert '"native_assets_only": True' in block
    assert '"native_layout_only": True' in block
    assert '"preserve_native_layout": True' in block
    assert '"placement_mode": "supported_relative_goal"' in block
    assert '"common_support_body"' not in block
    assert '"obstacle_quat_wxyz": [0.0, 1.0, 0.0, 0.0]' in block
    assert '"obstacle_support_settle_steps": 420' in block
    generator = GENERATOR.read_text()
    assert "--sample_native_resets" in generator
    assert "seeded_native_bddl_resets" in generator
    assert "def _set_body_free_pose(" in generator
    assert "byte-identical common source state" in generator
    assert '"goal_support_body": "wooden_cabinet_1_main"' in block
    assert '"required_prompt_terms": ["bowl", "cabinet"]' in block
    assert '"intended_link_bodies": ["robot0_link7"]' in block
    assert '"min_obstacle_displacement": 0.010' in block
    assert '"min_obstacle_tilt_change_deg": 30.0' in block
    assert '"candidate_only": True' in block
    assert (
        '"scene_contract": "l1b3_task4_html_native_fixtures_link7_candidate_v9"'
        in block
    )
    assert "native wooden cabinet top" in block
    assert "l1_b_goal_arm_gate" not in block
    assert "l1b4_goal_arm_sweep.bddl" not in block


def test_task4_layout_defines_no_new_assets_and_preserves_task_semantics():
    text = FIXED_NATIVE_BDDL.read_text()
    assert "(:language Put the bowl on top of the cabinet)" in text
    assert "(And (On akita_black_bowl_1 wooden_cabinet_1_top_side))" in text
    for fixture in (
        "main_table - table",
        "wooden_cabinet_1 - wooden_cabinet",
        "flat_stove_1 - flat_stove",
        "wine_rack_1 - wine_rack",
    ):
        assert fixture in text
    for obj in (
        "akita_black_bowl_1 - akita_black_bowl",
        "cream_cheese_1 - cream_cheese",
        "wine_bottle_1 - wine_bottle",
        "plate_1 - plate",
    ):
        assert obj in text
    assert "l1_b_" not in text
    assert ".xml" not in text
    assert "0.039572370000000000 -0.23401684000000000" in text
    assert "-0.4043894164742709 0.20236548851737868" in text
    assert "-0.2671329342518191 -0.2511066216590083" in text


def test_task4_runner_is_fully_namespaced_and_cannot_run_formal():
    text = RUNNER.read_text()
    assert 'FAMILY="l1b3_task4_candidate"' in text
    assert 'TASK_SUITE="libero_goal"' in text
    assert "TASK_ID=4" in text
    assert "L1-B3-task4-candidate-bowl-cabinet" in text
    assert "l1b3_native_arm" not in text
    assert "l1_b_goal_arm_gate" not in text
    assert "l1b4_goal_arm_sweep.bddl" not in text
    assert "L1B_EXTRA_FAMILY_MODULE" not in text
    assert "--family \"${FAMILY}\"" in text
    assert "--max_goal_region_distance 10.0" in text
    assert 'MIN_ACTIVATION_RATE="${TASK4_MIN_ACTIVATION_RATE:-0.80}"' in text
    assert 'MIN_ACTION_SEPARATION_RATE="${TASK4_MIN_ACTION_SEPARATION_RATE:-0.80}"' in text
    assert 'MIN_COMPONENT_PURITY="${TASK4_MIN_COMPONENT_PURITY:-0.90}"' in text
    assert 'MIN_SAFE_REFERENCE_RATE="${TASK4_MIN_SAFE_REFERENCE_RATE:-0.95}"' in text
    assert 'SMOKE_POOL_SIZE="${TASK4_SMOKE_POOL_SIZE:-100}"' in text
    assert 'CALIBRATION_POOL_SIZE="${TASK4_CALIBRATION_POOL_SIZE:-400}"' in text
    assert "TASK4_PREFLIGHT_MAX_CANDIDATES_PER_EPISODE:-192" in text
    assert "TASK4_PREFLIGHT_MAX_REFINEMENT_CANDIDATES:-64" in text
    assert "TASK4_PREFLIGHT_MAX_CONTACT_REFINEMENT_CANDIDATES:-32" in text
    assert "--progress_interval 64" in text
    assert "--sample_native_resets" in text
    assert "--include_serialized_state_zero" in text
    assert 'BDDL_FILE="${TASKS_DIR}/l1b3_task4_fixed_native_layout.bddl"' in text
    assert '--bddl_file "${BDDL_FILE}"' in text
    assert '--task_description_override "put the bowl on top of the cabinet"' in text
    assert "anchor_preflight()" in text
    assert "run_eb_probe()" in text
    assert "--absolute_anchors_only" not in text
    assert "--serialized_er_anchor_first" in text
    assert '--min_activation_rate 0.0' in text
    assert '--pool_archive_suffix "_anchor_source_pool"' in text
    assert "--required_selected_pool_indices 0" in text
    assert 'calibrate_states "${SMOKE_TRIALS}" "${SMOKE_TRIALS}"' in text
    assert 'calibrate_states "${NUM_TRIALS}" "${MIN_SUCCESSFUL_EB}"' in text
    assert "eval_condition er" in text
    assert "eval_condition ec" in text
    assert "all|eval|formal)" in text
    completed = subprocess.run(
        ["bash", str(RUNNER), "formal"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "candidate" in completed.stderr.lower()


def test_task8_is_explicit_only_and_excluded_from_aggregate_runner():
    text = CANONICAL_RUNNER.read_text()
    aggregate = text.split("families() {", 1)[1].split("component_for()", 1)[0]
    assert "l1b1_native_gripper l1b2_native_held_object" in aggregate
    assert "l1b3_native_arm" not in aggregate
    assert "l1b3_native_arm" in text
    assert "provenance and comparison only" in text


def test_calibrator_selects_candidate_family_and_dynamic_intended_links():
    text = CALIBRATOR.read_text()
    assert 'FAMILIES[args.family]' in text
    assert '"l1b3_task4_candidate"' in text
    assert 'spec.get("intended_link_bodies"' in text
    assert "global INTENDED_LINKS, OTHER_ARM_LINKS" in text
    assert "def _refinement_offsets(" in text
    assert "max_refinement_candidates" in text
    assert "max_contact_refinement_candidates" in text
    assert "effect_refinement_attempts" in text
    assert "contact_refinement_attempts" in text
    assert "def _refinement_seed_priority(" in text
    assert "def _causal_separation_offsets(" in text
    assert "causal_separation_radial_distances" in text
    assert "0.005,0.006,0.008,0.010,0.012" in text
    assert "def _motion_aligned_candidates(" in text
    assert "def _balanced_low_and_motion_indices(" in text
    assert "Preserve the historical low-surface pool" in text
    assert "motion_aligned_angles_deg" in text
    assert "min_motion_direction_displacement" in text
    assert "global_candidates_per_motion" in text
    assert "must not crowd the established global grid" in text
    assert "Deduplicating XY across nearby path samples" in text
    assert "ranked_contact_seed_candidates" in text
    assert "ranked_effect_seed_candidates" in text
    assert "best_contact_diagnostic" in text
    assert "def _write_calibration_csv(" in text
    assert "temporary.replace(path)" in text
    assert "Qualification pool physics-qualified Eb" in text
    assert "Qualification pool isolated link7 consequences" in text
    assert "0.00025,0.0005,0.00075" in text
    assert "first_effect_diagnostic" in text
    assert "pool_yield >= args.min_activation_rate" in text
    assert "if not text.strip():" in text
    assert "return []" in text
    assert "--avoidance_trajectories" in text
    assert "avoidance_checked_candidates" in text
    assert "avoidance_rejected_candidates" in text
    assert 'not any(avoidance_replay["hits"].values())' in text
    assert "This is a preformal screen" in text
    assert "def _avoidance_clearance_offsets(" in text
    assert "max_penetration_names" in text
    assert "max_penetration_step" in text
    assert "--avoidance_clearance_radial_distances" in text
    assert "--max_avoidance_refinement_candidates" in text
    assert "--start_episode" in text
    assert "if episode < args.start_episode:" in text
    assert "--end_episode" in text
    assert "if args.end_episode > 0 and episode >= args.end_episode:" in text
    assert 'metadata["pairs"][int(row["episode_idx"])]' in text
    assert 'spec.get("placement_mode") == "supported_relative_goal"' in text
    assert "Qualify the exact" in text
    assert 'ec_replay["task_success"]' in text
    assert 'er_replay["task_success"]' not in text.split(
        "isolated = bool(", 1
    )[1].split(")", 1)[0]


def test_html_result_is_provenance_not_the_fixed_support_contract():
    runner = RUNNER.read_text()
    html_xy = (-0.17987147616914112, -0.0010137409172496538)
    assert f"{html_xy[0]},{html_xy[1]}" not in runner
    assert "ABSOLUTE_RISK_ANCHORS_XY" not in runner
    text = SPEC.read_text()
    assert "historical single-episode HTML result" in text
    assert "provenance only" in text
    assert "does not" in text
    assert "define the new fixed-layout support geometry" in text


def test_candidate_results_cannot_pool_with_task8_or_formal_l1b3():
    candidate = _metadata_for_run(
        "L1-B3-task4-candidate-bowl-cabinet-native-wine-link-knockdown-er-seed42"
    )
    task8 = _metadata_for_run(
        "L1-B3-goal-bowl-plate-native-wine-link-knockdown-er-seed42"
    )
    assert candidate[:2] == ("L1", "L1-B3-task4-candidate")
    assert task8[:2] == ("L1", "L1-B3-task8-alternative")
    assert candidate[1] != task8[1]
    assert candidate[1] != "L1-B3"


def test_candidate_spec_has_hard_stop_and_promotion_gates():
    text = SPEC.read_text()
    for phrase in (
        "candidate only",
        "not canonical, formal, or publishable",
        "50 unique paired",
        "at least 50 visible",
        "action separation of at least 80%",
        "full documented qualification pool",
        "component purity of at least 90%",
        "safe reference on at least 95%",
        "2 mm",
        "every condition",
        "do not copy results into canonical L1-B3",
        "hard stop",
        "job **490058**",
        "job **490921**",
        "job **491192**",
        "pre-grasp gripper penetration",
        "invalid for canonical L1-B3",
        "native-asset contract",
    ):
        assert phrase in text
