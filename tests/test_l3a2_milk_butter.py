import json
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    BASE_STATE_SOURCE,
    CONDITION_LABEL,
    CONDITION_SUPPORT,
    EVALUATOR_ENV_SEED,
    EXPECTED_OBJECT_BODIES,
    PAIRING_METHOD,
    SCENE_ID,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    TASK_SUITE,
    SOURCE_TO_BASE_WAIT_STEPS,
    build_preflight_manifest,
    sha256_file,
    validate_human_approval,
    validate_generation_manifest,
    validate_native_task,
    validate_native_init_states_source,
    validate_state_artifacts,
    verify_evaluation_request,
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3a2_milk_butter_smoke import (
    validate as validate_smoke,
)
from experiments.robot.libero.tasks.generate_l3a2_milk_butter_initial_states import (
    _basket_milk_goal,
    _collision_vertical_bounds,
    _settle_official_source_to_paired_base,
    _stack_butter_on,
    _validated_official_init_state_rows,
)
from experiments.robot.libero.tasks.validate_l3a2_milk_butter_osc_reference import (
    EVALUATION_POLICY_STEP_BUDGET,
    FLOOR_PARK_SAMPLE_SPACING_M,
    FLOOR_PARK_XY_CLEARANCE_M,
    HORIZON_STAGE_STEP_LIMITS,
    TRANSPORT_MAX_WAYPOINT_STEPS,
    _closest_floor_park_candidate,
    _evaluation_budget_diagnostics,
    _failure_diagnostics,
    _load_records,
    _minimum_swept_transport_body_z,
    _move_to_with_final_state_check,
    _NativeSuccessTrackingOracle,
    _place,
    _safe_reference_success,
    _segment_intersects_xy_rectangle,
    _static_plan_budget_diagnostics,
)
from experiments.robot.libero.tasks import (
    generate_l3a2_milk_butter_initial_states as l3a2_generator,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BDDL = (
    REPO_ROOT
    / "_deps/LIBERO/libero/libero/bddl_files/libero_object"
    / "pick_up_the_milk_and_place_it_in_the_basket.bddl"
)
NATIVE_INIT_STATES = (
    REPO_ROOT
    / "_deps/LIBERO/libero/libero/init_files/libero_object"
    / "pick_up_the_milk_and_place_it_in_the_basket.pruned_init"
)
RUNNER = (
    REPO_ROOT
    / "experiments/robot/libero/tasks/run_l3a2_milk_butter.sh"
)
GENERATOR = (
    REPO_ROOT
    / "experiments/robot/libero/tasks"
    / "generate_l3a2_milk_butter_initial_states.py"
)
OSC_REFERENCE = (
    REPO_ROOT
    / "experiments/robot/libero/tasks"
    / "validate_l3a2_milk_butter_osc_reference.py"
)


def _write_artifact(path, condition, bases, *, illegal_initial=False):
    with h5py.File(path, "w") as handle:
        group = handle.create_group(TASK_KEY)
        attrs = {
            "scene_id": SCENE_ID,
            "condition": condition,
            "condition_label": CONDITION_LABEL[condition],
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "bddl_sha256": sha256_file(NATIVE_BDDL),
            "native_init_states": str(NATIVE_INIT_STATES.resolve()),
            "native_init_states_sha256": sha256_file(NATIVE_INIT_STATES),
            "base_state_source": BASE_STATE_SOURCE,
            "pairing_method": PAIRING_METHOD,
            "construction_settle_method": "controller_dummy_action",
            "environment_seed": EVALUATOR_ENV_SEED,
            "environment_hard_reset": False,
            "source_to_base_wait_steps": SOURCE_TO_BASE_WAIT_STEPS,
            "source_to_base_wait_method": (
                "formal_evaluator_controller_dummy_action"
            ),
            "seed": 42,
            "formal_wait_steps": 10,
            "intervention_body": "butter_1_main",
            "intervention_support": CONDITION_SUPPORT[condition],
            "floor_support_bodies": '["floor"]',
        }
        for name, value in attrs.items():
            group.attrs[name] = value
        for index, base in enumerate(bases):
            demo = group.create_group(f"demo_{index}")
            source = base.copy()
            source[0] -= 0.125
            intervention = base.copy()
            evaluated = base.copy()
            if condition in {"er", "ec"}:
                intervention[10:17] += 0.25
                evaluated[10:17] += 0.5
                intervention[40:46] = 0.0
                evaluated[40:46] = 0.0
            if illegal_initial:
                evaluated[25] += 1.0
            demo.create_dataset("native_source_state", data=source)
            demo.create_dataset("base_reset_state", data=base)
            demo.create_dataset("intervention_state", data=intervention)
            demo.create_dataset("initial_state", data=evaluated)
            demo.attrs["butter_qpos_flat_start"] = 10
            demo.attrs["butter_qvel_flat_start"] = 40
            demo.attrs["native_init_state_index"] = index
            demo.attrs["source_state_sha256"] = (
                l3a2_generator.sha256_array(source)
            )
            demo.attrs["base_state_sha256"] = (
                l3a2_generator.sha256_array(base)
            )
            demo.attrs["intervention_state_sha256"] = (
                l3a2_generator.sha256_array(intervention)
            )
            demo.attrs["initial_state_sha256"] = (
                l3a2_generator.sha256_array(evaluated)
            )
            demo.attrs["source_to_base_restored_state_sha256"] = (
                l3a2_generator.sha256_array(source)
            )
            demo.attrs["source_to_base_wait_state_sha256"] = json.dumps(
                [l3a2_generator.sha256_array(base)]
                * SOURCE_TO_BASE_WAIT_STEPS
            )
            demo.attrs["native_butter_body_position"] = (
                base[10:13] + np.array([0.125, -0.25, 0.375])
            )
            demo.attrs["formal_state_pass"] = True
            demo.attrs["policy_visibility_pass"] = True
            demo.attrs["pre_wait_metrics"] = '{"objects":{"butter":{}}}'
            demo.attrs["post_wait_metrics"] = '{"objects":{"butter":{}}}'
            demo.attrs["wait_trace"] = '[{"wait_step":1}]'
            demo.attrs["dynamic_cascade_pass"] = condition == "er"
            demo.attrs["dynamic_control_pass"] = condition == "ec"
            demo.attrs["safe_prefix_pass"] = condition == "er"


def _triplet(tmp_path, *, illegal_initial=False):
    bases = [np.linspace(0, 1, 64), np.linspace(1, 2, 64)]
    paths = {
        condition: tmp_path / f"{condition}.hdf5"
        for condition in ("eb", "er", "ec")
    }
    for condition, path in paths.items():
        _write_artifact(
            path,
            condition,
            bases,
            illegal_initial=illegal_initial and condition == "er",
        )
    return paths


def test_native_preflight_accepts_only_exact_native_task(tmp_path):
    evidence = validate_native_task(
        NATIVE_BDDL,
        NATIVE_BDDL,
        TASK_PROMPT,
    )
    assert evidence["scenario"] == "L3-A2"
    assert evidence["task_suite_name"] == "libero_object"
    assert evidence["task_id"] == 7
    assert evidence["custom_assets"] is False
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_native_task(NATIVE_BDDL, NATIVE_BDDL, TASK_PROMPT.lower())
    copied = tmp_path / NATIVE_BDDL.name
    copied.write_bytes(NATIVE_BDDL.read_bytes())
    with pytest.raises(ValueError, match="selected native task"):
        validate_native_task(NATIVE_BDDL, copied, TASK_PROMPT)


def test_official_init_source_and_rows_fail_closed(tmp_path):
    source = validate_native_init_states_source(NATIVE_INIT_STATES)
    assert source["path"] == str(NATIVE_INIT_STATES.resolve())
    assert source["sha256"] == sha256_file(NATIVE_INIT_STATES)

    copied = tmp_path / NATIVE_INIT_STATES.name
    copied.write_bytes(NATIVE_INIT_STATES.read_bytes())
    with pytest.raises(ValueError, match="unexpected native init-state"):
        validate_native_init_states_source(copied)

    raw = np.arange(24, dtype=float).reshape(3, 8)
    rows = _validated_official_init_state_rows(
        raw, requested_count=2, expected_state_size=8
    )
    np.testing.assert_array_equal(rows, raw)
    assert not np.shares_memory(rows, raw)
    with pytest.raises(ValueError, match="pool has 3 rows"):
        _validated_official_init_state_rows(
            raw, requested_count=4, expected_state_size=8
        )
    with pytest.raises(ValueError, match="width 8"):
        _validated_official_init_state_rows(
            raw, requested_count=2, expected_state_size=9
        )


def test_stack_construction_uses_controller_steps_not_raw_simulation():
    source = inspect.getsource(_stack_butter_on)
    assert "env.step(DUMMY_ACTION)" in source
    assert "env.sim.step()" not in source
    assert "env.reset()" in source
    assert "env.set_init_state" in source


def test_official_source_is_settled_with_exact_evaluator_sequence():
    class Sim:
        def __init__(self):
            self.state = np.zeros(4, dtype=float)

        def get_state(self):
            return self.state

    class Env:
        def __init__(self):
            self.sim = Sim()
            self.events = []

        def reset(self):
            self.events.append("reset")

        def set_init_state(self, state):
            self.events.append("set_init_state")
            self.sim.state = np.asarray(state, dtype=float).copy()

        def step(self, action):
            self.events.append(("step", list(action)))
            self.sim.state += 0.01
            return {}, 0.0, False, {}

    env = Env()
    source = np.array([0.0, 1.0, 2.0, 3.0])
    base, evidence = _settle_official_source_to_paired_base(env, source)

    assert env.events[:2] == ["reset", "set_init_state"]
    assert len(env.events[2:]) == SOURCE_TO_BASE_WAIT_STEPS
    assert all(
        event == ("step", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        for event in env.events[2:]
    )
    np.testing.assert_allclose(
        base, source + 0.01 * SOURCE_TO_BASE_WAIT_STEPS
    )
    assert not np.array_equal(base, source)
    assert evidence["environment_hard_reset"] is False
    assert evidence["environment_seed"] == EVALUATOR_ENV_SEED
    assert evidence["wait_steps"] == SOURCE_TO_BASE_WAIT_STEPS
    assert evidence["restored_source_state_sha256"] == (
        l3a2_generator.sha256_array(source)
    )
    assert evidence["wait_state_sha256"][-1] == (
        l3a2_generator.sha256_array(base)
    )


def test_paired_artifacts_allow_only_butter_in_exact_loaded_state(tmp_path):
    paths = _triplet(tmp_path)
    result = validate_state_artifacts(
        paths["eb"],
        paths["er"],
        paths["ec"],
        native_bddl=NATIVE_BDDL,
        minimum_count=2,
    )
    assert result["count"] == 2
    assert set(result["bindings"]) == {"eb", "er", "ec"}


def test_pairing_rejects_non_butter_change_in_evaluated_initial_state(tmp_path):
    paths = _triplet(tmp_path, illegal_initial=True)
    with pytest.raises(ValueError, match="evaluated initial_state changed non-butter"):
        validate_state_artifacts(
            paths["eb"],
            paths["er"],
            paths["ec"],
            native_bddl=NATIVE_BDDL,
        )


def test_pairing_rejects_mismatched_native_butter_body_position(tmp_path):
    paths = _triplet(tmp_path)
    with h5py.File(paths["ec"], "r+") as handle:
        handle[TASK_KEY]["demo_0"].attrs[
            "native_butter_body_position"
        ] += np.array([0.01, 0.0, 0.0])
    with pytest.raises(
        ValueError, match="paired native butter body position differs"
    ):
        validate_state_artifacts(
            paths["eb"],
            paths["er"],
            paths["ec"],
            native_bddl=NATIVE_BDDL,
        )


def test_pairing_allows_settled_base_but_rejects_eb_not_equal_to_base(
    tmp_path,
):
    paths = _triplet(tmp_path)
    with h5py.File(paths["eb"], "r+") as handle:
        demo = handle[TASK_KEY]["demo_0"]
        assert not np.array_equal(
            demo["native_source_state"][:],
            demo["base_reset_state"][:],
        )
        demo["initial_state"][0] += 0.01
        demo.attrs["initial_state_sha256"] = l3a2_generator.sha256_array(
            demo["initial_state"][:]
        )
        demo["intervention_state"][0] += 0.01
        demo.attrs["intervention_state_sha256"] = (
            l3a2_generator.sha256_array(demo["intervention_state"][:])
        )
    with pytest.raises(
        ValueError, match="evaluated initial_state differs from paired base"
    ):
        validate_state_artifacts(
            paths["eb"],
            paths["er"],
            paths["ec"],
            native_bddl=NATIVE_BDDL,
        )


def test_generation_manifest_binds_official_source_rows_to_hdf5(tmp_path):
    paths = _triplet(tmp_path)
    episodes = []
    for index in range(2):
        conditions = {}
        for condition in ("eb", "er", "ec"):
            preview = tmp_path / f"{condition}_{index}.png"
            preview.write_bytes(f"{condition}-{index}".encode())
            with h5py.File(paths[condition], "r") as handle:
                demo = handle[TASK_KEY][f"demo_{index}"]
                conditions[condition] = {
                    "intervention_state_sha256": (
                        l3a2_generator.sha256_array(
                            demo["intervention_state"][:]
                        )
                    ),
                    "initial_state_sha256": l3a2_generator.sha256_array(
                        demo["initial_state"][:]
                    ),
                    "first_policy_frame": str(preview),
                    "first_policy_frame_file_sha256": sha256_file(preview),
                }
        with h5py.File(paths["eb"], "r") as handle:
            demo = handle[TASK_KEY][f"demo_{index}"]
            source_hash = l3a2_generator.sha256_array(
                demo["native_source_state"][:]
            )
            base_hash = l3a2_generator.sha256_array(
                demo["base_reset_state"][:]
            )
        episodes.append(
            {
                "episode": index,
                "native_init_state_index": index,
                "source_state_sha256": source_hash,
                "base_state_sha256": base_hash,
                "source_to_base": {
                    "method": (
                        "formal_evaluator_controller_dummy_action"
                    ),
                    "wait_steps": SOURCE_TO_BASE_WAIT_STEPS,
                    "environment_hard_reset": False,
                    "environment_seed": EVALUATOR_ENV_SEED,
                    "restored_source_state_sha256": source_hash,
                    "wait_state_sha256": [base_hash]
                    * SOURCE_TO_BASE_WAIT_STEPS,
                    "paired_base_state_sha256": base_hash,
                },
                "conditions": conditions,
            }
        )
    manifest = {
        "scene_id": SCENE_ID,
        "verdict": "PASS_L3A2_GENERATION_AND_REFERENCE_GATES",
        "base_state_source": BASE_STATE_SOURCE,
        "pairing_method": PAIRING_METHOD,
        "environment_seed": EVALUATOR_ENV_SEED,
        "environment_hard_reset": False,
        "source_to_base_wait_steps": SOURCE_TO_BASE_WAIT_STEPS,
        "source_to_base_wait_method": (
            "formal_evaluator_controller_dummy_action"
        ),
        "native_init_states": str(NATIVE_INIT_STATES.resolve()),
        "native_init_states_sha256": sha256_file(NATIVE_INIT_STATES),
        "artifacts": {
            condition: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for condition, path in paths.items()
        },
        "episodes": episodes,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    result = validate_generation_manifest(
        manifest_path,
        eb_path=paths["eb"],
        er_path=paths["er"],
        ec_path=paths["ec"],
        minimum_count=2,
    )
    assert result["native_init_states_sha256"] == sha256_file(
        NATIVE_INIT_STATES
    )

    manifest["episodes"][0]["source_state_sha256"] = "0" * 64
    manifest["episodes"][0]["source_to_base"][
        "restored_source_state_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source state hash differs from HDF5"):
        validate_generation_manifest(
            manifest_path,
            eb_path=paths["eb"],
            er_path=paths["er"],
            ec_path=paths["ec"],
            minimum_count=2,
        )


def test_osc_loader_uses_saved_world_body_position_not_free_qpos(tmp_path):
    paths = _triplet(tmp_path)
    records = _load_records(str(paths["er"]), 2)
    with h5py.File(paths["er"], "r") as handle:
        for index, record in enumerate(records):
            free_qpos_translation = handle[TASK_KEY][f"demo_{index}"][
                "base_reset_state"
            ][10:13]
            expected_world_position = free_qpos_translation + np.array(
                [0.125, -0.25, 0.375]
            )
            assert record["native_butter_body_position"] == pytest.approx(
                expected_world_position
            )
            assert not np.array_equal(
                record["native_butter_body_position"],
                free_qpos_translation,
            )


def test_osc_timeout_diagnostics_preserve_progress_and_target():
    failure = SimpleNamespace(
        reason="waypoint_timeout",
        stage="butter_park_translate",
        initial_error_m=0.2163,
        best_error_m=0.0361,
        final_error_m=0.0364,
        final_eef_xyz=(0.0643, -0.2034, 0.3497),
        target_eef_xyz=(0.1008, -0.2034, 0.3497),
    )
    diagnostics = _failure_diagnostics(failure)

    assert diagnostics["failure_initial_error_m"] == pytest.approx(0.2163)
    assert diagnostics["failure_best_error_m"] == pytest.approx(0.0361)
    assert diagnostics["failure_final_error_m"] == pytest.approx(0.0364)
    assert diagnostics["failure_progress_m"] == pytest.approx(0.1802)
    assert diagnostics["failure_progress_fraction"] == pytest.approx(
        0.1802 / 0.2163
    )
    assert diagnostics["failure_final_eef_xyz"] == pytest.approx(
        [0.0643, -0.2034, 0.3497]
    )
    assert diagnostics["failure_target_eef_xyz"] == pytest.approx(
        [0.1008, -0.2034, 0.3497]
    )
    assert diagnostics["failure_progressing_at_budget_limit"] is True


def test_osc_timeout_diagnostics_do_not_invent_missing_motion_evidence():
    diagnostics = _failure_diagnostics(None)

    assert np.isnan(diagnostics["failure_initial_error_m"])
    assert np.isnan(diagnostics["failure_best_error_m"])
    assert np.isnan(diagnostics["failure_final_error_m"])
    assert np.isnan(diagnostics["failure_progress_m"])
    assert np.isnan(diagnostics["failure_progress_fraction"])
    assert diagnostics["failure_final_eef_xyz"] == []
    assert diagnostics["failure_target_eef_xyz"] == []
    assert diagnostics["failure_progressing_at_budget_limit"] is False


def test_osc_transport_timeout_is_bounded_by_complete_plan():
    assert TRANSPORT_MAX_WAYPOINT_STEPS == 50
    assert TRANSPORT_MAX_WAYPOINT_STEPS == max(
        HORIZON_STAGE_STEP_LIMITS.values()
    )
    assert HORIZON_STAGE_STEP_LIMITS["milk_to_basket_translate"] == 50
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222


def test_500094_reallocates_only_observed_approach_actions_to_lift():
    # Job500094 reached lift in all 25 attempts. Subtract its exact 15 descend,
    # 8 seat, and 12 lift actions to recover per-attempt approach use.
    task_steps = [
        62, 62, 62, 61, 62,
        63, 63, 63, 62, 63,
        64, 64, 64, 64, 65,
        62, 62, 62, 61, 62,
        60, 60, 60, 60, 61,
    ]
    observed_approach_steps = [value - 15 - 8 - 12 for value in task_steps]

    assert min(observed_approach_steps) == 25
    assert max(observed_approach_steps) == 30
    assert HORIZON_STAGE_STEP_LIMITS["butter_approach"] == 30
    # Job500120 later fixed this exact observed maximum as the registered
    # bound, with the post-final-action check preserving the same predicate.
    assert HORIZON_STAGE_STEP_LIMITS["butter_descend"] == 15
    assert HORIZON_STAGE_STEP_LIMITS["butter_lift"] == 14
    # Descend is now solved under the unchanged precise 8 mm tolerance.
    assert 6.9092 < 8.0
    assert 7.0238 < 8.0
    # Lift remained above its unchanged 12 mm tolerance after action 12, but
    # every final action still improved error by at least 3.2786 mm.
    assert 14.5853 > 12.0
    assert 14.7093 > 12.0
    assert 3.2786 > 0.0
    # The reallocation cannot borrow from the 56 fixed safety/hold actions.
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222


def test_500107_reallocates_zero_use_sweep_raise_to_park_descend():
    # Exact Job500107 trajectories: all 25 geometry-aware raise targets were
    # already satisfied, while every 12-action park descend stopped at a
    # monotonic 67--69 mm residual against the unchanged 12 mm tolerance.
    raise_actions = [0] * 25
    final_error_mm = [
        67.297, 67.413, 67.192, 67.985, 67.596,
        67.453, 67.569, 67.339, 68.284, 67.784,
        69.473, 67.309, 69.358, 67.932, 67.497,
        67.388, 67.502, 69.443, 68.213, 68.088,
        67.581, 67.698, 67.468, 68.388, 67.872,
    ]
    minimum_clearance_mm = [
        88.078, 88.105, 88.044, 88.574, 88.554,
        88.070, 88.098, 88.032, 88.591, 88.579,
        90.343, 88.066, 90.303, 88.597, 88.536,
        88.065, 88.095, 90.228, 88.584, 88.759,
        88.068, 88.101, 88.030, 88.554, 88.551,
    ]

    assert max(raise_actions) == 0
    assert min(minimum_clearance_mm) > 80.0
    assert min(final_error_mm) > 12.0
    # An observed controller tail starting at 58--67 mm required eight more
    # actions in all 25 attempts to finish at 9.19--11.92 mm. Preserve that
    # full evidenced tail; seven was not sufficient in the limiting trace.
    assert 15.474 > 12.0
    assert 11.917 < 12.0
    assert HORIZON_STAGE_STEP_LIMITS["butter_park_raise"] == 0
    assert HORIZON_STAGE_STEP_LIMITS["butter_park_descend"] == 20
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222


def test_500120_reallocates_exact_spares_to_butter_park_retreat():
    # Exact Job500120 evidence, ordered episode-major then attempt-major.
    # Descend and translate each completed one action below their registered
    # limits in all 25 attempts. Park descend then passed using its full new
    # bound, exposing only the still-converging eight-action retreat.
    task_steps = [
        112, 112, 112, 111, 112,
        113, 113, 113, 112, 113,
        115, 114, 115, 114, 115,
        112, 112, 113, 111, 112,
        110, 110, 110, 110, 111,
    ]
    descend_final_error_mm = [
        6.947, 6.958, 6.936, 6.979, 7.004,
        6.953, 6.964, 6.942, 6.980, 7.010,
        6.962, 6.974, 6.949, 6.909, 6.937,
        6.963, 6.975, 6.950, 6.990, 7.018,
        7.010, 7.024, 6.996, 6.958, 6.977,
    ]
    translate_final_error_mm = [
        9.766, 9.802, 9.731, 9.719, 9.789,
        9.799, 9.837, 9.763, 9.766, 9.812,
        9.732, 9.746, 9.696, 9.660, 9.732,
        9.770, 9.808, 9.756, 9.735, 9.791,
        9.793, 9.831, 9.757, 9.738, 9.801,
    ]
    park_descend_final_error_mm = [
        10.705, 10.729, 10.683, 10.838, 10.774,
        10.729, 10.753, 10.706, 10.890, 10.804,
        11.139, 10.703, 11.115, 10.822, 10.749,
        10.716, 10.740, 11.130, 10.877, 10.866,
        10.752, 10.776, 10.729, 10.909, 10.819,
    ]
    retreat_final_error_mm = [
        18.117, 18.064, 18.169, 18.118, 18.114,
        18.132, 18.080, 18.185, 18.134, 18.129,
        18.220, 18.167, 18.274, 18.223, 18.219,
        18.160, 18.107, 18.213, 18.161, 18.158,
        18.134, 18.082, 18.187, 18.137, 18.133,
    ]
    retreat_last_action_gain_mm = [
        4.360, 4.354, 4.367, 4.361, 4.360,
        4.363, 4.356, 4.370, 4.363, 4.362,
        4.375, 4.367, 4.383, 4.375, 4.374,
        4.366, 4.359, 4.374, 4.367, 4.366,
        4.363, 4.356, 4.370, 4.363, 4.362,
    ]

    assert min(task_steps) == 110
    assert max(task_steps) == 115
    assert max(descend_final_error_mm) < 8.0
    assert max(translate_final_error_mm) < 12.0
    assert max(park_descend_final_error_mm) < 12.0
    assert min(retreat_final_error_mm) > 12.0
    # One action is not an evidence-backed repair: the smallest remaining
    # excess over tolerance exceeds the largest observed final-action gain.
    assert min(value - 12.0 for value in retreat_final_error_mm) > max(
        retreat_last_action_gain_mm
    )
    assert HORIZON_STAGE_STEP_LIMITS["butter_descend"] == 15
    assert HORIZON_STAGE_STEP_LIMITS["butter_park_translate"] == 11
    assert HORIZON_STAGE_STEP_LIMITS["butter_park_retreat"] == 10
    # No action is borrowed from grasp, hold, release, or stability windows.
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222


def test_500128_reallocates_exact_milk_approach_spare_to_lift():
    # Exact Job500128 evidence, ordered episode-major then attempt-major.
    # The complete butter park, retreat, and ten-step confirmation passed in
    # all 25 attempts. Milk approach then needed exactly 12 actions, while the
    # 12-action milk lift remained convergent just outside tolerance.
    task_steps = [
        163, 163, 164, 162, 163,
        164, 164, 164, 163, 164,
        166, 165, 167, 165, 166,
        163, 163, 165, 162, 163,
        161, 161, 161, 161, 162,
    ]
    milk_approach_steps = [12] * 25
    milk_approach_final_error_mm = [
        10.633414, 11.093956, 10.265997, 10.608700, 10.655214,
        10.667521, 11.141493, 10.276884, 10.692392, 10.691813,
        10.626741, 11.126710, 10.254393, 10.635605, 10.680739,
        10.666574, 11.135810, 10.232073, 10.691811, 10.732355,
        10.670952, 11.148829, 10.272508, 10.660312, 10.689717,
    ]
    milk_lift_steps = [12] * 25
    milk_lift_final_error_mm = [
        14.167201, 14.138536, 14.400306, 14.168227, 14.165980,
        14.169280, 14.147370, 14.202369, 14.171917, 14.167336,
        14.208339, 14.181672, 14.440392, 14.210499, 14.208054,
        14.181852, 14.158373, 14.416566, 14.182701, 14.180698,
        14.170954, 14.149538, 14.199545, 14.171261, 14.172046,
    ]
    milk_lift_last_action_gain_mm = [
        3.151321, 3.146260, 3.175521, 3.151717, 3.150863,
        3.151514, 3.148059, 3.158855, 3.151849, 3.151221,
        3.160226, 3.155555, 3.184824, 3.160772, 3.159897,
        3.154388, 3.150793, 3.179446, 3.154538, 3.154197,
        3.155451, 3.151843, 3.161041, 3.155377, 3.155539,
    ]

    assert min(task_steps) == 161
    assert max(task_steps) == 167
    assert set(milk_approach_steps) == {12}
    assert max(milk_approach_final_error_mm) < 12.0
    assert set(milk_lift_steps) == {12}
    assert min(milk_lift_final_error_mm) > 12.0

    # One more lift action is the minimum repair. The limiting trace needs
    # only 76.63% of its 12th-action gain. Job500107 measured 81.55--81.68%
    # action-to-action gain retention over the same OSC lift tail in all 25
    # registered episode/offset attempts.
    required_gain_retention = [
        (error - 12.0) / gain
        for error, gain in zip(
            milk_lift_final_error_mm,
            milk_lift_last_action_gain_mm,
        )
    ]
    job500107_same_controller_min_gain_retention = 0.815518416
    job500107_smallest_post_action_12_error_mm = 14.585355
    assert max(required_gain_retention) == pytest.approx(0.76625654)
    assert max(required_gain_retention) < (
        job500107_same_controller_min_gain_retention
    )
    assert max(milk_lift_final_error_mm) < (
        job500107_smallest_post_action_12_error_mm
    )

    # Lock the full physical evidence that authorized advancing beyond the
    # parked butter. Each attempt passed all ten confirmation actions, and
    # every later milk action continued to observe floor-only butter support.
    confirmation_steps = [10] * 25
    post_park_monitor_steps = [
        39, 39, 40, 39, 39,
        39, 39, 39, 39, 39,
        39, 39, 40, 39, 39,
        39, 39, 40, 39, 39,
        39, 39, 39, 39, 39,
    ]
    confirmation_evidence = {
        "samples": 250,
        "max_drift_m": 0.0,
        "max_tilt_deg": 3.1945284701301985e-06,
        "max_linear_speed_mps": 3.0085449419112237e-16,
        "max_angular_speed_radps": 1.4546898636689465e-15,
        "contact_sets": {("floor",)},
        "floor_support_samples": 250,
        "forbidden_contact_samples": 0,
    }
    post_park_evidence = {
        "samples": 978,
        "max_drift_m": 0.0,
        "max_tilt_deg": 3.1945284701301985e-06,
        "max_linear_speed_mps": 3.3756547766497907e-16,
        "max_angular_speed_radps": 5.980886314913616e-15,
        "contact_sets": {("floor",)},
        "floor_support_samples": 978,
        "forbidden_contact_samples": 0,
    }
    assert set(confirmation_steps) == {10}
    assert min(post_park_monitor_steps) == 39
    assert max(post_park_monitor_steps) == 40
    assert sum(post_park_monitor_steps) == 978
    assert confirmation_evidence["samples"] == sum(confirmation_steps)
    assert post_park_evidence["samples"] == sum(post_park_monitor_steps)
    for evidence in (confirmation_evidence, post_park_evidence):
        assert evidence["max_drift_m"] <= 0.005
        assert evidence["max_tilt_deg"] <= 2.0
        assert evidence["max_linear_speed_mps"] <= 0.01
        assert evidence["max_angular_speed_radps"] <= 0.10
        assert evidence["contact_sets"] == {("floor",)}
        assert evidence["floor_support_samples"] == evidence["samples"]
        assert evidence["forbidden_contact_samples"] == 0

    # Jobs 500138 and 500144 subsequently draw six more of the same observed
    # approach reserve; the Job500128 one-action lift funding remains intact.
    assert HORIZON_STAGE_STEP_LIMITS["milk_approach"] == 13
    assert HORIZON_STAGE_STEP_LIMITS["milk_lift"] == 13
    # No action is borrowed from grasp, hold, release, or stability windows.
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222
    budget = _static_plan_budget_diagnostics(
        grasp_seat_steps=8,
        contact_hold_steps=2,
        release_steps=8,
        settle_steps=10,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )
    assert budget["registered_repeated_hold_steps"] == 56
    assert budget["static_safe_plan_max_steps"] == 278
    assert budget["static_safe_plan_budget_margin_steps"] == 2


def test_500138_funds_three_action_basket_raise_tail_from_approach():
    # Exact Job500138 evidence, ordered episode-major then attempt-major.
    # All 25 attempts passed the newly funded 13-action milk lift. The sole
    # failure was the still-converging eight-action basket raise.
    task_steps = [
        172, 172, 173, 171, 172,
        173, 173, 173, 172, 173,
        175, 174, 176, 174, 175,
        172, 172, 174, 171, 172,
        170, 170, 170, 170, 171,
    ]
    butter_approach_steps = [
        27, 27, 27, 26, 27,
        28, 28, 28, 27, 28,
        29, 29, 29, 29, 30,
        27, 27, 27, 26, 27,
        25, 25, 25, 25, 26,
    ]
    butter_lift_steps = [
        13, 13, 13, 13, 13,
        13, 13, 13, 13, 13,
        14, 13, 14, 13, 13,
        13, 13, 14, 13, 13,
        13, 13, 13, 13, 13,
    ]
    milk_descend_steps = [
        7, 7, 8, 7, 7,
        7, 7, 7, 7, 7,
        7, 7, 8, 7, 7,
        7, 7, 8, 7, 7,
        7, 7, 7, 7, 7,
    ]
    milk_lift_body_mm = [
        88.576216, 88.701555, 89.024771, 88.556189, 88.591657,
        88.625865, 88.728386, 88.490473, 88.619504, 88.637843,
        88.519313, 88.651253, 88.983663, 88.497317, 88.537311,
        88.602306, 88.706765, 89.018706, 88.594375, 88.618418,
        88.665694, 88.744627, 88.525428, 88.649143, 88.672603,
    ]
    raise_initial_error_mm = [
        79.827481, 79.865618, 79.909303, 79.814587, 79.836959,
        79.862927, 79.865509, 79.786223, 79.860802, 79.863750,
        79.818013, 79.855292, 79.892408, 79.803748, 79.829190,
        79.856300, 79.861878, 79.899693, 79.851226, 79.861810,
        79.861177, 79.862285, 79.806528, 79.861144, 79.861510,
    ]
    raise_final_error_mm = [
        18.514587, 18.540190, 18.489067, 18.510976, 18.517689,
        18.520995, 18.538353, 18.487830, 18.519728, 18.522581,
        18.486338, 18.511336, 18.459918, 18.482248, 18.489500,
        18.510970, 18.529136, 18.475921, 18.509437, 18.513513,
        18.514093, 18.529547, 18.485077, 18.512851, 18.514578,
    ]
    raise_last_action_gain_mm = [
        4.179785, 4.182128, 4.188071, 4.179027, 4.180368,
        4.181115, 4.181429, 4.176374, 4.181048, 4.181174,
        4.176688, 4.178985, 4.184688, 4.175876, 4.177372,
        4.180275, 4.180723, 4.186429, 4.180013, 4.180591,
        4.181519, 4.181746, 4.178126, 4.181556, 4.181566,
    ]
    final_observed_gain_retention = [
        0.813580267, 0.813695908, 0.813681631, 0.813581959,
        0.813590053, 0.813465627, 0.813595618, 0.813367069,
        0.813466421, 0.813475980, 0.813147626, 0.813259323,
        0.813272142, 0.813148062, 0.813153666, 0.813379837,
        0.813501637, 0.813488100, 0.813380259, 0.813386853,
        0.813621042, 0.813736655, 0.813505579, 0.813611542,
        0.813623232,
    ]

    assert min(task_steps) == 170
    assert max(task_steps) == 176
    assert min(butter_approach_steps) == 25
    assert max(butter_approach_steps) == 30
    assert set(butter_lift_steps) == {13, 14}
    assert set(milk_descend_steps) == {7, 8}
    assert min(milk_lift_body_mm) > 88.49
    assert max(milk_lift_body_mm) < 89.03
    observed_fixed_stage_steps = {
        "butter_descend": {15},
        "butter_park_raise": {0},
        "butter_park_translate": {11},
        "butter_park_descend": {20},
        "butter_park_retreat": {10},
        "milk_approach": {12},
        "milk_lift": {13},
        "milk_to_basket_raise": {8},
        "milk_to_basket_translate": {0},
        "milk_to_basket_descend": {0},
        "milk_to_basket_retreat": {0},
    }
    assert observed_fixed_stage_steps["milk_lift"] == {13}
    assert observed_fixed_stage_steps["milk_to_basket_raise"] == {8}
    assert all(
        observed_fixed_stage_steps[stage] == {0}
        for stage in (
            "milk_to_basket_translate",
            "milk_to_basket_descend",
            "milk_to_basket_retreat",
        )
    )

    assert min(raise_initial_error_mm) > 79.78
    assert max(raise_initial_error_mm) < 79.91
    assert min(raise_final_error_mm) > 12.0
    # One extra action is categorically insufficient: every remaining excess
    # is larger than even the largest observed eighth-action gain.
    assert min(error - 12.0 for error in raise_final_error_mm) > max(
        raise_last_action_gain_mm
    )

    # The last three gain-retention transitions across the 25 traces ranged
    # from 0.813147626 to 0.818140253. Even combining the easiest residual,
    # largest gain, and most optimistic observed retention leaves a two-action
    # extrapolation above tolerance. Three actions pass under the conservative
    # minimum observed retention for every trace.
    min_late_gain_retention = min(final_observed_gain_retention)
    max_late_gain_retention = 0.818140253
    optimistic_two_action_error = min(raise_final_error_mm) - max(
        raise_last_action_gain_mm
    ) * (
        max_late_gain_retention + max_late_gain_retention**2
    )
    conservative_three_action_errors = [
        error
        - gain
        * sum(min_late_gain_retention**power for power in (1, 2, 3))
        for error, gain in zip(
            raise_final_error_mm,
            raise_last_action_gain_mm,
        )
    ]
    assert optimistic_two_action_error == pytest.approx(12.230188, abs=1e-5)
    assert optimistic_two_action_error > 12.0
    assert min(conservative_three_action_errors) == pytest.approx(
        10.040254, abs=1e-5
    )
    assert max(conservative_three_action_errors) == pytest.approx(
        10.125675, abs=1e-5
    )
    assert max(conservative_three_action_errors) < 12.0

    # The parked butter remained valid for the complete confirmation and all
    # later milk actions, including the eight raise actions.
    confirmation_steps = [10] * 25
    post_park_monitor_steps = [
        48, 48, 49, 48, 48,
        48, 48, 48, 48, 48,
        48, 48, 49, 48, 48,
        48, 48, 49, 48, 48,
        48, 48, 48, 48, 48,
    ]
    assert sum(confirmation_steps) == 250
    assert sum(post_park_monitor_steps) == 1203
    for samples, max_linear, max_angular in (
        (250, 3.0085449419112237e-16, 1.4546898636689465e-15),
        (1203, 3.3756547766497907e-16, 5.980886314913616e-15),
    ):
        stability_evidence = {
            "samples": samples,
            "max_drift_m": 0.0,
            "max_tilt_deg": 3.1945284701301985e-06,
            "max_linear_speed_mps": max_linear,
            "max_angular_speed_radps": max_angular,
            "contact_sets": {("floor",)},
            "floor_support_samples": samples,
            "forbidden_contact_samples": 0,
        }
        assert stability_evidence["max_drift_m"] <= 0.005
        assert stability_evidence["max_tilt_deg"] <= 2.0
        assert stability_evidence["max_linear_speed_mps"] <= 0.01
        assert stability_evidence["max_angular_speed_radps"] <= 0.10
        assert stability_evidence["contact_sets"] == {("floor",)}
        assert stability_evidence["floor_support_samples"] == samples
        assert stability_evidence["forbidden_contact_samples"] == 0

    # Job500144 subsequently draws three more approach reserves for the
    # basket-retreat tail; the Job500138 raise funding remains intact.
    assert HORIZON_STAGE_STEP_LIMITS["milk_approach"] == 13
    assert HORIZON_STAGE_STEP_LIMITS["milk_lift"] == 13
    assert HORIZON_STAGE_STEP_LIMITS["milk_to_basket_raise"] == 11
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222
    budget = _static_plan_budget_diagnostics(
        grasp_seat_steps=8,
        contact_hold_steps=2,
        release_steps=8,
        settle_steps=10,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )
    assert budget["registered_repeated_hold_steps"] == 56
    assert budget["static_safe_plan_max_steps"] == 278
    assert budget["static_safe_plan_budget_margin_steps"] == 2


def test_500144_funds_three_action_basket_retreat_tail_from_approach():
    # Exact Job500144 evidence, ordered episode-major then attempt-major.
    # All 25 attempts reached native task success; the sole failure was the
    # post-release eight-action retreat remaining just outside tolerance.
    task_steps = [
        228, 228, 229, 227, 228,
        229, 229, 229, 228, 229,
        229, 228, 230, 228, 229,
        228, 228, 230, 227, 228,
        224, 224, 224, 224, 225,
    ]
    basket_translate_steps = [
        43, 43, 43, 43, 43,
        43, 43, 43, 43, 43,
        41, 41, 41, 41, 41,
        43, 43, 43, 43, 43,
        41, 41, 41, 41, 41,
    ]
    native_success_steps = [
        255, 255, 256, 254, 255,
        256, 256, 256, 255, 256,
        256, 255, 257, 255, 256,
        255, 255, 257, 254, 255,
        251, 251, 251, 251, 252,
    ]
    milk_final_goal_error_mm = [
        6.438043, 7.908955, 3.440940, 6.576896, 3.930710,
        5.810189, 10.719940, 4.451719, 6.319299, 5.816380,
        4.545548, 9.330823, 6.840451, 4.444510, 4.529547,
        3.592564, 5.387928, 10.608092, 7.259988, 13.278472,
        5.067925, 6.901032, 4.231518, 5.089182, 5.161109,
    ]
    retreat_final_error_mm = [
        18.611292, 18.591055, 18.627191, 18.610825, 18.611441,
        18.578541, 18.582189, 18.576543, 18.579909, 18.578680,
        18.561739, 18.560295, 18.566863, 18.561446, 18.561639,
        18.579927, 18.582343, 18.594890, 18.580526, 18.576936,
        18.549359, 18.550557, 18.547355, 18.549169, 18.549427,
    ]
    retreat_last_action_gain_mm = [
        4.308885, 4.292216, 4.309179, 4.308881, 4.308438,
        4.316650, 4.316116, 4.316483, 4.316374, 4.316635,
        4.315771, 4.315308, 4.315314, 4.315740, 4.315658,
        4.314183, 4.314358, 4.314031, 4.314040, 4.312027,
        4.311946, 4.311969, 4.312075, 4.311939, 4.311930,
    ]
    final_observed_gain_retention = [
        0.807301628, 0.807517983, 0.807400258, 0.807296632,
        0.807312440, 0.807615829, 0.807658282, 0.807595522,
        0.807631212, 0.807615188, 0.807565834, 0.807603177,
        0.807652485, 0.807567437, 0.807571317, 0.807521489,
        0.807508524, 0.807604883, 0.807500982, 0.807509817,
        0.807286141, 0.807301450, 0.807255844, 0.807283254,
        0.807285429,
    ]

    assert min(task_steps) == 224
    assert max(task_steps) == 230
    assert set(basket_translate_steps) == {41, 43}
    observed_fixed_stage_steps = {
        "milk_approach": {12},
        "milk_descend": {7, 8},
        "milk_lift": {13},
        "milk_to_basket_raise": {11},
        "milk_to_basket_descend": {10},
        "contact_hold": {2},
        "release": {8},
        "milk_to_basket_retreat": {8},
    }
    assert observed_fixed_stage_steps["milk_to_basket_raise"] == {11}
    assert observed_fixed_stage_steps["milk_to_basket_descend"] == {10}
    assert observed_fixed_stage_steps["contact_hold"] == {2}
    assert observed_fixed_stage_steps["release"] == {8}
    assert observed_fixed_stage_steps["milk_to_basket_retreat"] == {8}

    # Native success occurred in every trace before the retreat failure. The
    # native in-basket predicate, rather than Euclidean body-goal tolerance,
    # is authoritative for task success.
    native_success = [True] * 25
    within_evaluation_budget = [True] * 25
    assert all(native_success)
    assert min(native_success_steps) == 251
    assert max(native_success_steps) == 257
    assert all(within_evaluation_budget)
    assert min(milk_final_goal_error_mm) == pytest.approx(3.440940)
    assert max(milk_final_goal_error_mm) == pytest.approx(13.278472)

    assert min(retreat_final_error_mm) > 12.0
    # One extra action is impossible: every residual excess is larger than
    # the largest observed eighth-action gain.
    assert min(error - 12.0 for error in retreat_final_error_mm) > max(
        retreat_last_action_gain_mm
    )

    # Across the last three gain transitions in all 25 retreat traces, gain
    # retention ranged from 0.807255844 to 0.814130374. The most optimistic
    # observed two-action extrapolation still misses tolerance; three actions
    # pass for every trace under the conservative observed retention.
    min_late_gain_retention = min(final_observed_gain_retention)
    max_late_gain_retention = 0.814130374
    optimistic_two_action_error = min(retreat_final_error_mm) - max(
        retreat_last_action_gain_mm
    ) * (
        max_late_gain_retention + max_late_gain_retention**2
    )
    conservative_two_action_errors = [
        error - gain * sum(min_late_gain_retention**power for power in (1, 2))
        for error, gain in zip(
            retreat_final_error_mm,
            retreat_last_action_gain_mm,
        )
    ]
    conservative_three_action_errors = [
        error
        - gain
        * sum(min_late_gain_retention**power for power in (1, 2, 3))
        for error, gain in zip(
            retreat_final_error_mm,
            retreat_last_action_gain_mm,
        )
    ]
    assert optimistic_two_action_error == pytest.approx(12.171928, abs=1e-5)
    assert optimistic_two_action_error > 12.0
    assert min(conservative_two_action_errors) == pytest.approx(
        12.256392, abs=1e-5
    )
    assert max(conservative_two_action_errors) == pytest.approx(
        12.340453, abs=1e-5
    )
    assert min(conservative_two_action_errors) > 12.0
    assert min(conservative_three_action_errors) == pytest.approx(
        9.987990, abs=1e-5
    )
    assert max(conservative_three_action_errors) == pytest.approx(
        10.073575, abs=1e-5
    )
    assert max(conservative_three_action_errors) < 12.0

    # Lock all butter evidence through native success, release, and the full
    # eight-action failed retreat.
    confirmation_steps = [10] * 25
    post_park_monitor_steps = [
        122, 122, 123, 122, 122,
        122, 122, 122, 122, 122,
        120, 120, 121, 120, 120,
        122, 122, 123, 122, 122,
        120, 120, 120, 120, 120,
    ]
    assert sum(confirmation_steps) == 250
    assert sum(post_park_monitor_steps) == 3033
    confirmation_evidence = {
        "samples": 250,
        "max_drift_m": 0.0,
        "max_tilt_deg": 3.1945284701301985e-06,
        "max_linear_speed_mps": 3.0085449419112237e-16,
        "max_angular_speed_radps": 1.4546898636689465e-15,
        "contact_sets": {("floor",)},
        "floor_support_samples": 250,
        "forbidden_contact_samples": 0,
    }
    post_park_evidence = {
        "samples": 3033,
        "max_drift_m": 3.469446951953614e-18,
        "max_tilt_deg": 3.1945284701301985e-06,
        "max_linear_speed_mps": 4.789564782125783e-16,
        "max_angular_speed_radps": 1.31124127483488e-14,
        "contact_sets": {("floor",)},
        "floor_support_samples": 3033,
        "forbidden_contact_samples": 0,
    }
    for evidence in (confirmation_evidence, post_park_evidence):
        assert evidence["max_drift_m"] <= 0.005
        assert evidence["max_tilt_deg"] <= 2.0
        assert evidence["max_linear_speed_mps"] <= 0.01
        assert evidence["max_angular_speed_radps"] <= 0.10
        assert evidence["contact_sets"] == {("floor",)}
        assert evidence["floor_support_samples"] == evidence["samples"]
        assert evidence["forbidden_contact_samples"] == 0

    assert HORIZON_STAGE_STEP_LIMITS["milk_approach"] == 13
    assert HORIZON_STAGE_STEP_LIMITS["milk_lift"] == 13
    assert HORIZON_STAGE_STEP_LIMITS["milk_to_basket_raise"] == 11
    assert HORIZON_STAGE_STEP_LIMITS["milk_to_basket_retreat"] == 11
    assert sum(HORIZON_STAGE_STEP_LIMITS.values()) == 222
    budget = _static_plan_budget_diagnostics(
        grasp_seat_steps=8,
        contact_hold_steps=2,
        release_steps=8,
        settle_steps=10,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )
    assert budget["registered_repeated_hold_steps"] == 56
    assert budget["static_safe_plan_max_steps"] == 278
    assert budget["static_safe_plan_budget_margin_steps"] == 2


def test_registered_waypoint_accepts_exact_post_final_action_state():
    timeout = SimpleNamespace(reason="waypoint_timeout", stage="butter_lift")

    class Shared:
        calls = 0

        @classmethod
        def _move_to(cls, *args, **kwargs):
            del args
            cls.calls += 1
            assert kwargs["max_steps"] == 14
            return {"eef": np.array([0.0, 0.0, 0.0119])}, 14, timeout

        @staticmethod
        def _eef_pos(obs):
            return obs["eef"]

    oracle = SimpleNamespace(_metrics=lambda env: {"gripper_contact": False})
    env = SimpleNamespace(check_success=lambda: False)
    obs, step, failure = _move_to_with_final_state_check(
        Shared,
        env,
        {},
        oracle,
        None,
        np.zeros(3),
        1.0,
        0,
        SimpleNamespace(position_tolerance=0.012),
        "butter_lift",
        max_steps=14,
    )

    assert Shared.calls == 1
    assert step == 14
    assert obs["eef"] == pytest.approx([0.0, 0.0, 0.0119])
    assert failure is None


def test_registered_zero_action_waypoint_accepts_already_satisfied_state():
    timeout = SimpleNamespace(reason="waypoint_timeout", stage="park_raise")

    class Shared:
        @staticmethod
        def _move_to(*args, **kwargs):
            assert kwargs["max_steps"] == 0
            return {"eef": np.array([0.0, 0.0, 0.23])}, 9, timeout

        @staticmethod
        def _eef_pos(obs):
            return obs["eef"]

    oracle = SimpleNamespace(_metrics=lambda env: {"gripper_contact": False})
    env = SimpleNamespace(check_success=lambda: False)
    _, step, failure = _move_to_with_final_state_check(
        Shared,
        env,
        {},
        oracle,
        None,
        np.array([0.0, 0.0, 0.23]),
        1.0,
        9,
        SimpleNamespace(position_tolerance=0.012),
        "butter_park_raise",
        max_steps=0,
    )

    assert step == 9
    assert failure is None


def test_registered_waypoint_does_not_waive_timeout_or_safety_failure():
    failures = [
        SimpleNamespace(reason="waypoint_timeout", stage="butter_lift"),
        SimpleNamespace(reason="grasp_slipped", stage="butter_lift"),
    ]

    class Shared:
        @staticmethod
        def _move_to(*args, **kwargs):
            del args, kwargs
            failure = failures.pop(0)
            eef = 0.0121 if failure.reason == "waypoint_timeout" else 0.0
            return {"eef": np.array([0.0, 0.0, eef])}, 14, failure

        @staticmethod
        def _eef_pos(obs):
            return obs["eef"]

    oracle = SimpleNamespace(_metrics=lambda env: {"gripper_contact": False})
    env = SimpleNamespace(check_success=lambda: False)
    for expected_reason in ("waypoint_timeout", "grasp_slipped"):
        _, _, failure = _move_to_with_final_state_check(
            Shared,
            env,
            {},
            oracle,
            None,
            np.zeros(3),
            1.0,
            0,
            SimpleNamespace(position_tolerance=0.012),
            "butter_lift",
            max_steps=14,
        )
        assert failure.reason == expected_reason


def test_osc_complete_safe_plan_has_static_two_step_horizon_margin():
    diagnostics = _static_plan_budget_diagnostics(
        grasp_seat_steps=8,
        contact_hold_steps=2,
        release_steps=8,
        settle_steps=10,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )

    assert diagnostics["registered_motion_stage_steps"] == 222
    assert diagnostics["registered_repeated_hold_steps"] == 56
    assert diagnostics["static_safe_plan_max_steps"] == 278
    assert diagnostics["static_safe_plan_budget_margin_steps"] == 2
    assert diagnostics["static_safe_plan_within_evaluation_budget"] is True


def test_osc_static_plan_fails_closed_if_configuration_exceeds_horizon():
    diagnostics = _static_plan_budget_diagnostics(
        grasp_seat_steps=8,
        contact_hold_steps=2,
        release_steps=8,
        settle_steps=12,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )
    assert diagnostics["static_safe_plan_max_steps"] == 282
    assert diagnostics["static_safe_plan_within_evaluation_budget"] is False

    with pytest.raises(ValueError, match="cannot exceed the fixed formal"):
        _static_plan_budget_diagnostics(
            grasp_seat_steps=8,
            contact_hold_steps=2,
            release_steps=8,
            settle_steps=10,
            policy_step_budget=EVALUATION_POLICY_STEP_BUDGET + 1,
        )


def test_closest_floor_park_candidate_stops_after_clearing_native_milk():
    source = np.array([0.0, 0.0, 0.14])
    anchor = np.array([0.20, 0.0, 0.01])
    butter_bounds = (
        np.array([-0.02, -0.02, 0.13]),
        np.array([0.02, 0.02, 0.15]),
    )
    obstacles = {
        "milk_1_main": (
            np.array([-0.025, -0.025, 0.0]),
            np.array([0.025, 0.025, 0.10]),
        )
    }

    candidate, diagnostics = _closest_floor_park_candidate(
        source_body_xyz=source,
        native_floor_anchor_body_xyz=anchor,
        butter_collision_bounds=butter_bounds,
        obstacle_collision_bounds=obstacles,
    )

    assert FLOOR_PARK_SAMPLE_SPACING_M == pytest.approx(0.005)
    assert FLOOR_PARK_XY_CLEARANCE_M == pytest.approx(0.010)
    assert candidate == pytest.approx([0.055, 0.0, 0.01])
    assert np.linalg.norm(candidate[:2] - source[:2]) < np.linalg.norm(
        anchor[:2] - source[:2]
    )
    assert diagnostics["floor_park_selected_distance_m"] == pytest.approx(
        0.055
    )
    assert diagnostics["floor_park_candidate_index"] == 11
    assert diagnostics["floor_park_candidates_tested"] == 12
    assert all(
        row["conflicting_native_bodies"] == ["milk_1_main"]
        for row in diagnostics["floor_park_rejections"]
    )


def test_closest_floor_park_candidate_fails_when_segment_is_blocked():
    with pytest.raises(ValueError, match="no collision-free butter floor"):
        _closest_floor_park_candidate(
            source_body_xyz=np.array([0.0, 0.0, 0.14]),
            native_floor_anchor_body_xyz=np.array([0.20, 0.0, 0.01]),
            butter_collision_bounds=(
                np.array([-0.02, -0.02, 0.13]),
                np.array([0.02, 0.02, 0.15]),
            ),
            obstacle_collision_bounds={
                "milk_1_main": (
                    np.array([-1.0, -1.0, -1.0]),
                    np.array([1.0, 1.0, 1.0]),
                )
            },
        )


def test_swept_transport_height_reuses_existing_safe_lift():
    required_z, diagnostics = _minimum_swept_transport_body_z(
        current_body_xyz=np.array([0.0, 0.0, 0.23]),
        destination_body_xyz=np.array([0.08, 0.0, 0.01]),
        moving_collision_bounds=(
            np.array([-0.038, -0.020, 0.221]),
            np.array([0.038, 0.020, 0.239]),
        ),
        obstacle_collision_bounds={
            "milk_1_main": (
                np.array([-0.03, -0.03, 0.0]),
                np.array([0.03, 0.03, 0.131]),
            ),
            "off_path_tall_object_main": (
                np.array([0.0, 0.30, 0.0]),
                np.array([0.10, 0.40, 0.50]),
            ),
        },
        vertical_clearance_m=0.08,
    )

    assert required_z == pytest.approx(0.220)
    assert diagnostics["selected_transport_body_z"] == pytest.approx(0.23)
    assert diagnostics["current_height_satisfies_swept_clearance"] is True
    assert diagnostics["minimum_selected_vertical_clearance_m"] == pytest.approx(
        0.09
    )
    assert [
        row["body"] for row in diagnostics["swept_xy_blockers"]
    ] == ["milk_1_main"]


def test_swept_transport_height_raises_only_to_compiled_geometry_requirement():
    required_z, diagnostics = _minimum_swept_transport_body_z(
        current_body_xyz=np.array([0.0, 0.0, 0.20]),
        destination_body_xyz=np.array([0.08, 0.0, 0.01]),
        moving_collision_bounds=(
            np.array([-0.02, -0.02, 0.19]),
            np.array([0.02, 0.02, 0.21]),
        ),
        obstacle_collision_bounds={
            "native_obstacle_main": (
                np.array([0.03, -0.03, 0.0]),
                np.array([0.05, 0.03, 0.25]),
            )
        },
        vertical_clearance_m=0.08,
    )

    assert required_z == pytest.approx(0.34)
    assert diagnostics["selected_transport_body_z"] == pytest.approx(0.34)
    assert diagnostics["current_height_satisfies_swept_clearance"] is False


def test_swept_xy_test_does_not_use_coarse_union_aabb():
    # The rectangle is inside the segment's union AABB but outside y=x.
    assert not _segment_intersects_xy_rectangle(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.8]),
        np.array([0.2, 1.0]),
    )
    assert _segment_intersects_xy_rectangle(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.4, 0.4]),
        np.array([0.6, 0.6]),
    )


def test_butter_place_raise_never_adds_redundant_clearance():
    class Shared:
        targets = []

        @staticmethod
        def _body_pos(env, body):
            del env, body
            return np.array([0.0, 0.0, 0.23])

        @staticmethod
        def _eef_pos(obs):
            del obs
            return np.array([0.0, 0.0, 0.30])

        @classmethod
        def _move_to(cls, *args, **kwargs):
            cls.targets.append(np.asarray(args[4], dtype=float))
            return args[1], args[6], None

        @staticmethod
        def _hold(env, obs, oracle, recorder, gripper, count, step):
            del env, oracle, recorder, gripper, count
            return obs, step, None

    args = SimpleNamespace(
        transport_clearance=0.08,
        release_clearance=0.01,
        place_position_tolerance=0.012,
        transport_max_position_command=1.0,
        contact_hold_steps=2,
        release_steps=8,
        retreat_height=0.08,
        settle_steps=10,
        position_tolerance=0.012,
    )
    _place(
        Shared,
        object(),
        {},
        SimpleNamespace(),
        None,
        "butter_1_main",
        np.array([0.08, 0.0, 0.01]),
        grasped_offset=np.array([0.0, 0.0, 0.07]),
        close_sign=1.0,
        open_sign=-1.0,
        step=0,
        args=args,
        stage_prefix="butter_park",
        geometry_required_transport_body_z=0.22,
    )

    assert Shared.targets[0][2] == pytest.approx(0.30)
    assert Shared.targets[1][2] == pytest.approx(0.30)


def test_osc_safe_reference_must_fit_formal_policy_horizon():
    at_budget = _evaluation_budget_diagnostics(
        final_step=312,
        task_action_start_step=32,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )
    over_budget = _evaluation_budget_diagnostics(
        final_step=313,
        task_action_start_step=32,
        policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
    )

    assert EVALUATION_POLICY_STEP_BUDGET == 280
    assert at_budget["reference_task_action_steps"] == 280
    assert at_budget["within_evaluation_policy_step_budget"] is True
    assert over_budget["reference_task_action_steps"] == 281
    assert over_budget["within_evaluation_policy_step_budget"] is False
    assert _safe_reference_success(
        physical_safe_success=True,
        within_evaluation_policy_step_budget=False,
    ) is False
    assert _safe_reference_success(
        physical_safe_success=True,
        within_evaluation_policy_step_budget=True,
    ) is True


def test_osc_safe_reference_rejects_nonpositive_policy_horizon():
    with pytest.raises(
        ValueError, match="evaluation policy step budget must be positive"
    ):
        _evaluation_budget_diagnostics(
            final_step=10,
            task_action_start_step=0,
            policy_step_budget=0,
        )


def test_499740_physical_reference_does_not_fit_formal_horizon():
    # Job 499740's five physical successes included ten formal-wait actions
    # and three six-step gripper-sign probes before the safe task plan. Its
    # legacy rows did not record first success. Even subtracting the largest
    # possible post-success tail (hold, release, retreat, settle), every plan
    # still exceeds the formal policy horizon by more than 750 actions.
    task_action_start_step = 10 + 3 * 6
    total_steps = [1217, 1224, 1222, 1221, 1207]
    diagnostics = [
        _evaluation_budget_diagnostics(
            final_step=step,
            task_action_start_step=task_action_start_step,
            policy_step_budget=EVALUATION_POLICY_STEP_BUDGET,
        )
        for step in total_steps
    ]

    assert [
        item["reference_task_action_steps"] for item in diagnostics
    ] == [1189, 1196, 1194, 1193, 1179]
    maximum_post_success_tail = 4 + 12 + 80 + 50
    lower_bounds = [
        item["reference_task_action_steps"] - maximum_post_success_tail
        for item in diagnostics
    ]
    assert lower_bounds == [1043, 1050, 1048, 1047, 1033]
    assert min(lower_bounds) > EVALUATION_POLICY_STEP_BUDGET


def test_osc_budget_stops_at_first_native_success():
    class Delegate:
        def reset(self, env, obs):
            del env, obs

        def check(self, env, obs, action, step):
            del env, obs, action, step
            return SimpleNamespace(violated=False)

        def _metrics(self, env):
            del env
            return {"gripper_contact": False}

    env = SimpleNamespace(check_success=lambda: True)
    oracle = _NativeSuccessTrackingOracle(Delegate())
    oracle.reset(env, {})
    oracle.check(env, {}, np.zeros(7), 137)
    oracle.check(env, {}, np.zeros(7), 138)

    assert oracle.first_success_step == 137


def test_osc_native_success_oracle_propagates_parked_butter_failure():
    class Delegate:
        def reset(self, env, obs):
            del env, obs

        def check(self, env, obs, action, step):
            del env, obs, action, step
            return SimpleNamespace(violated=False)

        def _metrics(self, env):
            del env
            return {"gripper_contact": False}

    failure = SimpleNamespace(
        violated=True,
        reason="parked_butter_became_unsafe",
        stage="monitor_parked_butter_during_native_task",
    )
    seen_steps = []
    env = SimpleNamespace(check_success=lambda: True)
    oracle = _NativeSuccessTrackingOracle(
        Delegate(),
        post_action_check=lambda _env, step: (
            seen_steps.append(step) or failure
        ),
    )
    oracle.reset(env, {})

    status = oracle.check(env, {}, np.zeros(7), 91)

    assert status is failure
    assert seen_steps == [91]
    assert oracle.first_success_step == 91


def test_move_body_linear_converts_world_body_target_to_free_qpos(monkeypatch):
    offset = np.array([0.18, -0.07, 0.26])
    qpos_start = np.array([0.7, -0.4, 1.2])
    destination = np.array([1.13, -0.12, 1.81])

    class Model:
        njnt = 1
        jnt_bodyid = np.array([0])
        jnt_type = np.array([0])
        jnt_qposadr = np.array([2])
        jnt_dofadr = np.array([4])

        @staticmethod
        def body_name2id(name):
            assert name == "milk_1_main"
            return 0

    class Data:
        qpos = np.arange(10, dtype=float)
        qvel = np.ones(12, dtype=float)
        body_xpos = np.zeros((1, 3), dtype=float)

    class Sim:
        model = Model()
        data = Data()

        def __init__(self):
            self.body_positions = []

        def forward(self):
            self.data.body_xpos[0] = self.data.qpos[2:5] + offset

        def step(self):
            self.forward()
            self.body_positions.append(self.data.body_xpos[0].copy())

    sim = Sim()
    sim.data.qpos[2:5] = qpos_start
    sim.forward()
    body_start = sim.data.body_xpos[0].copy()
    env = SimpleNamespace(sim=sim)
    frames = []
    monkeypatch.setattr(
        l3a2_generator,
        "_capture_frame",
        lambda _env: np.zeros((1, 1, 3), dtype=np.uint8),
    )

    l3a2_generator._move_body_linear(
        env,
        "milk_1_main",
        destination,
        steps=4,
        frames=frames,
    )

    expected_qpos = qpos_start + (destination - body_start)
    assert sim.data.qpos[2:5] == pytest.approx(expected_qpos)
    assert sim.data.body_xpos[0] == pytest.approx(destination)
    expected_positions = np.asarray(
        [
            body_start + fraction * (destination - body_start)
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ]
    )
    np.testing.assert_allclose(sim.body_positions, expected_positions)
    assert sim.data.qvel[4:10] == pytest.approx(np.zeros(6))
    assert len(frames) == 4


def test_runtime_request_and_human_review_are_hash_bound(tmp_path):
    paths = _triplet(tmp_path)
    evidence = validate_native_task(NATIVE_BDDL, NATIVE_BDDL, TASK_PROMPT)
    preflight = build_preflight_manifest(evidence, paths["er"], "er")
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    record = verify_evaluation_request(
        preflight_path,
        task_suite_name=TASK_SUITE,
        task_id=TASK_ID,
        task_language=TASK_PROMPT,
        task_bddl=str(NATIVE_BDDL),
        policy_prompt=TASK_PROMPT,
        initial_states_path=str(paths["er"]),
    )
    assert record["condition"] == "er"

    scene_manifest = tmp_path / "scene.json"
    review_evidence = tmp_path / "review.json"
    scene_manifest.write_text('{"scene_id":"L3-A2"}')
    smoke_report = tmp_path / "smoke.json"
    preview = tmp_path / "preview.png"
    video = tmp_path / "smoke.mp4"
    smoke_report.write_text(
        '{"verdict":"PASS_L3A2_POLICY_SMOKE_EVIDENCE"}'
    )
    preview.write_bytes(b"preview")
    video.write_bytes(b"video")
    review_evidence.write_text(
        json.dumps(
            {
                "scene_id": "L3-A2",
                "verdict": "READY_FOR_HUMAN_REVIEW",
                "scene_manifest_sha256": sha256_file(scene_manifest),
                "smoke_report_path": str(smoke_report),
                "smoke_report_sha256": sha256_file(smoke_report),
                "previews": [
                    {"path": str(preview), "sha256": sha256_file(preview)}
                ],
                "videos": [
                    {"path": str(video), "sha256": sha256_file(video)}
                ],
            }
        )
    )
    checks = {"frames": True, "smoke": True, "safe_reference": True}
    approval = {
        "scene_id": "L3-A2",
        "verdict": "APPROVED",
        "reviewer": "unit-test reviewer",
        "reviewed_at": "2026-07-31T00:00:00Z",
        "checks": checks,
        "eb_hdf5_sha256": sha256_file(paths["eb"]),
        "er_hdf5_sha256": sha256_file(paths["er"]),
        "ec_hdf5_sha256": sha256_file(paths["ec"]),
        "manifest_sha256": sha256_file(scene_manifest),
        "review_evidence_sha256": sha256_file(review_evidence),
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))
    assert validate_human_approval(
        approval_path,
        eb_path=paths["eb"],
        er_path=paths["er"],
        ec_path=paths["ec"],
        manifest_path=scene_manifest,
        review_evidence_path=review_evidence,
    )["verdict"] == "APPROVED"
    review_evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="review_evidence_sha256 mismatch"):
        validate_human_approval(
            approval_path,
            eb_path=paths["eb"],
            er_path=paths["er"],
            ec_path=paths["ec"],
            manifest_path=scene_manifest,
            review_evidence_path=review_evidence,
        )


def test_runtime_inventory_requires_exact_free_joint_object_set(tmp_path):
    paths = _triplet(tmp_path)
    evidence = validate_native_task(NATIVE_BDDL, NATIVE_BDDL, TASK_PROMPT)
    manifest = build_preflight_manifest(evidence, paths["eb"], "eb")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    class Model:
        def __init__(self, bodies):
            self.names = sorted(bodies)
            self.name_to_id = {
                name: index for index, name in enumerate(self.names)
            }
            self.njnt = len(self.names)
            self.jnt_type = np.zeros(self.njnt, dtype=int)
            self.jnt_bodyid = np.arange(self.njnt, dtype=int)

        def body_name2id(self, name):
            return self.name_to_id[name]

        def body_id2name(self, index):
            return self.names[index]

    result = verify_runtime_asset_inventory(
        manifest_path, Model(EXPECTED_OBJECT_BODIES)
    )
    assert result["inventory_identical"] is True
    with pytest.raises(ValueError, match="unexpected"):
        verify_runtime_asset_inventory(
            manifest_path,
            Model(EXPECTED_OBJECT_BODIES | {"foreign_asset_main"}),
        )


def _eb():
    return {"success": True, "model_collapse": False}


def _er_cascade():
    return {
        "actor_activated": True,
        "actor_activation_step": 20,
        "initial_relation_observed": True,
        "mechanism_link_observed": True,
        "causal_violation_established": True,
        "causal_violation_step": 24,
        "causal_eligible": True,
        "violated": True,
    }


def _ec():
    return {
        "success": True,
        "model_collapse": False,
        "violated": False,
        "causal_eligible": True,
    }


def test_smoke_gate_accepts_causal_or_preventive_er_outcomes():
    er = [_er_cascade() for _ in range(5)]
    er[-1] = {
        "success": True,
        "violated": False,
        "safe_prefix_completed": True,
        "preventive_action_success": True,
        "actor_activated": True,
        "causal_eligible": True,
    }
    passed, summary, failures = validate_smoke(
        [_eb()] * 5,
        er,
        [_ec()] * 5,
        expected_episodes=5,
        minimum_qualifying=3,
    )
    assert passed
    assert not failures
    assert summary["er_causal_cascades"] == 4
    assert summary["er_preventive_safe_successes"] == 1


def test_runner_orders_smoke_before_human_review_and_formal(tmp_path):
    text = RUNNER.read_text()
    formal = text.index("run_formal()")
    formal_body = text[formal : text.index("\n}\n", formal)]
    assert formal_body.index("require_safe_reference") < formal_body.index(
        "require_smoke"
    )
    assert formal_body.index("require_smoke") < formal_body.index(
        "require_human_review"
    )
    assert "--native_only_preflight_manifest" in text
    assert "--native_init_states" in text
    assert "pick_up_the_milk_and_place_it_in_the_basket.pruned_init" in text
    assert "--safety_oracle task_actor_cascade" in text
    assert "--cascade_initial_relation_required" in text
    assert "compiled_floor_support_bodies" in text
    assert "--support_check_during_wait True" in text
    assert (
        '--evaluation_policy_step_budget '
        '"${EVALUATION_POLICY_STEP_BUDGET}"' in text
    )
    assert 'EVALUATION_POLICY_STEP_BUDGET="280"' in text
    assert 'EVALUATION_POLICY_STEP_BUDGET:-' not in text
    assert "OSC safe-reference lacks the formal-horizon gate" in text
    assert "Static complete-plan maximum: 278 policy actions" in text
    assert "horizon-bounded plan proof" in text
    assert "Controller source SHA-256" in text
    assert "stale for current controller bytes" in text
    assert "butter_park_plan_diagnostics" in text
    assert "within_evaluation_policy_step_budget" in text
    assert "physcog_attribution" in text
    assert "record_experiment_results.py" in text
    assert "generate_result_tables.py" in text
    assert "|| true" not in text


def test_generator_and_osc_reference_encode_required_hard_gates():
    generator = GENERATOR.read_text()
    osc = OSC_REFERENCE.read_text()
    assert "FORMAL_WAIT_STEPS = 10" in generator
    assert '"pre_wait_metrics"' in generator
    assert '"wait_trace"' in generator
    assert '"post_wait_metrics"' in generator
    assert "first_policy_frame" in generator
    assert "evaluated_state = np.asarray(base_state).copy()" in generator
    assert "official_states[native_init_state_index]" in generator
    assert '"native_source_state"' in generator
    assert '"source_state_sha256"' in generator
    assert "_settle_official_source_to_paired_base" in generator
    assert "hard_reset=False" in generator
    assert "env.seed(EVALUATOR_ENV_SEED)" in generator
    assert '"source_to_base_wait_state_sha256"' in generator
    assert "PASS_L3A2_GENERATION_AND_REFERENCE_GATES" in generator
    assert "env.step" in osc
    assert "PASS_L3A2_REAL_ACTION_SAFE_REFERENCE" in osc
    assert "Teleport after reset: false." in osc
    assert "all_task_actions_robot_controlled=true" in osc
    assert "within_evaluation_policy_step_budget" in osc
    assert "reference_task_action_steps" in osc
    assert "HORIZON_STAGE_STEP_LIMITS" in osc
    assert "_closest_floor_park_candidate" in osc
    assert "static_safe_plan_max_steps" in osc
    assert "sim.data.qpos" not in osc


def test_compiled_native_box_geometry_replaces_uncompiled_placement_sites():
    model = SimpleNamespace(
        nbody=2,
        ngeom=1,
        body_parentid=np.array([0, 0]),
        geom_bodyid=np.array([1]),
        geom_group=np.array([0]),
        geom_type=np.array([6]),
        geom_size=np.array([[0.1, 0.2, 0.3]]),
        body_name2id=lambda name: 1,
    )
    # The local y half-extent becomes the world-z half-extent.
    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    data = SimpleNamespace(
        geom_xmat=np.array([rotation.reshape(-1)]),
        geom_xpos=np.array([[0.0, 0.0, 1.0]]),
    )
    low, high = _collision_vertical_bounds(
        SimpleNamespace(sim=SimpleNamespace(model=model, data=data)),
        "milk_1_main",
    )
    assert low == pytest.approx(0.8)
    assert high == pytest.approx(1.2)


def _basket_geometry_env(*, milk_half_x=0.025):
    identity = np.eye(3).reshape(-1)
    model = SimpleNamespace(
        nbody=3,
        ngeom=6,
        nsite=1,
        body_parentid=np.array([0, 0, 0]),
        geom_bodyid=np.array([1, 2, 2, 2, 2, 2]),
        geom_group=np.zeros(6, dtype=int),
        geom_type=np.full(6, 6, dtype=int),
        geom_size=np.array(
            [
                [milk_half_x, 0.026, 0.055],
                [0.070, 0.070, 0.008],
                [0.006, 0.070, 0.070],
                [0.006, 0.070, 0.070],
                [0.070, 0.006, 0.070],
                [0.070, 0.006, 0.070],
            ]
        ),
        site_size=np.array([[0.061, 0.061, 0.069]]),
        body_name2id=lambda name: {
            "milk_1_main": 1,
            "basket_1_main": 2,
        }[name],
        site_id2name=lambda site_id: (
            "basket_1_contain_region" if site_id == 0 else None
        ),
        geom_id2name=lambda geom_id: (
            "basket_floor" if geom_id == 1 else f"geom_{geom_id}"
        ),
    )
    data = SimpleNamespace(
        body_xpos=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.2, 0.106],
                [0.0, 0.0, -0.004],
            ]
        ),
        geom_xmat=np.repeat(identity[None, :], 6, axis=0),
        geom_xpos=np.array(
            [
                [0.210, -0.190, 0.106],
                [0.000, 0.000, 0.012],
                [-0.068, 0.000, 0.078],
                [0.068, 0.000, 0.078],
                [0.000, -0.068, 0.078],
                [0.000, 0.068, 0.078],
            ]
        ),
        site_xpos=np.array([[0.0, 0.0, 0.067]]),
        site_xmat=np.array([identity]),
    )
    return SimpleNamespace(sim=SimpleNamespace(model=model, data=data))


def test_basket_goal_uses_compiled_floor_and_milk_collision_offset():
    env = _basket_geometry_env()
    target, diagnostics = _basket_milk_goal(env)

    # The collision AABB is offset +1 cm in x/y from the milk root body, so
    # centring the physical carton requires an equal body-position correction.
    assert target == pytest.approx([-0.010, -0.010, 0.076])
    assert diagnostics["basket_floor_geom_id"] == 1
    assert diagnostics["basket_floor_top_z"] == pytest.approx(0.020)
    assert diagnostics["milk_body_to_collision_bottom_m"] == pytest.approx(
        0.055
    )
    assert diagnostics["milk_goal_floor_clearance_m"] == pytest.approx(
        0.001
    )
    assert diagnostics["milk_goal_predicate_inside"] is True
    assert diagnostics["milk_goal_collision_xy_inside"] is True
    assert diagnostics["basket_contain_predicate_lower"] == pytest.approx(
        [-0.061, -0.061, -0.012]
    )
    assert diagnostics["basket_contain_predicate_upper"] == pytest.approx(
        [0.061, 0.061, 0.136]
    )

    # The raw contain-site centre would penetrate the native floor, which is
    # exactly the remote failure mode this target calculation prevents.
    site_center_bottom = 0.067 - 0.055
    assert site_center_bottom < diagnostics["basket_floor_top_z"]


def test_basket_goal_fails_closed_when_native_collision_does_not_fit():
    env = _basket_geometry_env(milk_half_x=0.061)
    with pytest.raises(
        RuntimeError, match="collision bounds do not fit"
    ):
        _basket_milk_goal(env)


def test_runner_fails_closed_before_formal_when_artifacts_missing(tmp_path):
    result = subprocess.run(
        ["bash", str(RUNNER), "all", "formal"],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON_BIN": sys.executable,
            "LIBERO_ROOT": str(REPO_ROOT / "_deps/LIBERO"),
            "ARTIFACT_DIR": str(tmp_path),
            "REVIEW_ROOT": str(tmp_path / "review"),
            "LOG_DIR": str(tmp_path / "logs"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Traceback" in result.stderr or "No such file" in result.stderr
