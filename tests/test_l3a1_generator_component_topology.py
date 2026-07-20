import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / (
    "experiments/robot/libero/tasks/"
    "generate_l3a1_drawer_bottle_initial_states.py"
)
SOURCE = GENERATOR.read_text()
TREE = ast.parse(SOURCE)


def _function_source(name: str) -> str:
    node = next(
        item for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node)


def _load_pure_function(name: str, namespace=None):
    node = next(
        item for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(GENERATOR), "exec"), namespace)
    return namespace[name]


def test_permanent_release_uses_last_component_contact_not_first_gap():
    release_step = _load_pure_function("_permanent_component_release_step")
    chatter = [
        {"step": 1, "roles": {"edge/front_outer"}},
        {"step": 2, "roles": set()},
        {"step": 3, "roles": set()},
        {"step": 4, "roles": {"edge/front_outer"}},
        {"step": 5, "roles": {"side/right"}},
        {"step": 6, "roles": set()},
    ]
    assert release_step(chatter) == 6
    assert release_step(chatter[:-1]) == -1


def test_factual_close_never_writes_bottle_velocity_and_zero_replay_is_explicit():
    source = _function_source("_component_close_response")
    assert "data.qvel[bottle_vadr:bottle_vadr + 6] = 0" in source
    assert "zero_momentum_at_release" in source
    assert "step == permanent_release_step_rC" in source
    assert "data.qvel[:]" not in source
    assert "bottle_qvel_overwritten" in source


def test_instant_removal_masks_and_restores_every_component_geom():
    source = _function_source("_instant_component_removal_response")
    assert "component_ids = [" in source
    assert "for geom_id in component_ids:" in source
    assert "for geom_id, (contype, conaffinity) in zip(component_ids, masks):" in source
    assert "model.geom_contype[geom_id] = contype" in source
    assert "model.geom_conaffinity[geom_id] = conaffinity" in source


def test_hazard_oracle_is_displacement_drop_or_attitude_change():
    for name in (
        "_component_close_response",
        "_instant_component_removal_response",
    ):
        source = _function_source(name)
        assert "displacement > oracle_displacement_threshold" in source
        assert "or height_drop > oracle_height_drop_threshold" in source
        assert "or attitude_change > oracle_tilt_change_threshold_deg" in source


def test_schema_v2_records_full_topology_and_rejects_panel_artifact_fields():
    assert 'choices=("right",)' in SOURCE
    assert '"edge/front_outer"' in SOURCE
    assert '"inner_front"' in SOURCE
    assert '"side/right"' in SOURCE
    assert '"l3a1_topology_schema_version"' in SOURCE
    assert '"compiled_support_component_signatures_json"' in SOURCE
    assert '"support_component_C_geoms"' in SOURCE
    assert '"support_qualification_algorithm_version"' in SOURCE
    assert '"compiled_support_panel_signature_json"' not in SOURCE
    assert '"support_panel_collision_geom"' not in SOURCE


def test_calibrated_defaults_and_force_qualification_are_bound():
    assert "DEFAULT_LEAN_DX = 0.147925" in SOURCE
    assert "DEFAULT_LEAN_DY = -0.060125" in SOURCE
    assert "DEFAULT_LEAN_DEG = -40.0" in SOURCE
    assert "DEFAULT_LEAN_DIRECTION_DEG = 105.0" in SOURCE
    assert "DEFAULT_STABLE_X_OFFSET = 0.00" in SOURCE
    assert "else DEFAULT_STABLE_X_OFFSET" in SOURCE
    assert "preserving paired" in SOURCE
    assert "outside-C absence gates" in SOURCE
    assert "MAX_EDGE_GAP_M = 0.006" in SOURCE
    assert "MIN_EDGE_FORCE_WEIGHT_FRACTION = 0.05" in SOURCE
    assert "MIN_TABLE_FORCE_WEIGHT_FRACTION = 0.25" in SOURCE
    assert "MIN_ABSOLUTE_FORCE_N = 1e-4" in SOURCE
    assert "mujoco.mj_contactForce" in SOURCE


def test_verified_natural_release_angular_speed_boundary_is_accepted():
    limits = {
        "MAX_PRE_RELEASE_DRAWER_AXIS_DISPLACEMENT_M": 0.002,
        "MAX_PRE_RELEASE_DRAWER_AXIS_SPEED_M_S": 0.02,
        "MAX_PRE_RELEASE_TOTAL_DISPLACEMENT_M": 0.002,
        "MAX_PRE_RELEASE_TILT_DELTA_DEG": 1.0,
        "MAX_PRE_RELEASE_ANGULAR_SPEED_RAD_S": 1.5,
    }
    acceptable = _load_pure_function(
        "_pre_release_motion_is_acceptable", limits
    )
    response = {
        "max_pre_release_drawer_axis_displacement_m": 0.0001,
        "max_pre_release_drawer_axis_speed_m_s": 0.0023,
        "max_pre_release_total_displacement_m": 0.0001,
        "max_pre_release_tilt_delta_deg": 0.31,
        "max_pre_release_angular_speed_rad_s": 1.5,
    }
    assert acceptable(response)
    response["max_pre_release_angular_speed_rad_s"] = 1.500001
    assert not acceptable(response)
    assert "MAX_PRE_RELEASE_DRAWER_AXIS_DISPLACEMENT_M = 0.002" in SOURCE
    assert "MAX_PRE_RELEASE_DRAWER_AXIS_SPEED_M_S = 0.02" in SOURCE
    assert "MAX_PRE_RELEASE_TOTAL_DISPLACEMENT_M = 0.002" in SOURCE
    assert "MAX_PRE_RELEASE_TILT_DELTA_DEG = 1.0" in SOURCE


def test_counterfactual_contacts_are_split_at_oracle_for_formal_gates():
    close = _function_source("_component_close_response")
    instant = _function_source("_instant_component_removal_response")
    for source in (close, instant):
        assert "pre_oracle_direct_contact_bodies" in source
        assert "post_oracle_direct_contact_bodies" in source
        assert "pre_oracle_other_cabinet_geoms" in source
        assert "post_oracle_other_cabinet_geoms" in source
    assert 'close_response["pre_oracle_direct_contact_bodies"]' in SOURCE
    assert 'close_response["post_oracle_direct_contact_bodies"]' not in SOURCE.split(
        'validation_records.append(', 1
    )[0]
    assert '"factual_close_pre_oracle_direct_contact_bodies"' in SOURCE
    assert '"factual_close_post_oracle_direct_contact_bodies"' in SOURCE
    assert '"instant_component_removal_pre_oracle_direct_contact_bodies"' in SOURCE
    assert '"instant_component_removal_post_oracle_direct_contact_bodies"' in SOURCE
    assert '"factual_close_direct_contact_bodies"' not in SOURCE
    assert '"zero_momentum_close_direct_contact_bodies"' not in SOURCE
    assert '"instant_component_removal_direct_contact_bodies"' not in SOURCE
