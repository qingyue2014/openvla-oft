from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks.export_l3a1_init_evidence import (
    array_sha256,
    contact_report,
    policy_agentview,
    validate_capture_topology,
)


def test_policy_agentview_matches_openvla_rotation():
    raw = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    actual = policy_agentview({"agentview_image": raw})
    np.testing.assert_array_equal(actual, raw[::-1, ::-1])
    assert actual.flags.c_contiguous


def test_state_hash_is_deterministic_and_byte_sensitive():
    state = np.arange(12, dtype=np.float64)
    assert array_sha256(state) == array_sha256(state.copy())
    changed = state.copy()
    changed[-1] += 1
    assert array_sha256(state) != array_sha256(changed)


def test_contact_report_includes_descendant_geom_contact():
    class Model:
        nbody = 4
        body_parentid = np.array([0, 0, 1, 0])
        geom_bodyid = np.array([2, 3])

        @staticmethod
        def body_name2id(name):
            if name == "tracked":
                return 1
            raise KeyError(name)

        @staticmethod
        def body_id2name(index):
            return ("world", "tracked", "tracked_child", "table")[index]

        @staticmethod
        def geom_id2name(index):
            return ("tracked_geom", "table_geom")[index]

    data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=1, dist=-0.001)])
    env = SimpleNamespace(sim=SimpleNamespace(model=Model(), data=data))
    row = contact_report(env, ("tracked",))[0]
    assert row["body1"] == "tracked_child"
    assert row["body2"] == "table"
    assert row["tracked_bodies"] == ["tracked"]


def test_exact_capture_topology_requires_qualified_edge_and_table():
    topology = {
        "support_edge_geom": "g33", "support_inner_front_geom": "g35",
        "support_side_geom": "g36", "min_absolute_support_force_n": 1e-4,
        "min_edge_force_weight_fraction": 0.05,
        "min_table_force_weight_fraction": 0.25,
        "max_support_penetration_m": 0.003, "max_edge_gap_m": 0.005,
        "min_edge_axial_m": 0.01,
    }
    common = {
        "tracked_bodies": ["wine_bottle_1_main"], "bottle_weight_n": 10.0,
        "penetration_m": 0.001, "world_contact_xyz_m": [0.0, 0.0, 0.0],
        "drawer_local_contact_xyz_m": [0.0, 0.0, 0.0],
        "normal_force_weight_fraction": 0.3, "bottle_axis_axial_m": 0.02,
        "bottle_axis_fraction": 0.7,
    }
    row = {"contacts": [
        {**common, "geom1": "bottle", "geom2": "g33", "normal_force_n": 0.6,
         "edge_gap_m": 0.004, "bottle_axis_axial_m": 0.02},
        {**common, "geom1": "bottle", "geom2": "table_collision", "normal_force_n": 2.6},
    ]}
    validate_capture_topology(row, "Er", topology)
    assert row["contacts"][0]["qualified_edge_witness"]
    assert row["contacts"][1]["qualified_table_witness"]

    contaminated = {"contacts": row["contacts"] + [
        {**common, "geom1": "bottle", "geom2": "g35", "normal_force_n": 1.0}
    ]}
    with np.testing.assert_raises_regex(RuntimeError, "g35/g36"):
        validate_capture_topology(contaminated, "Er", topology)
    with np.testing.assert_raises_regex(RuntimeError, "component C"):
        validate_capture_topology(contaminated, "Ec", topology)
