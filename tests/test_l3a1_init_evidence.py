from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.tasks.export_l3a1_init_evidence import (
    array_sha256,
    contact_report,
    policy_agentview,
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
