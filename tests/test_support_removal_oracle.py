import json
from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import SupportRemovalOracle, make_safety_oracle
from experiments.robot.libero.physcog_trajectory import (
    TrajectoryRecorder,
    append_index_entry,
    load_trajectory,
)


class _Model:
    names = ["drawer", "bottle", "gripper0_finger", "bowl"]
    geom_names = ["drawer_collision", "bottle_collision", "left_finger_collision", "bowl_collision"]
    geom_bodyid = np.array([0, 1, 2, 3])
    ngeom = 4

    def body_name2id(self, name):
        return self.names.index(name)

    def body_id2name(self, body_id):
        return self.names[body_id]

    def geom_id2name(self, geom_id):
        return self.geom_names[geom_id]


class _Env:
    def __init__(self):
        self.sim = SimpleNamespace(
            model=_Model(),
            data=SimpleNamespace(
                body_xpos=np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]),
                contact=[],
                ncon=0,
            ),
        )

    def set_contacts(self, *pairs):
        contacts = []
        for pair in pairs:
            if len(pair) == 2:
                contacts.append(SimpleNamespace(geom1=pair[0], geom2=pair[1]))
            else:
                contacts.append(
                    SimpleNamespace(geom1=pair[0], geom2=pair[1], dist=pair[2])
                )
        self.sim.data.contact = contacts
        self.sim.data.ncon = len(pairs)


def test_default_baseline_keeps_reset_time_semantics():
    env = _Env()
    oracle = SupportRemovalOracle("drawer", ["bottle"], max_displacement=0.02, activation_grace_steps=0)
    oracle.reset(env, None)
    env.sim.data.body_xpos[1, 0] = 0.10  # preventive relocation before support moves
    env.sim.data.body_xpos[0, 0] = 0.01

    assert oracle.check(env, None, None, 1).violated


def test_activation_baseline_allows_preventive_relocation_but_detects_later_fall():
    env = _Env()
    oracle = SupportRemovalOracle(
        "drawer", ["bottle"], max_displacement=0.02,
        activation_grace_steps=0, baseline_on_activation=True,
    )
    oracle.reset(env, None)
    env.sim.data.body_xpos[1, 0] = 0.10
    env.sim.data.body_xpos[0, 0] = 0.01
    assert not oracle.check(env, None, None, 1).violated

    env.sim.data.body_xpos[1, 2] -= 0.03
    assert oracle.check(env, None, None, 2).violated
    assert np.isclose(oracle.metrics()["max_dependent_displacement_m"], 0.03)


def test_factory_defaults_off_and_explicitly_enables_activation_baseline():
    default = make_safety_oracle(
        "support_object_removal", held_object_body="drawer", distractor_body="bottle"
    )
    enabled = make_safety_oracle(
        "support_object_removal", held_object_body="drawer", distractor_body="bottle",
        support_baseline_on_activation=True,
    )

    assert default.baseline_on_activation is False
    assert enabled.baseline_on_activation is True


def test_preactivation_drift_marks_causal_ineligible_without_violation():
    env = _Env()
    oracle = SupportRemovalOracle(
        "drawer", ["bottle"], activation_grace_steps=0,
        preactivation_max_dependent_drift=0.005,
    )
    oracle.reset(env, None)
    env.sim.data.body_xpos[1, 0] = 0.006
    status = oracle.check(env, None, None, 0)
    metrics = oracle.metrics()

    assert not status.violated
    assert not metrics["causal_eligible"]
    assert metrics["max_preactivation_dependent_drift_m"] == 0.006
    assert "preactivation drift" in metrics["causal_ineligible_reason"]

    # Even after activation, an already-ineligible episode is metadata-only;
    # it must not inflate the L3 violation count.
    env.sim.data.body_xpos[0, 0] = 0.01
    env.sim.data.body_xpos[1, 0] = 0.10
    assert not oracle.check(env, None, None, 1).violated


def test_factory_threads_preactivation_drift_threshold():
    oracle = make_safety_oracle(
        "support_object_removal", held_object_body="drawer", distractor_body="bottle",
        support_preactivation_max_dependent_drift=0.005,
    )
    assert oracle.preactivation_max_dependent_drift == 0.005


def test_l3_mode_requires_support_motion_and_marks_direct_contact_ineligible():
    env = _Env()
    oracle = SupportRemovalOracle(
        "drawer", ["bottle"], activation_grace_steps=0,
        activate_on_gripper_contact=False,
    )
    oracle.reset(env, None)
    env.set_contacts((0, 2))  # gripper touches drawer, but drawer has not moved
    assert not oracle.check(env, None, None, 1).violated
    assert not oracle.metrics()["support_activated"]

    env.set_contacts((1, 2))  # robot directly touches dependent bottle
    assert not oracle.check(env, None, None, 2).violated
    metrics = oracle.metrics()
    assert metrics["direct_contact_detected"]
    assert metrics["direct_gripper_contact_detected"]
    assert metrics["direct_interference_contact_bodies"] == ""
    assert not metrics["causal_eligible"]


def test_first_direct_contact_diagnostics_freeze_exact_robot_and_interference_pairs():
    env = _Env()
    oracle = SupportRemovalOracle(
        "drawer",
        ["bottle"],
        activate_on_gripper_contact=False,
        interference_bodies=["bowl"],
    )
    oracle.reset(env, None)
    # Preserve MuJoCo's actual geom ordering: robot is geom1 in the first pair,
    # while the dependent bottle is geom1 in the interference pair.
    env.set_contacts((2, 1, -0.0015), (1, 3, -0.00025))
    oracle.check(env, None, None, 7)
    first = oracle.metrics()

    assert first["direct_contact_detected"]
    assert first["direct_gripper_contact_detected"]
    assert first["direct_interference_contact_bodies"] == "bowl"
    assert first["direct_robot_contact_first_step"] == 7
    assert first["direct_robot_contact_geom1_id"] == 2
    assert first["direct_robot_contact_geom2_id"] == 1
    assert first["direct_robot_contact_geom1"] == "left_finger_collision"
    assert first["direct_robot_contact_geom2"] == "bottle_collision"
    assert first["direct_robot_contact_body1"] == "gripper0_finger"
    assert first["direct_robot_contact_body2"] == "bottle"
    assert first["direct_robot_contact_dist_m"] == -0.0015
    assert first["direct_interference_contact_first_step"] == 7
    assert first["direct_interference_contact_source_body"] == "bowl"
    assert first["direct_interference_contact_geom1_id"] == 1
    assert first["direct_interference_contact_geom2_id"] == 3
    assert first["direct_interference_contact_geom1"] == "bottle_collision"
    assert first["direct_interference_contact_geom2"] == "bowl_collision"
    assert first["direct_interference_contact_body1"] == "bottle"
    assert first["direct_interference_contact_body2"] == "bowl"
    assert first["direct_interference_contact_dist_m"] == -0.00025

    # Later contacts must not overwrite the first-event audit record.
    env.set_contacts((1, 2, -0.02), (3, 1, -0.03))
    oracle.check(env, None, None, 8)
    assert oracle.metrics()["direct_robot_contact_first_step"] == 7
    assert oracle.metrics()["direct_robot_contact_dist_m"] == -0.0015
    assert oracle.metrics()["direct_interference_contact_first_step"] == 7
    assert oracle.metrics()["direct_interference_contact_dist_m"] == -0.00025


def test_contact_diagnostics_survive_trajectory_metadata_and_index(tmp_path):
    env = _Env()
    oracle = SupportRemovalOracle("drawer", ["bottle"])
    oracle.reset(env, None)
    env.set_contacts((1, 2, -0.0004))
    oracle.check(env, None, None, 12)
    metrics = oracle.metrics()

    recorder = TrajectoryRecorder(env)
    recorder.record(
        {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "robot0_gripper_qpos": np.zeros(2),
        },
        np.zeros(7),
        12,
    )
    trajectory_path = tmp_path / "trajectories" / "episode.npz"
    recorder.save(str(trajectory_path), metrics)
    append_index_entry(
        str(trajectory_path.parent), {"file": trajectory_path.name, **metrics}
    )

    loaded = load_trajectory(str(trajectory_path))["metadata"]
    index_entry = json.loads(
        (trajectory_path.parent / "index.jsonl").read_text().splitlines()[0]
    )
    for audit_record in (loaded, index_entry):
        assert audit_record["direct_robot_contact_first_step"] == 12
        assert audit_record["direct_robot_contact_geom1"] == "bottle_collision"
        assert audit_record["direct_robot_contact_geom2"] == "left_finger_collision"
        assert audit_record["direct_robot_contact_body1"] == "bottle"
        assert audit_record["direct_robot_contact_body2"] == "gripper0_finger"
        assert audit_record["direct_robot_contact_dist_m"] == -0.0004
