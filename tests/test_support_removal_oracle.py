from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import SupportRemovalOracle, make_safety_oracle


class _Model:
    names = ["drawer", "bottle", "gripper0_finger"]
    geom_bodyid = np.array([0, 1, 2])
    ngeom = 3

    def body_name2id(self, name):
        return self.names.index(name)

    def body_id2name(self, body_id):
        return self.names[body_id]


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
        self.sim.data.contact = [SimpleNamespace(geom1=a, geom2=b) for a, b in pairs]
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
    assert not metrics["causal_eligible"]
