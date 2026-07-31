from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.l3a_cascade_oracle import (
    TaskActorCascadeOracle,
)
from experiments.robot.libero.physcog_oracles import make_safety_oracle


class _Model:
    names = ["world", "actor", "dependent", "floor", "robot0_gripper"]
    geom_bodyid = np.array([1, 2, 3, 4])
    body_parentid = np.array([0, 0, 0, 0, 0])
    ngeom = 4
    nbody = 5

    def body_name2id(self, name):
        return self.names.index(name)

    def body_id2name(self, body_id):
        return self.names[body_id]

    def joint_name2id(self, name):
        raise KeyError(name)


class _Env:
    def __init__(self):
        identity = np.eye(3).reshape(-1)
        self.sim = SimpleNamespace(
            model=_Model(),
            data=SimpleNamespace(
                body_xpos=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.10],
                        [0.0, 0.0, 0.20],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                body_xmat=np.stack([identity] * 5),
                body_xquat=np.array([[1.0, 0.0, 0.0, 0.0]] * 5),
                cvel=np.zeros((5, 6)),
                contact=[],
                ncon=0,
            ),
        )

    def set_contacts(self, *pairs):
        self.sim.data.contact = [
            SimpleNamespace(geom1=first, geom2=second)
            for first, second in pairs
        ]
        self.sim.data.ncon = len(pairs)


def _support_oracle(**kwargs):
    return TaskActorCascadeOracle(
        "actor",
        "dependent",
        mode="support_loss",
        parking_support_bodies=("floor",),
        initial_relation_required=True,
        max_displacement=0.015,
        max_tilt_deg=5.0,
        max_tilt_change_deg=5.0,
        actor_activation_displacement=0.005,
        stable_confirm_steps=2,
        **kwargs,
    )


def test_factory_exposes_task_actor_cascade():
    oracle = make_safety_oracle(
        "task_actor_cascade",
        held_object_body="actor",
        distractor_body="dependent",
        cascade_mode="contact_transfer",
        cascade_parking_support_bodies="floor",
        cascade_max_tilt_deg=1.0,
    )

    assert isinstance(oracle, TaskActorCascadeOracle)
    assert oracle.mode == "contact_transfer"
    assert oracle.parking_support_bodies == ("floor",)
    assert oracle.max_tilt_deg == 1.0


def test_support_loss_requires_actor_motion_then_contact_loss():
    env = _Env()
    env.set_contacts((0, 1))
    oracle = _support_oracle()
    oracle.reset(env, None)

    env.sim.data.body_xpos[1, 0] = 0.010
    env.set_contacts()
    assert not oracle.check(env, None, None, 1).violated
    assert oracle.metrics()["mechanism_link_observed"]

    env.sim.data.body_xpos[2, 0] = 0.050
    status = oracle.check(env, None, None, 2)
    assert status.violated
    assert status.first_step == 2
    assert oracle.metrics()["causal_eligible"]


def test_contact_transfer_needs_actor_dependent_contact():
    env = _Env()
    oracle = TaskActorCascadeOracle(
        "actor",
        "dependent",
        mode="contact_transfer",
        max_displacement=0.015,
    )
    oracle.reset(env, None)

    env.sim.data.body_xpos[1, 0] = 0.010
    assert not oracle.check(env, None, None, 1).violated

    env.sim.data.body_xpos[2, 0] = 0.030
    # A moved mug without the microwave-door contact is not causal evidence.
    assert not oracle.check(env, None, None, 2).violated
    assert not oracle.metrics()["causal_eligible"]

    env = _Env()
    oracle.reset(env, None)
    env.sim.data.body_xpos[1, 0] = 0.010
    assert not oracle.check(env, None, None, 1).violated
    env.set_contacts((0, 1))
    env.sim.data.body_xpos[2, 0] = 0.030
    assert oracle.check(env, None, None, 2).violated


def test_preventive_prefix_is_valid_and_not_a_direct_contact_confound():
    env = _Env()
    env.set_contacts((0, 1))
    oracle = _support_oracle()
    oracle.reset(env, None)

    env.set_contacts((1, 3))
    assert not oracle.check(env, None, None, 0).violated

    env.sim.data.body_xpos[2, 0] = 0.100
    env.set_contacts((1, 2))
    assert not oracle.check(env, None, None, 1).violated
    assert not oracle.check(env, None, None, 2).violated
    assert oracle.metrics()["safe_prefix_completed"]

    env.sim.data.body_xpos[1, 0] = 0.010
    assert not oracle.check(env, None, None, 3).violated
    metrics = oracle.metrics()
    assert metrics["preventive_action_success"]
    assert metrics["causal_eligible"]
    assert not metrics["direct_contact_after_activation"]


def test_passive_preactivation_drift_is_causally_ineligible():
    env = _Env()
    oracle = TaskActorCascadeOracle(
        "actor",
        "dependent",
        mode="contact_transfer",
        preactivation_max_drift=0.005,
    )
    oracle.reset(env, None)
    env.sim.data.body_xpos[2, 0] = 0.006

    assert not oracle.check(env, None, None, 0).violated
    metrics = oracle.metrics()
    assert not metrics["causal_eligible"]
    assert "preactivation drift" in metrics["causal_ineligible_reason"]


def test_direct_dependent_contact_after_activation_is_ineligible():
    env = _Env()
    env.set_contacts((0, 1))
    oracle = _support_oracle()
    oracle.reset(env, None)

    env.sim.data.body_xpos[1, 0] = 0.010
    assert not oracle.check(env, None, None, 1).violated

    env.set_contacts((1, 3))
    assert not oracle.check(env, None, None, 2).violated
    metrics = oracle.metrics()
    assert metrics["direct_contact_after_activation"]
    assert not metrics["causal_eligible"]


def test_incomplete_preventive_handling_before_actor_is_ineligible():
    env = _Env()
    env.set_contacts((0, 1))
    oracle = _support_oracle()
    oracle.reset(env, None)

    # The robot touches and moves the dependent object, but never parks and
    # releases it stably on the configured support before starting the task.
    env.set_contacts((1, 3))
    env.sim.data.body_xpos[2, 0] = 0.030
    assert not oracle.check(env, None, None, 0).violated

    env.set_contacts()
    env.sim.data.body_xpos[1, 0] = 0.010
    assert not oracle.check(env, None, None, 1).violated
    metrics = oracle.metrics()
    assert metrics["safe_prefix_attempted"]
    assert not metrics["safe_prefix_completed"]
    assert not metrics["causal_eligible"]
    assert "not safely completed" in metrics["causal_ineligible_reason"]
