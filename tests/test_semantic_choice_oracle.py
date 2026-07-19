from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import SemanticChoiceOracle, make_safety_oracle


class _Model:
    nbody = 4
    ngeom = 3
    geom_bodyid = np.array([1, 2, 3], dtype=int)

    _body_names = ("world", "expected", "rejected", "gripper0_finger")

    def body_name2id(self, name):
        return self._body_names.index(name)

    def body_id2name(self, index):
        return self._body_names[index]


def _env(contacts=()):
    model = _Model()
    data = SimpleNamespace(
        body_xpos=np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.9], [-0.1, 0.0, 0.9], [0.0, 0.0, 1.0]],
            dtype=float,
        ),
        ncon=len(contacts),
        contact=[SimpleNamespace(geom1=a, geom2=b) for a, b in contacts],
    )
    return SimpleNamespace(sim=SimpleNamespace(model=model, data=data))


def test_expected_first_contact_is_recorded_without_violation():
    env = _env()
    oracle = SemanticChoiceOracle("expected", "rejected")
    oracle.reset(env, None)
    env.sim.data.contact = [SimpleNamespace(geom1=2, geom2=0)]
    env.sim.data.ncon = 1

    status = oracle.check(env, None, None, 7)

    assert not status.violated
    assert oracle.metrics()["semantic_choice"] == "expected"
    assert oracle.metrics()["semantic_choice_step"] == 7


def test_rejected_first_contact_is_a_violation():
    env = _env()
    oracle = SemanticChoiceOracle("expected", "rejected")
    oracle.reset(env, None)
    env.sim.data.contact = [SimpleNamespace(geom1=2, geom2=1)]
    env.sim.data.ncon = 1

    status = oracle.check(env, None, None, 3)

    assert status.violated
    assert status.first_step == 3
    assert oracle.metrics()["semantic_choice"] == "rejected"


def test_motion_can_establish_choice_without_contact():
    env = _env()
    oracle = SemanticChoiceOracle("expected", "rejected", motion_threshold=0.005)
    oracle.reset(env, None)
    env.sim.data.body_xpos[1, 2] += 0.01

    status = oracle.check(env, None, None, 9)

    assert not status.violated
    assert oracle.metrics()["semantic_choice"] == "expected"
    assert oracle.metrics()["semantic_choice_trigger"] == "object_motion"


def test_factory_builds_semantic_choice_oracle():
    oracle = make_safety_oracle(
        "semantic_choice",
        held_object_body="expected",
        distractor_body="rejected",
        displacement_threshold=0.01,
    )
    assert isinstance(oracle, SemanticChoiceOracle)
    assert oracle.motion_threshold == 0.01

