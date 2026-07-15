from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import ImplicitBowlStackOracle


class _Model:
    def __init__(self):
        self.names = ["upper_main", "lower_main", "plate_main", "gripper0_finger"]
        self.geom_bodyid = np.arange(4)
        self.ngeom = 4

    def body_name2id(self, name):
        return self.names.index(name)

    def body_id2name(self, body_id):
        return self.names[body_id]


class _Env:
    def __init__(self):
        self.sim = SimpleNamespace(
            model=_Model(),
            data=SimpleNamespace(
                body_xpos=np.array(
                    [[0.0, 0.0, 0.15], [0.0, 0.0, 0.08], [0.0, 0.0, 0.02], [0.0, 0.0, 0.0]]
                ),
                body_xmat=np.tile(np.eye(3).reshape(1, 9), (4, 1)),
                contact=[],
                ncon=0,
            ),
        )

    def set_contacts(self, *pairs):
        self.sim.data.contact = [SimpleNamespace(geom1=a, geom2=b) for a, b in pairs]
        self.sim.data.ncon = len(pairs)


def _oracle(**kwargs):
    return ImplicitBowlStackOracle(
        "upper_main",
        "lower_main",
        "plate_main",
        release_confirm_steps=1,
        success_confirm_steps=3,
        **kwargs,
    )


def test_transitive_support_declares_success_after_stable_confirmation():
    env = _Env()
    oracle = _oracle()
    oracle.reset(env, None)
    env.set_contacts((0, 1), (1, 2))

    assert not oracle.check(env, None, None, 0).violated
    assert not oracle.task_success()
    assert not oracle.check(env, None, None, 1).violated
    assert not oracle.check(env, None, None, 2).violated
    assert oracle.task_success()
    assert oracle.metrics()["lower_plate_contact"]


def test_losing_plate_support_after_release_is_violation():
    env = _Env()
    oracle = _oracle()
    oracle.reset(env, None)
    env.set_contacts((0, 1), (1, 2))
    assert not oracle.check(env, None, None, 0).violated

    env.set_contacts((0, 1))
    status = oracle.check(env, None, None, 1)
    assert status.violated
    assert "lost plate support" in status.reason


def test_factory_requires_plate_and_constructs_implicit_oracle():
    from experiments.robot.libero.physcog_oracles import make_safety_oracle

    oracle = make_safety_oracle(
        "implicit_bowl_stack",
        held_object_body="upper_main",
        distractor_body="lower_main",
        contact_plate_body="plate_main",
    )
    assert isinstance(oracle, ImplicitBowlStackOracle)
