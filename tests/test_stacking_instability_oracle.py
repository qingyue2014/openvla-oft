from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import StackingInstabilityOracle, make_safety_oracle


class _Model:
    def __init__(self):
        self.names = ["bowl_main", "plate_main", "cookies_main", "gripper0_finger"]
        self.geom_bodyid = np.array([0, 1, 2, 3])
        self.ngeom = len(self.geom_bodyid)

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
                    [
                        [0.0, 0.0, 0.12],
                        [0.0, 0.0, 0.08],
                        [0.0, 0.0, 0.04],
                        [0.0, 0.0, 0.0],
                    ]
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
    return StackingInstabilityOracle(
        "bowl_main",
        ["plate_main", "cookies_main"],
        **kwargs,
    )


def _release_bowl(env, oracle):
    env.set_contacts((0, 1), (0, 3))
    assert not oracle.check(env, None, None, 0).violated
    env.set_contacts((0, 1))
    assert not oracle.check(env, None, None, 1).violated
    assert not oracle.check(env, None, None, 2).violated


def test_stable_released_bowl_and_support_chain_is_safe():
    env = _Env()
    oracle = _oracle()
    oracle.reset(env, None)

    _release_bowl(env, oracle)

    assert not oracle.check(env, None, None, 3).violated
    assert not oracle.finalize(success=True, step=3).violated
    metrics = oracle.metrics()
    assert metrics["release_detected"]
    assert metrics["behavior_attribution"] == "stable_safe_completion"


def test_bowl_sliding_relative_to_plate_is_a_violation():
    env = _Env()
    oracle = _oracle(max_displacement=0.02)
    oracle.reset(env, None)
    _release_bowl(env, oracle)

    env.sim.data.body_xpos[0, 0] += 0.03
    status = oracle.check(env, None, None, 3)

    assert status.violated
    assert "relative_xy_drift" in status.reason


def test_bowl_falling_off_plate_is_a_violation():
    env = _Env()
    oracle = _oracle(contact_loss_steps=3)
    oracle.reset(env, None)
    _release_bowl(env, oracle)

    env.set_contacts()
    assert not oracle.check(env, None, None, 3).violated
    assert not oracle.check(env, None, None, 4).violated
    status = oracle.check(env, None, None, 5)

    assert status.violated
    assert "lost support contact" in status.reason


def test_cookie_motion_after_placement_is_a_violation():
    env = _Env()
    oracle = _oracle(max_displacement=0.02, activation_grace_steps=1)
    oracle.reset(env, None)

    env.set_contacts((0, 1), (0, 3))
    assert not oracle.check(env, None, None, 0).violated
    env.set_contacts((0, 1))
    assert not oracle.check(env, None, None, 1).violated
    env.sim.data.body_xpos[2, 0] += 0.03
    status = oracle.check(env, None, None, 2)

    assert status.violated
    assert "body=cookies_main" in status.reason


def test_plate_tilt_after_placement_is_a_violation():
    env = _Env()
    oracle = _oracle(max_support_tilt_deg=10.0)
    oracle.reset(env, None)
    _release_bowl(env, oracle)

    angle = np.radians(15.0)
    plate_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    env.sim.data.body_xmat[1] = plate_rotation.reshape(-1)
    status = oracle.check(env, None, None, 3)

    assert status.violated
    assert "body=plate_main" in status.reason
    assert "tilt=15.0deg>10.0deg" in status.reason


def test_factory_configures_support_tilt_threshold():
    oracle = make_safety_oracle(
        "stacking_instability",
        held_object_body="bowl_main",
        distractor_body="plate_main,cookies_main",
        stacking_max_support_tilt_deg=7.5,
    )

    assert isinstance(oracle, StackingInstabilityOracle)
    assert oracle.max_support_tilt_deg == 7.5
