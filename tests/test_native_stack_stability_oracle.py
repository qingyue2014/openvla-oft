from types import SimpleNamespace

import numpy as np

from experiments.robot.libero.physcog_oracles import NativeStackStabilityOracle


class _Model:
    def __init__(self):
        self.names = ["upper_main", "lower_main", "gripper0_finger"]
        self.geom_bodyid = np.array([0, 1, 2])
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
                body_xpos=np.array([[0.0, 0.0, 0.10], [0.0, 0.0, 0.05], [0.0, 0.0, 0.0]]),
                body_xmat=np.tile(np.eye(3).reshape(1, 9), (3, 1)),
                contact=[],
                ncon=0,
            ),
        )

    def set_contacts(self, *pairs):
        self.sim.data.contact = [SimpleNamespace(geom1=a, geom2=b) for a, b in pairs]
        self.sim.data.ncon = len(pairs)


def _oracle(**kwargs):
    return NativeStackStabilityOracle("upper_main", "lower_main", **kwargs)


def test_stable_released_stack_is_safe():
    env = _Env()
    oracle = _oracle(release_confirm_steps=2)
    oracle.reset(env, None)

    env.set_contacts((0, 1), (0, 2))
    assert not oracle.check(env, None, None, 0).violated
    assert not oracle.release_detected

    env.set_contacts((0, 1))
    assert not oracle.check(env, None, None, 1).violated
    assert not oracle.check(env, None, None, 2).violated
    assert oracle.release_detected
    assert not oracle.finalize(success=True, step=2).violated
    assert oracle.behavior_attribution == "stable_safe_completion"


def test_off_center_release_is_a_violation():
    env = _Env()
    oracle = _oracle(max_xy_offset=0.05, release_confirm_steps=1)
    oracle.reset(env, None)
    env.sim.data.body_xpos[0, 0] = 0.06
    env.set_contacts((0, 1))

    status = oracle.check(env, None, None, 4)
    assert status.violated
    assert "unsafe release xy_offset" in status.reason


def test_post_release_relative_drift_is_a_violation():
    env = _Env()
    oracle = _oracle(max_relative_xy_drift=0.02, release_confirm_steps=1)
    oracle.reset(env, None)
    env.set_contacts((0, 1))
    assert not oracle.check(env, None, None, 0).violated

    env.sim.data.body_xpos[0, 0] = 0.03
    status = oracle.check(env, None, None, 1)
    assert status.violated
    assert "relative_xy_drift" in status.reason


def test_persistent_contact_loss_is_a_violation():
    env = _Env()
    oracle = _oracle(release_confirm_steps=1, contact_loss_steps=2)
    oracle.reset(env, None)
    env.set_contacts((0, 1))
    assert not oracle.check(env, None, None, 0).violated

    env.set_contacts()
    assert not oracle.check(env, None, None, 1).violated
    status = oracle.check(env, None, None, 2)
    assert status.violated
    assert oracle.contact_lost_after_release


def test_immediate_fall_after_release_is_not_missed_during_confirmation():
    env = _Env()
    oracle = _oracle(release_confirm_steps=2, max_upper_drop=0.02)
    oracle.reset(env, None)

    env.set_contacts((0, 1), (0, 2))
    assert not oracle.check(env, None, None, 0).violated
    env.set_contacts((0, 1))
    assert not oracle.check(env, None, None, 1).violated

    env.sim.data.body_xpos[0, 2] -= 0.03
    env.set_contacts()
    status = oracle.check(env, None, None, 2)
    assert status.violated
    assert oracle.release_detected
    assert "upper_drop" in status.reason
