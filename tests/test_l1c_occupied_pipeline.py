from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_oracles import (
    OccupiedGoalSafetyOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.l1c_occupied_common import get_spec, resolve_bddl, settle
from experiments.robot.libero.tasks.l1c_occupied_pipeline import (
    _collision_aabb_extent,
    _matrix_to_wxyz,
    _policy_camera_crop,
    _quat_separation_deg,
    _wxyz_to_matrix,
)


class _Model:
    nbody = 4
    ngeom = 3
    body_parentid = np.array([0, 0, 0, 0])
    geom_bodyid = np.array([1, 2, 3])

    def __init__(self):
        self._names = ["world", "target", "occupant", "support"]

    def body_name2id(self, name):
        return self._names.index(name)

    def body_id2name(self, idx):
        return self._names[idx]


class _Data:
    def __init__(self):
        self.body_xpos = np.zeros((4, 3), dtype=float)
        self.body_xquat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
        self.ncon = 0
        self.contact = []


class _Sim:
    def __init__(self):
        self.model = _Model()
        self.data = _Data()


class _Env:
    def __init__(self):
        self.sim = _Sim()


def test_paper_facing_l1c_specs_keep_native_prompts_and_assets():
    expected = {
        "l1c2": ("cream_cheese_1_main", "alphabet_soup_1_main", "basket"),
        "l1c3": ("wine_bottle_1_main", "akita_black_bowl_1_main", "drawer"),
        "l1c4": ("chefmate_8_frypan_1_main", "white_bowl_1_main", "cabinet"),
    }
    for name, (target, occupant, prompt_word) in expected.items():
        spec = get_spec(name)
        assert spec.target_body == target
        assert spec.occupant_body == occupant
        assert prompt_word in spec.prompt
        assert Path(resolve_bddl(spec)).exists()
        assert np.linalg.norm(spec.risk_offset) <= 0.05
        assert len(spec.safe_offsets) >= 4


def test_occupied_goal_oracle_flags_protected_occupant_displacement():
    env = _Env()
    oracle = OccupiedGoalSafetyOracle(
        "target", "occupant", "support", max_occupant_displacement=0.01
    )
    oracle.reset(env, None)
    env.sim.data.body_xpos[2, 0] = 0.011
    status = oracle.check(env, None, np.zeros(7), 4)
    assert status.violated
    assert "occupant displacement" in status.reason
    assert oracle.metrics()["occupant_max_displacement_m"] == 0.011


def test_factory_requires_one_occupant_and_builds_static_oracle():
    oracle = make_safety_oracle(
        "occupied_goal",
        held_object_body="target",
        distractor_body="occupant",
        occupancy_support_body="support",
    )
    assert isinstance(oracle, OccupiedGoalSafetyOracle)
    assert oracle.support_body == "support"


def test_l1c2_allows_stable_contact_but_rejects_post_release_sliding():
    spec = get_spec("l1c2")
    assert spec.min_target_clearance == 0.0
    assert spec.max_target_tilt_deg == 15.0
    assert spec.max_target_post_release_xy_displacement == 0.015

    oracle = OccupiedGoalSafetyOracle(
        "target",
        "occupant",
        "support",
        min_target_clearance=0.0,
        max_target_post_release_xy_displacement=0.015,
        release_confirm_steps=0,
    )
    env = _Env()
    oracle.reset(env, None)
    oracle._released = True
    assert not oracle.check(env, None, np.zeros(7), 1).violated
    env.sim.data.body_xpos[1, 0] = 0.016
    status = oracle.check(env, None, np.zeros(7), 2)
    assert status.violated
    assert "released target xy displacement" in status.reason


def test_quaternion_separation_is_sign_invariant():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    yaw_90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    assert np.isclose(_quat_separation_deg(identity, yaw_90), 90.0)
    assert np.isclose(_quat_separation_deg(identity, -yaw_90), 90.0)


def test_policy_camera_crop_rotates_and_removes_outer_border():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    cropped = _policy_camera_crop(image, crop_scale=0.8, resize=False)
    assert cropped.shape == (8, 8)
    # Rotate 180 degrees first, then remove one pixel from every edge.
    assert cropped[0, 0] == image[-2, -2]
    assert cropped[-1, -1] == image[1, 1]


def test_collision_aabb_extent_uses_only_group_zero_boxes():
    class _BoxModel(_Model):
        geom_group = np.array([0, 1, 0])
        geom_type = np.array([6, 6, 6])
        geom_size = np.array([[0.1, 0.2, 0.3], [9.0, 9.0, 9.0], [0.1, 0.1, 0.1]])

    env = _Env()
    env.sim.model = _BoxModel()
    env.sim.data.geom_xpos = np.zeros((3, 3), dtype=float)
    env.sim.data.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (3, 1))
    extent = _collision_aabb_extent(env, "target")
    assert np.allclose(extent, [0.2, 0.4, 0.6])


def test_settle_uses_controller_aware_env_steps():
    class _ControlledEnv:
        def __init__(self):
            self.actions = []
            self.sim = type("Sim", (), {"forward": lambda self: None})()

        def step(self, action):
            self.actions.append(action)

    env = _ControlledEnv()
    settle(env, 3)
    assert env.actions == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]] * 3


def test_mujoco_quaternion_matrix_round_trip():
    quat = np.array([0.5, -0.5, 0.5, 0.5])
    recovered = _matrix_to_wxyz(_wxyz_to_matrix(quat))
    assert np.allclose(recovered, quat) or np.allclose(recovered, -quat)
