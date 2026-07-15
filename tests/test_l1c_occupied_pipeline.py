from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_oracles import (
    OccupiedGoalSafetyOracle,
    make_safety_oracle,
)
from experiments.robot.libero.tasks.l1c_occupied_common import get_spec, resolve_bddl


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
        assert spec.risk_offset == (0.0, 0.0)
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
