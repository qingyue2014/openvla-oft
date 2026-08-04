import numpy as np

from experiments.robot.libero.tasks.repair_l1c1_eb_states import _outside_max


def test_outside_max_ignores_only_preregistered_joint_indices():
    baseline = np.zeros(6)
    changed = baseline.copy()
    changed[2:4] = [1.0, 2.0]
    assert _outside_max(baseline, changed, {2, 3}) == 0.0
    changed[5] = 0.25
    assert _outside_max(baseline, changed, {2, 3}) == 0.25
