import numpy as np

from experiments.robot.libero.physcog_frame_integrity import (
    frame_mad,
    select_consistent_policy_frame,
)


def _frame(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_frame_mad_uses_unsigned_rgb_without_wraparound():
    assert frame_mad(_frame(250), _frame(5)) == 245.0


def test_normal_transition_is_accepted_without_rerender():
    index, resolution = select_consistent_policy_frame(
        _frame(10), [_frame(12)], 25.0, 10.0
    )
    assert (index, resolution) == (0, "accepted_transition")


def test_corrupt_frame_is_replaced_by_previous_consistent_rerender():
    index, resolution = select_consistent_policy_frame(
        _frame(10), [_frame(250), _frame(12)], 25.0, 10.0
    )
    assert (index, resolution) == (1, "recovered_corrupt_frame")


def test_reproducible_large_transition_is_not_a_false_positive():
    index, resolution = select_consistent_policy_frame(
        _frame(10), [_frame(100), _frame(102)], 25.0, 10.0
    )
    assert (index, resolution) == (1, "verified_large_transition")


def test_initial_frame_is_verified_by_one_matching_rerender():
    index, resolution = select_consistent_policy_frame(
        None, [_frame(10), _frame(11)], 25.0, 10.0
    )
    assert (index, resolution) == (1, "verified_initial_frame")


def test_initial_corruption_requires_two_consistent_clean_rerenders():
    index, resolution = select_consistent_policy_frame(
        None, [_frame(250), _frame(10), _frame(11)], 25.0, 10.0
    )
    assert (index, resolution) == (2, "recovered_initial_frame")


def test_unresolved_frames_are_rejected():
    index, resolution = select_consistent_policy_frame(
        _frame(10), [_frame(250), _frame(100), _frame(180)], 25.0, 10.0
    )
    assert index is None
    assert resolution == "unresolved_large_transition"
