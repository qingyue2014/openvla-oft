import numpy as np

from experiments.robot.libero.tasks.validate_l2a_native_rollout_videos import (
    _isolated_spikes,
)


def test_isolated_spikes_detects_single_corrupt_frame():
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (10, 11, 250, 12, 13)]
    spikes = _isolated_spikes(frames, transition_threshold=25.0, bridge_threshold=10.0)
    assert [row["frame"] for row in spikes] == [2]


def test_isolated_spikes_allows_smooth_motion():
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(0, 50, 5)]
    assert not _isolated_spikes(
        frames, transition_threshold=25.0, bridge_threshold=10.0
    )
