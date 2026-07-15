from pathlib import Path

import numpy as np
import pytest

from experiments.robot.libero.physcog_attribution import Episode, score_against_benign
from experiments.robot.libero.tasks.record_experiment_results import _metadata_for_attribution


def _episode(path: str, x: float) -> Episode:
    return Episode(
        path=path,
        eef_path=np.full((4, 3), (x, 0.0, 0.0), dtype=float),
        success=True,
        violated=False,
    )


def test_ec_reference_scoring_excludes_the_episode_itself() -> None:
    first = _episode("ec/task2_ep000.npz", 0.0)
    second = _episode("ec/task2_ep001.npz", 1.0)

    score_against_benign([first, second], [first, second], threshold=0.1, exclude_self=True)

    assert first.dist_to_benign > 0.1
    assert second.dist_to_benign > 0.1
    assert first.diverged
    assert second.diverged


def test_leave_one_out_requires_another_reference_episode() -> None:
    episode = _episode("ec/task2_ep000.npz", 0.0)

    with pytest.raises(ValueError, match="no other reference episodes"):
        score_against_benign([episode], [episode], threshold=0.1, exclude_self=True)


def test_l1c1_attribution_report_metadata_is_registered() -> None:
    assert _metadata_for_attribution(
        Path("experiments/logs/l1c1_attribution.md"),
        "L1-C1 implicit support chain",
    ) == ("L1", "L1-C1")
