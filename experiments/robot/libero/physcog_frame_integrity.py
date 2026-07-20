"""Helpers for validating policy-view frames before VLA inference."""

from __future__ import annotations

from typing import Optional

import numpy as np


class PolicyFrameIntegrityError(RuntimeError):
    """Raised when no trusted same-state policy render can be established."""


def force_refresh_observation(env):
    """Rerender all observables without advancing the simulator state.

    LIBERO's ControlEnv wrapper forwards the refresh methods but, unlike its
    wrapped robosuite environment, does not expose ``_get_observations``.
    Support both shapes explicitly instead of depending on only one API.
    """
    env._post_process()
    env._update_observables(force=True)
    observation_source = env if hasattr(env, "_get_observations") else env.env
    return observation_source._get_observations()


def frame_mad(left: np.ndarray, right: np.ndarray) -> float:
    """Return the per-channel mean absolute difference between two RGB frames."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"policy frame shapes differ: {left_array.shape} != {right_array.shape}"
        )
    return float(
        np.mean(
            np.abs(
                left_array.astype(np.float32, copy=False)
                - right_array.astype(np.float32, copy=False)
            )
        )
    )


def select_consistent_policy_frame(
    previous_frame: Optional[np.ndarray],
    same_state_frames: list[np.ndarray],
    transition_threshold: float,
    same_state_threshold: float,
) -> tuple[Optional[int], str]:
    """Choose a trustworthy frame from repeated renders of one simulator state.

    Index zero is the original observation; later entries are same-state
    rerenders. A normal transition is accepted immediately. A large transition
    must either be reproduced by a rerender or be replaced by a rerender that
    agrees with the preceding clean policy frame. The first policy frame has no
    temporal reference, so it always needs two mutually consistent renders.
    """
    if not same_state_frames:
        return None, "no_frames"

    if previous_frame is not None:
        original_transition = frame_mad(previous_frame, same_state_frames[0])
        if original_transition <= transition_threshold:
            return 0, "accepted_transition"

        for index in range(1, len(same_state_frames)):
            candidate = same_state_frames[index]
            if frame_mad(previous_frame, candidate) <= transition_threshold:
                return index, "recovered_corrupt_frame"
            if frame_mad(same_state_frames[0], candidate) <= same_state_threshold:
                return index, "verified_large_transition"
        return None, "unresolved_large_transition"

    for index in range(1, len(same_state_frames)):
        for earlier in range(index):
            if (
                frame_mad(same_state_frames[earlier], same_state_frames[index])
                <= same_state_threshold
            ):
                resolution = (
                    "verified_initial_frame"
                    if earlier == 0
                    else "recovered_initial_frame"
                )
                return index, resolution
    return None, "unresolved_initial_frame"
