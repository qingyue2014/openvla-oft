"""Outcome-aware rollout-video retention for formal LIBERO evaluations."""


def _below_cap(saved: int, cap: int) -> bool:
    """Return whether another video may be saved; zero means unlimited."""
    return cap == 0 or saved < cap


def should_save_rollout_video(
    *,
    mode: str,
    violated: bool,
    safe_success: bool,
    violation_videos: int,
    success_videos: int,
    failure_videos: int,
    max_violation_videos: int,
    max_success_videos: int,
    max_failure_videos: int,
) -> bool:
    """Apply the configured outcome filter and its per-task retention cap."""
    if mode == "none":
        return False
    if violated:
        return mode in {"all", "violation"} and _below_cap(
            violation_videos, max_violation_videos
        )
    if safe_success:
        return mode in {"all", "safe_success"} and _below_cap(
            success_videos, max_success_videos
        )
    return mode == "all" and _below_cap(failure_videos, max_failure_videos)
