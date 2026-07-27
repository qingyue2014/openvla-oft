from experiments.robot.libero.video_retention import (
    is_safe_success,
    should_save_rollout_video,
)


def _decision(mode, *, violated=False, safe_success=False, saved=0, cap=10):
    return should_save_rollout_video(
        mode=mode,
        violated=violated,
        safe_success=safe_success,
        violation_videos=saved,
        success_videos=saved,
        failure_videos=saved,
        max_violation_videos=cap,
        max_success_videos=cap,
        max_failure_videos=cap,
    )


def test_safe_success_mode_keeps_only_safe_completions():
    assert _decision("safe_success", safe_success=True)
    assert not _decision("safe_success", violated=True)
    assert not _decision("safe_success")


def test_safe_success_cap_is_inclusive_of_first_ten_only():
    assert _decision("safe_success", safe_success=True, saved=9)
    assert not _decision("safe_success", safe_success=True, saved=10)


def test_all_mode_honors_each_outcome_cap():
    assert _decision("all", violated=True, saved=9)
    assert _decision("all", safe_success=True, saved=9)
    assert _decision("all", saved=9)
    assert not _decision("all", violated=True, saved=10)
    assert not _decision("all", safe_success=True, saved=10)
    assert not _decision("all", saved=10)


def test_zero_cap_means_unlimited():
    assert _decision("safe_success", safe_success=True, saved=100, cap=0)


def test_causal_ineligibility_cannot_be_published_as_safe_success():
    assert is_safe_success(
        task_success=True, violated=False, causal_eligible=True
    )
    assert not is_safe_success(
        task_success=True, violated=False, causal_eligible=False
    )
