from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = REPO_ROOT / "experiments/robot/libero/tasks"


def test_paper_matrix_full_mode_enables_capped_outcome_videos():
    text = (TASKS / "run_paper_matrix.sh").read_text()
    for setting in (
        'export SAVE_VIDEO_MODE="${SAVE_VIDEO_MODE:-all}"',
        'export MAX_VIOLATION_VIDEOS="${MAX_VIOLATION_VIDEOS:-10}"',
        'export MAX_SUCCESS_VIDEOS="${MAX_SUCCESS_VIDEOS:-10}"',
        'export MAX_FAILURE_VIDEOS="${MAX_FAILURE_VIDEOS:-10}"',
    ):
        assert setting in text


def test_every_paper_matrix_eval_runner_forwards_all_video_caps():
    runners = (
        "run_l1a_evals.sh",
        "run_l1b2_task6.sh",
        "run_l1b4_task6.sh",
        "run_l2b2_basket_stove.sh",
        "run_l2c2_bowl.sh",
        "run_l3a1_drawer_bottle.sh",
    )
    required = (
        "--save_video_mode",
        "--max_violation_videos",
        "--max_success_videos",
        "--max_failure_videos",
    )
    for runner in runners:
        text = (TASKS / runner).read_text()
        for argument in required:
            assert argument in text, f"{runner} does not forward {argument}"
