import json
from pathlib import Path

from experiments.robot.libero.tasks.summarize_l2a_affordance_prompt_gate import (
    PROMPTS,
    _summarize,
)


def _write_rows(root: Path, prompt_id: str, successes: int, total: int = 5) -> None:
    path = (
        root
        / f"L2-A-Affordance-PromptGate-{prompt_id}"
        / "trajectories"
        / "index.jsonl"
    )
    path.parent.mkdir(parents=True)
    rows = [
        {
            "success": index < successes,
            "model_collapse": False,
            "seed": 42,
            "episode_idx": index,
            "initial_state_sha256_f64le": f"state-{index}",
            "pretrained_checkpoint": "checkpoint",
            "git_commit": "commit",
            "task_id": 2,
            "policy_task_description": PROMPTS[prompt_id],
        }
        for index in range(total)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_container_passes_only_when_close_to_competent_native_baseline(tmp_path: Path):
    for prompt_id in PROMPTS:
        _write_rows(tmp_path, prompt_id, 4)

    result = _summarize(tmp_path, expected_trials=5)

    assert result["verdict"] == "PASS_L2A_AFFORDANCE_CONTAINER_LANGUAGE"
    assert result["stages"]["container"]["language_gate_pass"] is True


def test_explicit_upper_bound_does_not_rescue_container_failure(tmp_path: Path):
    _write_rows(tmp_path, "native", 5)
    _write_rows(tmp_path, "pot", 5)
    _write_rows(tmp_path, "container", 1)
    _write_rows(tmp_path, "heat-safe-container", 5)

    result = _summarize(tmp_path, expected_trials=5)

    assert result["verdict"] == "FAIL_L2A_AFFORDANCE_CONTAINER_LANGUAGE"
    assert result["stages"]["heat-safe-container"]["language_gate_pass"] is True


def test_incomplete_runs_fail_the_gate(tmp_path: Path):
    for prompt_id in PROMPTS:
        _write_rows(tmp_path, prompt_id, 3, total=4)

    result = _summarize(tmp_path, expected_trials=5)

    assert result["verdict"] == "FAIL_L2A_AFFORDANCE_PROMPT_GATE_INTEGRITY"


def test_mismatched_initial_state_hash_fails_integrity(tmp_path: Path):
    for prompt_id in PROMPTS:
        _write_rows(tmp_path, prompt_id, 5)
    index = (
        tmp_path
        / "L2-A-Affordance-PromptGate-container"
        / "trajectories"
        / "index.jsonl"
    )
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    rows[0]["initial_state_sha256_f64le"] = "different-state"
    index.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = _summarize(tmp_path, expected_trials=5)

    assert result["verdict"] == "FAIL_L2A_AFFORDANCE_PROMPT_GATE_INTEGRITY"
