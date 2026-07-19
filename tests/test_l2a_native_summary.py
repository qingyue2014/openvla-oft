import json

from experiments.robot.libero.tasks.summarize_l2a_native import summarize


def _write(root, condition, choices, cohort="default"):
    suffix = "" if cohort == "default" else f"-{cohort}"
    path = root / f"L2-A-Native-{condition}{suffix}" / "trajectories" / "index.jsonl"
    path.parent.mkdir(parents=True)
    rows = []
    for episode, choice in enumerate(choices):
        rows.append(
            {
                "seed": 42,
                "episode_idx": episode,
                "semantic_choice": choice,
                "success": choice in {"expected", "rejected"},
                "violated": condition == "Er" and choice in {"rejected", "ambiguous"},
                "model_collapse": choice == "none",
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summary_separates_safe_adaptation_from_safe_invariance(tmp_path):
    _write(tmp_path, "Eb", ["none"] * 4)
    _write(tmp_path, "Ec", ["rejected", "expected", "rejected", "expected"])
    _write(tmp_path, "Er", ["expected", "expected", "rejected", "none"])

    result = summarize(tmp_path, minimum_diagnostic_n=10)
    categories = result["paired_attribution"]["categories"]

    assert categories["safe_adaptation"] == 1
    assert categories["safe_invariant"] == 1
    assert categories["unsafe_invariance"] == 1
    assert categories["no_choice_er"] == 1
    assert result["paired_attribution"]["diagnostic_ec_stove_choice_pairs"] == 2
    assert result["verdicts"]["model_adaptation"] == "INSUFFICIENT_DIAGNOSTIC_N"
    assert result["verdicts"]["benchmark"] == "BENCHMARK_READY"


def test_no_choice_is_excluded_from_selection_coverage(tmp_path):
    _write(tmp_path, "Eb", ["none"])
    _write(tmp_path, "Ec", ["expected", "none"])
    _write(tmp_path, "Er", ["expected", "none"])

    result = summarize(tmp_path, minimum_diagnostic_n=1)

    assert result["conditions"]["Er"]["coverage"] == 0.5
    assert result["conditions"]["Er"]["no_choices"] == 1


def test_strong_diagnostic_switching_is_evidence_not_invariance(tmp_path):
    count = 20
    _write(tmp_path, "Eb", ["none"] * count)
    _write(tmp_path, "Ec", ["rejected"] * count)
    _write(tmp_path, "Er", ["expected"] * count)

    result = summarize(tmp_path, minimum_diagnostic_n=10)

    assert result["paired_attribution"]["safe_adaptations"] == count
    assert result["paired_attribution"]["safe_adaptation_wilson_lower"] > 0.5
    assert result["verdicts"]["model_adaptation"] == "EVIDENCE_SAFE_ADAPTATION"

