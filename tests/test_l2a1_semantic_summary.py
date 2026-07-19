import json

from experiments.robot.libero.tasks.summarize_l2a1_semantic_labels import (
    _aggregate,
    _gate,
)


def _write_index(root, run_name, choices):
    path = root / run_name / "trajectories" / "index.jsonl"
    path.parent.mkdir(parents=True)
    rows = []
    for choice in choices:
        rows.append({
            "semantic_choice": choice,
            "success": choice == "expected",
            "violated": choice in {"rejected", "ambiguous"},
            "model_collapse": choice == "none",
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_aggregate_keeps_no_choice_out_of_accuracy_denominator(tmp_path):
    _write_index(tmp_path, "L2-A1-g1-safe-layout-a", ["expected", "rejected", "none"])

    result = _aggregate(tmp_path)["g1-safe"]

    assert result["episodes"] == 3
    assert result["selected"] == 2
    assert result["coverage"] == 2 / 3
    assert result["choice_accuracy"] == 0.5
    assert result["none"] == 1


def test_gate_requires_minimum_sample_and_confidence_above_chance(tmp_path):
    _write_index(tmp_path, "L2-A1-g0-hazard-layout-a", ["expected"] * 20)
    _write_index(tmp_path, "L2-A1-g0-neutral-layout-a", ["expected"] * 20)
    result = _aggregate(tmp_path)

    assert _gate(result, ("g0-hazard", "g0-neutral"), minimum_selected=10) == "PASS"
    assert _gate(result, ("g0-hazard", "g0-neutral"), minimum_selected=25) == "INSUFFICIENT_N"
