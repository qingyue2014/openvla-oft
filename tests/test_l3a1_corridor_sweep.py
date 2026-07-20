import json
from pathlib import Path

from experiments.robot.libero.tasks.summarize_l3a1_corridor_sweep import write_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "experiments/robot/libero/tasks/sweep_l3a1_corridor.sh"


def _row(index: int, *, direct: bool = False) -> dict:
    row = {
        "episode_idx": index,
        "success": True,
        "model_collapse": False,
        "support_activated": True,
        "support_activation_step": 200,
        "causal_eligible": not direct,
        "violated": not direct,
        "violation_step": None if direct else 205,
        "direct_contact_detected": direct,
        "direct_contact_step": 202 if direct else None,
    }
    if direct:
        row.update({
            "direct_robot_contact_geom1_id": 3,
            "direct_robot_contact_geom2_id": 19,
            "direct_robot_contact_geom1": "bottle_g3",
            "direct_robot_contact_geom2": "gripper0_hand_collision",
            "direct_robot_contact_dist_m": -0.00125,
        })
    return row


def test_corridor_summary_counts_qualifying_and_exact_contacts(tmp_path):
    candidates = []
    for dx, direct_count in (("-0.064", 1), ("-0.068", 2), ("-0.072", 0)):
        trajectory = tmp_path / dx / "trajectories"
        trajectory.mkdir(parents=True)
        rows = [_row(index, direct=index < direct_count) for index in range(5)]
        (trajectory / "index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        candidates.append((dx, trajectory))

    report = tmp_path / "report.md"
    write_report(candidates, 5, report)
    text = report.read_text()
    assert "| -0.064 | 5 | 4 | 1 | bottle_g3[3] ↔ gripper0_hand_collision[19] |" in text
    assert "| -0.068 | 5 | 3 | 2 | bottle_g3[3] ↔ gripper0_hand_collision[19] |" in text
    assert "| -0.072 | 5 | 5 | 0 | — |" in text
    assert "bottle_g3[3] ↔ gripper0_hand_collision[19], dist=-1.250 mm" in text


def test_corridor_sweep_is_er_only_and_cleans_unique_rollouts():
    text = SWEEP.read_text()
    assert "candidates=(-0.140 -0.145 -0.150)" in text
    assert 'LEAN_DX="${dx}" LEAN_DY=-0.184 LEAN_DEG=-30.0' in text
    assert "LEAN_DIRECTION_DEG=0.0" in text
    assert text.count('bash "${RUNNER}" risk eval') == 1
    assert 'rm -rf -- "${rollout}"' in text
    assert "NUM_TRIALS=5 SAVE_VIDEO_MODE=all" in text
    assert "stable eval" not in text
    assert "eb eval" not in text
