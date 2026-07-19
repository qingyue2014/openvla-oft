import csv

from experiments.robot.libero.tasks.summarize_l1c1_direction_sweep import summarize


def _write(path, eligible, total=5):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attribution_eligible"])
        writer.writeheader()
        for index in range(total):
            writer.writerow({"attribution_eligible": int(index < eligible)})


def test_direction_sweep_ranks_by_eligibility_then_angle(tmp_path):
    _write(tmp_path / "angle_90.csv", eligible=4)
    _write(tmp_path / "angle_45.csv", eligible=4)
    _write(tmp_path / "angle_180.csv", eligible=3)

    rows = summarize(tmp_path)

    assert [row["angle_deg"] for row in rows] == [45, 90, 180]
    assert rows[0]["eligibility_rate"] == 0.8
