import csv

from experiments.robot.libero.tasks.generate_result_tables import (
    _build_scenario_rows,
    generate_tables_from_records,
)
from experiments.robot.libero.tasks.record_experiment_results import (
    RECORD_FIELDS,
    load_records_csv,
)


def _eval(run_id, scenario, condition, task_rate, safe_rate):
    n = 50
    return {
        "record_type": "eval",
        "timestamp": "2026_07_17_00_00_00",
        "model": "openvla",
        "run_id": run_id,
        "level": "L1",
        "scenario": scenario,
        "condition": condition,
        "n": n,
        "successes": round(task_rate * n),
        "violations": 0,
        "safe_successes": round(safe_rate * n),
        "task_success_rate": task_rate,
        "svr": 0.0,
        "safe_success_rate": safe_rate,
    }


def test_l1a2_table_reuses_l1a1_native_baseline_as_shared_eb():
    records = [
        _eval("L1-A1-native-baseline", "L1-A1", "Eb Native Gate", 1.0, 1.0),
        _eval(
            "L1-A2-upright-cookie-occlusion",
            "L1-A2",
            "Er Upright Cookie Occlusion",
            0.2,
            0.2,
        ),
        _eval(
            "L1-A2-upright-cookie-matched-safe",
            "L1-A2",
            "Ec Matched-Safe",
            0.8,
            0.8,
        ),
    ]

    rows = _build_scenario_rows(records, default_model="openvla")
    l1a2 = next(row for row in rows if row["scenario"] == "L1-A2")

    assert l1a2["n_eb"] == 50
    assert l1a2["eb_task_sr"] == 1.0
    assert "missing Eb" not in l1a2["notes"]


def test_archived_l1c2_records_backfill_condition_and_fill_tables(tmp_path):
    records_path = tmp_path / "experiment_records.csv"
    rows = [
        _eval("L1-C2-occupied-tray-eb", "L1-C2", "", 1.0, 1.0),
        _eval("L1-C2-occupied-tray-risk", "L1-C2", "", 0.54, 0.0),
        _eval("L1-C2-occupied-tray-ec", "L1-C2", "", 1.0, 1.0),
    ]
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    records = load_records_csv(records_path)
    assert [row["condition"] for row in records] == [
        "Eb Empty Tray",
        "Er Occupied Tray",
        "Ec Nearby Object",
    ]

    rendered = generate_tables_from_records(records, "test archive", default_model="openvla")
    assert "| L1 | L1-C2 | openvla | 50 | 50 | 50 |" in rendered
    assert "no data" not in rendered
