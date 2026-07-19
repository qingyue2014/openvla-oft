from experiments.robot.libero.tasks.generate_result_tables import _build_scenario_rows


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


def test_incomplete_attribution_is_not_treated_as_paper_ready():
    records = [
        _eval("L1-C1-eb", "L1-C1", "Eb Native Gate", 1.0, 1.0),
        _eval("L1-C1-er", "L1-C1", "Er Occupied Plate", 0.2, 0.2),
        _eval("L1-C1-ec", "L1-C1", "Ec Nearby Bowl", 0.9, 0.9),
        {
            "record_type": "attribution",
            "timestamp": "2026_07_17_01_00_00",
            "model": "openvla",
            "scenario": "L1-C1",
            "notes": "BENCHMARK_INCOMPLETE;SAR_N=35",
            "sar": 0.086,
        },
    ]

    rows = _build_scenario_rows(records, default_model="openvla")
    l1c1 = next(row for row in rows if row["scenario"] == "L1-C1")

    assert l1c1["sar"] == ""
    assert "missing attribution" in l1c1["notes"]
