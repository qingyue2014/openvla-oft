import json

import numpy as np

from experiments.robot.libero.physcog_attribution import run_attribution


def _trajectory(path, success=True, violated=False, x=0.0):
    eef = np.array([[x, 0.0, 0.8], [x, 0.0, 0.9], [x, 0.0, 1.0]])
    np.savez_compressed(
        path,
        metadata=json.dumps({"success": success, "violated": violated}),
        phases=np.array(["policy", "policy", "policy"]),
        eef_pos=eef,
    )


def test_btf_and_paired_risk_eligibility_are_registered(tmp_path):
    eb = tmp_path / "eb"
    er = tmp_path / "er"
    eb.mkdir()
    er.mkdir()
    _trajectory(eb / "ep0.npz", success=True, x=0.000)
    _trajectory(eb / "ep1.npz", success=True, x=0.001)
    _trajectory(eb / "ep2.npz", success=False, x=0.002)
    _trajectory(er / "ep0.npz", success=True, x=0.000)
    _trajectory(er / "ep1.npz", success=True, x=0.000)

    result = run_attribution(
        [str(eb)],
        [str(er)],
        None,
        risk_eligible_episodes={"ep1.npz"},
        risk_divergence_override={"ep1.npz": True},
        n_boot=10,
    )
    assert result["BTF"].tolist() == [0.0, 0.0, 1.0]
    assert result["n_risk_total"] == 2
    assert result["n_risk"] == 1
    assert result["n_risk_excluded"] == 1
    assert result["SAR"].tolist() == [1.0]


def test_zero_eligible_risk_episodes_produces_empty_metrics_not_exception(tmp_path):
    eb = tmp_path / "eb"
    er = tmp_path / "er"
    eb.mkdir()
    er.mkdir()
    _trajectory(eb / "ep0.npz", success=True, x=0.000)
    _trajectory(eb / "ep1.npz", success=True, x=0.001)
    _trajectory(er / "ep0.npz", success=True, x=0.000)

    result = run_attribution(
        [str(eb)], [str(er)], None, risk_eligible_episodes=set(), n_boot=10
    )
    assert result["n_risk_total"] == 1
    assert result["n_risk"] == 0
    assert result["n_risk_excluded"] == 1
    assert result["SAR"].size == 0
    assert result["UIR"].size == 0


def test_scenario_specific_violation_override_relabels_saved_safe_episode(tmp_path):
    eb = tmp_path / "eb"
    er = tmp_path / "er"
    eb.mkdir()
    er.mkdir()
    _trajectory(eb / "ep0.npz", success=True, x=0.000)
    _trajectory(eb / "ep1.npz", success=True, x=0.001)
    _trajectory(er / "ep0.npz", success=True, x=0.000)

    result = run_attribution(
        [str(eb)],
        [str(er)],
        None,
        risk_violation_override={"ep0.npz": True},
        n_boot=10,
    )
    assert result["UIR"].tolist() == [1.0]
    assert result["SAR"].tolist() == [0.0]
