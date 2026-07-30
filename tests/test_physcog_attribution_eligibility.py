import json

import numpy as np
import pytest

from experiments.robot.libero.physcog_attribution import (
    episode_pairing_key,
    load_risk_eligibility_csv,
    run_attribution,
)


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


def test_complete_benign_pool_drives_btf_without_breaking_exact_pairs(tmp_path):
    paired_eb = tmp_path / "paired_eb"
    competence_eb = tmp_path / "competence_eb"
    er = tmp_path / "er"
    ec = tmp_path / "ec"
    for root in (paired_eb, competence_eb, er, ec):
        root.mkdir()
    for episode in range(2):
        _trajectory(
            paired_eb / f"task4_ep{episode:03d}.npz",
            success=True,
            x=episode * 0.001,
        )
        _trajectory(
            er / f"task4_ep{episode:03d}.npz",
            success=True,
            x=episode * 0.001,
        )
        _trajectory(
            ec / f"task4_ep{episode:03d}.npz",
            success=True,
            x=episode * 0.001,
        )
    for episode, success in enumerate((True, True, False)):
        _trajectory(
            competence_eb / f"task4_ep{episode:03d}.npz",
            success=success,
            x=episode * 0.001,
        )

    result = run_attribution(
        [str(paired_eb)],
        [str(er)],
        [str(ec)],
        benign_competence_dirs=[str(competence_eb)],
        require_exact_pairing=True,
        n_boot=10,
    )

    assert result["exact_episode_pairing"] is True
    assert result["n_benign"] == 2
    assert result["n_benign_competence"] == 3
    assert result["benign_success_rate"] == pytest.approx(2 / 3)
    assert result["BTF"].tolist() == [0.0, 0.0, 1.0]


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


def test_condition_named_files_pair_by_episode_number(tmp_path):
    roots = {}
    for condition in ("eb", "er", "ec"):
        roots[condition] = tmp_path / condition
        roots[condition].mkdir()
        for episode in range(2):
            _trajectory(
                roots[condition] / f"L1-B3-{condition}_ep{episode:03d}.npz",
                success=True,
                x=episode * 0.001,
            )

    eligibility = tmp_path / "eligibility.csv"
    eligibility.write_text(
        "episode,attribution_eligible\n"
        "L1-B3-eb_ep000.npz,1\n"
        "L1-B3-eb_ep001.npz,0\n"
    )
    eligible = load_risk_eligibility_csv(str(eligibility))
    assert eligible == {"ep000000"}
    assert episode_pairing_key("L1-B3-er_ep000.npz") == "ep000000"

    result = run_attribution(
        [str(roots["eb"])],
        [str(roots["er"])],
        [str(roots["ec"])],
        risk_eligible_episodes=eligible,
        require_exact_pairing=True,
        n_boot=10,
    )
    assert result["exact_episode_pairing"] is True
    assert result["n_risk"] == 1


def test_exact_pairing_gate_rejects_missing_condition_episode(tmp_path):
    eb = tmp_path / "eb"
    er = tmp_path / "er"
    ec = tmp_path / "ec"
    for root in (eb, er, ec):
        root.mkdir()
    for episode in range(2):
        _trajectory(eb / f"L1-B3-eb_ep{episode:03d}.npz")
        _trajectory(er / f"L1-B3-er_ep{episode:03d}.npz")
    _trajectory(ec / "L1-B3-ec_ep000.npz")

    with pytest.raises(ValueError, match="not exactly paired"):
        run_attribution(
            [str(eb)],
            [str(er)],
            [str(ec)],
            require_exact_pairing=True,
            n_boot=10,
        )
