from pathlib import Path

from experiments.robot.libero.tasks.l3b_order_candidates import (
    CANDIDATES,
    SUITE,
    native_bddl_path,
    parse_native_bddl,
    static_preflight,
)


def test_candidates_use_allowed_native_libero_10_tasks():
    assert SUITE == "libero_10"
    assert set(CANDIDATES) == {"mugs", "moka"}
    assert {candidate.task_id for candidate in CANDIDATES.values()} == {4, 8}
    for candidate in CANDIDATES.values():
        assert Path(candidate.task_file).suffix == ".bddl"
        assert native_bddl_path(candidate).is_file()


def test_native_prompts_and_inventories_are_exact():
    for candidate in CANDIDATES.values():
        record = parse_native_bddl(candidate)
        assert record["prompt"] == candidate.prompt
        assert record["fixtures"] == candidate.fixtures
        assert record["objects"] == candidate.objects
        assert len(record["sha256"]) == 64


def test_static_preflight_forbids_custom_content():
    for candidate in CANDIDATES.values():
        report = static_preflight(candidate)
        assert report["status"] == "PASS_STATIC_NATIVE_ONLY_PREFLIGHT"
        assert report["native_only"] is True
        assert report["custom_assets"] == []
        assert report["custom_bddl"] is False
        assert report["modified_prompt"] is False
