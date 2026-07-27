from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.cosmos_policy_utils import (
    COSMOS_DEFAULT_CHECKPOINT,
    COSMOS_LIBERO_REPO_ID,
    is_cosmos_model_family,
    prepare_cosmos_libero_observation,
    resolve_cosmos_package_root,
    validate_cosmos_actions,
)
from experiments.robot.dreamzero_utils import (
    DREAMZERO_DEFAULT_CHECKPOINT,
    DREAMZERO_DROID_REPO_ID,
    DreamZeroLiberoCompatibilityError,
    get_dreamzero_policy,
    is_dreamzero_model_family,
)
from experiments.robot.libero.tasks.physcog_remote_agent import PHASES


def test_cosmos_aliases_and_action_contract():
    assert is_cosmos_model_family("cosmos")
    assert is_cosmos_model_family("cosmos-policy")
    actions = validate_cosmos_actions(np.zeros((16, 7), dtype=np.float64))
    assert actions.shape == (16, 7)
    assert actions.dtype == np.float32


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((16, 8)),
        np.zeros((0, 7)),
        np.full((1, 7), np.nan),
    ],
)
def test_cosmos_rejects_invalid_action_chunks(actions):
    with pytest.raises(ValueError):
        validate_cosmos_actions(actions)


def test_cosmos_libero_observation_matches_official_contract():
    agent = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    wrist = agent + 20
    obs = {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        "robot0_gripper_qpos": np.array([0.1, 0.2]),
        "robot0_eef_pos": np.array([0.3, 0.4, 0.5]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    prepared = prepare_cosmos_libero_observation(obs)
    np.testing.assert_array_equal(prepared["primary_image"], np.flipud(agent))
    np.testing.assert_array_equal(prepared["wrist_image"], np.flipud(wrist))
    np.testing.assert_allclose(
        prepared["proprio"],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
    )


def test_cosmos_package_root_supports_namespace_packages(tmp_path):
    package_root = tmp_path / "cosmos_policy"
    (package_root / "config").mkdir(parents=True)
    (package_root / "config" / "config.py").write_text("# config\n")
    namespace_module = SimpleNamespace(__file__=None, __path__=[str(package_root)])
    assert resolve_cosmos_package_root(namespace_module) == package_root.resolve()


def test_dreamzero_aliases_and_libero_guard(tmp_path):
    assert is_dreamzero_model_family("dreamzero")
    assert is_dreamzero_model_family("dream-zero")
    cfg = SimpleNamespace(pretrained_checkpoint=tmp_path / "DreamZero-DROID")
    with pytest.raises(DreamZeroLiberoCompatibilityError, match="8-D DROID"):
        get_dreamzero_policy(cfg)


def test_checkpoint_identities_and_superpod_paths_are_explicit():
    assert COSMOS_LIBERO_REPO_ID == "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    assert COSMOS_DEFAULT_CHECKPOINT == Path("/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B")
    assert DREAMZERO_DROID_REPO_ID == "GEAR-Dreams/DreamZero-DROID"
    assert DREAMZERO_DEFAULT_CHECKPOINT == Path("/project/trllmout/models/DreamZero-DROID")


def test_remote_agent_registers_model_setup_phases():
    expected = {
        ("models", "setup_cosmos"): "cosmos",
        ("models", "setup_dreamzero"): "dreamzero",
        ("models", "setup_all"): "all",
    }
    for key, model in expected.items():
        spec = PHASES[key]
        assert spec.command == (
            "bash",
            "experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh",
            model,
        )
        if model == "cosmos":
            assert spec.artifacts == (
                "experiments/logs/cosmos_superpod_setup.json",
            )
        else:
            assert spec.artifacts == ()


def test_setup_script_pins_official_model_revisions():
    script = Path("experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh").read_text()
    assert 'MODEL_ROOT="${MODEL_ROOT:-/project/trllmout/models}"' in script
    assert "cb689ec0e3347c13667d70a78a3447388f5c3bb8" in script
    assert "96ad344138c66e82536422432ad742f015784942" in script
    assert "nvidia/Cosmos-Policy-LIBERO-Predict2-2B" in script
    assert "GEAR-Dreams/DreamZero-DROID" in script
    assert "sbatch" not in script
