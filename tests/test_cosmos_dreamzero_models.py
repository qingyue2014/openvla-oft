from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.cosmos_policy_utils import (
    COSMOS_CHECKPOINT_FILENAME,
    COSMOS_CONFIG_MODULE_PATH,
    COSMOS_DEFAULT_CHECKPOINT,
    COSMOS_DEFAULT_TOKENIZER,
    COSMOS_LIBERO_REPO_ID,
    COSMOS_TOKENIZER_REPO_ID,
    COSMOS_TOKENIZER_REVISION,
    CosmosPolicyClient,
    _encode_wire_value,
    _recv_message,
    _send_message,
    defer_unused_cosmos_base_checkpoint_downloads,
    get_cosmos_policy,
    is_cosmos_model_family,
    prepare_cosmos_libero_observation,
    resolve_cosmos_package_root,
    use_local_cosmos_tokenizer,
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


def test_cosmos_loopback_protocol_round_trips_numpy_arrays():
    sender, receiver = socket.socketpair()
    try:
        payload = {"actions": np.zeros((16, 7), dtype=np.float32)}
        _send_message(sender, payload)
        received = _recv_message(receiver)
    finally:
        sender.close()
        receiver.close()
    np.testing.assert_array_equal(received["actions"], payload["actions"])


def test_cosmos_wire_payload_has_no_numpy_module_dependency():
    import pickle

    payload = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "scalar": np.float32(0.25),
    }
    encoded = pickle.dumps(_encode_wire_value(payload), protocol=pickle.HIGHEST_PROTOCOL)
    assert b"numpy" not in encoded


def test_cosmos_remote_client_is_selected_without_importing_heavy_runtime(monkeypatch):
    monkeypatch.setattr(CosmosPolicyClient, "_request", lambda self, request: "pong")
    cfg = SimpleNamespace(
        cosmos_host="127.0.0.1",
        cosmos_port=18001,
        cosmos_connect_timeout_s=1.0,
    )
    client = get_cosmos_policy(cfg)
    assert isinstance(client, CosmosPolicyClient)
    monkeypatch.setattr(
        client,
        "_request",
        lambda request: np.zeros((16, 7), dtype=np.float32),
    )
    assert client.infer({}, "task").shape == (16, 7)


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


def test_cosmos_defers_only_eager_hf_config_resolution():
    cache_clears = []

    def get_checkpoint_path(uri):
        return checkpoint_db.get_checkpoint_by_hf(uri)

    get_checkpoint_path.cache_clear = lambda: cache_clears.append(True)
    original = lambda uri: f"/downloaded/{uri}"
    checkpoint_db = SimpleNamespace(
        get_checkpoint_by_hf=original,
        get_checkpoint_path=get_checkpoint_path,
    )

    with defer_unused_cosmos_base_checkpoint_downloads(checkpoint_db):
        assert checkpoint_db.get_checkpoint_path("hf://base/model.pt") == "hf://base/model.pt"

    assert checkpoint_db.get_checkpoint_by_hf is original
    assert cache_clears == [True, True]


def test_cosmos_uses_pinned_local_tokenizer(tmp_path):
    tokenizer = tmp_path / "tokenizer.pth"
    tokenizer.write_bytes(b"tokenizer")
    remote_calls = []
    original = lambda **kwargs: remote_calls.append(kwargs) or "/remote/file"
    checkpoint_utils = SimpleNamespace(hf_hub_download=original)

    with use_local_cosmos_tokenizer(checkpoint_utils, tokenizer):
        resolved = checkpoint_utils.hf_hub_download(
            repo_id=COSMOS_TOKENIZER_REPO_ID,
            filename="tokenizer/tokenizer.pth",
        )
        assert resolved == str(tokenizer)
        assert checkpoint_utils.hf_hub_download(
            repo_id="other/repo",
            filename="weights.pt",
        ) == "/remote/file"

    assert checkpoint_utils.hf_hub_download is original
    assert remote_calls == [{"repo_id": "other/repo", "filename": "weights.pt"}]


def test_dreamzero_aliases_and_libero_guard(tmp_path):
    assert is_dreamzero_model_family("dreamzero")
    assert is_dreamzero_model_family("dream-zero")
    cfg = SimpleNamespace(pretrained_checkpoint=tmp_path / "DreamZero-DROID")
    with pytest.raises(DreamZeroLiberoCompatibilityError, match="8-D DROID"):
        get_dreamzero_policy(cfg)


def test_checkpoint_identities_and_superpod_paths_are_explicit():
    assert COSMOS_LIBERO_REPO_ID == "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    assert COSMOS_DEFAULT_CHECKPOINT == Path("/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B")
    assert COSMOS_CHECKPOINT_FILENAME == "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    assert COSMOS_CONFIG_MODULE_PATH == "cosmos_policy/config/config.py"
    assert COSMOS_TOKENIZER_REPO_ID == "nvidia/Cosmos-Predict2-2B-Video2World"
    assert COSMOS_TOKENIZER_REVISION == "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
    assert COSMOS_DEFAULT_TOKENIZER == Path(
        "/project/trllmout/models/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
    )
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


def test_remote_agent_registers_all_l1c_model_evaluations():
    scenario = "l1c4"
    for model in ("pi05", "cosmos"):
        for kind, count_env in (
            ("preview", "L1C_PREVIEW_TRIALS"),
            ("smoke", "L1C_SMOKE_TRIALS"),
            ("formal", "L1C_FORMAL_TRIALS"),
        ):
            spec = PHASES[(scenario, f"{model}_{kind}")]
            assert spec.command[-3:] == (model, scenario, kind)
            assert spec.count_env == count_env
            assert any("L1-C4_task" in value for value in spec.artifacts)
    assert not any(
        scenario in {"l1c1", "l1c2", "l1c3"}
        and phase.startswith(("pi05_", "cosmos_"))
        for scenario, phase in PHASES
    )


def test_setup_script_pins_official_model_revisions():
    script = Path("experiments/robot/libero/tasks/setup_cosmos_dreamzero_models.sh").read_text()
    assert 'MODEL_ROOT="${MODEL_ROOT:-/project/trllmout/models}"' in script
    assert "cb689ec0e3347c13667d70a78a3447388f5c3bb8" in script
    assert "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18" in script
    assert "96ad344138c66e82536422432ad742f015784942" in script
    assert "nvidia/Cosmos-Policy-LIBERO-Predict2-2B" in script
    assert "GEAR-Dreams/DreamZero-DROID" in script
    assert "sbatch" not in script
