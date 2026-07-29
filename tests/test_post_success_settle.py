from types import SimpleNamespace

from experiments.robot.libero.post_success import settle_after_success


class _Env:
    def __init__(self, transitions):
        self.transitions = iter(transitions)

    def step(self, action):
        obs, done = next(self.transitions)
        return obs, 0.0, done, {}


class _Recorder:
    def __init__(self):
        self.rows = []

    def record(self, obs, action, step, phase):
        self.rows.append((obs, action, step, phase))


def test_settle_records_every_observation_and_uses_final_native_success():
    env = _Env([("settle-1", True), ("settle-2", False)])
    recorder = _Recorder()
    captured = []

    obs, success = settle_after_success(
        env,
        initial_obs="terminal",
        dummy_action="noop",
        num_steps=2,
        start_step=11,
        recorder=recorder,
        capture_observation=captured.append,
    )

    assert obs == "settle-2"
    assert success is False
    assert captured == ["terminal", "settle-1", "settle-2"]
    assert recorder.rows == [
        ("settle-1", "noop", 11, "settle"),
        ("settle-2", "noop", 12, "settle"),
    ]


def test_settle_can_recheck_oracle_success_and_stop_after_safety_event():
    env = _Env([("settle-1", False), ("settle-2", False), ("unused", True)])
    captured = []
    oracle = SimpleNamespace(values=iter([True, False]))

    obs, success = settle_after_success(
        env,
        initial_obs="terminal",
        dummy_action="noop",
        num_steps=3,
        start_step=20,
        capture_observation=captured.append,
        success_after_step=lambda _done: next(oracle.values),
        check_safety=lambda _obs, _action, step: step == 21,
    )

    assert obs == "settle-2"
    assert success is False
    assert captured == ["terminal", "settle-1", "settle-2"]


def test_zero_settle_steps_preserve_initial_success():
    captured = []
    obs, success = settle_after_success(
        _Env([]),
        initial_obs="terminal",
        dummy_action="noop",
        num_steps=0,
        start_step=1,
        initial_success=True,
        capture_observation=captured.append,
    )

    assert obs == "terminal"
    assert success is True
    assert captured == ["terminal"]
