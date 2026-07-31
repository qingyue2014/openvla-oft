"""
physcog_trajectory.py

Per-episode trajectory recording for PhysCogSafe LIBERO evaluations.

Each episode is saved as one compressed .npz containing per-step:
  - end-effector position / quaternion and gripper joint positions
  - the action applied at that step
  - world poses of tracked MuJoCo bodies (target, distractors, hazards)
  - the step index and phase ("wait" | "policy" | "settle")

plus a JSON metadata blob (run id, seed, oracle, success/violation labels).

These files are the input for trajectory-level attribution: computing
Sim(tau_r, tau_b) between risk and benign rollouts, estimating the model's
natural trajectory variance from repeated benign rollouts, and classifying
episodes into SAR / UIR / OCR / NOR outcomes.
"""

import json
import os
from datetime import datetime

import numpy as np

_OBS_EEF_POS = "robot0_eef_pos"
_OBS_EEF_QUAT = "robot0_eef_quat"
_OBS_GRIPPER_QPOS = "robot0_gripper_qpos"


class TrajectoryRecorder:
    """Accumulates per-step state during one episode and saves it as .npz."""

    def __init__(self, env, tracked_bodies=None):
        self.env = env
        self._body_ids = {}
        for name in tracked_bodies or []:
            try:
                self._body_ids[name] = env.sim.model.body_name2id(name)
            except Exception:
                continue

        self.steps = []
        self.phases = []
        self.eef_pos = []
        self.eef_quat = []
        self.gripper_qpos = []
        self.actions = []
        self.body_pos = {name: [] for name in self._body_ids}
        self.body_quat = {name: [] for name in self._body_ids}
        self._warned_missing_obs = False

    def record(self, obs, action, step: int, phase: str = "policy"):
        self.steps.append(int(step))
        self.phases.append(phase)

        eef_pos = obs.get(_OBS_EEF_POS) if hasattr(obs, "get") else None
        eef_quat = obs.get(_OBS_EEF_QUAT) if hasattr(obs, "get") else None
        gripper = obs.get(_OBS_GRIPPER_QPOS) if hasattr(obs, "get") else None
        if eef_pos is None and not self._warned_missing_obs:
            print(f"[TrajectoryRecorder] obs missing '{_OBS_EEF_POS}'; recording NaN")
            self._warned_missing_obs = True
        self.eef_pos.append(_as_f32(eef_pos, 3))
        self.eef_quat.append(_as_f32(eef_quat, 4))
        self.gripper_qpos.append(_as_f32(gripper, 2))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))

        for name, body_id in self._body_ids.items():
            self.body_pos[name].append(
                self.env.sim.data.body_xpos[body_id].astype(np.float32).copy()
            )
            self.body_quat[name].append(
                self.env.sim.data.body_xquat[body_id].astype(np.float32).copy()
            )

    def save(self, path: str, metadata: dict = None) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {
            "steps": np.asarray(self.steps, dtype=np.int32),
            # Keep full task-specific phase labels (for example,
            # "reference_close_lift") in the auditable trajectory.
            "phases": np.asarray(self.phases, dtype="U32"),
            "eef_pos": _stack(self.eef_pos, 3),
            "eef_quat": _stack(self.eef_quat, 4),
            "gripper_qpos": _stack(self.gripper_qpos, 2),
            "actions": _stack_ragged(self.actions),
        }
        for name in self._body_ids:
            arrays[f"body_pos__{name}"] = _stack(self.body_pos[name], 3)
            arrays[f"body_quat__{name}"] = _stack(self.body_quat[name], 4)
        meta = dict(metadata or {})
        meta.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
        meta["tracked_bodies"] = sorted(self._body_ids)
        np.savez_compressed(path, metadata=json.dumps(meta), **arrays)
        return path


def load_trajectory(path: str) -> dict:
    """Load a saved episode into a dict of arrays plus parsed 'metadata'."""
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files if key != "metadata"}
        out["metadata"] = json.loads(str(data["metadata"]))
    return out


def append_index_entry(traj_dir: str, entry: dict) -> None:
    """Append one episode summary line to <traj_dir>/index.jsonl."""
    os.makedirs(traj_dir, exist_ok=True)
    with open(os.path.join(traj_dir, "index.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def collect_tracked_bodies(*comma_separated_lists) -> list:
    """Merge comma-separated body-name strings into a deduplicated list."""
    names = []
    for value in comma_separated_lists:
        if not value:
            continue
        for name in str(value).split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names


def _as_f32(value, dim: int) -> np.ndarray:
    if value is None:
        return np.full(dim, np.nan, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        out = np.full(dim, np.nan, dtype=np.float32)
        out[: min(dim, arr.shape[0])] = arr[:dim]
        return out
    return arr


def _stack(rows, dim: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    return np.stack(rows)


def _stack_ragged(rows) -> np.ndarray:
    """Stack action vectors, padding with NaN if dimensions differ."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    width = max(row.shape[0] for row in rows)
    out = np.full((len(rows), width), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : row.shape[0]] = row
    return out
