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
import hashlib
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
            "phases": np.asarray(self.phases, dtype="U8"),
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


def load_l3a1_episode_evidence(
    initial_states_path: str, task_description: str, episode_idx: int
) -> dict:
    """Load fail-closed L3-A1 topology evidence for one serialized episode."""
    import h5py

    digest = hashlib.sha256()
    with open(initial_states_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    key = task_description.replace(" ", "_")
    with h5py.File(initial_states_path, "r") as handle:
        if key not in handle or f"demo_{episode_idx}" not in handle[key]:
            raise ValueError(
                f"missing L3-A1 episode binding {key}/demo_{episode_idx}"
            )
        group = handle[key]
        demo = group[f"demo_{episode_idx}"]
        if int(group.attrs.get("l3a1_topology_schema_version", -1)) != 2:
            raise ValueError("L3-A1 artifact is not topology schema v2")
        group_map = {
            "l3a1_topology_id": "l3a1_topology_id",
            "support_topology_contract_sha256": "support_topology_contract_sha256",
            "native_cabinet_xml_sha256": "native_cabinet_xml_sha256",
            "compiled_support_component_signatures_sha256": (
                "compiled_support_component_signatures_sha256"
            ),
            "support_component_role_hashes_sha256": (
                "support_component_role_hashes_sha256"
            ),
        }
        demo_map = {
            "support_initial_component_roles": "initial_component_roles",
            "support_initial_edge_table_qualified": "initial_edge_table_qualified",
            "support_component_release_step_rC": "factual_close_component_release_step_rC",
            "support_first_oracle_step": "factual_close_first_oracle_step",
            "support_component_recontact_after_rC": (
                "factual_close_component_recontact_after_rC"
            ),
            "support_pre_oracle_other_cabinet_geoms": (
                "factual_close_pre_oracle_other_cabinet_geoms"
            ),
            "support_pre_oracle_direct_contact_bodies": (
                "factual_close_pre_oracle_direct_contact_bodies"
            ),
            "support_post_oracle_direct_contact_bodies": (
                "factual_close_post_oracle_direct_contact_bodies"
            ),
            "support_bottle_qvel_overwritten": "factual_close_bottle_qvel_overwritten",
        }
        missing = [source for source in group_map.values() if source not in group.attrs]
        missing += [source for source in demo_map.values() if source not in demo.attrs]
        if missing:
            raise ValueError(f"missing L3-A1 episode evidence fields: {sorted(missing)}")
        evidence = {
            target: _json_attr(group.attrs[source])
            for target, source in group_map.items()
        }
        evidence.update({
            target: _json_attr(demo.attrs[source])
            for target, source in demo_map.items()
        })
        evidence.update({
            "initial_states_artifact_sha256": digest.hexdigest(),
            "initial_states_demo_index": int(episode_idx),
            "initial_state_sha256": hashlib.sha256(
                demo["initial_state"][:].tobytes()
            ).hexdigest(),
            "l3a1_variant": str(group.attrs.get("l3a1_variant", "")),
        })
        return evidence


def load_l3a4_episode_evidence(
    initial_states_path: str, task_description: str, episode_idx: int
) -> dict:
    """Load fail-closed L3-A4 artifact and exact-state binding."""
    import h5py

    digest = hashlib.sha256()
    with open(initial_states_path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    key = task_description.replace(" ", "_")
    with h5py.File(initial_states_path, "r") as handle:
        if key not in handle or f"demo_{episode_idx}" not in handle[key]:
            raise ValueError(
                f"missing L3-A4 episode binding {key}/demo_{episode_idx}"
            )
        group = handle[key]
        demo = group[f"demo_{episode_idx}"]
        if int(group.attrs.get("l3a4_schema_version", -1)) != 1:
            raise ValueError("L3-A4 artifact is not schema v1")
        required = (
            "l3a4_topology_id",
            "l3a4_variant",
            "contract_sha256",
            "bddl_sha256",
        )
        missing = [name for name in required if name not in group.attrs]
        if missing:
            raise ValueError(f"missing L3-A4 binding fields: {missing}")
        return {
            "initial_states_artifact_sha256": digest.hexdigest(),
            "initial_states_demo_index": int(episode_idx),
            "initial_state_sha256": hashlib.sha256(
                demo["initial_state"][:].tobytes()
            ).hexdigest(),
            "l3a4_topology_id": str(group.attrs["l3a4_topology_id"]),
            "l3a4_variant": str(group.attrs["l3a4_variant"]),
            "l3a4_contract_sha256": str(group.attrs["contract_sha256"]),
            "l3a4_bddl_sha256": str(group.attrs["bddl_sha256"]),
        }


def _json_attr(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


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
