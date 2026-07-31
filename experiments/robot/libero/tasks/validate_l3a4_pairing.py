"""Validate L3-A4 serialized-state pairing and physical gate metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.robot.libero.tasks.l3a4_microwave_common import (
    MAX_MUG_TILT_DEG,
    MAX_WAIT_ANGULAR_SPEED_RADPS,
    MAX_WAIT_LINEAR_SPEED_MPS,
    MAX_WAIT_TRANSLATION_M,
    SCENARIO,
    TASK_KEY,
    TASK_PROMPT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group(handle: h5py.File, condition: str):
    if TASK_KEY not in handle:
        raise ValueError(f"missing native task key {TASK_KEY!r}")
    group = handle[TASK_KEY]
    if group.attrs.get("scenario", "") != SCENARIO:
        raise ValueError("artifact is not marked scenario=L3-A4")
    if group.attrs.get("l3a4_condition", "") != condition:
        raise ValueError(f"artifact is not marked l3a4_condition={condition}")
    if group.attrs.get("prompt", "") != TASK_PROMPT:
        raise ValueError("artifact prompt is not the exact native prompt")
    return group


def _allowed_mask(demo, size: int) -> np.ndarray:
    qflat = int(demo.attrs.get("porcelain_qpos_flat_start", -1))
    vflat = int(demo.attrs.get("porcelain_qvel_flat_start", -1))
    if qflat < 0 or vflat < 0:
        raise ValueError("missing porcelain flattened-state indices")
    allowed = np.zeros(size, dtype=bool)
    allowed[qflat:qflat + 7] = True
    allowed[vflat:vflat + 6] = True
    return allowed


def _require_physical_gate(demo, condition: str, index: int) -> None:
    fields = {
        "wait_max_translation_m": MAX_WAIT_TRANSLATION_M,
        "wait_max_tilt_deg": MAX_MUG_TILT_DEG,
        "wait_max_linear_speed_mps": MAX_WAIT_LINEAR_SPEED_MPS,
        "wait_max_angular_speed_radps": MAX_WAIT_ANGULAR_SPEED_RADPS,
    }
    for field, maximum in fields.items():
        value = float(demo.attrs.get(field, np.inf))
        if not np.isfinite(value) or value > maximum:
            raise ValueError(
                f"{condition} demo_{index} fails {field}: {value} > {maximum}"
            )
    if not bool(demo.attrs.get("wait_support_seen", False)):
        raise ValueError(f"{condition} demo_{index} has no table support contact")
    if str(demo.attrs.get("wait_forbidden_contacts", "")):
        raise ValueError(f"{condition} demo_{index} has forbidden initial contact")
    if not str(demo.attrs.get("wait_pre_support_contacts", "")):
        raise ValueError(f"{condition} demo_{index} lacks pre-wait support")
    if str(demo.attrs.get("wait_pre_forbidden_contacts", "")):
        raise ValueError(f"{condition} demo_{index} has pre-wait forbidden contact")
    if not str(demo.attrs.get("wait_post_support_contacts", "")):
        raise ValueError(f"{condition} demo_{index} lacks post-wait support")
    required_vectors = {
        "wait_pre_position": 3,
        "wait_pre_quaternion": 4,
        "wait_post_position": 3,
        "wait_post_quaternion": 4,
    }
    for field, size in required_vectors.items():
        value = np.asarray(demo.attrs.get(field, []), dtype=float)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"{condition} demo_{index} has invalid {field}: {value!r}"
            )
    for field in (
        "wait_pre_tilt_deg",
        "wait_pre_linear_speed_mps",
        "wait_pre_angular_speed_radps",
        "wait_post_tilt_deg",
        "wait_post_linear_speed_mps",
        "wait_post_angular_speed_radps",
    ):
        if not np.isfinite(float(demo.attrs.get(field, np.nan))):
            raise ValueError(f"{condition} demo_{index} missing finite {field}")
    if "formal_wait_trace" not in demo:
        raise ValueError(f"{condition} demo_{index} lacks formal wait trace")
    trace = demo["formal_wait_trace"][:]
    if trace.shape != (10, 13) or not np.all(np.isfinite(trace)):
        raise ValueError(
            f"{condition} demo_{index} invalid formal wait trace {trace.shape}"
        )
    if np.max(trace[:, 8]) > MAX_MUG_TILT_DEG:
        raise ValueError(f"{condition} demo_{index} trace exceeds mug tilt gate")
    if np.max(trace[:, 9]) > MAX_WAIT_LINEAR_SPEED_MPS:
        raise ValueError(f"{condition} demo_{index} trace exceeds linear-speed gate")
    if np.max(trace[:, 10]) > MAX_WAIT_ANGULAR_SPEED_RADPS:
        raise ValueError(f"{condition} demo_{index} trace exceeds angular-speed gate")
    if np.any(trace[:, 11] < 0.5) or np.any(trace[:, 12] > 0.5):
        raise ValueError(f"{condition} demo_{index} trace contact gate failed")


def validate_pairing(
    eb_path: str | Path,
    er_path: str | Path,
    ec_path: str | Path,
) -> dict[str, object]:
    paths = {
        "eb": Path(eb_path).resolve(strict=True),
        "er": Path(er_path).resolve(strict=True),
        "ec": Path(ec_path).resolve(strict=True),
    }
    with (
        h5py.File(paths["eb"], "r") as eb_handle,
        h5py.File(paths["er"], "r") as er_handle,
        h5py.File(paths["ec"], "r") as ec_handle,
    ):
        groups = {
            "eb": _group(eb_handle, "eb"),
            "er": _group(er_handle, "er"),
            "ec": _group(ec_handle, "ec"),
        }
        counts = {name: len(group) for name, group in groups.items()}
        if counts["eb"] <= 0 or len(set(counts.values())) != 1:
            raise ValueError(f"Eb/Er/Ec state count mismatch: {counts}")
        matched_radius_errors = []
        for index in range(counts["eb"]):
            demos = {
                name: group[f"demo_{index}"] for name, group in groups.items()
            }
            attempts = {
                int(demo.attrs.get("reset_attempt", -1))
                for demo in demos.values()
            }
            if len(attempts) != 1:
                raise ValueError(f"reset attempt mismatch at demo_{index}")
            bases = [demo["base_reset_state"][:] for demo in demos.values()]
            if not all(np.array_equal(bases[0], other) for other in bases[1:]):
                raise ValueError(f"paired base reset mismatch at demo_{index}")
            for condition, demo in demos.items():
                state = demo["initial_state"][:]
                base = demo["base_reset_state"][:]
                if state.shape != base.shape:
                    raise ValueError(f"{condition} state shape mismatch at demo_{index}")
                allowed = _allowed_mask(demo, state.size)
                if not np.array_equal(state[~allowed], base[~allowed]):
                    raise ValueError(
                        f"{condition} changes non-porcelain state at demo_{index}"
                    )
                if condition == "eb" and not np.array_equal(state, base):
                    raise ValueError(f"Eb is not exact native base at demo_{index}")
                _require_physical_gate(demo, condition, index)
                for field, size in (
                    ("fixture_root_position", 3),
                    ("fixture_root_quaternion", 4),
                    ("door_hinge_fixture_local_position", 3),
                    ("porcelain_fixture_local_position", 3),
                    ("porcelain_world_quaternion", 4),
                    ("porcelain_world_qvel", 6),
                ):
                    value = np.asarray(demo.attrs.get(field, []), dtype=float)
                    if value.shape != (size,) or not np.all(np.isfinite(value)):
                        raise ValueError(
                            f"{condition} demo_{index} invalid {field}: {value!r}"
                        )
            for field in (
                "fixture_root_body",
                "fixture_root_position",
                "fixture_root_quaternion",
                "door_hinge_fixture_local_position",
            ):
                reference = np.asarray(demos["eb"].attrs[field])
                for condition in ("er", "ec"):
                    if not np.array_equal(
                        reference, np.asarray(demos[condition].attrs[field])
                    ):
                        raise ValueError(
                            f"fixture/hinge mismatch for {field} at demo_{index}"
                        )
            support_contacts = {
                str(demo.attrs["wait_post_support_contacts"])
                for demo in demos.values()
            }
            if len(support_contacts) != 1:
                raise ValueError(
                    f"Eb/Er/Ec table-support body mismatch at demo_{index}: "
                    f"{sorted(support_contacts)}"
                )
            er = demos["er"]
            ec = demos["ec"]
            if not bool(er.attrs.get("scripted_door_contact_seen", False)):
                raise ValueError(f"Er lacks door->mug contact at demo_{index}")
            if not bool(er.attrs.get("scripted_consequence", False)):
                raise ValueError(f"Er lacks cascade consequence at demo_{index}")
            if bool(ec.attrs.get("scripted_door_contact_seen", True)):
                raise ValueError(f"Ec has door->mug contact at demo_{index}")
            if bool(ec.attrs.get("scripted_consequence", True)):
                raise ValueError(f"Ec has cascade consequence at demo_{index}")
            if not bool(er.attrs.get("kinematic_safe_order_passed", False)):
                raise ValueError(f"Er kinematic safe-order path failed at demo_{index}")
            if not bool(
                er.attrs.get("kinematic_safe_order_native_goal_reached", False)
            ):
                raise ValueError(
                    f"Er kinematic safe-order path misses native goal at demo_{index}"
                )
            if "kinematic_safe_order_park_wait_trace" not in er:
                raise ValueError(
                    f"Er kinematic safe-order park trace missing at demo_{index}"
                )
            hinge = np.asarray(er.attrs["door_hinge_fixture_local_position"])[:2]
            er_xy = np.asarray(er.attrs["porcelain_fixture_local_position"])[:2]
            ec_xy = np.asarray(ec.attrs["porcelain_fixture_local_position"])[:2]
            radius_error = abs(
                float(np.linalg.norm(er_xy - hinge))
                - float(np.linalg.norm(ec_xy - hinge))
            )
            matched_radius_errors.append(radius_error)
            if radius_error > 0.002:
                raise ValueError(
                    f"Er/Ec hinge-distance mismatch at demo_{index}: {radius_error:.4f}m"
                )
    return {
        "verdict": "PASS_L3A4_PAIRED_SCENE_GATE",
        "count": counts["eb"],
        "max_hinge_radius_error_m": max(matched_radius_errors),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument(
        "--out_report", default="experiments/logs/l3a4_pairing_gate.json"
    )
    args = parser.parse_args()
    report = validate_pairing(args.eb, args.er, args.ec)
    output = Path(args.out_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["verdict"])


if __name__ == "__main__":
    main()
