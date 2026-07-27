"""Validate exact serialized-state pairing for L3-A1 Er/Ec artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


PAIRING_METHOD = "serialized_er_state_bottle_transform"
BINDING_FIELDS = (
    "l3a1_variant", "seed", "bddl", "lean_dx", "lean_dy", "lean_dz",
    "lean_deg", "lean_axis", "settle_steps", "validation_hold_steps",
    "verify_close_steps", "min_topple_deg", "oracle_displacement_threshold",
    "oracle_height_drop_threshold", "stable_x_offset", "fixture_pose_replay",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_binding(path: str, task_description: str) -> str:
    """Return deterministic JSON binding an artifact's bytes and formal metadata."""
    key = task_description.replace(" ", "_")
    artifact_sha256 = _sha256(path)
    with h5py.File(path, "r") as handle:
        group = handle[key]
        binding = {
            "artifact_sha256": artifact_sha256,
            "count": len(group),
            "task_key": key,
        }
        for field in BINDING_FIELDS:
            value = group.attrs.get(field, None)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode()
            binding[field] = value
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def validate_expected_config(
    path: str,
    task_description: str,
    *,
    variant: str | None = None,
    seed: int | None = None,
    bddl: str | None = None,
    displacement_threshold: float | None = None,
    lean_dx: float | None = None,
    lean_dy: float | None = None,
    lean_deg: float | None = None,
    minimum_count: int | None = None,
) -> None:
    key = task_description.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        group = handle[key]
        attrs = group.attrs
        if minimum_count is not None and len(group) < minimum_count:
            raise ValueError(
                f"artifact count {len(group)} is below required {minimum_count}"
            )
        expected = {
            "l3a1_variant": variant,
            "seed": seed,
            "bddl": bddl,
            "oracle_displacement_threshold": displacement_threshold,
            "lean_dx": lean_dx,
            "lean_dy": lean_dy,
            "lean_deg": lean_deg,
        }
        for field, wanted in expected.items():
            if wanted is None:
                continue
            actual = attrs.get(field, None)
            if isinstance(wanted, float):
                matches = actual is not None and np.isclose(
                    float(actual), wanted, rtol=0.0, atol=1e-12
                )
            else:
                matches = str(actual) == str(wanted)
            if not matches:
                raise ValueError(
                    f"artifact config mismatch for {field}: actual={actual!r}, expected={wanted!r}"
                )


def validate_base_preservation(path: str, task_description: str) -> int:
    key = task_description.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        group = handle[key]
        variant = str(group.attrs.get("l3a1_variant", ""))
        count = len(group)
        for index in range(len(group)):
            demo = group[f"demo_{index}"]
            state = demo["initial_state"][:]
            if "base_reset_state" not in demo:
                raise ValueError(f"missing base_reset_state at demo_{index}")
            base = demo["base_reset_state"][:]
            qpos_start = int(demo.attrs.get("bottle_qpos_flat_start", -1))
            qvel_start = int(demo.attrs.get("bottle_qvel_flat_start", -1))
            if qpos_start < 0 or qvel_start < 0 or state.shape != base.shape:
                raise ValueError(f"invalid bottle/base metadata at demo_{index}")
            allowed = np.zeros(state.size, dtype=bool)
            allowed[qpos_start:qpos_start + 7] = True
            allowed[qvel_start:qvel_start + 6] = True
            if not np.array_equal(state[~allowed], base[~allowed]):
                raise ValueError(f"non-bottle state differs from base reset at demo_{index}")
            if float(demo.attrs.get("initial_eef_drift_m", np.inf)) > 1e-10:
                raise ValueError(f"initial EEF drift is nonzero at demo_{index}")
            if float(demo.attrs.get("runtime_wait_displacement_m", np.inf)) > 0.005:
                raise ValueError(f"runtime wait drift exceeds 5 mm at demo_{index}")
            fixture_required = (
                "fixture_root_body",
                "fixture_root_position",
                "fixture_root_quaternion",
            )
            fixture_missing = [
                name for name in fixture_required if name not in demo.attrs
            ]
            if fixture_missing:
                raise ValueError(
                    f"missing native fixture replay metadata at demo_{index}: "
                    f"{fixture_missing}"
                )
            for name, size in (
                ("fixture_root_position", 3),
                ("fixture_root_quaternion", 4),
            ):
                value = np.asarray(demo.attrs[name], dtype=float)
                if value.shape != (size,) or not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"invalid {name} metadata at demo_{index}: {value!r}"
                    )
            if variant in {"risk", "stable"}:
                required = (
                    "support_body",
                    "bottle_body",
                    "support_relative_position",
                    "bottle_world_quaternion",
                    "bottle_world_qvel",
                )
                missing = [name for name in required if name not in demo.attrs]
                if missing:
                    raise ValueError(
                        f"missing native replay metadata at demo_{index}: {missing}"
                    )
                for name, size in (
                    ("support_relative_position", 3),
                    ("bottle_world_quaternion", 4),
                    ("bottle_world_qvel", 6),
                ):
                    value = np.asarray(demo.attrs[name], dtype=float)
                    if value.shape != (size,) or not np.all(np.isfinite(value)):
                        raise ValueError(
                            f"invalid {name} metadata at demo_{index}: {value!r}"
                        )
                if "runtime_wait_contacts" not in demo.attrs:
                    raise ValueError(
                        f"missing runtime_wait_contacts at demo_{index}"
                    )
                runtime_contacts = {
                    name
                    for name in str(demo.attrs["runtime_wait_contacts"]).split(",")
                    if name
                }
                forbidden_prefixes = (
                    ("akita_black_bowl_1", "wine_rack_1")
                    if variant == "risk"
                    else (
                        "akita_black_bowl_1",
                        "white_cabinet_1",
                        "wine_rack_1",
                    )
                )
                contamination = {
                    name for name in runtime_contacts
                    if name.startswith(forbidden_prefixes)
                }
                if contamination:
                    raise ValueError(
                        f"runtime-wait contamination at demo_{index}: "
                        f"{sorted(contamination)}"
                    )
    return count


def validate_baseline_pairing(
    eb_path: str, er_path: str, task_description: str
) -> int:
    """Require Eb to be the exact pre-intervention native base for every Er demo."""
    validate_base_preservation(eb_path, task_description)
    validate_base_preservation(er_path, task_description)
    key = task_description.replace(" ", "_")
    with h5py.File(eb_path, "r") as eb_handle, h5py.File(er_path, "r") as er_handle:
        eb_group, er_group = eb_handle[key], er_handle[key]
        if str(eb_group.attrs.get("l3a1_variant", "")) != "baseline":
            raise ValueError("Eb artifact is not marked l3a1_variant=baseline")
        for field in ("seed", "bddl", "source_task_key"):
            eb_value = eb_group.attrs.get(field, "")
            er_value = (
                key if field == "source_task_key"
                else er_group.attrs.get(field, "")
            )
            if str(eb_value) != str(er_value):
                raise ValueError(
                    f"Eb/Er metadata mismatch for {field}: "
                    f"{eb_value!r} != {er_value!r}"
                )
        if len(eb_group) == 0 or len(eb_group) != len(er_group):
            raise ValueError(
                f"Eb/Er state count mismatch: Eb={len(eb_group)}, Er={len(er_group)}"
            )
        for index in range(len(er_group)):
            eb_demo = eb_group[f"demo_{index}"]
            er_demo = er_group[f"demo_{index}"]
            if int(eb_demo.attrs.get("reset_attempt", -1)) != int(
                er_demo.attrs.get("reset_attempt", -2)
            ):
                raise ValueError(f"Eb/Er reset_attempt mismatch at demo_{index}")
            if not np.array_equal(
                eb_demo["initial_state"][:], er_demo["base_reset_state"][:]
            ):
                raise ValueError(
                    f"Eb is not the exact Er base_reset_state at demo_{index}"
                )
            for field in (
                "fixture_root_body",
                "fixture_root_position",
                "fixture_root_quaternion",
            ):
                if not np.array_equal(
                    np.asarray(eb_demo.attrs[field]),
                    np.asarray(er_demo.attrs[field]),
                ):
                    raise ValueError(
                        f"Eb/Er native fixture mismatch for {field} at demo_{index}"
                    )
        count = len(er_group)
    return count


def validate_pairing(er_path: str, ec_path: str, task_description: str) -> list[int]:
    validate_base_preservation(er_path, task_description)
    validate_base_preservation(ec_path, task_description)
    key = task_description.replace(" ", "_")
    with h5py.File(er_path, "r") as er_handle, h5py.File(ec_path, "r") as ec_handle:
        er_group, ec_group = er_handle[key], ec_handle[key]
        if len(er_group) == 0 or len(er_group) != len(ec_group):
            raise ValueError(f"Er/Ec state count mismatch: Er={len(er_group)}, Ec={len(ec_group)}")
        if ec_group.attrs.get("pairing_method", "") != PAIRING_METHOD:
            raise ValueError("Ec pairing_method does not identify a serialized Er-state transform")
        source = str(ec_group.attrs.get("paired_er_states", ""))
        if not source or Path(source).resolve() != Path(er_path).resolve():
            raise ValueError(f"Ec paired_er_states source mismatch: {source!r} != {er_path!r}")
        if ec_group.attrs.get("source_task_key", "") != key:
            raise ValueError("Ec source_task_key metadata mismatch")

        attempts = []
        for index in range(len(er_group)):
            er_demo = er_group[f"demo_{index}"]
            ec_demo = ec_group[f"demo_{index}"]
            er_attempt = int(er_demo.attrs["reset_attempt"])
            ec_attempt = int(ec_demo.attrs["reset_attempt"])
            if er_attempt != ec_attempt:
                raise ValueError(
                    f"Er/Ec reset_attempt mismatch at demo_{index}: {er_attempt} != {ec_attempt}"
                )
            if int(ec_demo.attrs.get("source_demo_index", -1)) != index:
                raise ValueError(f"Ec source_demo_index mismatch at demo_{index}")

            er_state = er_demo["initial_state"][:]
            ec_state = ec_demo["initial_state"][:]
            if er_state.shape != ec_state.shape:
                raise ValueError(f"Er/Ec state shape mismatch at demo_{index}")
            qpos_start = int(ec_demo.attrs.get("bottle_qpos_flat_start", -1))
            qvel_start = int(ec_demo.attrs.get("bottle_qvel_flat_start", -1))
            allowed = np.zeros(er_state.size, dtype=bool)
            if qpos_start < 0 or qvel_start < 0:
                raise ValueError(f"missing bottle flat-index metadata at demo_{index}")
            allowed[qpos_start:qpos_start + 7] = True
            allowed[qvel_start:qvel_start + 6] = True
            for condition, demo, state in (
                ("Er", er_demo, er_state), ("Ec", ec_demo, ec_state)
            ):
                if "base_reset_state" not in demo:
                    raise ValueError(f"{condition} missing base_reset_state at demo_{index}")
                base_state = demo["base_reset_state"][:]
                if base_state.shape != state.shape or not np.array_equal(
                    base_state[~allowed], state[~allowed]
                ):
                    raise ValueError(
                        f"{condition} non-bottle state differs from base reset at demo_{index}"
                    )
                if float(demo.attrs.get("initial_eef_drift_m", np.inf)) > 1e-10:
                    raise ValueError(f"{condition} initial EEF drift is nonzero at demo_{index}")
            if not np.array_equal(
                er_demo["base_reset_state"][:], ec_demo["base_reset_state"][:]
            ):
                raise ValueError(f"Er/Ec base_reset_state mismatch at demo_{index}")
            if not np.array_equal(er_state[~allowed], ec_state[~allowed]):
                changed = np.flatnonzero((er_state != ec_state) & ~allowed)
                raise ValueError(
                    f"non-bottle state mismatch at demo_{index}; changed indices={changed[:10].tolist()}"
                )
            for field in (
                "fixture_root_body",
                "fixture_root_position",
                "fixture_root_quaternion",
            ):
                if not np.array_equal(
                    np.asarray(er_demo.attrs[field]),
                    np.asarray(ec_demo.attrs[field]),
                ):
                    raise ValueError(
                        f"Er/Ec native fixture mismatch for {field} at demo_{index}"
                    )
            if np.array_equal(
                er_state[qpos_start:qpos_start + 7], ec_state[qpos_start:qpos_start + 7]
            ):
                raise ValueError(f"Ec bottle pose was not transformed at demo_{index}")
            attempts.append(er_attempt)

    if len(set(attempts)) != len(attempts):
        raise ValueError(f"Er reset_attempt values are not unique: {attempts}")
    return attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb")
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec")
    parser.add_argument("--task_description", required=True)
    parser.add_argument("--print_binding", action="store_true")
    parser.add_argument("--expected_variant")
    parser.add_argument("--expected_seed", type=int)
    parser.add_argument("--expected_bddl")
    parser.add_argument("--expected_displacement_threshold", type=float)
    parser.add_argument("--expected_lean_dx", type=float)
    parser.add_argument("--expected_lean_dy", type=float)
    parser.add_argument("--expected_lean_deg", type=float)
    parser.add_argument("--minimum_count", type=int)
    args = parser.parse_args()
    validate_expected_config(
        args.er,
        args.task_description,
        variant=args.expected_variant,
        seed=args.expected_seed,
        bddl=args.expected_bddl,
        displacement_threshold=args.expected_displacement_threshold,
        lean_dx=args.expected_lean_dx,
        lean_dy=args.expected_lean_dy,
        lean_deg=args.expected_lean_deg,
        minimum_count=args.minimum_count,
    )
    if args.print_binding:
        validate_base_preservation(args.er, args.task_description)
        print(artifact_binding(args.er, args.task_description))
        return
    baseline_count = None
    if args.eb:
        baseline_count = validate_baseline_pairing(
            args.eb, args.er, args.task_description
        )
    if not args.ec:
        count = validate_base_preservation(args.er, args.task_description)
        if baseline_count is not None:
            print(f"PASS_L3A1_PAIRED_NATIVE_BASELINE count={baseline_count}")
            return
        print(f"PASS_L3A1_BASE_STATE_PRESERVED count={count}")
        return
    attempts = validate_pairing(args.er, args.ec, args.task_description)
    print(
        f"PASS_L3A1_PAIRED_SERIALIZED_STATES count={len(attempts)} "
        f"attempts={','.join(map(str, attempts))}"
        + (
            f" baseline_count={baseline_count}"
            if baseline_count is not None else ""
        )
    )


if __name__ == "__main__":
    main()
