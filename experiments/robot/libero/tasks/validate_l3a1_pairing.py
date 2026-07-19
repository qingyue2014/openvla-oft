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
    "lean_deg", "lean_axis", "lean_direction_deg", "policy_entry_probe_actions",
    "settle_steps", "validation_hold_steps",
    "verify_close_steps", "min_topple_deg", "oracle_displacement_threshold",
    "oracle_height_drop_threshold", "stable_x_offset", "initialization_strategy",
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
            if isinstance(value, np.ndarray):
                value = value.tolist()
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
    lean_direction_deg: float | None = None,
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
            "lean_direction_deg": lean_direction_deg,
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
        count = len(group)
        base_state_hashes = []
        variant = str(group.attrs.get("l3a1_variant", ""))
        template_sha = None
        template_source_attempt = None
        for index in range(len(group)):
            demo = group[f"demo_{index}"]
            state = demo["initial_state"][:]
            if "base_reset_state" not in demo:
                raise ValueError(f"missing base_reset_state at demo_{index}")
            base = demo["base_reset_state"][:]
            if "initialization_mode" in demo.attrs:
                base_sha = hashlib.sha256(base.tobytes()).hexdigest()
                base_state_hashes.append(base_sha)
                if str(demo.attrs.get("base_state_sha256", "")) != base_sha:
                    raise ValueError(f"base_state_sha256 mismatch at demo_{index}")
                mode = str(demo.attrs["initialization_mode"])
                if variant == "risk":
                    expected_mode = (
                        "sampled_lean" if index == 0
                        else "support_relative_equilibrium_template"
                    )
                    if mode != expected_mode:
                        raise ValueError(
                            f"risk initialization_mode mismatch at demo_{index}: {mode!r}"
                        )
                    demo_template_sha = str(demo.attrs.get("template_sha256", ""))
                    demo_template_source = int(
                        demo.attrs.get("template_source_attempt", -1)
                    )
                    if index == 0:
                        template_sha = demo_template_sha
                        template_source_attempt = int(demo.attrs["reset_attempt"])
                        if not template_sha or demo_template_source != template_source_attempt:
                            raise ValueError("invalid risk template source metadata at demo_0")
                    elif (
                        demo_template_sha != template_sha
                        or demo_template_source != template_source_attempt
                    ):
                        raise ValueError(f"risk template reference mismatch at demo_{index}")
                elif variant == "stable" and mode != "paired_safe_transform":
                    raise ValueError(
                        f"stable initialization_mode mismatch at demo_{index}: {mode!r}"
                    )
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
            if float(demo.attrs.get("policy_entry_displacement_m", np.inf)) > 0.005:
                raise ValueError(f"policy entry drift exceeds 5 mm at demo_{index}")
            if int(demo.attrs.get("policy_entry_probe_count", -1)) != 3:
                raise ValueError(f"policy entry probe count is not 3 at demo_{index}")
            if str(demo.attrs.get("policy_entry_direct_contacts", "missing")):
                raise ValueError(f"policy entry has direct contact at demo_{index}")
        if base_state_hashes and len(set(base_state_hashes)) != count:
            raise ValueError("formal artifact reuses duplicate native base reset states")
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
    parser.add_argument("--expected_lean_direction_deg", type=float)
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
        lean_direction_deg=args.expected_lean_direction_deg,
        minimum_count=args.minimum_count,
    )
    if args.print_binding:
        validate_base_preservation(args.er, args.task_description)
        print(artifact_binding(args.er, args.task_description))
        return
    if not args.ec:
        count = validate_base_preservation(args.er, args.task_description)
        print(f"PASS_L3A1_BASE_STATE_PRESERVED count={count}")
        return
    attempts = validate_pairing(args.er, args.ec, args.task_description)
    print(
        f"PASS_L3A1_PAIRED_SERIALIZED_STATES count={len(attempts)} "
        f"attempts={','.join(map(str, attempts))}"
    )


if __name__ == "__main__":
    main()
