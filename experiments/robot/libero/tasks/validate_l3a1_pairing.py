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
    "bddl_sha256", "fixture_layout_contract", "native_cabinet_xml_path",
    "native_cabinet_xml_sha256",
    "support_wing_contract_json", "support_wing_contract_sha256",
    "fixture_python_sha256",
    "support_restore_position_tolerance_m", "support_restore_angle_tolerance_deg",
    "controller_neutral_hold_steps",
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
        artifact_bddl = Path(str(group.attrs.get("bddl", "")))
        if not artifact_bddl.is_file():
            raise ValueError(f"artifact BDDL does not exist: {artifact_bddl}")
        current_bddl_sha = hashlib.sha256(artifact_bddl.read_bytes()).hexdigest()
        if str(group.attrs.get("bddl_sha256", "")) != current_bddl_sha:
            raise ValueError("artifact BDDL SHA256 does not match current task bytes")


def validate_base_preservation(path: str, task_description: str) -> int:
    key = task_description.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        group = handle[key]
        count = len(group)
        base_state_hashes = []
        variant = str(group.attrs.get("l3a1_variant", ""))
        if str(group.attrs.get("fixture_layout_contract", "")) != (
            "fixed_physcog_white_cabinet_native_center_with_support_wing"
        ):
            raise ValueError("missing fixed-cabinet fixture layout contract")
        fixture_python = Path(__file__).resolve().parents[1] / "physcog_objects.py"
        asset_hashes = {
            "native_cabinet_xml_sha256": str(
                group.attrs.get("native_cabinet_xml_sha256", "")
            ),
            "support_wing_contract_sha256": str(
                group.attrs.get("support_wing_contract_sha256", "")
            ),
            "fixture_python_sha256": str(
                group.attrs.get("fixture_python_sha256", "")
            ),
        }
        if any(len(value) != 64 for value in asset_hashes.values()):
            raise ValueError("missing L3-A1 fixture asset SHA256 metadata")
        native_xml = Path(str(group.attrs.get("native_cabinet_xml_path", "")))
        if not native_xml.is_file():
            raise ValueError(f"current native WhiteCabinet XML does not exist: {native_xml}")
        if _sha256(str(native_xml)) != asset_hashes["native_cabinet_xml_sha256"]:
            raise ValueError("artifact native WhiteCabinet XML SHA256 is stale")
        wing_json = str(group.attrs.get("support_wing_contract_json", ""))
        if hashlib.sha256(wing_json.encode()).hexdigest() != asset_hashes[
            "support_wing_contract_sha256"
        ]:
            raise ValueError("support-wing contract SHA256 mismatch")
        if _sha256(str(fixture_python)) != asset_hashes["fixture_python_sha256"]:
            raise ValueError("artifact fixture Python SHA256 does not match current source")
        support_position_tolerance = float(
            group.attrs.get("support_restore_position_tolerance_m", np.nan)
        )
        support_angle_tolerance = float(
            group.attrs.get("support_restore_angle_tolerance_deg", np.nan)
        )
        controller_hold_steps = int(
            group.attrs.get("controller_neutral_hold_steps", -1)
        )
        if not (
            np.isfinite(support_position_tolerance)
            and support_position_tolerance == 1e-9
            and np.isfinite(support_angle_tolerance)
            and support_angle_tolerance == 1e-6
        ):
            raise ValueError("invalid fixed-support replay tolerance metadata")
        if controller_hold_steps != 220:
            raise ValueError("controller neutral hold count is not 220")
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
            if str(demo.attrs.get("policy_entry_wing_interference", "missing")):
                raise ValueError(f"policy entry has support-wing interference at demo_{index}")
            support_relative_xyz = np.asarray([
                demo.attrs.get("policy_entry_support_relative_x_m", np.nan),
                demo.attrs.get("policy_entry_support_relative_y_m", np.nan),
                demo.attrs.get("policy_entry_support_relative_z_m", np.nan),
            ], dtype=float)
            if not np.all(np.isfinite(support_relative_xyz)):
                raise ValueError(
                    f"missing policy-entry support-relative pose at demo_{index}"
                )
            support_world_xyz = np.asarray([
                demo.attrs.get("support_world_x_m", np.nan),
                demo.attrs.get("support_world_y_m", np.nan),
                demo.attrs.get("support_world_z_m", np.nan),
            ], dtype=float)
            if not np.all(np.isfinite(support_world_xyz)):
                raise ValueError(
                    f"missing fixed support world pose at demo_{index}"
                )
            if float(demo.attrs.get(
                "support_restore_position_error_m", np.inf
            )) > support_position_tolerance:
                raise ValueError(f"support position replay mismatch at demo_{index}")
            if float(demo.attrs.get(
                "support_restore_angle_error_deg", np.inf
            )) > support_angle_tolerance:
                raise ValueError(f"support rotation replay mismatch at demo_{index}")
            if variant == "risk" and index > 0:
                if float(demo.attrs.get("template_position_error_m", np.inf)) > 1e-9:
                    raise ValueError(f"risk template position mismatch at demo_{index}")
                if float(demo.attrs.get("template_angle_error_deg", np.inf)) > 1e-6:
                    raise ValueError(f"risk template rotation mismatch at demo_{index}")
            if int(demo.attrs.get("controller_neutral_hold_steps", -1)) != controller_hold_steps:
                raise ValueError(
                    f"controller neutral hold count mismatch at demo_{index}"
                )
            if float(demo.attrs.get(
                "controller_neutral_hold_max_displacement_m", np.inf
            )) > 0.005:
                raise ValueError(
                    f"controller neutral hold drift exceeds 5 mm at demo_{index}"
                )
            if str(demo.attrs.get(
                "controller_neutral_hold_direct_contacts", "missing"
            )):
                raise ValueError(
                    f"controller neutral hold has direct contact at demo_{index}"
                )
            if str(demo.attrs.get(
                "controller_neutral_hold_wing_interference", "missing"
            )):
                raise ValueError(
                    f"controller neutral hold has support-wing interference at demo_{index}"
                )
            if str(demo.attrs.get("hold_wing_interference", "missing")):
                raise ValueError(f"open hold has support-wing interference at demo_{index}")
            support_wing_geom = str(
                demo.attrs.get("support_wing_collision_geom", "")
            )
            contact_geoms = set(filter(None, str(
                demo.attrs.get("contact_geoms", "")
            ).split(",")))
            close_final_geoms = set(filter(None, str(
                demo.attrs.get("close_final_contact_geoms", "")
            ).split(",")))
            if not support_wing_geom.endswith("l3a1_support_wing_collision"):
                raise ValueError(f"missing exact support-wing geom at demo_{index}")
            if variant == "risk":
                if support_wing_geom not in contact_geoms:
                    raise ValueError(f"risk state misses exact support wing at demo_{index}")
                for field in (
                    "policy_entry_support_wing_contact_all",
                    "controller_neutral_hold_support_wing_contact_all",
                    "hold_support_wing_contact_all",
                ):
                    if not bool(demo.attrs.get(field, False)):
                        raise ValueError(
                            f"risk state loses support-wing contact during {field} at demo_{index}"
                        )
                if support_wing_geom in close_final_geoms:
                    raise ValueError(
                        f"risk state retains wing contact after drawer close at demo_{index}"
                    )
            else:
                if support_wing_geom in contact_geoms:
                    raise ValueError(f"stable state touches support wing at demo_{index}")
                for field in (
                    "policy_entry_support_wing_contact_any",
                    "controller_neutral_hold_support_wing_contact_any",
                    "hold_support_wing_contact_any",
                ):
                    if bool(demo.attrs.get(field, True)):
                        raise ValueError(
                            f"stable state contacts support wing during {field} at demo_{index}"
                        )
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
        for field in (
            "native_cabinet_xml_path",
            "native_cabinet_xml_sha256",
            "support_wing_contract_json",
            "support_wing_contract_sha256",
            "fixture_python_sha256",
        ):
            if str(er_group.attrs.get(field, "")) != str(ec_group.attrs.get(field, "")):
                raise ValueError(f"Er/Ec fixture asset mismatch for {field}")

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
