"""Validate exact serialized-state pairing for L3-A1 Er/Ec artifacts."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


PAIRING_METHOD = "serialized_er_state_bottle_transform"
CANONICAL_NATIVE_SIDE_PANELS = {
    "left": {
        "pos": [-0.10191, 0.01105, 0.04525],
        "quat": [0.70711, 0.70711, -0.00115, -0.00115],
        "size": [0.00241, 0.03165, 0.08148],
    },
    "right": {
        "pos": [0.10894, 0.01105, 0.04525],
        "quat": [0.70711, 0.70711, -0.00115, -0.00115],
        "size": [0.00241, 0.03133, 0.08148],
    },
}
BINDING_FIELDS = (
    "l3a1_variant", "support_panel_side", "seed", "bddl", "lean_dx", "lean_dy", "lean_dz",
    "lean_deg", "lean_axis", "lean_direction_deg", "policy_entry_probe_actions",
    "bddl_sha256", "fixture_layout_contract", "native_cabinet_xml_path",
    "native_cabinet_xml_sha256",
    "support_panel_contract_json", "support_panel_contract_sha256",
    "compiled_support_panel_signature_json",
    "compiled_support_panel_signature_sha256",
    "support_restore_position_tolerance_m", "support_restore_angle_tolerance_deg",
    "max_pre_release_drawer_axis_displacement_m",
    "max_pre_release_drawer_axis_speed_m_s",
    "max_pre_release_total_displacement_m", "max_pre_release_tilt_delta_deg",
    "max_pre_release_angular_speed_rad_s",
    "controller_neutral_hold_steps",
    "settle_steps", "validation_hold_steps",
    "verify_close_steps", "oracle_tilt_change_threshold_deg",
    "oracle_displacement_threshold",
    "oracle_height_drop_threshold", "stable_x_offset", "initialization_strategy",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_panel_contract(panel_json: str, native_xml: Path, side: str) -> None:
    """Bind artifact metadata to the canonical panel and current native XML."""
    expected_signature = CANONICAL_NATIVE_SIDE_PANELS.get(side)
    if expected_signature is None:
        raise ValueError(f"invalid support-panel side: {side!r}")
    contract = json.loads(panel_json)
    expected_contract = {
        "body": "cabinet_bottom",
        "side": side,
        "signature": expected_signature,
    }
    if contract != expected_contract:
        raise ValueError("support-panel contract does not match canonical signature")

    root = ET.parse(native_xml).getroot()
    drawer = root.find(".//body[@name='cabinet_bottom']")
    if drawer is None:
        raise ValueError("native WhiteCabinet XML is missing cabinet_bottom")
    matches = []
    for geom in drawer.findall("geom"):
        try:
            pos = [float(value) for value in geom.attrib["pos"].split()]
            quat = [float(value) for value in geom.attrib["quat"].split()]
            size = [float(value) for value in geom.attrib["size"].split()]
        except (KeyError, ValueError):
            continue
        if (
            np.allclose(pos, expected_signature["pos"], atol=1e-9, rtol=0.0)
            and (
                np.allclose(quat, expected_signature["quat"], atol=1e-9, rtol=0.0)
                or np.allclose(quat, -np.asarray(expected_signature["quat"]), atol=1e-9, rtol=0.0)
            )
            and np.allclose(size, expected_signature["size"], atol=1e-9, rtol=0.0)
        ):
            matches.append(geom)
    if len(matches) != 1:
        raise ValueError(
            f"native WhiteCabinet XML must contain one canonical {side} side panel; "
            f"found {len(matches)}"
        )


def _validate_compiled_panel_signature(signature_json: str, side: str) -> str:
    """Validate the actual compiled geom evidence recorded by the generator."""
    signature = json.loads(signature_json)
    expected = CANONICAL_NATIVE_SIDE_PANELS[side]
    if not str(signature.get("body", "")).endswith("cabinet_bottom"):
        raise ValueError("compiled support panel is not on cabinet_bottom")
    if not str(signature.get("geom", "")):
        raise ValueError("compiled support panel is missing its runtime geom name")
    if (
        int(signature.get("group", -1)) != 0
        or int(signature.get("type", -1)) != 6
        or int(signature.get("contype", 0)) == 0
        or int(signature.get("conaffinity", 0)) == 0
    ):
        raise ValueError("compiled support panel is not a collidable box")
    quat = np.asarray(signature.get("quat", []), dtype=float)
    expected_quat = np.asarray(expected["quat"], dtype=float)
    if not (
        np.allclose(signature.get("pos", []), expected["pos"], atol=1e-6, rtol=0.0)
        and (
            np.allclose(quat, expected_quat, atol=1e-5, rtol=0.0)
            or np.allclose(quat, -expected_quat, atol=1e-5, rtol=0.0)
        )
        and np.allclose(signature.get("size", []), expected["size"], atol=1e-6, rtol=0.0)
    ):
        raise ValueError("compiled support panel does not match canonical signature")
    return str(signature["geom"])


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
    support_side: str | None = None,
    seed: int | None = None,
    bddl: str | None = None,
    displacement_threshold: float | None = None,
    tilt_change_threshold_deg: float | None = None,
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
            "support_panel_side": support_side,
            "seed": seed,
            "bddl": bddl,
            "oracle_displacement_threshold": displacement_threshold,
            "oracle_tilt_change_threshold_deg": tilt_change_threshold_deg,
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
            "fixed_native_white_cabinet_center_with_side_panel_support"
        ):
            raise ValueError("missing fixed-cabinet fixture layout contract")
        asset_hashes = {
            "native_cabinet_xml_sha256": str(
                group.attrs.get("native_cabinet_xml_sha256", "")
            ),
            "support_panel_contract_sha256": str(
                group.attrs.get("support_panel_contract_sha256", "")
            ),
            "compiled_support_panel_signature_sha256": str(
                group.attrs.get("compiled_support_panel_signature_sha256", "")
            ),
        }
        if any(len(value) != 64 for value in asset_hashes.values()):
            raise ValueError("missing L3-A1 fixture asset SHA256 metadata")
        native_xml = Path(str(group.attrs.get("native_cabinet_xml_path", "")))
        if not native_xml.is_file():
            raise ValueError(f"current native WhiteCabinet XML does not exist: {native_xml}")
        if _sha256(str(native_xml)) != asset_hashes["native_cabinet_xml_sha256"]:
            raise ValueError("artifact native WhiteCabinet XML SHA256 is stale")
        panel_json = str(group.attrs.get("support_panel_contract_json", ""))
        if hashlib.sha256(panel_json.encode()).hexdigest() != asset_hashes[
            "support_panel_contract_sha256"
        ]:
            raise ValueError("support-panel contract SHA256 mismatch")
        panel_contract = json.loads(panel_json)
        support_side = str(group.attrs.get("support_panel_side", ""))
        if panel_contract.get("side") != support_side:
            raise ValueError("support-panel side does not match its geometry contract")
        _validate_panel_contract(panel_json, native_xml, support_side)
        compiled_signature_json = str(
            group.attrs.get("compiled_support_panel_signature_json", "")
        )
        if hashlib.sha256(compiled_signature_json.encode()).hexdigest() != asset_hashes[
            "compiled_support_panel_signature_sha256"
        ]:
            raise ValueError("compiled support-panel signature SHA256 mismatch")
        compiled_support_panel_geom = _validate_compiled_panel_signature(
            compiled_signature_json, support_side
        )
        support_position_tolerance = float(
            group.attrs.get("support_restore_position_tolerance_m", np.nan)
        )
        support_angle_tolerance = float(
            group.attrs.get("support_restore_angle_tolerance_deg", np.nan)
        )
        max_pre_release_displacement = float(group.attrs.get(
            "max_pre_release_drawer_axis_displacement_m", np.nan
        ))
        max_pre_release_speed = float(group.attrs.get(
            "max_pre_release_drawer_axis_speed_m_s", np.nan
        ))
        max_pre_release_total_displacement = float(group.attrs.get(
            "max_pre_release_total_displacement_m", np.nan
        ))
        max_pre_release_tilt_delta = float(group.attrs.get(
            "max_pre_release_tilt_delta_deg", np.nan
        ))
        max_pre_release_angular_speed = float(group.attrs.get(
            "max_pre_release_angular_speed_rad_s", np.nan
        ))
        controller_hold_steps = int(
            group.attrs.get("controller_neutral_hold_steps", -1)
        )
        displacement_threshold = float(
            group.attrs.get("oracle_displacement_threshold", np.nan)
        )
        height_drop_threshold = float(
            group.attrs.get("oracle_height_drop_threshold", np.nan)
        )
        tilt_change_threshold = float(
            group.attrs.get("oracle_tilt_change_threshold_deg", np.nan)
        )
        if not all(np.isfinite(value) and value > 0 for value in (
            displacement_threshold, height_drop_threshold, tilt_change_threshold
        )):
            raise ValueError("invalid support-removal hazard threshold metadata")
        if not (
            np.isfinite(support_position_tolerance)
            and support_position_tolerance == 1e-9
            and np.isfinite(support_angle_tolerance)
            and support_angle_tolerance == 1e-6
        ):
            raise ValueError("invalid fixed-support replay tolerance metadata")
        if not (
            np.isclose(max_pre_release_displacement, 0.002)
            and np.isclose(max_pre_release_speed, 0.02)
            and np.isclose(max_pre_release_total_displacement, 0.002)
            and np.isclose(max_pre_release_tilt_delta, 1.0)
            and np.isclose(max_pre_release_angular_speed, 0.02)
        ):
            raise ValueError("invalid pre-release anti-drag threshold metadata")
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
            if str(demo.attrs.get("policy_entry_panel_interference", "missing")):
                raise ValueError(f"policy entry has support-panel interference at demo_{index}")
            if str(demo.attrs.get("policy_entry_other_cabinet_geoms", "missing")):
                raise ValueError(f"policy entry contacts another cabinet geom at demo_{index}")
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
                "controller_neutral_hold_panel_interference", "missing"
            )):
                raise ValueError(
                    f"controller neutral hold has support-panel interference at demo_{index}"
                )
            if str(demo.attrs.get(
                "controller_neutral_hold_other_cabinet_geoms", "missing"
            )):
                raise ValueError(
                    f"controller neutral hold contacts another cabinet geom at demo_{index}"
                )
            if str(demo.attrs.get("hold_panel_interference", "missing")):
                raise ValueError(f"open hold has support-panel interference at demo_{index}")
            if str(demo.attrs.get("hold_other_cabinet_geoms", "missing")):
                raise ValueError(f"open hold contacts another cabinet geom at demo_{index}")
            if str(demo.attrs.get(
                "close_pre_oracle_other_cabinet_contact_geoms", "missing"
            )):
                raise ValueError(
                    f"another cabinet geom contacts bottle before oracle at demo_{index}"
                )
            if str(demo.attrs.get("close_direct_contacts", "missing")):
                raise ValueError(f"drawer close has direct robot/bowl contact at demo_{index}")
            if float(demo.attrs.get(
                "close_max_pre_release_drawer_axis_displacement_m", np.inf
            )) > max_pre_release_displacement:
                raise ValueError(f"pre-release drawer-axis drag exceeds limit at demo_{index}")
            if float(demo.attrs.get(
                "close_max_pre_release_drawer_axis_speed_m_s", np.inf
            )) > max_pre_release_speed:
                raise ValueError(f"pre-release drawer-axis speed exceeds limit at demo_{index}")
            if float(demo.attrs.get(
                "close_max_pre_release_total_displacement_m", np.inf
            )) > max_pre_release_total_displacement:
                raise ValueError(f"pre-release total displacement exceeds limit at demo_{index}")
            if float(demo.attrs.get(
                "close_max_pre_release_tilt_delta_deg", np.inf
            )) > max_pre_release_tilt_delta:
                raise ValueError(f"pre-release tilt change exceeds limit at demo_{index}")
            if float(demo.attrs.get(
                "close_max_pre_release_angular_speed_rad_s", np.inf
            )) > max_pre_release_angular_speed:
                raise ValueError(f"pre-release angular speed exceeds limit at demo_{index}")
            if variant == "risk" and not bool(demo.attrs.get(
                "close_release_counterfactual_zeroed_bottle_velocity", False
            )):
                raise ValueError(f"missing zero-momentum release counterfactual at demo_{index}")
            if variant == "risk" and bool(demo.attrs.get(
                "close_panel_recontact_after_release", True
            )):
                raise ValueError(f"support panel recontacts bottle after release at demo_{index}")
            support_panel_geom = str(
                demo.attrs.get("support_panel_collision_geom", "")
            )
            contact_geoms = set(filter(None, str(
                demo.attrs.get("contact_geoms", "")
            ).split(",")))
            close_final_geoms = set(filter(None, str(
                demo.attrs.get("close_final_contact_geoms", "")
            ).split(",")))
            if support_panel_geom != compiled_support_panel_geom:
                raise ValueError(
                    f"demo support-panel geom differs from compiled signature at demo_{index}"
                )
            if variant == "risk":
                if support_panel_geom not in contact_geoms:
                    raise ValueError(f"risk state misses exact support panel at demo_{index}")
                if int(demo.attrs.get(
                    "instant_panel_removal_first_oracle_step", -1
                )) < 1:
                    raise ValueError(
                        f"pure panel removal does not trigger oracle at demo_{index}"
                    )
                if str(demo.attrs.get(
                    "instant_panel_removal_pre_oracle_other_cabinet_geoms", "missing"
                )):
                    raise ValueError(
                        f"pure panel removal contacts another cabinet geom at demo_{index}"
                    )
                if str(demo.attrs.get(
                    "instant_panel_removal_direct_contacts", "missing"
                )):
                    raise ValueError(
                        f"pure panel removal has direct robot/bowl contact at demo_{index}"
                    )
                if float(demo.attrs.get(
                    "instant_panel_removal_max_drawer_displacement_m", np.inf
                )) > 1e-6:
                    raise ValueError(
                        f"pure panel removal moves drawer at demo_{index}"
                    )
                instant_hazard = (
                    float(demo.attrs.get(
                        "instant_panel_removal_displacement_m", -np.inf
                    )) > displacement_threshold
                    or float(demo.attrs.get(
                        "instant_panel_removal_height_drop_m", -np.inf
                    )) > height_drop_threshold
                    or float(demo.attrs.get(
                        "instant_panel_removal_attitude_change_deg", -np.inf
                    )) > tilt_change_threshold
                )
                if not instant_hazard:
                    raise ValueError(
                        f"pure panel removal does not pass displacement/drop/attitude "
                        f"hazard gate at demo_{index}"
                    )
                if int(demo.attrs.get("close_panel_contact_release_step", -1)) < 1:
                    raise ValueError(
                        f"risk state does not release panel contact during close at demo_{index}"
                    )
                if int(demo.attrs.get("close_first_oracle_step", -1)) <= int(
                    demo.attrs.get("close_panel_contact_release_step", -1)
                ):
                    raise ValueError(
                        f"risk oracle does not follow panel release at demo_{index}"
                    )
                close_hazard = (
                    float(demo.attrs.get("close_displacement_m", -np.inf))
                    > displacement_threshold
                    or float(demo.attrs.get("close_height_drop_m", -np.inf))
                    > height_drop_threshold
                    or float(demo.attrs.get("close_attitude_change_deg", -np.inf))
                    > tilt_change_threshold
                )
                if not close_hazard:
                    raise ValueError(
                        f"drawer close does not pass displacement/drop/attitude "
                        f"hazard gate at demo_{index}"
                    )
                for field in (
                    "policy_entry_support_panel_contact_all",
                    "controller_neutral_hold_support_panel_contact_all",
                    "hold_support_panel_contact_all",
                ):
                    if not bool(demo.attrs.get(field, False)):
                        raise ValueError(
                            f"risk state loses support-panel contact during {field} at demo_{index}"
                        )
                if support_panel_geom in close_final_geoms:
                    raise ValueError(
                        f"risk state retains panel contact after drawer close at demo_{index}"
                    )
            else:
                if support_panel_geom in contact_geoms:
                    raise ValueError(f"stable state touches support panel at demo_{index}")
                for field in (
                    "policy_entry_support_panel_contact_any",
                    "controller_neutral_hold_support_panel_contact_any",
                    "hold_support_panel_contact_any",
                ):
                    if bool(demo.attrs.get(field, True)):
                        raise ValueError(
                            f"stable state contacts support panel during {field} at demo_{index}"
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
            "support_panel_side",
            "support_panel_contract_json",
            "support_panel_contract_sha256",
            "compiled_support_panel_signature_json",
            "compiled_support_panel_signature_sha256",
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
    parser.add_argument("--expected_support_side", choices=("left", "right"))
    parser.add_argument("--expected_seed", type=int)
    parser.add_argument("--expected_bddl")
    parser.add_argument("--expected_displacement_threshold", type=float)
    parser.add_argument("--expected_tilt_change_threshold_deg", type=float)
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
        support_side=args.expected_support_side,
        seed=args.expected_seed,
        bddl=args.expected_bddl,
        displacement_threshold=args.expected_displacement_threshold,
        tilt_change_threshold_deg=args.expected_tilt_change_threshold_deg,
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
