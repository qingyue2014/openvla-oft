"""
PhysCogSafe custom LIBERO object classes.

Registers GlassCup / SteelCup (L2-C1 cup experiment),
GlassAkitaBlackBowl (L2-C2 in-distribution bowl experiment), and the
L3-A3 support-chain pivot assets.

Import this module before building any LIBERO environment that uses these
objects so that the @register_object decorators fire and the BDDL parser can
resolve the type names.

Usage (in eval scripts):
    import experiments.robot.libero.physcog_objects  # noqa: F401 — side-effect import
"""

import hashlib
import json
import os
import pathlib
import re
import tempfile
import xml.etree.ElementTree as ET

import libero
import libero.libero as libero_pkg
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import OBJECTS_DICT, register_object

# ── L2-C1 custom cylinder cups ────────────────────────────────────────────────

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

L3A1_SUPPORT_PANEL_BODY = "cabinet_bottom"
# Native white_cabinet.xml collision geoms are unnamed. MuJoCo assigns unstable
# gNN names at compile time, so bind each panel by its body-local signature.
L3A1_NATIVE_SIDE_PANELS = {
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

# Versioned native-cabinet topology used by the formal L3-A1 edge condition.
# The native XML leaves collision geoms unnamed, and MuJoCo's compiled gNN
# names are not stable.  Every role is therefore bound exclusively by its
# body-local collision signature.
L3A1_NATIVE_CORNER_EDGE_SIGNATURES = {
    "edge/front_outer": {
        "pos": [0.00334, -0.07524, 0.04476],
        "quat": [0.5, 0.5, -0.5, -0.5],
        "size": [0.00271, 0.03427, 0.10934],
    },
    "inner_front": {
        "pos": [0.00334, -0.06839, 0.04525],
        "quat": [0.5, 0.5, 0.5, 0.5],
        "size": [0.00356, 0.03214, 0.10679],
    },
    "side/right": {
        "pos": [0.10894, 0.01105, 0.04525],
        "quat": [0.70711, 0.70711, -0.00115, -0.00115],
        "size": [0.00241, 0.03133, 0.08148],
    },
}

L3A1_NATIVE_CORNER_EDGE_V1 = {
    "schema_version": 2,
    "topology_id": "native_white_cabinet_bottom_front_right_edge_v1",
    "body": L3A1_SUPPORT_PANEL_BODY,
    "roles": L3A1_NATIVE_CORNER_EDGE_SIGNATURES,
    "initial_support_roles": ["edge/front_outer"],
    "removal_component": ["edge/front_outer", "inner_front", "side/right"],
    "forbidden_initial_roles": ["inner_front", "side/right"],
}

_L3A1_GEOM_POS_ATOL = 1e-6
_L3A1_GEOM_QUAT_ATOL = 1e-5
_L3A1_GEOM_SIZE_ATOL = 1e-6
_MUJOCO_GEOM_BOX = 6


def l3a1_contract_json_and_sha256(contract: dict) -> tuple[str, str]:
    """Return canonical JSON and its SHA256 for an L3-A1 contract."""
    contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return contract_json, hashlib.sha256(contract_json.encode()).hexdigest()


def l3a1_native_corner_edge_contract_hash() -> str:
    """Return the deterministic hash of ``L3A1_NATIVE_CORNER_EDGE_V1``."""
    return l3a1_contract_json_and_sha256(L3A1_NATIVE_CORNER_EDGE_V1)[1]


def l3a1_native_edge_component_contract() -> dict:
    """Return an isolated copy of the canonical formal topology contract."""
    contract_json, _ = l3a1_contract_json_and_sha256(
        L3A1_NATIVE_CORNER_EDGE_V1
    )
    return json.loads(contract_json)


def _l3a1_model(env_or_model):
    """Accept a LIBERO env, MuJoCo sim, or compiled MuJoCo model."""
    candidate = env_or_model
    if hasattr(candidate, "sim"):
        candidate = candidate.sim
    if hasattr(candidate, "model"):
        candidate = candidate.model
    required = (
        "ngeom", "geom_bodyid", "geom_group", "geom_type", "geom_contype",
        "geom_conaffinity", "geom_pos", "geom_quat", "geom_size",
        "body_name2id", "geom_id2name", "geom_name2id",
    )
    missing = [name for name in required if not hasattr(candidate, name)]
    if missing:
        raise TypeError(f"object is not a compiled MuJoCo model; missing={missing}")
    return candidate


def _l3a1_quat_matches(actual, expected) -> bool:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return bool(
        np.allclose(actual, expected, atol=_L3A1_GEOM_QUAT_ATOL, rtol=0.0)
        or np.allclose(actual, -expected, atol=_L3A1_GEOM_QUAT_ATOL, rtol=0.0)
    )


def _l3a1_geom_matches_signature(model, geom_id: int, signature: dict) -> bool:
    return bool(
        np.allclose(
            model.geom_pos[geom_id], signature["pos"],
            atol=_L3A1_GEOM_POS_ATOL, rtol=0.0,
        )
        and _l3a1_quat_matches(model.geom_quat[geom_id], signature["quat"])
        and np.allclose(
            model.geom_size[geom_id], signature["size"],
            atol=_L3A1_GEOM_SIZE_ATOL, rtol=0.0,
        )
    )


def resolve_l3a1_native_corner_edge_geoms(
    env_or_model, support_body: str
) -> dict[str, str]:
    """Resolve the three canonical edge-component roles in a compiled model.

    Resolution is deliberately independent of compiled geom names.  Each role
    must match exactly one collidable box on ``support_body``; missing,
    duplicate, cross-body, non-box, non-collidable, or unnamed matches fail
    closed.
    """
    model = _l3a1_model(env_or_model)
    support_id = int(model.body_name2id(support_body))
    resolved: dict[str, str] = {}
    resolved_ids: set[int] = set()
    for role, signature in L3A1_NATIVE_CORNER_EDGE_SIGNATURES.items():
        signature_matches = [
            geom_id
            for geom_id in range(int(model.ngeom))
            if _l3a1_geom_matches_signature(model, geom_id, signature)
        ]
        cross_body = [
            geom_id for geom_id in signature_matches
            if int(model.geom_bodyid[geom_id]) != support_id
        ]
        if cross_body:
            raise RuntimeError(
                f"L3-A1 topology role {role!r} has cross-body signature "
                f"matches: geom_ids={cross_body}"
            )
        body_matches = [
            geom_id for geom_id in signature_matches
            if int(model.geom_bodyid[geom_id]) == support_id
        ]
        if len(body_matches) != 1:
            raise RuntimeError(
                f"L3-A1 topology role {role!r} must match exactly one geom on "
                f"{support_body!r}; geom_ids={body_matches}"
            )
        geom_id = body_matches[0]
        if (
            int(model.geom_group[geom_id]) != 0
            or int(model.geom_type[geom_id]) != _MUJOCO_GEOM_BOX
            or int(model.geom_contype[geom_id]) == 0
            or int(model.geom_conaffinity[geom_id]) == 0
        ):
            raise RuntimeError(
                f"L3-A1 topology role {role!r} is not a group-0 collidable box"
            )
        if geom_id in resolved_ids:
            raise RuntimeError(
                f"one compiled geom cannot satisfy multiple L3-A1 roles: {geom_id}"
            )
        geom_name = model.geom_id2name(geom_id)
        if not geom_name:
            raise RuntimeError(
                f"L3-A1 topology role {role!r} has no addressable runtime geom name"
            )
        # Round-trip the name to prevent an ambiguous or inconsistent model API
        # from silently binding the intervention to a different geom.
        if int(model.geom_name2id(geom_name)) != geom_id:
            raise RuntimeError(
                f"L3-A1 topology role {role!r} has an inconsistent runtime geom name"
            )
        resolved[role] = str(geom_name)
        resolved_ids.add(geom_id)
    return resolved


def validate_l3a1_native_corner_edge_runtime_binding(
    env_or_model, support_body: str, role_to_geom: dict[str, str]
) -> dict[str, str]:
    """Validate an explicit role binding and return the canonical resolution."""
    expected_roles = set(L3A1_NATIVE_CORNER_EDGE_SIGNATURES)
    if set(role_to_geom) != expected_roles:
        raise RuntimeError(
            "L3-A1 runtime binding roles differ from the canonical topology: "
            f"actual={sorted(role_to_geom)}, expected={sorted(expected_roles)}"
        )
    resolved = resolve_l3a1_native_corner_edge_geoms(env_or_model, support_body)
    if role_to_geom != resolved:
        raise RuntimeError(
            f"L3-A1 runtime role binding mismatch: actual={role_to_geom}, "
            f"expected={resolved}"
        )
    return resolved


class PhyscogXMLObject(MujocoXMLObject):
    """Base class for PhysCogSafe-injected objects.

    Loads XML from the project-local assets/ directory instead of LIBERO's
    built-in asset tree, so LIBERO itself does not need to be modified.
    """

    def __init__(self, name, obj_name, joints=None):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        xml_path = str(_ASSETS_DIR / obj_name / f"{obj_name}.xml")
        super().__init__(
            xml_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


def _l3a1_native_cabinet_xml() -> pathlib.Path:
    return (
        _libero_package_root()
        / "assets"
        / "articulated_objects"
        / "white_cabinet.xml"
    )


def l3a1_native_cabinet_asset_contract(side: str) -> dict[str, str]:
    """Return deterministic hashes for one native drawer side-panel binding."""
    if side not in L3A1_NATIVE_SIDE_PANELS:
        raise ValueError(f"unknown L3-A1 support side: {side!r}")
    native_xml = _l3a1_native_cabinet_xml()
    panel_contract = {
        "body": L3A1_SUPPORT_PANEL_BODY,
        "side": side,
        "signature": L3A1_NATIVE_SIDE_PANELS[side],
    }
    panel_json = json.dumps(panel_contract, sort_keys=True, separators=(",", ":"))
    return {
        "native_cabinet_xml_path": str(native_xml.resolve()),
        "native_cabinet_xml_sha256": hashlib.sha256(native_xml.read_bytes()).hexdigest(),
        "support_panel_contract_json": panel_json,
        "support_panel_contract_sha256": hashlib.sha256(panel_json.encode()).hexdigest(),
    }


def l3a1_native_corner_edge_asset_contract() -> dict[str, str]:
    """Bind the versioned edge topology to the current native cabinet XML."""
    native_xml = _l3a1_native_cabinet_xml()
    topology_json, topology_sha256 = l3a1_contract_json_and_sha256(
        L3A1_NATIVE_CORNER_EDGE_V1
    )
    return {
        "native_cabinet_xml_path": str(native_xml.resolve()),
        "native_cabinet_xml_sha256": hashlib.sha256(
            native_xml.read_bytes()
        ).hexdigest(),
        "l3a1_topology_id": L3A1_NATIVE_CORNER_EDGE_V1["topology_id"],
        "support_topology_contract_json": topology_json,
        "support_topology_contract_sha256": topology_sha256,
    }


@register_object
class GlassCup(PhyscogXMLObject):
    """Semi-transparent drinking glass — fragile condition for L2-C1."""

    def __init__(
        self,
        name="glass_cup",
        obj_name="glass_cup",
        joints=None,
    ):
        super().__init__(name, obj_name, joints)


@register_object
class SteelCup(PhyscogXMLObject):
    """Opaque stainless-steel cup — sturdy baseline for L2-C1."""

    def __init__(
        self,
        name="steel_cup",
        obj_name="steel_cup",
        joints=None,
    ):
        super().__init__(name, obj_name, joints)


# ── L3-A3 force-chain pivot assets ───────────────────────────────────────────

@register_object
class LThreeAThreeSupportPad(PhyscogXMLObject):
    """Broad cyan low-friction support pad (chain member A)."""

    def __init__(
        self,
        name="l_three_a_three_support_pad",
        obj_name="l3a3_support_pad",
        joints=None,
    ):
        super().__init__(name, obj_name, joints)


@register_object
class LThreeAThreeTopBlock(PhyscogXMLObject):
    """Compact magenta cylindrical top load (chain member B)."""

    def __init__(
        self,
        name="l_three_a_three_top_block",
        obj_name="l3a3_top_block",
        joints=None,
    ):
        super().__init__(name, obj_name, joints)


def assert_l3a3_pivot_objects_registered() -> None:
    """Fail before env construction if either custom BDDL type is unavailable."""
    expected = {
        "l_three_a_three_support_pad": LThreeAThreeSupportPad,
        "l_three_a_three_top_block": LThreeAThreeTopBlock,
    }
    mismatched = {
        key: OBJECTS_DICT.get(key)
        for key, target in expected.items()
        if OBJECTS_DICT.get(key) is not target
    }
    if mismatched:
        raise RuntimeError(
            "L3-A3 pivot object registry is incomplete: "
            f"actual={mismatched}, expected={sorted(expected)}"
        )


# ── L2-C2 in-distribution glass bowl ──────────────────────────────────────────

# Glass material properties: semi-transparent blue-white, high specular.
_GLASS_MAT = {
    "rgba":        "0.75 0.90 1.00 0.40",
    "reflectance": "0.92",
    "specular":    "1.0",
    "shininess":   "1.0",
}
# Attributes that reference a texture file — remove them for a clean glass look.
_TEXTURE_ATTRS = ("texture", "texrepeat", "texuniform")


def _libero_package_root() -> pathlib.Path:
    """Return the inner LIBERO package root that contains the assets directory."""
    candidates = []

    pkg_file = getattr(libero_pkg, "__file__", None)
    if pkg_file:
        candidates.append(pathlib.Path(pkg_file).resolve().parent)

    root_file = getattr(libero, "__file__", None)
    if root_file:
        root_dir = pathlib.Path(root_file).resolve().parent
        candidates.extend((root_dir, root_dir / "libero"))

    for package_path in getattr(libero, "__path__", []):
        root_dir = pathlib.Path(package_path).resolve()
        candidates.extend((root_dir, root_dir / "libero"))

    for candidate in candidates:
        if (candidate / "assets").is_dir():
            return candidate

    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not infer LIBERO package assets directory. Checked:\n  " + checked
    )


def _build_glass_xml(libero_obj_name: str) -> str:
    """Return path to a temporary XML with glass material replacing the original.

    The temp file uses absolute mesh paths so it can be placed anywhere.
    The caller is responsible for deleting it after MujocoXMLObject has parsed it.
    """
    libero_root = _libero_package_root()
    orig_xml = libero_root / "assets" / "stable_scanned_objects" / libero_obj_name / f"{libero_obj_name}.xml"
    orig_dir = orig_xml.parent
    if not orig_xml.is_file():
        raise FileNotFoundError(f"Could not find LIBERO object XML: {orig_xml}")

    tree = ET.parse(str(orig_xml))
    root = tree.getroot()
    asset_el = root.find("asset")

    # Rewrite mesh file paths to absolute so the temp XML can find them.
    for mesh in asset_el.findall("mesh"):
        fpath = mesh.get("file", "")
        if fpath and not os.path.isabs(fpath):
            mesh.set("file", str(orig_dir / fpath))

    # Remove texture elements (we replace with solid rgba glass look).
    for tex in list(asset_el.findall("texture")):
        asset_el.remove(tex)

    # Update material: strip texture refs, apply glass properties.
    for mat in asset_el.findall("material"):
        for attr in _TEXTURE_ATTRS:
            mat.attrib.pop(attr, None)
        for attr, val in _GLASS_MAT.items():
            mat.set(attr, val)

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    tmp.close()
    tree.write(tmp.name, encoding="unicode", xml_declaration=False)
    return tmp.name


@register_object
class GlassAkitaBlackBowl(MujocoXMLObject):
    """AkitaBlackBowl with glass visual material — in-distribution L2-C2 object.

    Geometry, mass, friction, and collision geoms are identical to the original
    akita_black_bowl used in libero_spatial fine-tuning.  Only the visual
    material is changed (semi-transparent blue-white, high reflectance) to
    signal fragility.  Any behavioral change the VLA shows is therefore
    attributable solely to the material appearance cue, not to shape OOD.

    MuJoCo body name convention: glass_akita_black_bowl_1_main
    """

    def __init__(self, name="glass_akita_black_bowl", joints=None):
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]
        tmp_path = _build_glass_xml("akita_black_bowl")
        try:
            super().__init__(
                tmp_path,
                name=name,
                joints=joints,
                obj_type="all",
                duplicate_collision_geoms=False,
            )
        finally:
            os.unlink(tmp_path)
        self.category_name = "glass_akita_black_bowl"
        self.rotation = (3.14159 / 2, 3.14159 / 2)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}
