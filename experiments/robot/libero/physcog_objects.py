"""
PhysCogSafe custom LIBERO object classes.

Registers GlassCup / SteelCup (L2-C1 cup experiment) and
GlassAkitaBlackBowl (L2-C2 in-distribution bowl experiment).

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
from libero.libero.envs.base_object import register_object

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
