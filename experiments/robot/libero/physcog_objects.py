"""
PhysCogSafe custom LIBERO object classes.

Registers GlassCup and SteelCup — two objects with identical geometry but
different visual materials (same-shape / different-material control).  Import
this module before building any LIBERO environment that uses these objects so
that the @register_object decorators fire and the BDDL parser can resolve the
type names ``glass_cup`` and ``steel_cup``.

Usage (in eval scripts):
    import experiments.robot.libero.physcog_objects  # noqa: F401 — side-effect import
"""

import os
import pathlib
import re

from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object

_ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


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
