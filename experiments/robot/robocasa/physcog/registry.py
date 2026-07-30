"""Scene registry for the RoboCasa PhysCogSafe suite.

Each sub-level module under ``experiments/robot/robocasa/envs/`` exposes a
module-level ``SCENES`` tuple of scene classes. This module aggregates them so
runners and the static checker can iterate the whole suite by id.
"""

from __future__ import annotations

import importlib
from typing import Dict, List

# Official source checkout used for the initial RoboCasa implementation audit.
# Live manifests additionally record the actually imported source file, SHA256,
# and Git commit; this constant is provenance, not a substitute for that check.
VALIDATED_ROBOCASA_SOURCE_COMMIT = (
    "b4684e6ee37d377cc392e98302a6b916d588b415"
)

#: sub-level module name -> sub-level id
SUBLEVEL_MODULES = {
    "l1_a": "L1-A",
    "l1_b": "L1-B",
    "l1_c": "L1-C",
    "l2_a": "L2-A",
    "l2_b": "L2-B",
    "l2_c": "L2-C",
    "l3_a": "L3-A",
    "l3_b": "L3-B",
    "l3_c": "L3-C",
}

SUBLEVEL_TITLES = {
    "L1-A": "static geometry perception (depth, occlusion, surface normals)",
    "L1-B": "swept-volume cognition (arm arc, links, held object)",
    "L1-C": "static configuration safety (stack stability, support dependency)",
    "L2-A": "inter-object semantic compatibility (hazard sources, exclusion zones)",
    "L2-B": "single-object safety properties (material-conditioned force/speed)",
    "L2-C": "referential safety (labels, latent hazard attributes, disambiguation)",
    "L3-A": "cascading physical consequences (support removal, momentum chains)",
    "L3-B": "residual risk state (irreversibility, preconditions)",
    "L3-C": "temporal shared-space conflict (external dynamics, concurrent interference)",
}


def _load(module_name: str) -> List[type]:
    mod = importlib.import_module(f"experiments.robot.robocasa.envs.{module_name}")
    return list(getattr(mod, "SCENES", ()))


def all_scenes() -> Dict[str, type]:
    """Return ``{scene_id: scene_class}`` for every implemented sub-level."""
    out: Dict[str, type] = {}
    for module_name in SUBLEVEL_MODULES:
        full_name = f"experiments.robot.robocasa.envs.{module_name}"
        try:
            scenes = _load(module_name)
        except ModuleNotFoundError as exc:
            # An unimplemented sub-level module is allowed while the suite is
            # being built. A missing dependency *inside* an existing module is
            # not: silently skipping it would bypass native-only preflight.
            if exc.name == full_name:
                continue
            raise RuntimeError(
                f"cannot load {full_name}; missing dependency {exc.name!r}"
            ) from exc
        for cls in scenes:
            sid = cls.physcog_scene_id
            if not isinstance(sid, str) or not sid:
                raise RuntimeError(f"{cls.__module__}.{cls.__name__}: invalid scene id")
            if sid in out:
                raise RuntimeError(f"duplicate scene id {sid}")
            out[sid] = cls
    return out


def get_scene(scene_id: str) -> type:
    scenes = all_scenes()
    if scene_id not in scenes:
        raise KeyError(f"unknown scene {scene_id!r}; known: {sorted(scenes)}")
    return scenes[scene_id]
