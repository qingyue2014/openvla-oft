import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


def _load_physcog_objects_without_libero_runtime():
    """Load the pure topology helpers without requiring a SuperPod env."""
    libero_root = types.ModuleType("libero")
    libero_pkg = types.ModuleType("libero.libero")
    libero_envs = types.ModuleType("libero.libero.envs")
    base_object = types.ModuleType("libero.libero.envs.base_object")
    robosuite = types.ModuleType("robosuite")
    robosuite_models = types.ModuleType("robosuite.models")
    robosuite_objects = types.ModuleType("robosuite.models.objects")

    class DummyMujocoXMLObject:
        pass

    def register_object(cls):
        return cls

    libero_root.libero = libero_pkg
    base_object.register_object = register_object
    robosuite_objects.MujocoXMLObject = DummyMujocoXMLObject
    stubs = {
        "libero": libero_root,
        "libero.libero": libero_pkg,
        "libero.libero.envs": libero_envs,
        "libero.libero.envs.base_object": base_object,
        "robosuite": robosuite,
        "robosuite.models": robosuite_models,
        "robosuite.models.objects": robosuite_objects,
    }
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/robot/libero/physcog_objects.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_l3a1_test_physcog_objects", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


_TOPOLOGY = _load_physcog_objects_without_libero_runtime()
L3A1_NATIVE_CORNER_EDGE_SIGNATURES = (
    _TOPOLOGY.L3A1_NATIVE_CORNER_EDGE_SIGNATURES
)
L3A1_NATIVE_CORNER_EDGE_V1 = _TOPOLOGY.L3A1_NATIVE_CORNER_EDGE_V1
l3a1_contract_json_and_sha256 = _TOPOLOGY.l3a1_contract_json_and_sha256
l3a1_native_corner_edge_contract_hash = (
    _TOPOLOGY.l3a1_native_corner_edge_contract_hash
)
l3a1_native_edge_component_contract = (
    _TOPOLOGY.l3a1_native_edge_component_contract
)
resolve_l3a1_native_corner_edge_geoms = (
    _TOPOLOGY.resolve_l3a1_native_corner_edge_geoms
)
validate_l3a1_native_corner_edge_runtime_binding = (
    _TOPOLOGY.validate_l3a1_native_corner_edge_runtime_binding
)


class FakeModel:
    def __init__(self, rows):
        self._body_names = {"white_cabinet_1_cabinet_bottom": 7, "other": 8}
        self._geom_names = [row["name"] for row in rows]
        self.ngeom = len(rows)
        self.geom_bodyid = np.asarray([row.get("body", 7) for row in rows])
        self.geom_group = np.asarray([row.get("group", 0) for row in rows])
        self.geom_type = np.asarray([row.get("type", 6) for row in rows])
        self.geom_contype = np.asarray([row.get("contype", 1) for row in rows])
        self.geom_conaffinity = np.asarray(
            [row.get("conaffinity", 1) for row in rows]
        )
        self.geom_pos = np.asarray([row["signature"]["pos"] for row in rows])
        self.geom_quat = np.asarray([row["signature"]["quat"] for row in rows])
        self.geom_size = np.asarray([row["signature"]["size"] for row in rows])

    def body_name2id(self, name):
        return self._body_names[name]

    def geom_id2name(self, geom_id):
        return self._geom_names[geom_id]

    def geom_name2id(self, name):
        return self._geom_names.index(name)


def _canonical_rows():
    # Deliberately avoid names resembling MuJoCo's unstable compiled gNN names.
    return [
        {"name": f"opaque_runtime_name_{index}", "signature": copy.deepcopy(signature)}
        for index, signature in enumerate(
            L3A1_NATIVE_CORNER_EDGE_SIGNATURES.values(), start=11
        )
    ]


def test_versioned_contract_has_exact_role_sets_and_stable_hash():
    assert L3A1_NATIVE_CORNER_EDGE_V1["schema_version"] == 2
    assert L3A1_NATIVE_CORNER_EDGE_V1["topology_id"] == (
        "native_white_cabinet_bottom_front_right_edge_v1"
    )
    assert L3A1_NATIVE_CORNER_EDGE_V1["body"] == "cabinet_bottom"
    assert L3A1_NATIVE_CORNER_EDGE_V1["initial_support_roles"] == [
        "edge/front_outer"
    ]
    assert set(L3A1_NATIVE_CORNER_EDGE_V1["removal_component"]) == {
        "edge/front_outer", "inner_front", "side/right"
    }
    assert set(L3A1_NATIVE_CORNER_EDGE_V1["forbidden_initial_roles"]) == {
        "inner_front", "side/right"
    }
    contract_json, digest = l3a1_contract_json_and_sha256(
        L3A1_NATIVE_CORNER_EDGE_V1
    )
    assert json.loads(contract_json) == L3A1_NATIVE_CORNER_EDGE_V1
    assert digest == hashlib.sha256(contract_json.encode()).hexdigest()
    assert digest == l3a1_native_corner_edge_contract_hash()
    isolated = l3a1_native_edge_component_contract()
    assert isolated == L3A1_NATIVE_CORNER_EDGE_V1
    isolated["roles"].clear()
    assert L3A1_NATIVE_CORNER_EDGE_V1["roles"]


def test_resolver_uses_signatures_not_runtime_names_and_accepts_quat_sign():
    rows = _canonical_rows()
    rows[0]["signature"]["quat"] = (
        -np.asarray(rows[0]["signature"]["quat"])
    ).tolist()
    resolved = resolve_l3a1_native_corner_edge_geoms(
        FakeModel(rows), "white_cabinet_1_cabinet_bottom"
    )
    assert resolved == {
        "edge/front_outer": "opaque_runtime_name_11",
        "inner_front": "opaque_runtime_name_12",
        "side/right": "opaque_runtime_name_13",
    }


def test_resolver_rejects_missing_signature():
    with pytest.raises(RuntimeError, match="must match exactly one"):
        resolve_l3a1_native_corner_edge_geoms(
            FakeModel(_canonical_rows()[1:]), "white_cabinet_1_cabinet_bottom"
        )


def test_resolver_rejects_duplicate_signature():
    rows = _canonical_rows()
    rows.append({
        "name": "duplicate_edge",
        "signature": copy.deepcopy(
            L3A1_NATIVE_CORNER_EDGE_SIGNATURES["edge/front_outer"]
        ),
    })
    with pytest.raises(RuntimeError, match="must match exactly one"):
        resolve_l3a1_native_corner_edge_geoms(
            FakeModel(rows), "white_cabinet_1_cabinet_bottom"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("group", 1), ("type", 5), ("contype", 0), ("conaffinity", 0)),
)
def test_resolver_rejects_noncollidable_or_noncanonical_geom(field, value):
    rows = _canonical_rows()
    rows[0][field] = value
    with pytest.raises(RuntimeError, match="group-0 collidable box"):
        resolve_l3a1_native_corner_edge_geoms(
            FakeModel(rows), "white_cabinet_1_cabinet_bottom"
        )


def test_resolver_rejects_cross_body_signature_match():
    rows = _canonical_rows()
    rows[0]["body"] = 8
    with pytest.raises(RuntimeError, match="cross-body"):
        resolve_l3a1_native_corner_edge_geoms(
            FakeModel(rows), "white_cabinet_1_cabinet_bottom"
        )


def test_explicit_runtime_binding_rejects_role_interchange():
    model = FakeModel(_canonical_rows())
    resolved = resolve_l3a1_native_corner_edge_geoms(
        model, "white_cabinet_1_cabinet_bottom"
    )
    swapped = dict(resolved)
    swapped["edge/front_outer"], swapped["inner_front"] = (
        swapped["inner_front"], swapped["edge/front_outer"]
    )
    with pytest.raises(RuntimeError, match="runtime role binding mismatch"):
        validate_l3a1_native_corner_edge_runtime_binding(
            model, "white_cabinet_1_cabinet_bottom", swapped
        )


def test_explicit_runtime_binding_rejects_missing_or_extra_role():
    model = FakeModel(_canonical_rows())
    with pytest.raises(RuntimeError, match="roles differ"):
        validate_l3a1_native_corner_edge_runtime_binding(
            model,
            "white_cabinet_1_cabinet_bottom",
            {
                "edge/front_outer": "opaque_runtime_name_11",
                "inner_front": "opaque_runtime_name_12",
            },
        )
