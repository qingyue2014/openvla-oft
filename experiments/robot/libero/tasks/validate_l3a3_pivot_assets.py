#!/usr/bin/env python3
"""Fail-closed static audit for the L3-A3 pad/block MuJoCo assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


VERDICT = "PASS_L3A3_ASSET_VISIBILITY_CONVENTION_GATE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(path: Path, role: str) -> dict:
    root = ET.parse(path).getroot()
    geoms = list(root.iter("geom"))
    collision = [geom for geom in geoms if geom.get("group") == "0"]
    visual = [geom for geom in geoms if geom.get("group") == "1"]
    if len(collision) != 1 or len(visual) != 1:
        raise ValueError(
            f"{role} requires exactly one group-0 collision and one group-1 visual"
        )
    collision_geom = collision[0]
    visual_geom = visual[0]
    if collision_geom.get("contype", "1") == "0":
        raise ValueError(f"{role} collision geom has contype=0")
    if collision_geom.get("conaffinity", "1") == "0":
        raise ValueError(f"{role} collision geom has conaffinity=0")
    if (
        visual_geom.get("contype") != "0"
        or visual_geom.get("conaffinity") != "0"
    ):
        raise ValueError(f"{role} visual duplicate participates in collision")
    materials = {
        material.get("name"): material for material in root.iter("material")
    }
    material = materials.get(visual_geom.get("material"))
    if material is None:
        raise ValueError(f"{role} visual geom has no explicit material")
    rgba = [float(value) for value in material.get("rgba", "").split()]
    if len(rgba) != 4 or rgba[3] != 1.0:
        raise ValueError(f"{role} visual material must be fully opaque")
    if max(rgba[:3]) - min(rgba[:3]) < 0.5:
        raise ValueError(f"{role} visual material is not high contrast")
    if role == "middle_A":
        friction = [
            float(value)
            for value in collision_geom.get("friction", "").split()
        ]
        if not friction or friction[0] > 0.2:
            raise ValueError("middle_A support pad is not low friction")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "collision_geom": collision_geom.get("name"),
        "visual_geom": visual_geom.get("name"),
        "visual_rgba": rgba,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    evidence = {
        "schema": "physcog_l3a3_pivot_asset_audit_v1",
        "verdict": VERDICT,
        "assets": {
            "middle_A": _audit(Path(args.pad), "middle_A"),
            "top_B": _audit(Path(args.block), "top_B"),
        },
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# L3-A3 pivot custom-asset audit\n\n"
        f"- Verdict: **{VERDICT}**\n"
        "- Contract: group-0 collision + opaque high-contrast group-1 visual-only.\n"
        "- Middle A additionally requires low sliding friction.\n\n"
        f"```json\n{json.dumps(evidence, indent=2, sort_keys=True)}\n```\n"
    )
    print(VERDICT)


if __name__ == "__main__":
    main()
