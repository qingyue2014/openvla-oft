#!/usr/bin/env python3
"""Hard native-only preflight for the L3-A1 V2 source task."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path


BDDL_BASENAME = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_"
    "cabinet_and_close_it.bddl"
)
EXPECTED_PROMPT = (
    "put the black bowl in the bottom drawer of the cabinet and close it"
)
EXPECTED_SHA256 = "5255fe54d7f25fad4dee8fa30a30033d8cb908f1708d953c91ab609264fb4fb8"
EXPECTED_FIXTURES = (
    "kitchen_table:kitchen_table",
    "white_cabinet_1:white_cabinet",
    "wine_rack_1:wine_rack",
)
EXPECTED_OBJECTS = (
    "akita_black_bowl_1:akita_black_bowl",
    "wine_bottle_1:wine_bottle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_language(text: str) -> str:
    match = re.search(r"\(:language\s+([^)]+)\)", text)
    if match is None:
        raise ValueError("native BDDL has no :language field")
    return " ".join(match.group(1).split())


def parse_typed_block(text: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"\(:{re.escape(name)}\s+(.*?)\n\s*\)",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"native BDDL has no :{name} block")
    rows = re.findall(r"([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)", match.group(1))
    return tuple(f"{instance}:{asset_type}" for instance, asset_type in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated_bddl", required=True)
    parser.add_argument("--evaluated_prompt", required=True)
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l3a1_v2_native_preflight.md",
    )
    args = parser.parse_args()

    libero_root = Path(os.environ.get("LIBERO_ROOT", "")).resolve()
    if not libero_root.is_dir():
        raise SystemExit("LIBERO_ROOT must point to the native LIBERO checkout")
    native = (
        libero_root
        / "libero"
        / "libero"
        / "bddl_files"
        / "libero_10"
        / BDDL_BASENAME
    ).resolve()
    evaluated = Path(args.evaluated_bddl).resolve()
    failures: list[str] = []
    if not native.is_file():
        failures.append(f"native BDDL missing: {native}")
    if not evaluated.is_file():
        failures.append(f"evaluated BDDL missing: {evaluated}")

    native_sha = evaluated_sha = ""
    prompt = ""
    fixtures: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    if not failures:
        if not evaluated.samefile(native):
            failures.append("evaluated BDDL is not the selected native source file")
        native_sha = sha256(native)
        evaluated_sha = sha256(evaluated)
        text = evaluated.read_text(encoding="utf-8")
        prompt = parse_language(text)
        fixtures = parse_typed_block(text, "fixtures")
        objects = parse_typed_block(text, "objects")
        if native_sha != EXPECTED_SHA256 or evaluated_sha != EXPECTED_SHA256:
            failures.append("native/evaluated BDDL SHA-256 mismatch")
        if prompt != EXPECTED_PROMPT or args.evaluated_prompt != EXPECTED_PROMPT:
            failures.append("evaluated prompt is not the exact native prompt")
        if fixtures != EXPECTED_FIXTURES:
            failures.append("native fixture inventory mismatch")
        if objects != EXPECTED_OBJECTS:
            failures.append("native object inventory mismatch")

    verdict = (
        "PASS_L3A1_V2_NATIVE_ONLY_PREFLIGHT"
        if not failures
        else "FAIL_L3A1_V2_NATIVE_ONLY_PREFLIGHT"
    )
    lines = [
        "# L3-A1 V2 native-only preflight",
        "",
        f"- Verdict: **{verdict}**",
        f"- Selected native task: `libero_10/{BDDL_BASENAME}`",
        f"- Exact original prompt: `{EXPECTED_PROMPT}`",
        f"- Native BDDL source: `{native}`",
        f"- Evaluated BDDL source: `{evaluated}`",
        f"- Native/evaluated BDDL SHA-256: `{native_sha or 'unavailable'}`",
        f"- Native/evaluated fixtures: `{', '.join(fixtures)}`",
        f"- Native/evaluated objects: `{', '.join(objects)}`",
        f"- Asset inventory identical: `{not failures and fixtures == EXPECTED_FIXTURES and objects == EXPECTED_OBJECTS}`",
        f"- Prompt identical: `{not failures and prompt == EXPECTED_PROMPT}`",
        "- Custom assets: `False`",
        "- Custom BDDL: `False`",
        "",
        "## Failures",
        "",
        *(f"- {failure}" for failure in failures),
    ]
    if not failures:
        lines.append("- None")
    output = Path(args.out_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(verdict)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
