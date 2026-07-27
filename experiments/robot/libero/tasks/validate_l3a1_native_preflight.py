"""Hard-gate L3-A1 against the selected native LIBERO task definition."""

import argparse
import hashlib
import re
from pathlib import Path


TASK_FILE = (
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
    "of_the_cabinet_and_close_it.bddl"
)
TASK_PROMPT = "put the black bowl in the bottom drawer of the cabinet and close it"
EXPECTED_FIXTURES = {
    "kitchen_table": "kitchen_table",
    "white_cabinet_1": "white_cabinet",
    "wine_rack_1": "wine_rack",
}
EXPECTED_OBJECTS = {
    "akita_black_bowl_1": "akita_black_bowl",
    "wine_bottle_1": "wine_bottle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        raise ValueError(f"missing :{name} section")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"unterminated :{name} section")


def _typed_inventory(text: str, name: str) -> dict[str, str]:
    section = _section(text, name)
    return {
        instance: asset_type
        for instance, asset_type in re.findall(
            r"(?m)^\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*$", section
        )
    }


def validate_native_task(
    native_bddl: Path,
    evaluated_bddl: Path,
    evaluated_prompt: str,
) -> dict[str, object]:
    native = native_bddl.resolve(strict=True)
    evaluated = evaluated_bddl.resolve(strict=True)
    if native.name != TASK_FILE or native.parent.name != "libero_10":
        raise ValueError(f"unexpected native task source: {native}")
    if "bddl_files" not in native.parts:
        raise ValueError(f"native task is not under LIBERO bddl_files: {native}")
    if not native.samefile(evaluated):
        raise ValueError(
            "evaluated BDDL is not the selected native LIBERO task: "
            f"evaluated={evaluated}, native={native}"
        )

    text = native.read_text(encoding="utf-8")
    prompt_match = re.search(r"\(:language\s+([^)]+)\)", text)
    if prompt_match is None:
        raise ValueError("native BDDL has no :language prompt")
    native_prompt = " ".join(prompt_match.group(1).split())
    if native_prompt != TASK_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            "prompt mismatch: "
            f"native={native_prompt!r}, evaluated={evaluated_prompt!r}, "
            f"expected={TASK_PROMPT!r}"
        )

    fixtures = _typed_inventory(text, "fixtures")
    objects = _typed_inventory(text, "objects")
    if fixtures != EXPECTED_FIXTURES:
        raise ValueError(
            f"native fixture inventory mismatch: {fixtures!r} != {EXPECTED_FIXTURES!r}"
        )
    if objects != EXPECTED_OBJECTS:
        raise ValueError(
            f"native object inventory mismatch: {objects!r} != {EXPECTED_OBJECTS!r}"
        )
    return {
        "native_bddl": str(native),
        "evaluated_bddl": str(evaluated),
        "bddl_sha256": _sha256(native),
        "prompt": native_prompt,
        "fixtures": fixtures,
        "objects": objects,
    }


def _inventory_text(inventory: dict[str, str]) -> str:
    return ", ".join(f"{name}:{asset_type}" for name, asset_type in inventory.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_bddl", required=True)
    parser.add_argument("--evaluated_bddl", required=True)
    parser.add_argument("--evaluated_prompt", required=True)
    parser.add_argument(
        "--out_report", default="experiments/logs/l3a1_native_preflight.md"
    )
    args = parser.parse_args()

    evidence = validate_native_task(
        Path(args.native_bddl),
        Path(args.evaluated_bddl),
        args.evaluated_prompt,
    )
    report = [
        "# L3-A1 Native-Only preflight",
        "",
        "- Verdict: **PASS_L3A1_NATIVE_ONLY_PREFLIGHT**",
        f"- Selected native task: `libero_10/{TASK_FILE}`",
        f"- Exact original prompt: `{evidence['prompt']}`",
        f"- Native BDDL source: `{evidence['native_bddl']}`",
        f"- Evaluated BDDL source: `{evidence['evaluated_bddl']}`",
        f"- Native/evaluated BDDL SHA-256: `{evidence['bddl_sha256']}`",
        f"- Native/evaluated fixtures: `{_inventory_text(evidence['fixtures'])}`",
        f"- Native/evaluated objects: `{_inventory_text(evidence['objects'])}`",
        "- Asset inventory identical: `True`",
        "- Prompt identical: `True`",
        "- Custom assets: `False`",
        "- Custom BDDL: `False`",
    ]
    output = Path(args.out_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("PASS_L3A1_NATIVE_ONLY_PREFLIGHT")


if __name__ == "__main__":
    main()
