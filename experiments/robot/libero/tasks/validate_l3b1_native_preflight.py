"""Hard-gate L3-B1 against the selected native LIBERO task definition."""

import argparse
import hashlib
import re
from pathlib import Path


TASK_FILE = "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl"
TASK_PROMPT = "close the bottom drawer of the cabinet"
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


def _inventory(text: str, section: str) -> dict[str, str]:
    return {
        instance: asset_type
        for instance, asset_type in re.findall(
            r"(?m)^\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*$",
            _section(text, section),
        )
    }


def validate_native_task(
    native_bddl: Path, evaluated_bddl: Path, evaluated_prompt: str
) -> dict[str, object]:
    native = native_bddl.resolve(strict=True)
    evaluated = evaluated_bddl.resolve(strict=True)
    if native.name != TASK_FILE or native.parent.name != "libero_90":
        raise ValueError(f"unexpected native task source: {native}")
    if "bddl_files" not in native.parts or not native.samefile(evaluated):
        raise ValueError(
            f"evaluated BDDL is not the selected native task: {evaluated}"
        )
    text = native.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+([^)]+)\)", text)
    if match is None:
        raise ValueError("native BDDL has no :language prompt")
    prompt = " ".join(match.group(1).split())
    if prompt != TASK_PROMPT or evaluated_prompt != TASK_PROMPT:
        raise ValueError(
            f"prompt mismatch: native={prompt!r}, evaluated={evaluated_prompt!r}"
        )
    fixtures = _inventory(text, "fixtures")
    objects = _inventory(text, "objects")
    if fixtures != EXPECTED_FIXTURES or objects != EXPECTED_OBJECTS:
        raise ValueError(
            f"native asset inventory mismatch: fixtures={fixtures}, objects={objects}"
        )
    return {
        "native": str(native),
        "evaluated": str(evaluated),
        "sha256": _sha256(native),
        "prompt": prompt,
        "fixtures": fixtures,
        "objects": objects,
    }


def _inventory_text(values: dict[str, str]) -> str:
    return ", ".join(f"{name}:{kind}" for name, kind in values.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_bddl", required=True)
    parser.add_argument("--evaluated_bddl", required=True)
    parser.add_argument("--evaluated_prompt", required=True)
    parser.add_argument(
        "--out_report", default="experiments/logs/l3b1_native_preflight.md"
    )
    args = parser.parse_args()
    evidence = validate_native_task(
        Path(args.native_bddl), Path(args.evaluated_bddl), args.evaluated_prompt
    )
    report = [
        "# L3-B1 Native-Only preflight",
        "",
        "- Verdict: **PASS_L3B1_NATIVE_ONLY_PREFLIGHT**",
        f"- Selected native task: `libero_90/{TASK_FILE}`",
        f"- Exact original prompt: `{evidence['prompt']}`",
        f"- Native BDDL source: `{evidence['native']}`",
        f"- Evaluated BDDL source: `{evidence['evaluated']}`",
        f"- Native/evaluated BDDL SHA-256: `{evidence['sha256']}`",
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
    print("PASS_L3B1_NATIVE_ONLY_PREFLIGHT")


if __name__ == "__main__":
    main()
