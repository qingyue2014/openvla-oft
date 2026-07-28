"""Verify that the L3-B1 capability baseline contains unmodified native states."""

import argparse
from pathlib import Path

import h5py
import numpy as np


PROMPT = "put the wine bottle on the wine rack"
BDDL = "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl"


def validate(path: str, minimum_count: int) -> int:
    with h5py.File(path, "r") as handle:
        key = PROMPT.replace(" ", "_")
        if key not in handle:
            raise ValueError(f"missing task group {key!r}")
        group = handle[key]
        expected_attrs = {
            "l3b1_variant": "capability_native",
            "task_suite_name": "libero_90",
            "task_description": PROMPT,
            "bddl_basename": BDDL,
            "custom_assets": False,
            "custom_bddl": False,
            "intervention_body": "",
        }
        for name, expected in expected_attrs.items():
            actual = group.attrs.get(name)
            if isinstance(actual, bytes):
                actual = actual.decode()
            if actual != expected:
                raise ValueError(
                    f"{path}: attribute {name}={actual!r}, expected {expected!r}"
                )
        demos = sorted(
            group.keys(), key=lambda name: int(name.removeprefix("demo_"))
        )
        if len(demos) < minimum_count:
            raise ValueError(
                f"{path}: found {len(demos)} states, need {minimum_count}"
            )
        source_indices = []
        for name in demos:
            demo = group[name]
            initial = np.asarray(demo["initial_state"])
            baseline = np.asarray(demo["base_reset_state"])
            if not np.array_equal(initial, baseline):
                raise ValueError(f"{path}:{name}: native state was modified")
            if float(demo.attrs.get("non_bottle_error", np.inf)) != 0.0:
                raise ValueError(f"{path}:{name}: non_bottle_error is not zero")
            source_indices.append(int(demo.attrs["source_state_index"]))
        if len(source_indices) != len(set(source_indices)):
            raise ValueError(f"{path}: duplicate official source indices")
    return len(demos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--minimum_count", type=int, default=20)
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l3b1_native_capability_states.md",
    )
    args = parser.parse_args()
    count = validate(args.states, args.minimum_count)
    report = [
        "# L3-B1 native capability state validation",
        "",
        "- Verdict: **PASS_L3B1_NATIVE_CAPABILITY_STATES**",
        f"- States: {count}",
        f"- Exact prompt: `{PROMPT}`",
        f"- Native BDDL: `{BDDL}`",
        "- Initial state equals official baseline state for every episode: `True`",
        "- State intervention: `None`",
        "- Custom assets: `False`",
        "- Custom BDDL: `False`",
    ]
    output = Path(args.out_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("PASS_L3B1_NATIVE_CAPABILITY_STATES")


if __name__ == "__main__":
    main()
