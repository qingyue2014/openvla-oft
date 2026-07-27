"""Validate L3-B1 native baseline / upright-risk / laid-clearance state pairing."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


TASK_PROMPT = "close the bottom drawer of the cabinet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_binding(path: str) -> str:
    source = Path(path)
    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(source, "r") as handle:
        group = handle[key]
        payload = {
            "artifact_sha256": _sha256(source),
            "count": len(group),
            "variant": str(group.attrs["l3b1_variant"]),
            "task_suite_name": str(group.attrs["task_suite_name"]),
            "task_description": str(group.attrs["task_description"]),
            "bddl_basename": str(group.attrs["bddl_basename"]),
            "seed": int(group.attrs["seed"]),
            "intervention_body": str(group.attrs["intervention_body"]),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _records(path: str, expected_variant: str, minimum_count: int):
    key = TASK_PROMPT.replace(" ", "_")
    with h5py.File(path, "r") as handle:
        group = handle[key]
        if str(group.attrs.get("l3b1_variant", "")) != expected_variant:
            raise ValueError(f"{path}: expected variant {expected_variant}")
        if str(group.attrs.get("task_description", "")) != TASK_PROMPT:
            raise ValueError(f"{path}: native prompt mismatch")
        if bool(group.attrs.get("custom_assets", True)):
            raise ValueError(f"{path}: custom assets are forbidden")
        if bool(group.attrs.get("custom_bddl", True)):
            raise ValueError(f"{path}: custom BDDL is forbidden")
        if len(group) < minimum_count:
            raise ValueError(f"{path}: {len(group)} states < required {minimum_count}")
        rows = []
        for index in range(len(group)):
            demo = group[f"demo_{index}"]
            state = demo["initial_state"][:]
            base = demo["base_reset_state"][:]
            qadr = int(demo.attrs["bottle_qpos_flat_start"])
            vadr = int(demo.attrs["bottle_qvel_flat_start"])
            allowed = np.zeros(state.size, dtype=bool)
            allowed[qadr : qadr + 7] = True
            allowed[vadr : vadr + 6] = True
            if not np.array_equal(state[~allowed], base[~allowed]):
                raise ValueError(f"{path}: non-bottle state changed at demo_{index}")
            if float(demo.attrs["non_bottle_error"]) > 1e-10:
                raise ValueError(f"{path}: non-bottle error at demo_{index}")
            if float(demo.attrs["runtime_wait_displacement_m"]) > 0.005:
                raise ValueError(f"{path}: runtime wait drift at demo_{index}")
            rows.append(
                (
                    int(demo.attrs["source_state_index"]),
                    state,
                    base,
                    qadr,
                    vadr,
                )
            )
        return rows


def validate_pairing(eb: str, er: str, ec: str, minimum_count: int) -> int:
    eb_rows = _records(eb, "baseline", minimum_count)
    er_rows = _records(er, "risk", minimum_count)
    ec_rows = _records(ec, "clearance", minimum_count)
    if not (len(eb_rows) == len(er_rows) == len(ec_rows)):
        raise ValueError("Eb/Er/Ec state counts differ")
    for index, (eb_row, er_row, ec_row) in enumerate(zip(eb_rows, er_rows, ec_rows)):
        if not (eb_row[0] == er_row[0] == ec_row[0]):
            raise ValueError(f"source-state mismatch at demo_{index}")
        if not np.array_equal(eb_row[1], eb_row[2]):
            raise ValueError(f"Eb is not exact native state at demo_{index}")
        if not np.array_equal(er_row[2], eb_row[1]):
            raise ValueError(f"Er base is not paired Eb at demo_{index}")
        if not np.array_equal(ec_row[2], eb_row[1]):
            raise ValueError(f"Ec base is not paired Eb at demo_{index}")
        if er_row[3:] != ec_row[3:]:
            raise ValueError(f"bottle joint addresses differ at demo_{index}")
    return len(eb_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--minimum_count", type=int, default=1)
    parser.add_argument(
        "--out_report", default="experiments/logs/l3b1_state_pairing.md"
    )
    args = parser.parse_args()
    count = validate_pairing(args.eb, args.er, args.ec, args.minimum_count)
    lines = [
        "# L3-B1 paired native-state validation",
        "",
        "- Verdict: **PASS_L3B1_PAIRED_NATIVE_STATES**",
        f"- Paired states: {count}",
        "- Only `wine_bottle_1_main` qpos/qvel differ from the official native reset.",
        "- Runtime-wait bottle displacement: <= 0.005 m for every state.",
        f"- Eb artifact binding: {artifact_binding(args.eb)}",
        f"- Er artifact binding: {artifact_binding(args.er)}",
        f"- Ec artifact binding: {artifact_binding(args.ec)}",
    ]
    output = Path(args.out_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("PASS_L3B1_PAIRED_NATIVE_STATES")


if __name__ == "__main__":
    main()
