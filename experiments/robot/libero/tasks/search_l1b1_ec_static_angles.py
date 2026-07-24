"""Search equal-radius L1-B1 Ec angles using the full serialized-state gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_ANGLES_DEG = (-66, -64, -62, -60, -58, -55, -50, 120, 150, 165)
REFERENCE_DELTA_XY = (0.090, 0.185)
RISK_FRACTION = 0.46
RISK_LATERAL_M = 0.065


def _control_parameters(angle_deg: float) -> tuple[float, float]:
    path_length = math.hypot(*REFERENCE_DELTA_XY)
    radius = math.hypot(RISK_FRACTION * path_length, RISK_LATERAL_M)
    angle_rad = math.radians(angle_deg)
    return radius * math.cos(angle_rad) / path_length, radius * math.sin(angle_rad)


def search(args: argparse.Namespace) -> list[dict]:
    generator = Path(__file__).with_name("generate_l1b_swept_initial_states.py")
    validator = Path(__file__).with_name("validate_l1b_swept_states.py")
    rows = []
    for angle_deg in args.angles_deg:
        control_fraction, control_lateral = _control_parameters(angle_deg)
        with tempfile.TemporaryDirectory(prefix="l1b1-ec-angle-") as temp_dir:
            command = [
                sys.executable,
                str(generator),
                "--family",
                "l1b1_native_gripper",
                "--task_suite_name",
                "libero_spatial",
                "--task_id",
                "6",
                "--num_states",
                str(args.states_per_angle),
                "--max_attempts",
                str(max(1000, args.states_per_angle * 5)),
                "--output_dir",
                temp_dir,
                "--control_fraction",
                repr(control_fraction),
                "--control_lateral",
                repr(control_lateral),
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            pairing_path = Path(temp_dir) / "l1b1_native_gripper_pairing.json"
            generated = completed.returncode == 0 and pairing_path.exists()
            pair = (
                json.loads(pairing_path.read_text())["pairs"][0]
                if generated
                else {}
            )
            validation_report = Path(temp_dir) / "scene_check.md"
            validation = None
            if generated:
                validation = subprocess.run(
                    [
                        sys.executable,
                        str(validator),
                        "--family",
                        "l1b1_native_gripper",
                        "--state_dir",
                        temp_dir,
                        "--task_suite_name",
                        "libero_spatial",
                        "--task_id",
                        "6",
                        "--num_previews",
                        "0",
                        "--preview_dir",
                        str(Path(temp_dir) / "preview"),
                        "--out_report",
                        str(validation_report),
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
            passed = bool(validation is not None and validation.returncode == 0)
            combined_output = completed.stdout + "\n" + completed.stderr
            report_text = (
                validation_report.read_text()
                if validation_report.exists()
                else ""
            )

            def report_number(pattern: str) -> str:
                match = re.search(pattern, report_text)
                return match.group(1) if match else ""

            rows.append(
                {
                    "angle_deg": float(angle_deg),
                    "control_fraction": control_fraction,
                    "control_lateral_m": control_lateral,
                    "states": args.states_per_angle,
                    "generation_valid": int(generated),
                    "static_valid": int(passed),
                    "ec_x_m": (
                        float(pair["ec_obstacle_xyz"][0]) if passed else ""
                    ),
                    "ec_y_m": (
                        float(pair["ec_obstacle_xyz"][1]) if passed else ""
                    ),
                    "ec_drift_m": (
                        float(pair["ec_obstacle_drift_m"]) if passed else ""
                    ),
                    "failure_has_cookie_contact": int(
                        "cookies_1_main <-> glazed_rim_porcelain_ramekin_1_main"
                        in combined_output
                    ),
                    "forbidden_initial_contacts": report_number(
                        r"Forbidden initial obstacle contacts/interpenetrations: `(\d+)`"
                    ),
                    "max_radius_mismatch_m": report_number(
                        r"Settled Er/Ec radius mismatch \(max\): `([0-9.]+) m`"
                    ),
                    "ec_min_visible_pixels": report_number(
                        r"EC obstacle pixels \(min/max\): `(\d+)/"
                    ),
                    "returncode": completed.returncode,
                }
            )

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["static_valid"]]
    report = [
        "# L1-B1 equal-radius Ec static-angle search",
        "",
        f"- Tested angles: `{len(rows)}`",
        f"- Serialized states per angle: `{args.states_per_angle}`",
        f"- Static-valid angles: `{len(valid)}`",
        "- Gate: exact serialized-state generation, settling, forbidden-contact "
        "audit, and obstacle-drift validation.",
        "",
        "| angle (deg) | fraction | lateral (m) | static | contacts | radius mismatch (m) | EC px |",
        "| ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
        *(
            f"| {row['angle_deg']:.1f} | {row['control_fraction']:.9f} | "
            f"{row['control_lateral_m']:.9f} | "
            f"{'PASS' if row['static_valid'] else 'FAIL'} | "
            f"{row['forbidden_initial_contacts'] or '-'} | "
            f"{row['max_radius_mismatch_m'] or '-'} | "
            f"{row['ec_min_visible_pixels'] or '-'} |"
            for row in rows
        ),
    ]
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    if not valid:
        raise RuntimeError("No static-valid equal-radius Ec angle")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angles_deg",
        type=float,
        nargs="+",
        default=DEFAULT_ANGLES_DEG,
    )
    parser.add_argument("--states_per_angle", type=int, default=1)
    parser.add_argument(
        "--out_csv",
        default="experiments/logs/l1b1_ec_static_angle_search.csv",
    )
    parser.add_argument(
        "--out_report",
        default="experiments/logs/l1b1_ec_static_angle_search.md",
    )
    search(parser.parse_args())


if __name__ == "__main__":
    main()
