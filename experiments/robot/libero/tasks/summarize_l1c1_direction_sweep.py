"""Summarize L1-C1 unchanged-Eb replay results across risk directions."""

import argparse
import csv
from pathlib import Path


def summarize(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("angle_*.csv")):
        angle = int(path.stem.split("_")[-1])
        with path.open(newline="") as handle:
            episodes = list(csv.DictReader(handle))
        eligible = sum(int(row["attribution_eligible"]) for row in episodes)
        rows.append(
            {
                "angle_deg": angle,
                "episodes": len(episodes),
                "eligible": eligible,
                "eligibility_rate": eligible / len(episodes) if episodes else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-row["eligibility_rate"], row["angle_deg"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.input_dir)
    if not rows:
        raise SystemExit(f"No angle CSV files in {args.input_dir}")
    best = rows[0]
    verdict = "PASS_DIRECTION_SWEEP_HAS_ELIGIBLE_CANDIDATE" if best["eligibility_rate"] >= 0.8 else "FAIL_DIRECTION_SWEEP_NO_ELIGIBLE_CANDIDATE"
    lines = [
        "# L1-C1 Risk-Direction Sweep",
        "",
        f"- Verdict: **{verdict}**",
        f"- Best direction: {best['angle_deg']} deg",
        f"- Best eligibility: {best['eligible']}/{best['episodes']} ({best['eligibility_rate']:.3f})",
        "- Scope: unchanged-Eb action separation only; the selected direction still requires physical and dynamic-reference gates.",
        "",
        "| Direction (deg) | Eligible | N | Rate |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['angle_deg']} | {row['eligible']} | {row['episodes']} | {row['eligibility_rate']:.3f} |"
        )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
