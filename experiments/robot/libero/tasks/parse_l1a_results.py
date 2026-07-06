"""
Parse PhysCog L1-A / L1-B2 eval logs and print a results table.

Usage:
    python experiments/robot/libero/tasks/parse_l1a_results.py
    python experiments/robot/libero/tasks/parse_l1a_results.py --log_dir ./experiments/logs
    python experiments/robot/libero/tasks/parse_l1a_results.py --out results/l1a_results.md
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

# Map run_id_note → (test_label, group_label)
# Order here controls table row order.
RUN_LABELS = {
    "L1-A1-ramekin-vs-plate-occlusion":    ("L1-A1", "Occlusion"),
    "L1-A1-ramekin-vs-plate-matched-safe": ("L1-A1", "Matched Safe"),
    "L1-A2-drawer-occlusion":              ("L1-A2", "Occlusion"),
    "L1-A2-drawer-matched-safe":           ("L1-A2", "Matched Safe"),
    "L1-B1-task6-cookies":                 ("L1-B1", "Contact"),
    "L1-B1-task6-matched-safe":            ("L1-B1", "Matched Safe"),
}

_FLOAT_RE = re.compile(r"([\d.]+)%")


def _extract(text: str, key: str) -> str:
    """Return 'XX.X%' for lines like 'Overall success rate: 0.7200 (72.0%)'."""
    for line in text.splitlines():
        if key in line:
            m = _FLOAT_RE.search(line)
            if m:
                return m.group(0)
    return "—"


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    return {
        "task_success":   _extract(text, "Overall success rate:"),
        "svr":            _extract(text, "Overall SVR:"),
        "safe_success":   _extract(text, "Overall safe success rate:"),
    }


def find_latest_logs(log_dir: Path) -> dict:
    """Return {run_id_note: Path} keeping only the newest file per note."""
    latest: dict[str, tuple[datetime, Path]] = {}
    for p in log_dir.glob("EVAL-*.txt"):
        # filename: EVAL-<suite>-<family>-<DATETIME>--<note>.txt
        parts = p.stem.split("--", 1)
        if len(parts) < 2:
            continue
        note = parts[1]
        if note not in RUN_LABELS:
            continue
        # parse timestamp from the prefix segment
        prefix = parts[0]  # e.g. EVAL-libero_spatial-openvla-2025_01_01_12_00_00
        ts_str = "_".join(prefix.split("_")[-6:])
        try:
            ts = datetime.strptime(ts_str, "%Y_%m_%d_%H_%M_%S")
        except ValueError:
            ts = datetime.min
        if note not in latest or ts > latest[note][0]:
            latest[note] = (ts, p)
    return {note: path for note, (_, path) in latest.items()}


def build_table(results: dict) -> str:
    rows = []
    order = list(RUN_LABELS.keys())
    for note in order:
        if note not in results:
            continue
        test, group = RUN_LABELS[note]
        r = results[note]
        rows.append((test, group, r["task_success"], r["svr"], r["safe_success"]))

    if not rows:
        return "(no matching log files found)"

    header = f"{'Test':<8} {'Group':<14} {'Task SR':>8} {'SVR':>8} {'Safe SR':>8}"
    sep    = "-" * len(header)
    lines  = [header, sep]
    for test, group, tsr, svr, ssr in rows:
        lines.append(f"{test:<8} {group:<14} {tsr:>8} {svr:>8} {ssr:>8}")

    # Delta rows (only for tests with matched safe control)
    lines.append(sep)
    for test_name in ("L1-A1", "L1-A2", "L1-B1"):
        occ_note  = next((k for k, v in RUN_LABELS.items() if v[0] == test_name and v[1] != "Matched Safe"), None)
        safe_note = next((k for k, v in RUN_LABELS.items() if v == (test_name, "Matched Safe")), None)
        if occ_note in results and safe_note in results:
            def _pct(s):
                m = re.search(r"([\d.]+)%", s)
                return float(m.group(1)) if m else None
            occ_sr  = _pct(results[occ_note]["task_success"])
            safe_sr = _pct(results[safe_note]["task_success"])
            if occ_sr is not None and safe_sr is not None:
                delta = occ_sr - safe_sr
                lines.append(
                    f"{test_name:<8} {'Δ (Occ-Safe)':<14} {delta:>+7.1f}%"
                    "          (task success drop due to perturbation)"
                )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse L1-A PhysCog eval logs")
    parser.add_argument("--log_dir", default="./experiments/logs")
    parser.add_argument("--out", default=None, help="Optional output file path (.md or .txt)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[WARN] log_dir not found: {log_dir}")
        return

    latest = find_latest_logs(log_dir)
    if not latest:
        print(f"[WARN] No matching log files found in {log_dir}")
        return

    results = {note: parse_log(path) for note, path in latest.items()}

    print(f"\nPhysCog L1-A/B2 Results  (parsed {datetime.now():%Y-%m-%d %H:%M})")
    print(f"Log dir: {log_dir.resolve()}\n")
    for note, path in sorted(latest.items()):
        print(f"  {RUN_LABELS[note][0]} {RUN_LABELS[note][1]:<14} ← {path.name}")
    print()
    table = build_table(results)
    print(table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"# PhysCog L1-A/B2 Results\n\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"```\n{table}\n```\n"
        )
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
