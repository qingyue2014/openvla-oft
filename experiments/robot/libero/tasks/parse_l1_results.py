"""Parse PhysCogSafe L1 pilot eval logs into a compact table.

Usage:
    python experiments/robot/libero/tasks/parse_l1_results.py
    python experiments/robot/libero/tasks/parse_l1_results.py --out experiments/logs/l1_pilot_results.md
"""

import argparse
import re
from datetime import datetime
from pathlib import Path


RUN_LABELS = {
    "L1-A1-native-baseline": ("L1-A1", "Eb: Native Gate"),
    "L1-A1-ramekin-vs-plate-occlusion": ("L1-A1", "Er: Occlusion Risk"),
    "L1-A1-ramekin-vs-plate-matched-safe": ("L1-A1", "Ec: Matched-Safe"),
    "L1-B1-task6-cookies": ("L1-B1", "Risk: Contact"),
    "L1-B1-task6-matched-safe": ("L1-B1", "Control: Matched Safe"),
    "L1-B2-task6-cookie-ramekin": ("L1-B2", "Risk: Narrow Corridor"),
    "L1-B2-task6-matched-safe": ("L1-B2", "Control: Open Corridor"),
    "L1-B4-task6-ramekin-retraction": ("L1-B4", "Risk: Post-Grasp Bystander"),
    "L1-B4-task6-no-insertion": ("L1-B4", "Control: No Insertion"),
    "L1-B4-task6-out-of-path-bystander": ("L1-B4", "Control: Out-of-Path"),
}

RUN_ORDER = list(RUN_LABELS)

_PCT_RE = re.compile(r"([\d.]+)%")
_INT_RE = re.compile(r"(-?\d+)")


def _extract_percent(text: str, key: str) -> str:
    for line in text.splitlines():
        if key in line:
            match = _PCT_RE.search(line)
            if match:
                return match.group(0)
    return "--"


def _extract_int(text: str, key: str) -> str:
    for line in text.splitlines():
        if key in line:
            match = _INT_RE.search(line)
            if match:
                return match.group(1)
    return "--"


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    return {
        "episodes": _extract_int(text, "Total episodes:"),
        "task_success": _extract_percent(text, "Overall success rate:"),
        "svr": _extract_percent(text, "Overall SVR:"),
        "valid_svr": _extract_percent(text, "Overall valid-execution violation rate:"),
        "collapse": _extract_percent(text, "Overall model collapse rate:"),
        "safe_success": _extract_percent(text, "Overall safe success rate:"),
    }


def _timestamp_from_eval_name(path: Path) -> datetime:
    prefix = path.stem.split("--", 1)[0]
    ts_str = "_".join(prefix.split("_")[-6:])
    try:
        return datetime.strptime(ts_str, "%Y_%m_%d_%H_%M_%S")
    except ValueError:
        return datetime.min


def find_latest_logs(log_dir: Path) -> dict[str, Path]:
    latest: dict[str, tuple[datetime, Path]] = {}
    for path in log_dir.glob("EVAL-*.txt"):
        parts = path.stem.split("--", 1)
        if len(parts) != 2:
            continue
        note = parts[1]
        if note not in RUN_LABELS:
            continue
        ts = _timestamp_from_eval_name(path)
        if note not in latest or ts > latest[note][0]:
            latest[note] = (ts, path)
    return {note: path for note, (_, path) in latest.items()}


def build_table(results: dict[str, dict]) -> str:
    rows = []
    for note in RUN_ORDER:
        if note not in results:
            rows.append((RUN_LABELS[note][0], RUN_LABELS[note][1], "missing", "--", "--", "--", "--", "--"))
            continue
        test, condition = RUN_LABELS[note]
        r = results[note]
        rows.append(
            (
                test,
                condition,
                r["episodes"],
                r["task_success"],
                r["svr"],
                r["safe_success"],
                r["valid_svr"],
                r["collapse"],
            )
        )

    header = (
        f"{'Test':<7} {'Condition':<30} {'N':>5} "
        f"{'Task SR':>8} {'SVR':>8} {'Safe SR':>8} {'Valid SVR':>10} {'Collapse':>9}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        test, condition, n, task_sr, svr, safe_sr, valid_svr, collapse = row
        lines.append(
            f"{test:<7} {condition:<30} {n:>5} "
            f"{task_sr:>8} {svr:>8} {safe_sr:>8} {valid_svr:>10} {collapse:>9}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PhysCogSafe L1 pilot eval logs")
    parser.add_argument("--log_dir", default="experiments/logs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[WARN] log_dir not found: {log_dir}")
        return

    latest = find_latest_logs(log_dir)
    results = {note: parse_log(path) for note, path in latest.items()}

    print(f"\nPhysCogSafe L1 Pilot Results  (parsed {datetime.now():%Y-%m-%d %H:%M})")
    print(f"Log dir: {log_dir.resolve()}\n")
    for note in RUN_ORDER:
        if note in latest:
            print(f"  {RUN_LABELS[note][0]} {RUN_LABELS[note][1]:<30} <- {latest[note].name}")
    print()
    table = build_table(results)
    print(table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# PhysCogSafe L1 Pilot Results\n\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"```\n{table}\n```\n"
        )
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
