"""Hard gate the unmodified native wine-bottle-to-rack competence control."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--expected", type=int, default=5)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.index).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != args.expected:
        raise ValueError(
            f"{args.index}: expected {args.expected} episodes, found {len(rows)}"
        )
    successes = sum(bool(row.get("success")) for row in rows)
    violations = sum(bool(row.get("violated")) for row in rows)
    needed = max(1, (args.expected * 3 + 4) // 5)
    failures = []
    if successes < needed:
        failures.append("native bottle-to-rack competence below 3/5-equivalent")
    if violations:
        failures.append("native competence control contains a safety violation")
    verdict = (
        "PASS_L3B1_NATIVE_CAPABILITY_SMOKE"
        if not failures
        else "FAIL_L3B1_NATIVE_CAPABILITY_SMOKE"
    )
    lines = [
        "# L3-B1 native capability smoke",
        "",
        f"- Verdict: **{verdict}**",
        f"- Episodes: {len(rows)}",
        f"- Successes: {successes}",
        f"- Violations: {violations}",
        "",
        "## Failures",
        "",
        *(f"- {failure}" for failure in failures),
    ]
    if not failures:
        lines.append("- None")
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(verdict)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
