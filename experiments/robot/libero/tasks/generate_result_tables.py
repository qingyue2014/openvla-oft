"""Generate filled paper-style PhysCog result tables from experiment logs.

This script turns the raw experiment registry into the concrete tables described
in RESULT_TABLE_DESIGN.md:

  experiments/logs/result_tables.md

It does not edit RESULT_TABLE_DESIGN.md; that file remains the table schema and
reporting guide.
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.record_experiment_results import collect_records
from experiments.robot.libero.tasks.physcog_stats import (
    format_p,
    format_rate_ci,
    mcnemar_exact_p,
    newcombe_diff_interval,
    paired_discordant_counts,
    parse_binary_seq,
    run_level_mean_ci,
    two_proportion_z_p,
)


def _num(value) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_rate(value) -> str:
    value = _num(value)
    return "--" if value is None else f"{value * 100:.1f}%"


def _fmt_count(value) -> str:
    return "--" if value in ("", None) else str(value)


def _mean(values: Iterable) -> Optional[float]:
    nums = [_num(v) for v in values]
    nums = [v for v in nums if v is not None]
    return sum(nums) / len(nums) if nums else None


def _role(condition: str) -> str:
    if condition.startswith("Eb"):
        return "eb"
    if condition.startswith("Er"):
        return "er"
    if condition.startswith("Ec"):
        return "ec"
    return ""


def _int_or_none(value) -> Optional[int]:
    num = _num(value)
    return None if num is None else int(round(num))


def _count(row: Dict[str, object], count_key: str, rate_key: str) -> Optional[int]:
    """Event count for one run, falling back to round(rate * n) for old logs."""
    count = _int_or_none(row.get(count_key))
    if count is not None:
        return count
    rate, n = _num(row.get(rate_key)), _int_or_none(row.get("n"))
    if rate is None or n is None:
        return None
    return int(round(rate * n))


def _aggregate_eval_rows(
    rows: List[Dict[str, object]],
    pool_mode: str = "latest",
    pool_since: str = "",
) -> List[Dict[str, object]]:
    """Aggregate repeated evaluation runs of the same (model, run_id).

    Runs whose run_id ends in an explicit `-seed<N>` suffix are recognized as
    intentional seed repeats: they are grouped under the canonical run_id
    (suffix stripped) and pooled in every pool_mode. For runs without a seed
    suffix, pool_mode controls pooling:
      - "latest" (default): only the most recent run per run_id. Safe when the
        log dir still contains historical runs from scenario-design iterations
        whose configs differ from the final one.
      - "all": pool every run of the run_id. Only valid when all retained runs
        are seed repeats of the identical final config.
    pool_since (YYYY_MM_DD) additionally drops runs before the given date from
    pooling, so final seed-repeat batches can be pooled without deleting old
    diagnostic logs.

    Pooled runs sum episode counts for Wilson intervals and keep per-run rates
    for the run-level mean +/- CI. Metadata comes from the most recent run.
    """
    seed_suffix = re.compile(r"-seed\d+$")
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") != "eval":
            continue
        model = str(row.get("model") or "unknown")
        run_id = str(row.get("run_id") or row.get("source_path"))
        grouped[(model, seed_suffix.sub("", run_id))].append(row)

    aggregates = []
    for (model, canonical_run_id), runs in grouped.items():
        runs = sorted(runs, key=lambda r: str(r.get("timestamp", "")))
        if pool_since:
            recent = [r for r in runs if str(r.get("timestamp", "")) >= pool_since]
            runs = recent or runs[-1:]
        seed_runs = [r for r in runs if seed_suffix.search(str(r.get("run_id") or ""))]
        if seed_runs:
            # Explicit seed repeats supersede legacy unsuffixed runs. Keep the
            # latest run per seed so a rerun of one seed does not double-count.
            by_seed: Dict[str, Dict[str, object]] = {}
            for run in seed_runs:
                by_seed[str(run.get("run_id"))] = run
            runs = sorted(by_seed.values(), key=lambda r: str(r.get("timestamp", "")))
        elif pool_mode == "latest":
            runs = runs[-1:]
        agg = dict(runs[-1])
        agg["run_id"] = canonical_run_id
        agg["n_runs"] = len(runs)
        agg["run_task_rates"] = [_num(r.get("task_success_rate")) for r in runs]
        agg["run_safe_rates"] = [_num(r.get("safe_success_rate")) for r in runs]
        agg["run_svr_rates"] = [_num(r.get("svr")) for r in runs]
        for seq_key in ("episode_success_seq", "episode_violation_seq", "episode_safe_seq"):
            agg[seq_key] = "".join(str(r.get(seq_key) or "") for r in runs)

        ns = [_int_or_none(r.get("n")) for r in runs]
        successes = [_count(r, "successes", "task_success_rate") for r in runs]
        violations = [_count(r, "violations", "svr") for r in runs]
        safe_successes = [_count(r, "safe_successes", "safe_success_rate") for r in runs]
        if all(v is not None for v in ns):
            agg["n"] = sum(ns)
            for field, counts, rate_field in (
                ("successes", successes, "task_success_rate"),
                ("violations", violations, "svr"),
                ("safe_successes", safe_successes, "safe_success_rate"),
            ):
                if all(c is not None for c in counts):
                    agg[field] = sum(counts)
                    agg[rate_field] = sum(counts) / agg["n"] if agg["n"] else None
        aggregates.append(agg)
    return aggregates


def _infer_attribution_model(
    attribution: Dict[str, object],
    eval_models_by_scenario: Dict[str, set],
    default_model: str,
) -> str:
    if attribution.get("model"):
        return str(attribution["model"])
    source = Path(str(attribution.get("source_path", ""))).stem.lower()
    for model in sorted({m for models in eval_models_by_scenario.values() for m in models}):
        if model and model.lower() in source:
            return model
    scenario = str(attribution.get("scenario") or "")
    models = eval_models_by_scenario.get(scenario, set())
    if len(models) == 1:
        return next(iter(models))
    return default_model or "unknown"


def _latest_attribution_rows(
    rows: List[Dict[str, object]],
    eval_rows: List[Dict[str, object]],
    default_model: str,
) -> List[Dict[str, object]]:
    eval_models_by_scenario: Dict[str, set] = defaultdict(set)
    for row in eval_rows:
        if row.get("scenario"):
            eval_models_by_scenario[str(row["scenario"])].add(str(row.get("model") or "unknown"))

    latest: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        if row.get("record_type") != "attribution":
            continue
        copied = dict(row)
        copied["model"] = _infer_attribution_model(copied, eval_models_by_scenario, default_model)
        key = (str(copied["model"]), str(copied.get("scenario") or copied.get("family")))
        if key not in latest or str(copied.get("timestamp", "")) > str(latest[key].get("timestamp", "")):
            latest[key] = copied
    return list(latest.values())


def _build_scenario_rows(
    records: List[Dict[str, object]],
    default_model: str,
    pool_mode: str = "latest",
    pool_since: str = "",
) -> List[Dict[str, object]]:
    eval_rows = _aggregate_eval_rows(records, pool_mode=pool_mode, pool_since=pool_since)
    # L1-A2 intentionally shares the unchanged native task-1 competence gate
    # with L1-A1. Duplicate that aggregate only for table assembly so L1-A2's
    # Eb column is populated without rerunning or relabeling the native data.
    shared_l1a2_eb = []
    for row in eval_rows:
        if str(row.get("run_id") or "") == "L1-A1-native-baseline":
            copied = dict(row)
            copied.update(
                {
                    "scenario": "L1-A2",
                    "family": "L1-A2",
                    "condition": "Eb Shared Native Gate",
                    "notes": "Shared unchanged native task-1 gate from L1-A1",
                }
            )
            shared_l1a2_eb.append(copied)
    eval_rows.extend(shared_l1a2_eb)
    attribution_rows = _latest_attribution_rows(records, eval_rows, default_model)
    attr_by_model_scenario = {
        (str(row.get("model") or "unknown"), str(row.get("scenario") or "")): row
        for row in attribution_rows
        if "BENCHMARK_INCOMPLETE" not in str(row.get("notes") or "")
    }

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in eval_rows:
        model = str(row.get("model") or "unknown")
        scenario = str(row.get("scenario") or "")
        if not scenario:
            continue
        grouped[(model, scenario)].append(row)

    scenario_rows = []
    for (model, scenario), rows in sorted(grouped.items()):
        by_role: Dict[str, Dict[str, object]] = {}
        for row in rows:
            role = _role(str(row.get("condition") or ""))
            if role and (role not in by_role or str(row.get("timestamp", "")) > str(by_role[role].get("timestamp", ""))):
                by_role[role] = row
        if not by_role:
            continue
        level = next((str(row.get("level") or "") for row in rows if row.get("level")), "")
        attr = attr_by_model_scenario.get((model, scenario), {})
        eb, er, ec = by_role.get("eb", {}), by_role.get("er", {}), by_role.get("ec", {})
        notes = []
        if not eb:
            notes.append("missing Eb")
        if not er:
            notes.append("missing Er")
        if not ec:
            notes.append("missing Ec")
        if not attr:
            notes.append("missing attribution")
        scenario_rows.append({
            "model": model,
            "level": level,
            "scenario": scenario,
            "n_eb": eb.get("n", ""),
            "n_er": er.get("n", ""),
            "n_ec": ec.get("n", ""),
            "eb_task_sr": eb.get("task_success_rate", ""),
            "er_task_sr": er.get("task_success_rate", ""),
            "er_safe_sr": er.get("safe_success_rate", ""),
            "er_svr": er.get("svr", ""),
            "ec_task_sr": ec.get("task_success_rate", ""),
            "ec_safe_sr": ec.get("safe_success_rate", ""),
            "ec_svr": ec.get("svr", ""),
            "sar": attr.get("sar", ""),
            "btf": attr.get("btf", ""),
            "uir": attr.get("uir", ""),
            "ocr": attr.get("ocr", ""),
            "nor": attr.get("nor", ""),
            "notes": "; ".join(notes),
            "_eb": eb,
            "_er": er,
            "_ec": ec,
        })
    return scenario_rows


def _table1(scenario_rows: List[Dict[str, object]]) -> List[str]:
    by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in scenario_rows:
        by_model[row["model"]].append(row)
    lines = [
        "## Table 1. Model-level statistical summary",
        "",
        "| VLA Model | # Families | Eb Task SR ↑ | Er Task SR ↑ | Er Safe SR ↑ | Er SVR ↓ | Ec Task SR ↑ | Ec Safe SR ↑ | Ec SVR ↓ | BTF ↓ | SAR ↑ | UIR ↓ | OCR | NOR ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, rows in sorted(by_model.items()):
        lines.append(
            f"| {model} | {len(rows)} | {_fmt_rate(_mean(r['eb_task_sr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['er_task_sr'] for r in rows))} | {_fmt_rate(_mean(r['er_safe_sr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['er_svr'] for r in rows))} | {_fmt_rate(_mean(r['ec_task_sr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['ec_safe_sr'] for r in rows))} | {_fmt_rate(_mean(r['ec_svr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['btf'] for r in rows))} | {_fmt_rate(_mean(r['sar'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['uir'] for r in rows))} | {_fmt_rate(_mean(r['ocr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['nor'] for r in rows))} |"
        )
    if not by_model:
        lines.append("| -- | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    return lines


def _table2(scenario_rows: List[Dict[str, object]]) -> List[str]:
    by_model_level: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in scenario_rows:
        by_model_level[(row["model"], row["level"])].append(row)
    lines = [
        "## Table 2. Per-level model breakdown",
        "",
        "| VLA Model | Level | # Families | Eb Task SR ↑ | Er Task SR ↑ | Er Safe SR ↑ | Er SVR ↓ | Ec Task SR ↑ | Ec Safe SR ↑ | Ec SVR ↓ | BTF ↓ | SAR ↑ | UIR ↓ | OCR | NOR ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (model, level), rows in sorted(by_model_level.items()):
        lines.append(
            f"| {model} | {level or '--'} | {len(rows)} | {_fmt_rate(_mean(r['eb_task_sr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['er_task_sr'] for r in rows))} | {_fmt_rate(_mean(r['er_safe_sr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['er_svr'] for r in rows))} | {_fmt_rate(_mean(r['ec_task_sr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['ec_safe_sr'] for r in rows))} | {_fmt_rate(_mean(r['ec_svr'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['btf'] for r in rows))} | {_fmt_rate(_mean(r['sar'] for r in rows))} | {_fmt_rate(_mean(r['uir'] for r in rows))} | "
            f"{_fmt_rate(_mean(r['ocr'] for r in rows))} | {_fmt_rate(_mean(r['nor'] for r in rows))} |"
        )
    if not by_model_level:
        lines.append("| -- | -- | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    return lines


def _table3(scenario_rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "## Table 3. Scenario-level result matrix",
        "",
        "| Level | Scenario | VLA Model | N Eb | N Er | N Ec | Eb Task SR ↑ | Er Task SR ↑ | Er Safe SR ↑ | Er SVR ↓ | Ec Task SR ↑ | Ec Safe SR ↑ | Ec SVR ↓ | BTF ↓ | SAR ↑ | UIR ↓ | OCR | NOR ↓ | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(scenario_rows, key=lambda r: (r["level"], r["scenario"], r["model"])):
        lines.append(
            f"| {row['level'] or '--'} | {row['scenario']} | {row['model']} | {_fmt_count(row['n_eb'])} | "
            f"{_fmt_count(row['n_er'])} | {_fmt_count(row['n_ec'])} | {_fmt_rate(row['eb_task_sr'])} | "
            f"{_fmt_rate(row['er_task_sr'])} | {_fmt_rate(row['er_safe_sr'])} | {_fmt_rate(row['er_svr'])} | "
            f"{_fmt_rate(row['ec_task_sr'])} | {_fmt_rate(row['ec_safe_sr'])} | {_fmt_rate(row['ec_svr'])} | "
            f"{_fmt_rate(row['btf'])} | {_fmt_rate(row['sar'])} | {_fmt_rate(row['uir'])} | {_fmt_rate(row['ocr'])} | {_fmt_rate(row['nor'])} | "
            f"{row['notes']} |"
        )
    if not scenario_rows:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | no data |")
    return lines


def _rate_ci_cell(row: Dict[str, object], count_field: str, rate_field: str) -> str:
    if not row:
        return "--"
    n = _int_or_none(row.get("n"))
    count = _count(row, count_field, rate_field)
    if n is None or count is None:
        return _fmt_rate(row.get(rate_field))
    return format_rate_ci(count, n)


def _run_ci_cell(row: Dict[str, object], rates_field: str) -> str:
    if not row or int(row.get("n_runs") or 1) < 2:
        return "--"
    stats = run_level_mean_ci([r for r in row.get(rates_field, []) if r is not None])
    if stats is None:
        return "--"
    mean, half = stats
    return f"{mean * 100:.1f}% ± {half * 100:.1f}"


def _er_ec_contrast(er: Dict[str, object], ec: Dict[str, object]) -> Tuple[str, str]:
    """Delta Safe SR (Ec - Er) with 95% CI, and the significance test result."""
    n_er, n_ec = _int_or_none(er.get("n")), _int_or_none(ec.get("n"))
    k_er = _count(er, "safe_successes", "safe_success_rate")
    k_ec = _count(ec, "safe_successes", "safe_success_rate")
    if None in (n_er, n_ec, k_er, k_ec) or 0 in (n_er, n_ec):
        return "--", "--"
    interval = newcombe_diff_interval(k_ec, n_ec, k_er, n_er)
    diff_cell = "--"
    if interval is not None:
        diff, lo, hi = interval
        diff_cell = f"{diff * 100:+.1f}pp [{lo * 100:+.1f}, {hi * 100:+.1f}]"

    seq_er = parse_binary_seq(str(er.get("episode_safe_seq") or ""))
    seq_ec = parse_binary_seq(str(ec.get("episode_safe_seq") or ""))
    discordant = paired_discordant_counts(seq_ec, seq_er)
    if discordant is not None and (len(seq_er) == n_er and len(seq_ec) == n_ec):
        p = mcnemar_exact_p(*discordant)
        return diff_cell, f"{format_p(p)} (McNemar, paired)"
    p = two_proportion_z_p(k_ec, n_ec, k_er, n_er)
    return diff_cell, f"{format_p(p)} (z, unpaired)"


def _table5(scenario_rows: List[Dict[str, object]]) -> List[str]:
    lines = [
        "## Table 5. Statistical reliability (95% CI and Er-vs-Ec contrast)",
        "",
        "Per-condition rates carry Wilson 95% intervals over pooled episodes.",
        "`Run-level` is mean ± 95% CI half-width across repeated evaluation runs",
        "(reported only when a condition has ≥ 2 runs). `Δ Safe SR` is the",
        "Newcombe 95% interval for Ec − Er; the p-value uses the exact McNemar",
        "test when Er/Ec episodes are index-paired, otherwise an unpaired",
        "two-proportion z-test. Degenerate cases (no discordant pairs, or no",
        "outcome variation) are reported as `no test` instead of a p-value.",
        "",
        "| Level | Scenario | VLA Model | Runs Eb/Er/Ec | Eb Task SR [95% CI] | Er Safe SR [95% CI] | Ec Safe SR [95% CI] | Er Safe SR run-level | Δ Safe SR (Ec−Er) [95% CI] | p (test) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(scenario_rows, key=lambda r: (r["level"], r["scenario"], r["model"])):
        eb, er, ec = row["_eb"], row["_er"], row["_ec"]
        runs = "/".join(str(cond.get("n_runs", 0) or 0) if cond else "0" for cond in (eb, er, ec))
        diff_cell, p_cell = _er_ec_contrast(er, ec) if er and ec else ("--", "--")
        lines.append(
            f"| {row['level'] or '--'} | {row['scenario']} | {row['model']} | {runs} | "
            f"{_rate_ci_cell(eb, 'successes', 'task_success_rate')} | "
            f"{_rate_ci_cell(er, 'safe_successes', 'safe_success_rate')} | "
            f"{_rate_ci_cell(ec, 'safe_successes', 'safe_success_rate')} | "
            f"{_run_ci_cell(er, 'run_safe_rates')} | {diff_cell} | {p_cell} |"
        )
    if not scenario_rows:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | -- | no data |")
    return lines


def generate_tables(
    log_dir: Path,
    default_model: str = "",
    pool_mode: str = "latest",
    pool_since: str = "",
) -> str:
    records = collect_records(log_dir)
    scenario_rows = _build_scenario_rows(
        records, default_model=default_model, pool_mode=pool_mode, pool_since=pool_since
    )
    lines = [
        "# Filled PhysCogTest Result Tables",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Source: parsed from `experiments/logs/EVAL-*.txt` and `experiments/logs/*attribution*.md`.",
        "",
        "Aggregation: macro-average across scenario families unless otherwise noted.",
        "",
        f"Run pooling: `{pool_mode}`" + (f" since `{pool_since}`" if pool_since else "")
        + " (latest = most recent run per run_id; all = pool seed repeats).",
        "",
    ]
    lines += _table1(scenario_rows)
    lines += [""] + _table2(scenario_rows)
    lines += [""] + _table3(scenario_rows)
    lines += [""] + _table5(scenario_rows)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filled PhysCog result tables")
    parser.add_argument("--log_dir", default="experiments/logs")
    parser.add_argument("--out", default="experiments/logs/result_tables.md")
    parser.add_argument("--default_model", default="",
                        help="Model name used for attribution reports when it cannot be inferred automatically")
    parser.add_argument("--pool_mode", choices=("latest", "all"), default="latest",
                        help="latest: use only the most recent run per run_id (default); "
                             "all: pool repeated runs of the same run_id as seed repeats")
    parser.add_argument("--pool_since", default="",
                        help="Only pool runs with timestamp >= this date (YYYY_MM_DD); "
                             "use with --pool_mode all so stale design-iteration runs are excluded")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log_dir not found: {log_dir}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_tables(
        log_dir,
        default_model=args.default_model,
        pool_mode=args.pool_mode,
        pool_since=args.pool_since,
    ))
    print(f"Wrote filled result tables to {output}")


if __name__ == "__main__":
    main()
