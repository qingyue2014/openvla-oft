"""Dependency-free statistical helpers for PhysCog result reporting.

Implements the confidence-interval protocol described in
RESULT_TABLE_DESIGN.md ("Statistical reporting protocol"):

- Wilson score intervals for per-condition binomial rates (pooled episodes).
- Run-level mean +/- t-based 95% CI over repeated evaluation runs
  (the repeated-run scheme used by LIBERO-Gen, Appendix B.1).
- Newcombe score interval for the difference of two independent proportions
  (used for the primary Er-vs-Ec Safe SR contrast).
- Exact McNemar test for episode-paired Er/Ec families.
- Unpaired two-proportion z-test as a fallback when episodes are not paired.

Degenerate cases (no discordant pairs, or both groups at 0%/100%) return
None for the p-value; report them as "no test" instead of fabricating a
significance number.
"""

import math
from typing import List, Optional, Sequence, Tuple

Z_975 = 1.959963984540054

# Two-sided 95% t critical values by degrees of freedom.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980,
}


def t_critical_975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_975:
        return _T_975[df]
    for bound in sorted(_T_975):
        if df < bound:
            return _T_975[bound]
    return Z_975


def wilson_interval(successes: int, n: int) -> Optional[Tuple[float, float]]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None
    z = Z_975
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def run_level_mean_ci(rates: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Mean and 95% CI half-width across repeated evaluation runs.

    This is the LIBERO-Gen Appendix B.1 scheme: each run contributes one
    success rate; report mean +/- half-width. Requires >= 2 runs.
    """
    rates = [r for r in rates if r is not None]
    if len(rates) < 2:
        return None
    k = len(rates)
    mean = sum(rates) / k
    var = sum((r - mean) ** 2 for r in rates) / (k - 1)
    half = t_critical_975(k - 1) * math.sqrt(var / k)
    return (mean, half)


def newcombe_diff_interval(
    k1: int, n1: int, k2: int, n2: int
) -> Optional[Tuple[float, float, float]]:
    """95% Newcombe score interval for p1 - p2 (independent samples).

    Returns (diff, lo, hi). Built from the two Wilson intervals, so it stays
    valid at 0% and 100% rates where the Wald interval degenerates.
    """
    w1, w2 = wilson_interval(k1, n1), wilson_interval(k2, n2)
    if w1 is None or w2 is None:
        return None
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = w1
    l2, u2 = w2
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (diff, max(-1.0, lo), min(1.0, hi))


def mcnemar_exact_p(b: int, c: int) -> Optional[float]:
    """Exact two-sided McNemar p-value from discordant pair counts.

    b = pairs where condition A succeeded and B failed; c = the reverse.
    Returns None when there are no discordant pairs (degenerate: the paired
    outcomes are identical, so no test is meaningful).
    """
    n = b + c
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) * 0.5 ** n
    return min(1.0, 2.0 * tail)


def paired_discordant_counts(
    seq_a: Sequence[int], seq_b: Sequence[int]
) -> Optional[Tuple[int, int]]:
    """Discordant pair counts (b, c) for two equal-length binary sequences."""
    if not seq_a or len(seq_a) != len(seq_b):
        return None
    b = sum(1 for x, y in zip(seq_a, seq_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(seq_a, seq_b) if x == 0 and y == 1)
    return (b, c)


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_proportion_z_p(k1: int, n1: int, k2: int, n2: int) -> Optional[float]:
    """Two-sided pooled z-test for two independent proportions.

    Returns None in degenerate cases (pooled rate 0 or 1, i.e. no variation
    in either group), matching the LIBERO-Gen convention of not reporting a
    significance test when all runs are identical.
    """
    if n1 <= 0 or n2 <= 0:
        return None
    pooled = (k1 + k2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return None
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = abs(k1 / n1 - k2 / n2) / se
    return 2.0 * _normal_sf(z)


def parse_binary_seq(compact: str) -> List[int]:
    """Parse a compact '1011...' episode-outcome string into ints."""
    return [1 if ch == "1" else 0 for ch in compact if ch in "01"]


def format_rate_ci(successes: Optional[int], n: Optional[int]) -> str:
    """Format 'rate% [lo, hi]' with a Wilson interval, or '--'."""
    if successes is None or n in (None, 0):
        return "--"
    interval = wilson_interval(successes, n)
    if interval is None:
        return "--"
    lo, hi = interval
    return f"{successes / n * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"


def format_p(p: Optional[float]) -> str:
    if p is None:
        return "no test (degenerate)"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"
