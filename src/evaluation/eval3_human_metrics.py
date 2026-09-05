# -*- coding: utf-8 -*-
"""
Core metrics for EVALUATION3 human-alignment evaluation.

This module is intentionally standard-library only so it can be imported by
the repository's portability test workflow without NumPy/SciPy.

Metric contract:
- compatibility_logit: higher = more compatible
- raw Cmt: Good=1, Normal=2, Bad=3
- derived human_ordinal_quality = 4 - Cmt
  => Bad=1, Normal=2, Good=3
- primary: Kendall tau-b
- secondary: Spearman rho
- bootstrap: outfit-level, percentile CI
- pairwise exact logit ties contribute 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Iterable, Mapping, Sequence


CLASS_ORDER = ("Bad", "Normal", "Good")
PAIRWISE_SPECS = (
    ("Good", "Bad", "P_Good_gt_Bad"),
    ("Good", "Normal", "P_Good_gt_Normal"),
    ("Normal", "Bad", "P_Normal_gt_Bad"),
)


@dataclass(frozen=True)
class MetricCI:
    estimate: float
    ci_low: float
    ci_high: float
    valid_resamples: int


def _as_finite_floats(values: Sequence[float], name: str) -> list[float]:
    out = [float(v) for v in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(v) for v in out):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_pair(
    logits: Sequence[float],
    quality: Sequence[float],
) -> tuple[list[float], list[float]]:
    x = _as_finite_floats(logits, "logits")
    y = _as_finite_floats(quality, "quality")
    if len(x) != len(y):
        raise ValueError(
            f"logits/quality length mismatch: {len(x)} != {len(y)}"
        )
    if len(x) < 2:
        raise ValueError("At least two observations are required")
    return x, y


def kendall_value(
    logits: Sequence[float],
    quality: Sequence[float],
) -> float:
    """Kendall tau-b with tie correction on both variables."""
    x, y = _validate_pair(logits, quality)

    concordant = 0
    discordant = 0
    ties_x_only = 0
    ties_y_only = 0

    n = len(x)
    for i in range(n - 1):
        xi = x[i]
        yi = y[i]
        for j in range(i + 1, n):
            dx = xi - x[j]
            dy = yi - y[j]

            if dx == 0.0 and dy == 0.0:
                continue
            if dx == 0.0:
                ties_x_only += 1
                continue
            if dy == 0.0:
                ties_y_only += 1
                continue

            if (dx > 0.0) == (dy > 0.0):
                concordant += 1
            else:
                discordant += 1

    numerator = concordant - discordant
    denominator = math.sqrt(
        (concordant + discordant + ties_x_only)
        * (concordant + discordant + ties_y_only)
    )

    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda row: row[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i + 1
        value = indexed[i][1]
        while j < len(indexed) and indexed[j][1] == value:
            j += 1

        average_rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            original_index = indexed[k][0]
            ranks[original_index] = average_rank
        i = j

    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("Pearson inputs must have the same length")
    if len(x) < 2:
        return math.nan

    mx = statistics.fmean(x)
    my = statistics.fmean(y)

    dx = [v - mx for v in x]
    dy = [v - my for v in y]

    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = sum(a * a for a in dx)
    denom_y = sum(b * b for b in dy)
    denominator = math.sqrt(denom_x * denom_y)

    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def spearman_value(
    logits: Sequence[float],
    quality: Sequence[float],
) -> float:
    """Spearman rho using average ranks for ties."""
    x, y = _validate_pair(logits, quality)
    return _pearson(_average_ranks(x), _average_ranks(y))


def linear_quantile(values: Sequence[float], q: float) -> float:
    """
    Linear quantile compatible with NumPy's method='linear'.

    Position = (n - 1) * q with linear interpolation between adjacent values.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    xs = sorted(float(v) for v in values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]

    position = (len(xs) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return xs[lo]

    weight = position - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def percentile_ci(
    values: Sequence[float],
    ci_level: float,
) -> tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        raise ValueError("No finite bootstrap values")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    alpha = 1.0 - ci_level
    return (
        linear_quantile(finite, alpha / 2.0),
        linear_quantile(finite, 1.0 - alpha / 2.0),
    )


def bootstrap_both(
    logits: Sequence[float],
    quality: Sequence[float],
    seed: int,
    resamples: int,
    ci_level: float,
    *,
    backend: str = "frozen",
) -> tuple[MetricCI, MetricCI]:
    """
    Bootstrap Kendall tau-b and Spearman rho with identical outfit indices.

    backend="frozen"
        Reproduces the frozen evaluation implementation:
        NumPy default_rng + SciPy kendalltau/spearmanr. Imports are lazy so
        merely importing this module does not require NumPy/SciPy.

    backend="reference"
        Standard-library reference backend intended for small unit tests only.
        It is deterministic but does NOT use the same RNG stream as NumPy and
        therefore is not used for frozen reported confidence intervals.
    """
    x, y = _validate_pair(logits, quality)

    if resamples <= 0:
        raise ValueError("resamples must be positive")

    if backend == "frozen":
        try:
            import numpy as np
            from scipy.stats import kendalltau, spearmanr
        except ImportError as exc:
            raise RuntimeError(
                "Frozen bootstrap backend requires numpy and scipy. "
                "Install evaluation dependencies or use backend='reference' "
                "only for small tests."
            ) from exc

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        def k_metric(a, b):
            return float(
                kendalltau(
                    a,
                    b,
                    variant="b",
                    nan_policy="raise",
                ).statistic
            )

        def s_metric(a, b):
            return float(
                spearmanr(
                    a,
                    b,
                    nan_policy="raise",
                ).statistic
            )

        k_point = k_metric(x_arr, y_arr)
        s_point = s_metric(x_arr, y_arr)

        if not math.isfinite(k_point):
            raise ValueError("Non-finite Kendall point estimate")
        if not math.isfinite(s_point):
            raise ValueError("Non-finite Spearman point estimate")

        rng = np.random.default_rng(int(seed))
        k_values = np.empty(int(resamples), dtype=np.float64)
        s_values = np.empty(int(resamples), dtype=np.float64)
        valid = 0
        n = len(x_arr)

        for _ in range(int(resamples)):
            indices = rng.integers(0, n, size=n)
            k = k_metric(x_arr[indices], y_arr[indices])
            s = s_metric(x_arr[indices], y_arr[indices])

            if math.isfinite(k) and math.isfinite(s):
                k_values[valid] = k
                s_values[valid] = s
                valid += 1

        min_valid = max(100, int(0.99 * int(resamples)))
        if valid < min_valid:
            raise ValueError(
                f"Too many invalid bootstrap resamples: {valid}/{resamples}"
            )

        alpha = 1.0 - float(ci_level)
        k_low, k_high = np.quantile(
            k_values[:valid],
            [alpha / 2.0, 1.0 - alpha / 2.0],
            method="linear",
        )
        s_low, s_high = np.quantile(
            s_values[:valid],
            [alpha / 2.0, 1.0 - alpha / 2.0],
            method="linear",
        )

        return (
            MetricCI(k_point, float(k_low), float(k_high), valid),
            MetricCI(s_point, float(s_low), float(s_high), valid),
        )

    if backend != "reference":
        raise ValueError(
            f"Unknown bootstrap backend {backend!r}; "
            "expected 'frozen' or 'reference'"
        )

    k_point = kendall_value(x, y)
    s_point = spearman_value(x, y)
    if not math.isfinite(k_point) or not math.isfinite(s_point):
        raise ValueError("Non-finite point estimate")

    rng = random.Random(int(seed))
    k_values: list[float] = []
    s_values: list[float] = []
    n = len(x)

    for _ in range(int(resamples)):
        indices = [rng.randrange(n) for _ in range(n)]
        bx = [x[i] for i in indices]
        by = [y[i] for i in indices]
        k = kendall_value(bx, by)
        s = spearman_value(bx, by)
        if math.isfinite(k) and math.isfinite(s):
            k_values.append(k)
            s_values.append(s)

    valid = len(k_values)
    min_valid = max(100, int(0.99 * int(resamples)))
    if valid < min_valid:
        raise ValueError(
            f"Too many invalid bootstrap resamples: {valid}/{resamples}"
        )

    k_low, k_high = percentile_ci(k_values, ci_level)
    s_low, s_high = percentile_ci(s_values, ci_level)
    return (
        MetricCI(k_point, k_low, k_high, valid),
        MetricCI(s_point, s_low, s_high, valid),
    )


def ordering_probability(
    higher_logits: Sequence[float],
    lower_logits: Sequence[float],
) -> dict[str, float | int]:
    """P(higher-class logit > lower-class logit), exact ties count 0.5."""
    high = [float(v) for v in higher_logits]
    low = [float(v) for v in lower_logits]

    if any(not math.isfinite(v) for v in high + low):
        raise ValueError("Pairwise logits contain non-finite values")

    if not high or not low:
        return {
            "probability": math.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "cross_class_pairs": 0,
        }

    wins = ties = losses = 0
    for h in high:
        for l in low:
            if h > l:
                wins += 1
            elif h < l:
                losses += 1
            else:
                ties += 1

    total = wins + ties + losses
    return {
        "probability": (wins + 0.5 * ties) / total,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cross_class_pairs": total,
    }


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    xs = _as_finite_floats(values, "values")
    q1 = linear_quantile(xs, 0.25)
    median = linear_quantile(xs, 0.50)
    q3 = linear_quantile(xs, 0.75)

    return {
        "N": len(xs),
        "median_logit": median,
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "mean_logit": statistics.fmean(xs),
        "std_logit": statistics.stdev(xs) if len(xs) > 1 else math.nan,
    }


def evaluate_subset(
    records: Sequence[Mapping[str, object]],
    subset_name: str,
    bootstrap_policy: Mapping[str, object],
    *,
    bootstrap_backend: str = "frozen",
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """
    Evaluate one already-filtered subset.

    Each record must provide:
        compatibility_logit
        human_ordinal_quality
        human_label
    """
    if not records:
        raise ValueError(f"{subset_name} is empty")

    logits: list[float] = []
    quality: list[float] = []
    class_values = {label: [] for label in CLASS_ORDER}

    for row in records:
        logit = float(row["compatibility_logit"])
        q = float(row["human_ordinal_quality"])
        label = str(row["human_label"])

        if not math.isfinite(logit) or not math.isfinite(q):
            raise ValueError(f"{subset_name} contains non-finite metric inputs")
        if label not in class_values:
            raise ValueError(f"Unexpected human label: {label!r}")

        logits.append(logit)
        quality.append(q)
        class_values[label].append(logit)

    missing = [label for label in CLASS_ORDER if not class_values[label]]
    if missing:
        raise ValueError(f"{subset_name} is missing human classes: {missing}")

    k, s = bootstrap_both(
        logits,
        quality,
        seed=int(bootstrap_policy["seed"]),
        resamples=int(bootstrap_policy["resamples"]),
        ci_level=float(bootstrap_policy["ci_level"]),
        backend=bootstrap_backend,
    )

    main = {
        "subset": subset_name,
        "N_total": len(records),
        "N_Bad": len(class_values["Bad"]),
        "N_Normal": len(class_values["Normal"]),
        "N_Good": len(class_values["Good"]),
        "kendall_tau_b": k.estimate,
        "kendall_ci_low": k.ci_low,
        "kendall_ci_high": k.ci_high,
        "spearman_rho": s.estimate,
        "spearman_ci_low": s.ci_low,
        "spearman_ci_high": s.ci_high,
        "bootstrap_seed": int(bootstrap_policy["seed"]),
        "bootstrap_resamples": int(bootstrap_policy["resamples"]),
        "bootstrap_ci_level": float(bootstrap_policy["ci_level"]),
        "bootstrap_ci_method": str(bootstrap_policy.get("ci_method", "percentile")),
        "bootstrap_unit": str(bootstrap_policy.get("unit", "outfit")),
        "bootstrap_valid_resamples": k.valid_resamples,
    }

    class_rows: list[dict[str, object]] = []
    for label in CLASS_ORDER:
        class_rows.append(
            {
                "subset": subset_name,
                "human_class": label,
                **summarize_values(class_values[label]),
            }
        )

    pair_rows: list[dict[str, object]] = []
    for high_label, low_label, name in PAIRWISE_SPECS:
        pair_rows.append(
            {
                "subset": subset_name,
                "diagnostic": name,
                "higher_class": high_label,
                "lower_class": low_label,
                **ordering_probability(
                    class_values[high_label],
                    class_values[low_label],
                ),
            }
        )

    return main, class_rows, pair_rows
