"""Me9 (seed-variance / replication index). Spec §5.2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import bootstrap  # type: ignore[import-untyped]


def me9_bootstrap(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap-based mean + confidence interval over seed variance.

    Returns `(mean, ci_low, ci_high)` using scipy's BCa bootstrap with
    `n_resamples` resamples (spec §4.5 default = 1000) at `confidence`
    level (spec default 95 %). Reproducible via `seed`.

    Me9 is the cross-cell variance summary: AdaptationLoop collects a
    single metric value per seed, and me9_bootstrap compresses the
    per-seed list into (mean, lo, hi) for tabulation in the HTML report
    (Task 3.9) and the invariant-threshold decisions (§1.2 B-1/B-2/B-3).
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        # Bootstrap needs at least 2 samples; for a singleton return
        # (x, x, x) so callers do not need to special-case.
        value = float(data.mean()) if data.size else 0.0
        return value, value, value

    rng = np.random.default_rng(seed)
    result = bootstrap(
        (data,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence,
        random_state=rng,
    )
    return (
        float(data.mean()),
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )
