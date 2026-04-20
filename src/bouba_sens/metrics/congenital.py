"""Me7 (congenital vs late lesion gap). Spec §5.2."""

from __future__ import annotations


def me7_congenital_gap(perf_t1: float, perf_t2: float) -> float:
    """Scalar B-1 invariant: `perf_T1 - perf_T2` at SNR_floor.

    Positive value signals that congenital (T1) adaptation yields higher
    asymptotic performance than late-acquired (T2) per spec §1.2 B-1.
    Threshold: > 0.05 over bootstrap 95 % CI.
    """
    return perf_t1 - perf_t2
