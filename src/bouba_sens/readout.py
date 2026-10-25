"""K-draw Gumbel probe readout and code-structure statistics (PI decisions 2026-10-25).

After the v0.7 B-2 pilot the PI decided (2026-10-25, 13:12) that every
Gumbel probe readout averages over **K = 8** independent Gumbel draws of
the same network and probe (v0.7: the MI estimate is averaged; v0.8: the
codes are averaged), and that v0.7 reports a per-seed binary outcome
"class-separating switch learned".

Draw seeding (frozen): draw r (r = 0 ... K - 1) of phase p (0 = before
the lesion phase, 1 = after it) for experiment seed s is computed inside
``torch.random.fork_rng`` after ``torch.manual_seed(gumbel_draw_seed(s,
p, r))`` with gumbel_draw_seed = 2 * 10**9 + 1000 * s + 100 * p + r. The
fork restores the global RNG afterwards, so the K draws neither depend
on nor change the training stream.

Switch rule (frozen; v0.7 section 4.2): for a code matrix Z (512 x 896)
and the probe labels y, with the centred covariance eigenvalues
lambda_1 >= lambda_2 >= ...:

* top-1 share = lambda_1 / sum(lambda);
* between-class share = (sum_c n_c ||mean_c - mean||^2) / (N - 1) / sum(lambda),
  the fraction of the total variance carried by the class means;

each averaged over the K draws. A switch is detected when the
post-lesion top-1 share is >= ``SWITCH_TOP1`` (0.25) **and** the
post-lesion between-class share is >= ``SWITCH_BETWEEN`` (0.10).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

GUMBEL_DRAWS = 8
PHASE_PRE = 0
PHASE_POST = 1
SWITCH_TOP1 = 0.25
SWITCH_BETWEEN = 0.10


def gumbel_draw_seed(seed: int, phase: int, draw: int) -> int:
    """Generator seed of Gumbel draw ``draw`` in phase ``phase`` (module docstring)."""
    if phase not in (PHASE_PRE, PHASE_POST) or not 0 <= draw < 100:
        raise ValueError(f"phase must be 0 or 1 and draw in [0, 100); got {phase}, {draw}")
    return 2 * 10**9 + 1000 * seed + 100 * phase + draw


def gumbel_draws(capture: Callable[[], Tensor], seed: int, phase: int, k: int) -> list[Tensor]:
    """``k`` captures, each under its own seed, without touching the global RNG."""
    out: list[Tensor] = []
    for r in range(k):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(gumbel_draw_seed(seed, phase, r))
            out.append(capture())
    return out


def code_structure(z: Tensor, labels: Tensor) -> dict[str, float]:
    """Top-1 eigenvalue share and between-class share of the variance of ``z``."""
    x = z.detach().double()
    y = labels.detach().long().flatten()
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("z must be (N, d) with N >= 2 labels")
    xc = x - x.mean(0)
    lam = torch.linalg.svdvals(xc) ** 2 / (x.shape[0] - 1)
    total = float(lam.sum())
    if total == 0.0:
        return {"top1_share": 0.0, "between_class_share": 0.0}
    between = 0.0
    for c in torch.unique(y):
        m = y == c
        between += float(m.sum()) * float(((x[m].mean(0) - x.mean(0)) ** 2).sum())
    between /= x.shape[0] - 1
    return {"top1_share": float(lam[0]) / total, "between_class_share": between / total}


def mean_structure(zs: Sequence[Tensor], labels: Tensor) -> dict[str, float]:
    """``code_structure`` averaged over draws."""
    stats = [code_structure(z, labels) for z in zs]
    return {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}


def switch_detected(post: dict[str, float]) -> bool:
    """The frozen rule on the post-lesion (K-averaged) structure statistics."""
    return post["top1_share"] >= SWITCH_TOP1 and post["between_class_share"] >= SWITCH_BETWEEN
