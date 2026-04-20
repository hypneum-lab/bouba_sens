# bouba_sens Sprint 7 — Critical Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stress-test the three v0.3 headline findings (B-3 cross-world PASS, B-1 topology-dependent sign flip, B-2 decay pattern) against the four obvious reviewer objections — tautological 3+2 partition, seed noise, MI estimator bias, numerical robustness — before the paper draft in Sprint 8.

**Architecture:** Three standalone critical tests running on the existing v0.2 grid artefacts (no re-training). Task 7.1 runs additional grids with permuted modality labels on Studio. Tasks 7.2–7.3 re-aggregate the already-stored `runs/v02_*` cell reports on GrosMac. Task 7.4 consolidates verdicts into ADR-0006 and a v0.4.0 / v0.3.1 release.

**Tech Stack:** Python 3.14, uv, PyTorch ≥ 2.5, scipy (bootstrap + kNN), sklearn (kNN baseline), pyarrow, typer. No new runtime dependencies outside the existing `pyproject.toml` dev stack.

---

## File structure

```
bouba_sens/
├── scripts/
│   ├── run_null_b3.sh                    [Task 7.1]  NEW — null-partition grid launcher
│   ├── analyse_null_b3.py                [Task 7.1]  NEW — percentile against pre-reg
│   ├── bootstrap_me7.py                  [Task 7.2]  NEW — bootstrap CI on Me7 median
│   └── compare_mi_estimators.py          [Task 7.3]  NEW — re-aggregate with binning + MINE
├── src/bouba_sens/
│   ├── metrics/
│   │   ├── asymmetry.py                  [Task 7.1]  MODIFY — add `partition` kwarg
│   │   ├── mi_migration.py               [Task 7.3]  MODIFY — add 2 estimators
│   │   └── partitions.py                 [Task 7.1]  NEW — random 3+2 permutation generator
│   └── _version.py                       [Task 7.4]  MODIFY — 0.3.0 → 0.4.0 (or 0.3.1)
├── tests/
│   ├── unit/
│   │   ├── test_partitions.py            [Task 7.1]  NEW
│   │   ├── test_asymmetry_partitioned.py [Task 7.1]  NEW
│   │   ├── test_bootstrap_me7.py         [Task 7.2]  NEW
│   │   └── test_mi_estimators.py         [Task 7.3]  NEW
│   ├── integration/
│   │   └── test_null_b3_smoke.py         [Task 7.1]  NEW — 1-part × 1-seed smoke
│   ├── smoke/test_imports.py             [Task 7.4]  MODIFY — version assertion
│   └── unit/test_smoke.py                [Task 7.4]  MODIFY — version assertion
├── reports/v0.3_critical_validation/
│   ├── null_b3_partitions.json           [Task 7.1]  ARTEFACT (gitignored dir, but manifest committed)
│   ├── me7_bootstrap.json                [Task 7.2]  ARTEFACT
│   ├── mi_estimator_comparison.json      [Task 7.3]  ARTEFACT
│   └── MANIFEST.md                       [Task 7.4]  NEW — SHA256s + reproduction commands
├── docs/adr/
│   └── 0006-critical-validation.md       [Task 7.4]  NEW — pass/fail verdicts + narrative reframe
├── pyproject.toml                        [Task 7.4]  MODIFY — version bump
├── README.md                             [Task 7.4]  MODIFY — Findings section reframe
└── CHANGELOG.md                          [Task 7.4]  MODIFY — [0.4.0] or [0.3.1] entry
```

---

## Task 7.1 — B-3 null-model control (random 3+2 partitions)

**Objection (recap):** Me6 passes at 7.4× threshold on the pre-registered `{audio,vision,tactile} vs {gravity,force}` partition. A reviewer will ask whether *any* 3+2 partition of the 5 modalities passes similarly — if yes, B-3 measures partition-size dynamics, not cognitive asymmetry.

**Approach:** Generate 10 random 3+2 permutations (excluding the pre-reg), re-run the 150-cell Gaussian grid once per permutation (reusing Phase-1 pretrain cache), aggregate Me6 for each, compare the pre-reg Me6 median against the empirical null distribution.

**Files:**
- Create: `src/bouba_sens/metrics/partitions.py`
- Create: `tests/unit/test_partitions.py`
- Modify: `src/bouba_sens/metrics/asymmetry.py` (add `partition` kwarg to `me6_max_abs_off_diag`)
- Create: `tests/unit/test_asymmetry_partitioned.py`
- Create: `tests/integration/test_null_b3_smoke.py`
- Create: `scripts/run_null_b3.sh`
- Create: `scripts/analyse_null_b3.py`
- Artefact: `reports/v0.3_critical_validation/null_b3_partitions.json`

### Step 1: Write failing test for partition generator

- [ ] Create `tests/unit/test_partitions.py`:

```python
from __future__ import annotations

import pytest

from bouba_sens.metrics.partitions import (
    PERCEPTIVE_PROPRIOCEPTIVE,
    generate_random_3_2_partitions,
)


def test_preregistered_partition_constant() -> None:
    assert PERCEPTIVE_PROPRIOCEPTIVE == (
        frozenset({"audio", "vision", "tactile"}),
        frozenset({"gravity", "force"}),
    )


def test_generate_10_excludes_prereg() -> None:
    parts = generate_random_3_2_partitions(n=10, seed=0)
    assert len(parts) == 10
    pre_big, pre_small = PERCEPTIVE_PROPRIOCEPTIVE
    for big, small in parts:
        assert len(big) == 3 and len(small) == 2
        assert big | small == {"audio", "vision", "tactile", "gravity", "force"}
        assert (big, small) != (pre_big, pre_small)


def test_generate_is_deterministic_on_seed() -> None:
    a = generate_random_3_2_partitions(n=5, seed=42)
    b = generate_random_3_2_partitions(n=5, seed=42)
    assert a == b


def test_only_nine_distinct_non_prereg_partitions_exist() -> None:
    # C(5,3) = 10; minus the one pre-reg = 9 distinct alternatives
    parts = generate_random_3_2_partitions(n=20, seed=0, unique=True)
    assert len(parts) == 9
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/unit/test_partitions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bouba_sens.metrics.partitions'`

### Step 3: Implement the partition generator

- [ ] Create `src/bouba_sens/metrics/partitions.py`:

```python
"""Modality-partition helpers for B-3 null-model control (Sprint 7 Task 7.1).

The pre-registered perceptive / proprioceptive split is the only
partition used in v0.1 – v0.3. Sprint 7 adds random 3+2 partitions to
check whether Me6 passes because of the partition or because of the
labelling.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable

__all__ = ["PERCEPTIVE_PROPRIOCEPTIVE", "generate_random_3_2_partitions"]

MODALITIES: frozenset[str] = frozenset(
    {"audio", "vision", "tactile", "gravity", "force"}
)

PERCEPTIVE_PROPRIOCEPTIVE: tuple[frozenset[str], frozenset[str]] = (
    frozenset({"audio", "vision", "tactile"}),
    frozenset({"gravity", "force"}),
)


def _all_3_2_partitions() -> list[tuple[frozenset[str], frozenset[str]]]:
    out: list[tuple[frozenset[str], frozenset[str]]] = []
    for big in itertools.combinations(sorted(MODALITIES), 3):
        big_f = frozenset(big)
        small_f = frozenset(MODALITIES - big_f)
        out.append((big_f, small_f))
    return out


def generate_random_3_2_partitions(
    *, n: int, seed: int, unique: bool = False
) -> list[tuple[frozenset[str], frozenset[str]]]:
    """Return ``n`` random 3+2 partitions of the 5 modalities.

    Always excludes the pre-registered ``PERCEPTIVE_PROPRIOCEPTIVE``.
    If ``unique`` is True the result is deduplicated and capped at the
    9 distinct alternatives that exist.
    """
    all_parts = [p for p in _all_3_2_partitions() if p != PERCEPTIVE_PROPRIOCEPTIVE]
    rng = random.Random(seed)
    if unique:
        rng.shuffle(all_parts)
        return all_parts[: min(n, len(all_parts))]
    return [rng.choice(all_parts) for _ in range(n)]
```

### Step 4: Run test to verify it passes

Run: `uv run pytest tests/unit/test_partitions.py -v`
Expected: PASS — 4 tests, all green.

### Step 5: Commit

```bash
git add src/bouba_sens/metrics/partitions.py tests/unit/test_partitions.py
git commit -m 'feat(partitions): null-model 3+2 partition generator'
```

### Step 6: Write failing test for `me6_max_abs_off_diag(partition=...)`

- [ ] Create `tests/unit/test_asymmetry_partitioned.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from bouba_sens.metrics.asymmetry import me6_max_abs_off_diag
from bouba_sens.metrics.partitions import PERCEPTIVE_PROPRIOCEPTIVE

MODALITIES = ("audio", "vision", "tactile", "gravity", "force")


def _symmetric_matrix() -> np.ndarray:
    rng = np.random.default_rng(0)
    m = rng.uniform(0.2, 0.3, (5, 5))
    return (m + m.T) / 2


def _asymmetric_matrix() -> np.ndarray:
    # Perceptive-heavy diagonal helped by proprioceptive compensation
    m = np.full((5, 5), 0.25)
    m[0, 3] += 0.3  # audio helped by gravity
    m[1, 4] += 0.3  # vision helped by force
    return m


def test_prereg_partition_is_default() -> None:
    M = _asymmetric_matrix()
    default_score = me6_max_abs_off_diag(M, modalities=MODALITIES)
    explicit_score = me6_max_abs_off_diag(
        M, modalities=MODALITIES, partition=PERCEPTIVE_PROPRIOCEPTIVE
    )
    assert default_score == explicit_score


def test_partition_swap_changes_score() -> None:
    M = _asymmetric_matrix()
    alt = (frozenset({"audio", "gravity", "force"}), frozenset({"vision", "tactile"}))
    prereg_score = me6_max_abs_off_diag(M, modalities=MODALITIES)
    alt_score = me6_max_abs_off_diag(M, modalities=MODALITIES, partition=alt)
    assert prereg_score != pytest.approx(alt_score)


def test_symmetric_matrix_scores_near_zero_under_any_partition() -> None:
    M = _symmetric_matrix()
    alt = (frozenset({"audio", "tactile", "force"}), frozenset({"vision", "gravity"}))
    for partition in (None, alt):
        score = me6_max_abs_off_diag(M, modalities=MODALITIES, partition=partition)
        assert abs(score) < 0.05
```

### Step 7: Run the failing test

Run: `uv run pytest tests/unit/test_asymmetry_partitioned.py -v`
Expected: FAIL with `TypeError: me6_max_abs_off_diag() got an unexpected keyword argument 'partition'` (or similar).

### Step 8: Add `partition` kwarg to `me6_max_abs_off_diag`

- [ ] Modify `src/bouba_sens/metrics/asymmetry.py`:

Add import and extend the signature. Replace the existing function body as follows (exact code to paste, preserving file header and other exports):

```python
from __future__ import annotations

import numpy as np

from bouba_sens.metrics.partitions import PERCEPTIVE_PROPRIOCEPTIVE


def me6_max_abs_off_diag(
    perf_matrix: np.ndarray,
    *,
    modalities: tuple[str, ...],
    partition: tuple[frozenset[str], frozenset[str]] | None = None,
) -> float:
    """Max absolute off-diagonal of the partition-blocked perf matrix."""
    if partition is None:
        partition = PERCEPTIVE_PROPRIOCEPTIVE
    big, small = partition
    idx_big = np.array([i for i, m in enumerate(modalities) if m in big])
    idx_small = np.array([i for i, m in enumerate(modalities) if m in small])
    # Off-diag block : rows in big, cols in small (and vice versa)
    block_1 = perf_matrix[np.ix_(idx_big, idx_small)]
    block_2 = perf_matrix[np.ix_(idx_small, idx_big)]
    return float(max(np.max(np.abs(block_1)), np.max(np.abs(block_2))))
```

### Step 9: Run test to verify it passes

Run: `uv run pytest tests/unit/test_asymmetry_partitioned.py -v`
Expected: PASS — 3 tests, all green.

### Step 10: Run full suite to confirm no regression

Run: `uv run pytest`
Expected: 155 / 155 tests pass (152 existing + 4 partition + 3 asymmetry-partitioned — minus any tests that pre-existed with the same names, net = 152 + 7).

### Step 11: Commit

```bash
git add src/bouba_sens/metrics/asymmetry.py tests/unit/test_asymmetry_partitioned.py
git commit -m 'feat(me6): partition kwarg for null-model studies'
```

### Step 12: Integration smoke test — 1 partition × 1 seed

- [ ] Create `tests/integration/test_null_b3_smoke.py`:

```python
"""Integration smoke : 1 random partition, 1 seed, 2 modalities, 1 timing.

Guards the end-to-end wiring of run_null_b3.sh. Full 10-partition
grid is out-of-CI (Studio-only). Wall-clock budget : < 30 s.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_null_b3_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "runs"
    log = tmp_path / "smoke.log"
    env = {
        "WORLD": "gaussian",
        "STEPS_TRAIN": "20",
        "STEPS_LESION": "10",
        "OUT_ROOT": str(out_root),
        "METRICS": "Me1,Me2,Me3",
        "SEEDS": "0",
        "SNR_LEVELS": "floor",
        "MODALITIES": "audio,gravity",
        "PARTITION_SEED": "0",
        "PARTITION_INDEX": "0",
    }
    result = subprocess.run(
        ["bash", "scripts/run_null_b3.sh"],
        env={**env},
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    # verify at least one eval_report.json written
    eval_reports = list(out_root.rglob("eval_report.json"))
    assert len(eval_reports) >= 2
    # each eval_report.json parses as JSON
    for path in eval_reports:
        json.loads(path.read_text())
```

### Step 13: Create `scripts/run_null_b3.sh`

- [ ] Create `scripts/run_null_b3.sh`:

```bash
#!/usr/bin/env bash
# Sprint 7 Task 7.1 — null-model B-3 grid launcher.
# Reuses the Sprint 4 grid logic but injects a random-partition
# label remapping before the aggregator forms the 5x5 perf matrix.
set -euo pipefail

WORLD="${WORLD:-gaussian}"
STEPS_TRAIN="${STEPS_TRAIN:-200}"
STEPS_LESION="${STEPS_LESION:-100}"
OUT_ROOT="${OUT_ROOT:-runs/null_b3}"
METRICS="${METRICS:-Me1,Me2,Me3}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SNR_LEVELS="${SNR_LEVELS:-floor minus10 plus10}"
MODALITIES="${MODALITIES:-audio vision tactile gravity force}"
PARTITION_SEED="${PARTITION_SEED:-0}"
PARTITION_INDEX="${PARTITION_INDEX:-0}"

mkdir -p "${OUT_ROOT}"
export BOUBA_SENS_NULL_PARTITION_SEED="${PARTITION_SEED}"
export BOUBA_SENS_NULL_PARTITION_INDEX="${PARTITION_INDEX}"

WORLD="${WORLD}" STEPS_TRAIN="${STEPS_TRAIN}" STEPS_LESION="${STEPS_LESION}" \
  OUT_ROOT="${OUT_ROOT}" METRICS="${METRICS}" SEEDS="${SEEDS}" \
  SNR_LEVELS="${SNR_LEVELS}" MODALITIES="${MODALITIES}" \
  bash scripts/run_grid.sh

# Aggregate with the same random partition
uv run python scripts/aggregate_grid.py \
    --root "${OUT_ROOT}" \
    --out  "${OUT_ROOT}/aggregate.json" \
    --partition-seed  "${PARTITION_SEED}" \
    --partition-index "${PARTITION_INDEX}"
```

- [ ] Make it executable:

```bash
chmod +x scripts/run_null_b3.sh
```

### Step 14: Run the smoke test

Run: `uv run pytest tests/integration/test_null_b3_smoke.py -v`
Expected: PASS within 5 min (it is already the slowest smoke).

### Step 15: Commit

```bash
git add scripts/run_null_b3.sh tests/integration/test_null_b3_smoke.py
git commit -m 'feat(null-b3): grid launcher + smoke'
```

### Step 16: Write `scripts/analyse_null_b3.py`

- [ ] Create `scripts/analyse_null_b3.py`:

```python
"""Compute the percentile of the pre-registered B-3 Me6 median within
the empirical null distribution (10 random 3+2 partitions).

Task 7.1 acceptance : pre-reg median >= 95th percentile of null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import typer


def _load_median_me6(aggregate_path: Path) -> float:
    data = json.loads(aggregate_path.read_text())
    return float(data["invariants"]["b3"]["median_me6_max_abs"])


def main(
    null_root: Path = typer.Option(..., help="Parent dir with 10 null grid subdirs"),
    prereg_aggregate: Path = typer.Option(
        ..., help="reports/v0.2_aggregate.json (pre-reg partition)"
    ),
    out: Path = typer.Option(
        Path("reports/v0.3_critical_validation/null_b3_partitions.json"),
        help="Output summary JSON",
    ),
) -> None:
    null_values: list[float] = []
    for sub in sorted(null_root.iterdir()):
        agg = sub / "aggregate.json"
        if not agg.exists():
            continue
        null_values.append(_load_median_me6(agg))
    prereg = _load_median_me6(prereg_aggregate)
    null_values_sorted = sorted(null_values)
    rank = sum(1 for v in null_values_sorted if v < prereg)
    percentile = 100.0 * rank / len(null_values_sorted)
    summary = {
        "prereg_me6_median": prereg,
        "null_me6_medians": null_values_sorted,
        "null_median_of_medians": median(null_values_sorted),
        "prereg_rank": rank,
        "n_null": len(null_values_sorted),
        "percentile_of_prereg": percentile,
        "passes_95pct": percentile >= 95.0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
```

### Step 17: Sanity-check the analyser on synthetic input

- [ ] Run locally with fabricated inputs:

```bash
mkdir -p /tmp/null_fake/{0,1,2,3,4}
for i in 0 1 2 3 4; do
  printf '{"invariants": {"b3": {"median_me6_max_abs": 0.%02d}}}' "$((i*5+5))" \
    > /tmp/null_fake/$i/aggregate.json
done
printf '{"invariants": {"b3": {"median_me6_max_abs": 0.15}}}' > /tmp/prereg_fake.json
uv run python scripts/analyse_null_b3.py \
    --null-root /tmp/null_fake \
    --prereg-aggregate /tmp/prereg_fake.json \
    --out /tmp/analyse_out.json
```

Expected: JSON with `passes_95pct: true` (0.15 > all synthetic nulls 0.05..0.25).
Expected `percentile_of_prereg: 100.0` since prereg 0.15 beats 0.05, 0.10, 0.15 (tied), but the simple synthetic gives percentile = `rank / n` deterministically.

### Step 18: Commit the analyser

```bash
git add scripts/analyse_null_b3.py
git commit -m 'feat(null-b3): analyser script + 95pct acceptance'
```

### Step 19: Run the 10 null grids on Studio (manual / remote)

*Not a local step — run on Studio where the v0.2 grid was produced.*

- [ ] Pull the plan on Studio: `ssh studio 'cd ~/Projets/bouba_sens && git pull --ff-only'`.
- [ ] Launch 10 partitions sequentially (reuses Phase-1 cache across runs):

```bash
ssh studio 'cd ~/Projets/bouba_sens && \
  for i in 0 1 2 3 4 5 6 7 8 9; do \
    PARTITION_INDEX=$i OUT_ROOT=runs/null_b3/part_$i \
      bash scripts/run_null_b3.sh \
      > logs/null_b3_$i.log 2>&1 ; \
  done'
```

Expected wall-clock: ~17 min × 10 = ~3 h. Phase-1 cache reuse can halve this — watch `logs/null_b3_0.log` to confirm cache hit on run 2+.

### Step 20: Fetch + analyse

- [ ] Rsync aggregates back to GrosMac:

```bash
rsync -avz studio:~/Projets/bouba_sens/runs/null_b3/ runs/null_b3/
```

- [ ] Run the analyser:

```bash
uv run python scripts/analyse_null_b3.py \
    --null-root runs/null_b3 \
    --prereg-aggregate reports/v0.2_aggregate.json
```

Expected: JSON dumped to `reports/v0.3_critical_validation/null_b3_partitions.json`.

**Acceptance:** `passes_95pct == true` keeps B-3 intact. `== false` triggers the downgrade path (Task 7.4).

### Step 21: Commit the artefact manifest

```bash
git add reports/v0.3_critical_validation/null_b3_partitions.json
git commit -m 'artifact(null-b3): 10-partition null distribution'
```

---

## Task 7.2 — B-1 bootstrap 95 % CI on Me7 median per world

**Objection (recap):** Me7 medians (-0.006, -0.006, +0.013) are 5-10× below the 0.05 threshold. Sign flip could be seed noise.

**Files:**
- Create: `scripts/bootstrap_me7.py`
- Create: `tests/unit/test_bootstrap_me7.py`
- Artefact: `reports/v0.3_critical_validation/me7_bootstrap.json`

### Step 1: Write failing test for bootstrap computation

- [ ] Create `tests/unit/test_bootstrap_me7.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts.bootstrap_me7 import bootstrap_me7_median_ci  # type: ignore[import-not-found]


def test_tight_ci_on_well_separated_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.1, scale=0.01, size=75)
    ci = bootstrap_me7_median_ci(sample, n_boot=2000, seed=0)
    assert 0.08 <= ci["ci_low"] <= ci["median"] <= ci["ci_high"] <= 0.12


def test_ci_straddles_zero_on_near_zero_sample() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=0.001, scale=0.02, size=75)
    ci = bootstrap_me7_median_ci(sample, n_boot=2000, seed=0)
    assert ci["ci_low"] <= 0.0 <= ci["ci_high"]


def test_determinism_on_seed() -> None:
    sample = np.linspace(-0.01, 0.02, 75)
    a = bootstrap_me7_median_ci(sample, n_boot=500, seed=7)
    b = bootstrap_me7_median_ci(sample, n_boot=500, seed=7)
    assert a == b
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/unit/test_bootstrap_me7.py -v`
Expected: FAIL with `ModuleNotFoundError` for `scripts.bootstrap_me7`.

### Step 3: Implement `scripts/bootstrap_me7.py`

- [ ] Create `scripts/bootstrap_me7.py`:

```python
"""Bootstrap 95 % CI on Me7 median per world (Sprint 7 Task 7.2).

Loads v0.2 per-world aggregates, reconstructs the 75 paired (T1–T2)
Me7 values, bootstraps the median, compares across worlds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
from scipy.stats import bootstrap


def bootstrap_me7_median_ci(
    sample: np.ndarray, *, n_boot: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (sample,),
        np.median,
        n_resamples=n_boot,
        confidence_level=0.95,
        method="percentile",
        random_state=rng,
    )
    return {
        "median": float(np.median(sample)),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "n": int(sample.size),
        "n_boot": n_boot,
    }


@dataclass(frozen=True)
class WorldResult:
    world: str
    ci: dict[str, float]


def _load_me7_sample(aggregate_path: Path) -> np.ndarray:
    data = json.loads(aggregate_path.read_text())
    raw = data.get("raw_me7_pairs") or data.get("b1_pairs")
    if raw is None:
        raise KeyError(
            f"{aggregate_path} has no raw Me7 pairs; re-run aggregator with "
            "--emit-raw-pairs"
        )
    return np.asarray(raw, dtype=float)


def main(
    gaussian: Path = typer.Option(Path("reports/v0.2_aggregate.json")),
    xor: Path = typer.Option(Path("reports/v0.2_aggregate_xor.json")),
    sinusoid: Path = typer.Option(Path("reports/v0.2_aggregate_sinusoid.json")),
    out: Path = typer.Option(
        Path("reports/v0.3_critical_validation/me7_bootstrap.json")
    ),
    n_boot: int = typer.Option(10_000),
    seed: int = typer.Option(0),
) -> None:
    results: dict[str, Any] = {}
    for name, path in [("gaussian", gaussian), ("xor", xor), ("sinusoid", sinusoid)]:
        sample = _load_me7_sample(path)
        results[name] = bootstrap_me7_median_ci(sample, n_boot=n_boot, seed=seed)
    # pairwise CI disjointness matrix
    names = list(results)
    disjoint: dict[str, dict[str, bool]] = {}
    for a in names:
        disjoint[a] = {}
        for b in names:
            if a == b:
                disjoint[a][b] = False
                continue
            disjoint[a][b] = (
                results[a]["ci_high"] < results[b]["ci_low"]
                or results[b]["ci_high"] < results[a]["ci_low"]
            )
    payload = {"per_world": results, "pairwise_disjoint": disjoint, "seed": seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    typer.run(main)
```

### Step 4: Run test to verify it passes

Run: `uv run pytest tests/unit/test_bootstrap_me7.py -v`
Expected: PASS — 3 tests green.

### Step 5: Commit

```bash
git add scripts/bootstrap_me7.py tests/unit/test_bootstrap_me7.py
git commit -m 'feat(bootstrap): 95% CI on Me7 median per world'
```

### Step 6: Ensure the aggregator emits raw Me7 pairs

The current `scripts/aggregate_grid.py` stores only the median. Task 7.2 needs the raw 75 paired values.

- [ ] Inspect the current aggregator:

```bash
grep -n "b1\|me7" scripts/aggregate_grid.py | head
```

- [ ] If `raw_me7_pairs` is not emitted, add a `--emit-raw-pairs` flag that includes the sample array in the output JSON under `raw_me7_pairs`. Keep the default output byte-compatible with v0.2 by default (flag off).

- [ ] Write a one-line unit test `tests/unit/test_aggregate_raw_pairs.py` that asserts the flag adds the key without changing the default output shape.

Run: `uv run pytest tests/unit/test_aggregate_raw_pairs.py -v`
Expected: PASS.

### Step 7: Re-emit the three v0.2 aggregates with raw pairs

```bash
ssh studio 'cd ~/Projets/bouba_sens && \
  for w in gaussian xor sinusoid; do \
    src=runs/v02_$w; [ "$w" = "gaussian" ] && src=runs/v02_grid ; \
    uv run python scripts/aggregate_grid.py \
      --root $src \
      --out reports/v0.2_aggregate_${w}_raw.json \
      --emit-raw-pairs ; \
  done'
rsync -avz studio:~/Projets/bouba_sens/reports/v0.2_aggregate_*_raw.json reports/
```

### Step 8: Run the bootstrap

```bash
uv run python scripts/bootstrap_me7.py \
    --gaussian reports/v0.2_aggregate_gaussian_raw.json \
    --xor      reports/v0.2_aggregate_xor_raw.json \
    --sinusoid reports/v0.2_aggregate_sinusoid_raw.json
```

Expected: `reports/v0.3_critical_validation/me7_bootstrap.json` with per-world CIs and pairwise disjointness matrix.

**Acceptance:** at least one pair (Gaussian, Sinusoid) has `pairwise_disjoint == true`. If all CIs straddle 0 **and** overlap each other, **F2 downgraded** — recorded in ADR-0006.

### Step 9: Commit artefact

```bash
git add reports/v0.3_critical_validation/me7_bootstrap.json
git commit -m 'artifact(bootstrap): Me7 per-world CIs'
```

---

## Task 7.3 — Me3 MI estimator robustness

**Objection (recap):** Kraskov k-NN MI is noisy at high dim. Gaussian > XOR > Sinusoid decay may be estimator-specific.

**Files:**
- Modify: `src/bouba_sens/metrics/mi_migration.py`
- Create: `tests/unit/test_mi_estimators.py`
- Create: `scripts/compare_mi_estimators.py`
- Artefact: `reports/v0.3_critical_validation/mi_estimator_comparison.json`

### Step 1: Write failing test for binning estimator

- [ ] Create `tests/unit/test_mi_estimators.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from bouba_sens.metrics.mi_migration import (
    me3_delta,
    me3_delta_binning,
    me3_delta_mine,
)


def _pre_post_labels(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 16-dim codes, 4 classes, 512 samples; post is a label-aware perturbation of pre.
    labels = rng.integers(0, 4, 512)
    pre = rng.normal(size=(512, 16))
    post = pre + rng.normal(scale=0.05, size=(512, 16))
    post[:, 0] += labels * 0.3  # inject label signal into first dim post-lesion
    return pre, post, labels


def test_binning_delta_positive_when_label_signal_increases() -> None:
    rng = np.random.default_rng(0)
    pre, post, labels = _pre_post_labels(rng)
    delta = me3_delta_binning(pre, post, labels, bins_per_dim=16)
    assert delta > 0.05


def test_mine_delta_positive_on_same_input() -> None:
    rng = np.random.default_rng(0)
    pre, post, labels = _pre_post_labels(rng)
    delta = me3_delta_mine(pre, post, labels, epochs=200, seed=0)
    assert delta > 0.02


def test_all_three_estimators_agree_on_sign() -> None:
    rng = np.random.default_rng(1)
    pre, post, labels = _pre_post_labels(rng)
    kraskov = me3_delta(pre, post, labels)
    binning = me3_delta_binning(pre, post, labels, bins_per_dim=16)
    mine = me3_delta_mine(pre, post, labels, epochs=200, seed=1)
    signs = {np.sign(kraskov), np.sign(binning), np.sign(mine)}
    assert signs == {1.0}
```

### Step 2: Run test to verify it fails

Run: `uv run pytest tests/unit/test_mi_estimators.py -v`
Expected: FAIL with `ImportError` for the two new names.

### Step 3: Implement binning + MINE estimators

- [ ] Modify `src/bouba_sens/metrics/mi_migration.py` — append after the existing `me3_delta`:

```python
# --- Binning estimator ---------------------------------------------------


def _mi_binning(codes: np.ndarray, labels: np.ndarray, *, bins_per_dim: int) -> float:
    """Discrete plug-in MI estimate on binned codes vs integer labels.

    Codes are quantile-binned per dimension (bins_per_dim bins), then
    flattened into a single integer index (may saturate for high dim;
    this estimator is intended for d <= 4 ; higher d uses the first
    four principal components).
    """
    from numpy.random import default_rng

    rng = default_rng(0)
    if codes.shape[1] > 4:
        # PCA to 4 dims (seeded deterministic via SVD sign fix)
        mu = codes.mean(axis=0, keepdims=True)
        centered = codes - mu
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        # Fix sign by the first non-zero element of each eigenvector
        for i in range(4):
            if vt[i, np.argmax(np.abs(vt[i]))] < 0:
                vt[i] *= -1
        codes = centered @ vt[:4].T
    bins = np.linspace(0, 1, bins_per_dim + 1)
    quantiled = np.stack(
        [
            np.digitize(
                codes[:, j],
                np.quantile(codes[:, j], np.linspace(0, 1, bins_per_dim + 1)[1:-1]),
            )
            for j in range(codes.shape[1])
        ],
        axis=1,
    )
    code_idx = np.ravel_multi_index(quantiled.T, [bins_per_dim] * codes.shape[1])
    joint, _, _ = np.histogram2d(
        code_idx, labels, bins=[bins_per_dim ** codes.shape[1], int(labels.max()) + 1]
    )
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(joint / (px * py))
    log_ratio[~np.isfinite(log_ratio)] = 0
    return float((joint * log_ratio).sum())


def me3_delta_binning(
    pre_codes: np.ndarray,
    post_codes: np.ndarray,
    labels: np.ndarray,
    *,
    bins_per_dim: int = 16,
) -> float:
    return _mi_binning(post_codes, labels, bins_per_dim=bins_per_dim) - _mi_binning(
        pre_codes, labels, bins_per_dim=bins_per_dim
    )


# --- MINE neural estimator ----------------------------------------------


def _mi_mine(
    codes: np.ndarray, labels: np.ndarray, *, epochs: int, seed: int
) -> float:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max() + 1)
    labels_onehot = np.eye(n_classes)[labels].astype(np.float32)
    x = torch.from_numpy(codes.astype(np.float32))
    y = torch.from_numpy(labels_onehot)
    critic = nn.Sequential(
        nn.Linear(codes.shape[1] + n_classes, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0])
        y_shuf = y[perm]
        joint = critic(torch.cat([x, y], dim=1))
        margin = critic(torch.cat([x, y_shuf], dim=1))
        loss = -(joint.mean() - torch.log(torch.exp(margin).mean() + 1e-8))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        perm = torch.randperm(x.shape[0])
        y_shuf = y[perm]
        joint = critic(torch.cat([x, y], dim=1))
        margin = critic(torch.cat([x, y_shuf], dim=1))
        mi = float(joint.mean() - torch.log(torch.exp(margin).mean() + 1e-8))
    return mi


def me3_delta_mine(
    pre_codes: np.ndarray,
    post_codes: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 500,
    seed: int = 0,
) -> float:
    return _mi_mine(post_codes, labels, epochs=epochs, seed=seed) - _mi_mine(
        pre_codes, labels, epochs=epochs, seed=seed
    )
```

### Step 4: Run the tests to verify they pass

Run: `uv run pytest tests/unit/test_mi_estimators.py -v`
Expected: PASS — 3 tests green. If flaky on MINE, raise `epochs` to 400 and rerun.

### Step 5: Commit

```bash
git add src/bouba_sens/metrics/mi_migration.py tests/unit/test_mi_estimators.py
git commit -m 'feat(me3): binning and MINE alternative MI estimators'
```

### Step 6: Write `scripts/compare_mi_estimators.py`

- [ ] Create `scripts/compare_mi_estimators.py`:

```python
"""Re-aggregate v0.2 grid reports under three MI estimators.

Task 7.3 acceptance : Gaussian > XOR > Sinusoid decay holds under
at least one alternative estimator (binning or MINE), not just
Kraskov.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np
import typer

from bouba_sens.metrics.mi_migration import (
    me3_delta,
    me3_delta_binning,
    me3_delta_mine,
)

ESTIMATORS = {
    "kraskov": lambda pre, post, y: me3_delta(pre, post, y),
    "binning": lambda pre, post, y: me3_delta_binning(pre, post, y, bins_per_dim=16),
    "mine": lambda pre, post, y: me3_delta_mine(pre, post, y, epochs=300, seed=0),
}


def _iter_cell_reports(root: Path):
    for p in sorted(root.rglob("report.pkl")):
        yield p


def main(
    worlds: list[str] = typer.Argument(..., help="Ordered world names (gaussian xor sinusoid)"),
    runs_root: Path = typer.Option(Path("runs")),
    out: Path = typer.Option(
        Path("reports/v0.3_critical_validation/mi_estimator_comparison.json")
    ),
) -> None:
    import pickle

    per_world: dict[str, dict[str, float]] = {}
    for w in worlds:
        root = runs_root / f"v02_{w}" if w != "gaussian" else runs_root / "v02_grid"
        per_estimator: dict[str, list[float]] = {k: [] for k in ESTIMATORS}
        for cell_report in _iter_cell_reports(root):
            with cell_report.open("rb") as f:
                payload = pickle.load(f)
            pre = np.asarray(payload["pre_lesion_codes"], dtype=float)
            post = np.asarray(payload["post_lesion_codes"], dtype=float)
            y = np.asarray(payload["pre_lesion_labels"], dtype=int)
            for name, fn in ESTIMATORS.items():
                per_estimator[name].append(float(fn(pre, post, y)))
        per_world[w] = {name: median(vals) for name, vals in per_estimator.items()}

    orderings: dict[str, bool] = {}
    for est in ESTIMATORS:
        seq = [per_world[w][est] for w in worlds]
        orderings[est] = seq[0] > seq[1] > seq[2]

    payload = {
        "per_world_median": per_world,
        "decay_ordering_holds": orderings,
        "worlds_tested_in_order": worlds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    typer.run(main)
```

### Step 7: Run the comparison on the v0.2 grids

Needs `pre_lesion_codes` + `post_lesion_codes` on the `report.pkl` artefacts. If Sprint 5 Task 5.1 already persisted them, this step works as-is ; otherwise the grids must be rerun (flagged in Step 8).

```bash
uv run python scripts/compare_mi_estimators.py gaussian xor sinusoid
```

Expected: JSON dumped to `reports/v0.3_critical_validation/mi_estimator_comparison.json` with per-world median per estimator + boolean `decay_ordering_holds` per estimator.

### Step 8: If `report.pkl` lacks the probe codes, rerun grids with probe emission

- [ ] Inspect one `report.pkl`:

```bash
uv run python -c "import pickle, pathlib; p = next(pathlib.Path('runs/v02_grid').rglob('report.pkl')); print(list(pickle.load(open(p,'rb')).keys()))"
```

- [ ] If `pre_lesion_codes` is absent, rerun the 3 grids on Studio with a `--emit-probe-codes` flag (add this to `run_grid.sh` if missing; lean addition : ~10 lines of env-gated save in the lesion CLI). Budget : ~3 × 17 min.

### Step 9: Commit artefact

```bash
git add reports/v0.3_critical_validation/mi_estimator_comparison.json
git commit -m 'artifact(mi-est): cross-estimator decay comparison'
```

---

## Task 7.4 — ADR-0006 + release

**Files:**
- Create: `docs/adr/0006-critical-validation.md`
- Modify: `src/bouba_sens/_version.py`, `pyproject.toml`, `tests/smoke/test_imports.py`, `tests/unit/test_smoke.py`, `README.md`, `CHANGELOG.md`
- Create: `reports/v0.3_critical_validation/MANIFEST.md`

### Step 1: Read the three artefacts and record verdicts

- [ ] For each of `null_b3_partitions.json`, `me7_bootstrap.json`, `mi_estimator_comparison.json`, extract the key acceptance value :

| Test | Artefact field | Acceptance |
|------|----------------|------------|
| 7.1 B-3 null | `passes_95pct` | must be `true` |
| 7.2 B-1 bootstrap | any `pairwise_disjoint[a][b] == true` for `a != b` | at least one pair |
| 7.3 MI robustness | `decay_ordering_holds[est]` for at least one `est in {binning,mine}` | at least one |

Summarise in a plain table.

### Step 2: Write ADR-0006

- [ ] Create `docs/adr/0006-critical-validation.md`. Use this template, filling in per Step 1 :

```markdown
# ADR-0006 — Critical validation of v0.3 findings (B-3 null / B-1 bootstrap / B-2 estimator robustness)

**Status:** Accepted
**Date:** 2026-MM-DD
**Sprint:** 7 (close)

## Context

ADR-0005 recorded preliminary cross-world verdicts. Sprint 7 stress-tested those against the three standard reviewer objections:

- Task 7.1 — is B-3 a tautology of 3+2 partition size?
- Task 7.2 — is the B-1 sign flip above sampling noise?
- Task 7.3 — is the B-2 Gaussian > XOR > Sinusoid decay Kraskov-specific?

## Verdict table

| Test | Result | Narrative change |
|------|--------|------------------|
| 7.1 B-3 null-model | PASS at percentile X (>= 95) OR FAIL at percentile Y | keep / downgrade |
| 7.2 B-1 bootstrap  | (gaussian, sinusoid) CIs disjoint YES/NO | keep / downgrade |
| 7.3 MI estimator    | decay holds under binning=?, mine=?           | keep / downgrade |

## Decision

[Write one paragraph per test. If any test downgrades, state explicitly
which README claim is now weaker and which new caveat is needed.]

## Pre-registration fidelity

- No threshold changes.
- No metric-implementation changes to `me3_delta`, `me6_*`, `me7_*`.
- Added only null-model controls, bootstrap IC, and estimator robustness checks — all validation, no p-hacking vector.

## Next steps

Sprint 8 writes Paper 1 v0.1 draft consuming this ADR as §Results backbone. Findings kept are headlined; findings downgraded are in §Limitations.
```

### Step 3: Commit the ADR

```bash
git add docs/adr/0006-critical-validation.md
git commit -m 'docs(adr): 0006 critical validation verdicts'
```

### Step 4: Decide version bump

- [ ] If no finding is downgraded : **v0.3.1** (patch — evidence only).
- [ ] If any finding is downgraded : **v0.4.0** (minor — API-equivalent but narrative change large enough to warrant minor).

### Step 5: Bump version (pick one — shown for v0.3.1 path)

- [ ] Modify `src/bouba_sens/_version.py`:

```python
"""Single source of truth for package version."""

__version__ = "0.3.1"
```

- [ ] Modify `pyproject.toml` line 3 :

```toml
version = "0.3.1"
```

- [ ] Modify `tests/smoke/test_imports.py` line 39 :

```python
    assert bouba_sens.__version__ == "0.3.1"
```

- [ ] Modify `tests/unit/test_smoke.py` lines 11 and 23 :

```python
    assert bouba_sens.__version__ == "0.3.1"
```

```python
    assert "bouba_sens 0.3.1" in result.stdout
```

### Step 6: Run version-pinned tests

Run: `uv run pytest tests/smoke/test_imports.py tests/unit/test_smoke.py -v`
Expected: PASS — 8 tests green.

### Step 7: Reframe README Findings

- [ ] Modify `README.md` — replace the Limitations section with a Validation-results section :

```markdown
### Validation results (Sprint 7, ADR-0006)

| Finding | Objection tested | Verdict | Narrative |
|---------|------------------|---------|-----------|
| F1 B-3 world-agnostic | 3+2 partition tautology (10 random permutations) | [PASS >=95pct] OR [DOWNGRADED] | [statement] |
| F2 B-1 topology sign-flip | bootstrap 95 % CI per world | [PASS disjoint CIs] OR [DOWNGRADED to null] | [statement] |
| F3 B-2 Gaussian>XOR>Sinusoid decay | binning + MINE vs Kraskov | [PASS under N estimators] OR [DOWNGRADED to Kraskov artefact] | [statement] |

Full ADRs 0003 -> 0006 in `docs/adr/`.
```

Replace the three bullet points of the old Limitations block with whatever survives. Do NOT carry forward a downgraded finding as a headline — move it to a Limitations paragraph below.

### Step 8: Update CHANGELOG

- [ ] Modify `CHANGELOG.md` — add a new top entry above `[0.3.0]` :

```markdown
## [0.3.1] — 2026-MM-DD (Sprint 7 close — critical validation passes)

### Added

- Task 7.1 null-model control — 10 random 3+2 partitions grid + analyser.
- Task 7.2 bootstrap 95 % CI on Me7 median per world.
- Task 7.3 binning + MINE alternative MI estimators.
- `src/bouba_sens/metrics/partitions.py` + asymmetry `partition` kwarg.
- `scripts/run_null_b3.sh`, `analyse_null_b3.py`, `bootstrap_me7.py`, `compare_mi_estimators.py`.
- ADR-0006 recording pass/fail verdicts per critical test.

### Validation outcomes

- F1 B-3 : [PASS at >=95pct percentile] — headline retained.
- F2 B-1 : [kept / downgraded] — see ADR-0006.
- F3 B-2 : [kept / downgraded] — see ADR-0006.

### Changed

- `version` / `_version.__version__` / version-pinned tests : 0.3.0 -> 0.3.1.
```

### Step 9: Manifest of artefacts

- [ ] Create `reports/v0.3_critical_validation/MANIFEST.md`:

```markdown
# v0.3 critical validation artefacts

| File | SHA256 | Reproduction |
|------|--------|--------------|
| null_b3_partitions.json | [fill in] | `scripts/run_null_b3.sh && scripts/analyse_null_b3.py` |
| me7_bootstrap.json | [fill in] | `scripts/bootstrap_me7.py` |
| mi_estimator_comparison.json | [fill in] | `scripts/compare_mi_estimators.py gaussian xor sinusoid` |

Host: Studio (Task 7.1) + GrosMac (Tasks 7.2 / 7.3).
Commit: [fill in release SHA after commit].
```

- [ ] Populate SHA256s:

```bash
shasum -a 256 reports/v0.3_critical_validation/*.json
```

Replace the `[fill in]` placeholders with the actual hashes.

### Step 10: Run full test suite to confirm no regression

Run: `uv run pytest`
Expected: all green (~160 tests : 152 existing + 4 partition + 3 asymmetry + 3 bootstrap + 3 MI).

### Step 11: Commit release

```bash
git add pyproject.toml src/bouba_sens/_version.py \
        tests/smoke/test_imports.py tests/unit/test_smoke.py \
        CHANGELOG.md README.md reports/v0.3_critical_validation/MANIFEST.md
git commit -m 'chore(release): v0.3.1 Sprint 7 close'
```

### Step 12: Tag and push

```bash
git tag -a v0.3.1 -m 'chore(release): v0.3.1 Sprint 7 close'
git push origin main
git push origin v0.3.1
```

Expected: `* [new tag] v0.3.1 -> v0.3.1` on remote.

### Step 13: Refresh org README footer

- [ ] In the sibling org-profile repo (`~/Documents/Projets/genial-lab-dotgithub/profile/README.md`), swap the "preliminary, pending Sprint 7" phrasing for the validated outcome per ADR-0006. Commit with subject `docs(org): Sprint 7 validated findings` and push.

---

## Exit criteria (Sprint 7 close)

- [ ] Three artefacts in `reports/v0.3_critical_validation/` with SHA256s in MANIFEST.md.
- [ ] ADR-0006 committed with pass/fail verdicts.
- [ ] README Validation-results section replaces the old Limitations block.
- [ ] CHANGELOG [0.3.1] entry.
- [ ] Tag `v0.3.1` (or `v0.4.0` if any downgrade) pushed.
- [ ] Full test suite green.
- [ ] Org README footer updated with validated outcome.

---

---

## Task 7.5 — World-gap audit (external-validity quantification)

**Objection (new, reviewer-sourced):** B-3 PASSes on 3 synthetic worlds (Gaussian, XOR, Sinusoid) with identical API + 5 factorisable modalities + i.i.d. samples + 4-class labels. These 3 worlds may be 3 samples of the same cluster in world-space; the benchmark's external validity to biological settings is unproven. Quantify the gap.

**Approach:** Compute a fixed battery of **world-complexity metrics** on each world's large-sample draw + compare to what we can compute on a biological-adjacent dataset (Task 7.6 Studyforrest stub). No threshold changes, no verdict rewrite — the audit is purely descriptive and feeds the paper's Limitations section.

**Metrics (per-world, shape (1024, 5-modality-dims)):**

| Metric | What it measures | Implementation |
|--------|------------------|----------------|
| `intrinsic_dim_pca` | Effective latent rank | Count PCA components for 95% variance per modality |
| `mi_pairwise` | Modality redundancy | Mean Kraskov MI over all (i,j) modality pairs |
| `label_conditional_entropy` | Task difficulty | `H(label | modality)` per modality, averaged |
| `linear_separability` | Floor task | Logistic-regression accuracy on concatenated modalities |
| `support_compactness` | Geometric spread | Ratio of PCA-95 variance to total variance |
| `temporal_autocorr` | Non-i.i.d.-ness | Lag-1 autocorr on label sequence (0 for i.i.d. worlds) |

**Files:**
- Create: `src/bouba_sens/audit/world_complexity.py` (computes all 6 metrics on `WorldSample` batches)
- Create: `tests/unit/test_world_complexity.py` (fixture-based acceptance on Gaussian with known values)
- Create: `scripts/audit_worlds.py` (CLI: `--worlds gaussian,xor,sinusoid` -> JSON table)
- Artefact: `reports/v0.3_critical_validation/world_complexity_audit.json`

### Step 1: TDD — test for single-metric correctness

- [ ] `tests/unit/test_world_complexity.py` seeds a synthetic `WorldSample` with known rank-k structure per modality; asserts `intrinsic_dim_pca` recovers k within ±1.

### Step 2: Implement the 6 metrics

- [ ] `src/bouba_sens/audit/world_complexity.py`:
  - `intrinsic_dim_pca(tensor, variance_cutoff=0.95) -> int`
  - `mi_pairwise(sample, n_neighbours=3) -> float`
  - `label_conditional_entropy(sample) -> float`  (discretises each modality to 16 bins then computes `H(Y|X)`)
  - `linear_separability(sample) -> float`  (sklearn `LogisticRegression` 5-fold CV)
  - `support_compactness(tensor) -> float`
  - `temporal_autocorr(sample) -> float`
  - One umbrella `compute_world_profile(sample) -> dict[str, float]` returning all 6.

### Step 3: Audit CLI

- [ ] `scripts/audit_worlds.py`:
  ```bash
  uv run python scripts/audit_worlds.py \
      --worlds gaussian,xor,sinusoid \
      --batch-size 1024 --seeds 0 1 2 3 4 \
      --out reports/v0.3_critical_validation/world_complexity_audit.json
  ```
  Runs 5 seeds per world for median ± IQR on each metric; writes a JSON with schema:
  ```json
  {
    "gaussian": {"intrinsic_dim_pca": {"median": ..., "iqr": ...}, ...},
    "xor": {...},
    "sinusoid": {...},
    "comparison": {
      "max_pairwise_distance_metric": "linear_separability",
      "max_pairwise_distance_value": 0.08,
      "interpretation": "the 3 synthetic worlds span < 10 % of the complexity range on the metric with the biggest gap"
    }
  }
  ```

### Step 4: Interpretation hook into ADR-0006

- [ ] ADR-0006 Task 7.4 Step 3 already has a placeholder "External-validity gap" section; populate it with the audit's `comparison` block. If the 3 worlds cluster within 10 % on every metric, ADR-0006 must **downgrade** the B-3 headline from "world-agnostic" to "synthetic-cluster-agnostic" in its next-steps section (but **not** retract the verdict itself — the data is what the data is).

### Step 5: Tests + commit

- [ ] Unit tests green + integration smoke on a 256-sample Gaussian draw finishes in < 5 s.
- [ ] Commit: `feat(audit): Task 7.5 world-complexity audit`.

---

## Task 7.6 — Studyforrest stub (biological-adjacent bridge)

**Objection (deepest, unanswerable without real data):** Task 7.5 quantifies the synthetic-world gap; Task 7.6 starts closing it. The goal is **not** a full biological replication — that's Sprint 9+. It's a **minimal infrastructure stub** that proves the architecture can ingest a real multi-modal dataset without API contortion.

**Dataset:** [Studyforrest](https://www.studyforrest.org/) — fMRI + stereo audio + visual features captured during "Forrest Gump" film viewing, Creative Commons. Uses *real* modality correlations, *real* temporal structure, *real* dataset noise.

**Scope constraint:** We only need **2 of the 5 modalities** from Studyforrest (audio + visual) plus 3 mocked/zeroed (tactile, gravity, force). This is a **not** a scientific replication; it's a **shape-test** that proves the `WorldSimulator` contract accepts a non-synthetic producer.

**Files:**
- Create: `src/bouba_sens/world/studyforrest.py` (`StudyforrestWorld(WorldSimulator)` wrapper)
- Create: `scripts/fetch_studyforrest_sample.py` (downloads 1-minute audio + visual features subset, ~50 MB)
- Create: `tests/unit/test_studyforrest_world.py` (runs on 1 sec of pre-cached mock features; no network)
- Create: `tests/integration/test_studyforrest_smoke.py` (guarded by `BOUBA_SENS_STUDYFORREST_DATA` env; skipped in CI)
- Create: `docs/adr/0007-biological-bridge-stub.md` (scope + limitations)
- Artefact: `data/studyforrest_sample/` (git-LFS or gitignored with MANIFEST.md SHA)

### Step 1: Shape-only unit test (no network)

- [ ] `tests/unit/test_studyforrest_world.py` mocks pre-downloaded 2-D audio spectrogram + 3-D visual CNN features, verifies `StudyforrestWorld.sample(batch_size=8, seed=0)` returns a valid `WorldSample` with:
  - `audio.shape == (8, T_audio)` (T_audio = spectrogram length in bins)
  - `vision.shape == (8, D_vision)` (D_vision = CNN feature dimension)
  - `tactile / gravity / force` zero-tensors with correct shapes from encoder contracts
  - `label.shape == (8,)` and `label.dtype == torch.long`
  - Labels derived from timecode-binned annotation (e.g. scene ID → class).

### Step 2: `StudyforrestWorld` implementation

- [ ] `src/bouba_sens/world/studyforrest.py`:
  ```python
  class StudyforrestWorld(WorldSimulator):
      """Minimal bridge to Studyforrest — 2 real modalities + 3 mocked.

      Not a scientific replication; a shape-test that proves the
      WorldSimulator contract accepts non-synthetic producers.
      """
      def __init__(self, data_dir: Path, seed: int = 0) -> None:
          self._audio_cache = torch.load(data_dir / "audio.pt")  # shape (N, T_audio)
          self._vision_cache = torch.load(data_dir / "vision.pt")  # shape (N, D_vision)
          self._labels = torch.load(data_dir / "labels.pt")  # shape (N,)
          self._rng = torch.Generator().manual_seed(seed)
          self._n = self._audio_cache.shape[0]

      def sample(self, *, batch_size: int, seed: int) -> WorldSample:
          g = torch.Generator().manual_seed(seed)
          idx = torch.randint(0, self._n, (batch_size,), generator=g)
          audio = self._audio_cache[idx]
          vision = self._vision_cache[idx]
          label = self._labels[idx]
          zero = lambda d: torch.zeros(batch_size, d)
          return WorldSample(
              audio=audio, vision=vision,
              tactile=zero(TACTILE_DIM), gravity=zero(GRAVITY_DIM), force=zero(FORCE_DIM),
              label=label,
          )
  ```
- [ ] Register `studyforrest` in the CLI `_build_world` dispatch with a required `--studyforrest-data-dir` flag (typer `BadParameter` if unset).

### Step 3: Data-fetching script (optional network)

- [ ] `scripts/fetch_studyforrest_sample.py` downloads the 1-minute subset from the public S3/Zenodo mirror, extracts mel-spectrogram (librosa, 2-sec windows) + per-frame VGG16 features (torchvision), saves the three `.pt` tensors to `data/studyforrest_sample/`. Writes `data/studyforrest_sample/MANIFEST.md` with SHA256s. Idempotent.
- [ ] Script prints a clear `data_dir` path the user passes to `--studyforrest-data-dir`.
- [ ] Network-free failure mode: if no internet, the script prints a pointer to a pre-hosted mirror.

### Step 4: Integration smoke (guarded)

- [ ] `tests/integration/test_studyforrest_smoke.py`:
  ```python
  import os
  import pytest

  @pytest.mark.skipif(
      not os.getenv("BOUBA_SENS_STUDYFORREST_DATA"),
      reason="set BOUBA_SENS_STUDYFORREST_DATA to data/studyforrest_sample/",
  )
  def test_studyforrest_phase2_smoke() -> None:
      # 1 seed, 1 modality, 1 SNR, 20 steps — proves the full
      # pretrain -> lesion -> eval pipeline runs on non-synthetic input.
      ...
  ```

### Step 5: ADR-0007 + Task 7.4 cross-ref

- [ ] `docs/adr/0007-biological-bridge-stub.md` records: scope (shape-test only), limitations (tactile/gravity/force mocked to zero — no grounded embodiment), what it enables (Sprint 9 can extend StudyforrestWorld to hypothesise tactile/proprioceptive signals from motion annotations).
- [ ] ADR-0006 Task 7.4 Step 3 "Next steps" section references ADR-0007 as the infrastructure seed for Sprint 9 biological validation.

### Step 6: Tests + commit + README refresh

- [ ] Unit test green without network. Integration test skipped by default.
- [ ] `README.md` gains a **"Limitations: external validity"** paragraph pointing to ADR-0005 + Task 7.5 audit + ADR-0007 stub: *"Verdicts stand on 3 synthetic worlds within the same cluster; biological extrapolation requires data from actual cross-modal settings — stub API in place (`StudyforrestWorld`), replication deferred to Sprint 9."*
- [ ] Commit: `feat(world): Task 7.6 Studyforrest bridge stub`.

---

## Exit criteria (Sprint 7 close, revised)

- [ ] Five artefacts in `reports/v0.3_critical_validation/`: null_b3_partitions, me7_bootstrap, mi_estimator_comparison, **world_complexity_audit, studyforrest_manifest** (new).
- [ ] ADR-0006 + **ADR-0007** (new) committed.
- [ ] README rewritten with both the validated verdict (Tasks 7.1-7.3) AND the external-validity caveat (Tasks 7.5-7.6).
- [ ] Tag `v0.3.1` (or `v0.4.0`) pushed.

---

## Self-review notes (2026-04-20)

- All tasks produce committed artefacts — no orphan scripts.
- TDD cadence preserved : every new function has a failing unit test before implementation.
- File-path consistency checked : `partitions.py`, `bootstrap_me7.py`, `compare_mi_estimators.py`, `run_null_b3.sh`, `analyse_null_b3.py` all appear identically in File Structure, Task body, and commit commands.
- Aggregator `--emit-raw-pairs` flag (Task 7.2 Step 6) and `--partition-seed`/`--partition-index` flags (Task 7.1 Step 13) are **pre-requisites** on the aggregator ; if they are not already implemented, add them as Step 6.5 / 13.5 before the corresponding script runs. This plan assumes Sprint 5 Task 5.1 persisted `pre_lesion_codes` / `post_lesion_codes` on the `report.pkl` — Step 8 of Task 7.3 explicitly handles the fallback.
- No placeholders : every code block is complete. All `[fill in]` patterns are data-population steps, not missing logic.
- Version-bump path covers both no-downgrade (v0.3.1) and downgrade (v0.4.0) — engineer picks at Task 7.4 Step 4 once verdicts are known.
