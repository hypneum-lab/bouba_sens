# ADR-0023 — B-3 probe v2: linear read-out of frozen per-modality representations

**Status:** Proposed (2026-10-25; results added 2026-10-26). Awaiting PI decision. This ADR modifies
no past metric, verdict, ADR or locked threshold; the v1 probe
(`query_accuracy` / `query_successes`) is kept bit for bit.
**Date:** 2026-10-25
**Sprint:** post-v0.5.10 audit (follows ADR-0022)
**Preregistration:** `docs/prereg/2026-10-25-b3-probe-v2.md`, tag
`prereg-b3-probe-v2-20261025`
**Artefacts:** `src/bouba_sens/linear_probe.py`,
`src/bouba_sens/loop.py` (`modality_features`, `probe_v2`,
`query_successes_v2`, `query_accuracy_v2`), `src/bouba_sens/cli.py`
(`--probe-v2`, `--probe-v2-seed`, `per_query_successes_v2.json`),
`scripts/me6_paired_null.py` (`--successes-file`),
`scripts/b3_probe_v2.py`, `scripts/run_grid_probe_v2.sh`,
`tests/unit/test_probe_v2.py`, `tests/unit/test_b3_probe_v2_script.py`,
`reports/b3_probe_v2_options_exploratory.json`

## Context

ADR-0022 (PI verdict, 2026-10-25) found that the B-3 probe does not
measure anything decodable: `AdaptationLoop.query_accuracy` zeroes the
four non-query modalities and reads the network's own head, and the mean
single-modality query accuracy is chance (0.254 Gaussian, 0.240 mock).
Zero inputs are outside the training distribution of every pathway, so
the read-out is not a measure of what one modality carries. The PI made
a probe whose per-modality accuracy is above chance a precondition of any
new B-3 reading, and asked (decision of 2026-10-25, 18:44) for a
corrected probe with a mandatory positive control. On the mock world the
probe items are contiguous windows (lag-1 autocorrelation of the paired
outcomes 0.36, ≈ 0 at lag 8; entry s.d. 0.124 vs 0.053 binomial), so the
per-item null is invalid there and the block-flip p-value depended on a
block length chosen after seeing the result (8 → 1e-4, 16 → 0.06,
32 → 0.31).

### What the network actually transmits (verified, exploratory)

`SensoryWML.step` computes a hidden state `h = core(input_proj(x))`
(128-d), takes the **argmax** of `emit_head_pi(h)` over the 64-code
alphabet, replicates that single code on the K gamma slots and modulates
it; `CrossModalNerve.fuse` demodulates and projects it through the
codebook. On the intact phase-1 networks of the PR #40 grids (5 seeds per
world; `reports/b3_probe_v2_options_exploratory.json`; 256 held-out
items, 512 training items, probe seed 12345 + seed; these grids were
already seen, so all of this is exploratory):

| mean over 5 intact nets | audio | vision | tactile | gravity | force |
|---|---:|---:|---:|---:|---:|
| Gaussian — distinct argmax codes on 256 items | 1.0 | 1.0 | 1.4 | 2.4 | 1.0 |
| Gaussian — v1 (zero masking), network head | 0.234 | 0.234 | 0.274 | 0.237 | 0.227 |
| Gaussian — (b) marginal replacement, network head | 0.245 | 0.233 | 0.264 | 0.252 | 0.238 |
| Gaussian — linear probe on the code embedding | 0.255 | 0.255 | 0.290 | 0.283 | 0.255 |
| Gaussian — **(a) linear probe on hidden state h** | **0.684** | **0.702** | **0.600** | 0.323 | 0.382 |
| Gaussian — majority-class baseline | 0.255 | 0.255 | 0.255 | 0.255 | 0.255 |
| mock — (b) marginal replacement, network head | 0.280 | 0.265 | 0.263 | 0.259 | 0.263 |
| mock — **(a) linear probe on hidden state h** | **0.533** | **0.511** | 0.181 | 0.181 | 0.181 |
| mock — majority-class baseline | 0.181 | 0.181 | 0.181 | 0.181 | 0.181 |

The intact network with **all five modalities live** scores 0.270
(Gaussian) and 0.274 (mock) on the same items: after the grid's 200 + 100
training steps the code channel is collapsed (one code for almost every
input), so nothing about the label reaches the head, whatever the probe.

## Options evaluated

**(b) Replace the masked modalities by draws from their training
marginal** (an independent world sample instead of zeros). This removes
the out-of-distribution input, but it still reads the network's head,
which is at chance even with every modality live. Its accuracy is chance
on both worlds (table), so it **fails the positive control by
construction** on these networks. Rejected.

**(a) Linear probe on frozen per-modality representations.** For each
query modality q, the frozen hidden state `h_q = core_q(input_proj_q(x_q))`
of q's own pathway, on clean in-distribution inputs of q only — no other
modality is computed, so no input is replaced and nothing leaves the
training distribution. A multinomial logistic regression is fit on a
training pool and scored item by item on held-out items. It is clearly
above the majority baseline on the full-rank Gaussian modalities and on
the two live mock modalities, at the baseline where the input is
constant (mock tactile / gravity / force), and near chance on a world
with labels independent of the inputs (unit test). **Chosen.**

A probe on the code embedding (what the nerve receives) was also tried:
it sits at the baseline because the code channel is collapsed. It is the
level that bears on cross-modal *transmission*; it cannot be used while
the code channel carries no information.

## Decision (proposed)

### Probe v2 (`AdaptationLoop.probe_v2`)

- Representation: `modality_features(q, x_q)` = the 128-d pre-argmax
  hidden state of q's sensory pathway, `no_grad`, deterministic (no
  dropout, no sampling, no global RNG).
- Items: held-out test = `world.sample(256, seed=s)`; training pool =
  `world.sample(512, seed=s + 1)` minus any item whose raw inputs equal a
  test item (contiguous-window worlds can overlap). Item i is the same
  world item for the five queries.
- Read-out (`linear_probe.fit_linear_probe`, scikit-learn, already a
  dependency): standardisation + multinomial logistic regression (lbfgs,
  2000 iterations); L2 strength C ∈ {0.01, 0.1, 1, 10} chosen by 5-fold
  shuffled cross-validation **on the training pool only** (fold seed 0,
  ties → smallest C); refit on the whole pool; per-item 0/1 on the test
  items. The training-pool majority class scored on the test items is
  recorded as the probe's own chance level (`baseline`).
- `query_successes_v2` / `query_accuracy_v2` expose it per modality; the
  v1 methods are untouched (`test_probe_v2_does_not_touch_v1_path_rng_or_state`,
  `test_cli_v1_outputs_bit_identical_with_and_without_probe_v2`).

### Files per cell (CLI, on by default, `--no-probe-v2` to skip)

- `bouba-sens lesion` writes, after the v1 files and from the same frozen
  state, `per_query_successes_v2.json` (`{modality: [0/1] * 256}`),
  `per_query_me1_v2.json` and `probe_v2_meta.json` (probe seed, CV
  accuracy, chosen C, baseline, pool size, items dropped, distinct
  argmax codes). `--probe-v2-seed` sets s (default lesion seed + 777).
- `bouba-sens train` writes the same three files for the intact network
  (positive control) and `init_*` versions for the untrained initial
  state (descriptive reference).
- `scripts/run_grid_probe_v2.sh` gives the five cells of a (seed,
  timing, SNR) trio **one shared probe seed** (`seed * 10000 + 5000 +
  trio`), so P[q, m] and P[m, q] are scored on the same world items —
  ADR-0022 limit 3 (pairing by index across two different probes) no
  longer applies.

### Reading of B-3 v2 (`scripts/b3_probe_v2.py`, preregistered)

Positive control on vision of the intact networks (mean margin over the
baseline ≥ 0.10); statistic, H0, paired sign-flip null with a block
length fixed on a separate pilot grid, and decision thresholds as in the
preregistration. The confirmatory grids use new seeds (200–204); the
PR #40 grids stay exploratory.

## What B-3 v2 measures, and what it does not

- It measures whether the **linear decodability of the label from one
  modality's own representation** differs between "q queried, m
  lesioned" and "m queried, q lesioned". A lesion of m acts on q's
  representation only through shared parameters and training dynamics
  (shared multiplexer, joint optimiser, replay), so asymmetries can
  arise; they are not cross-modal transfer through the nerve.
- It does **not** measure what the nerve or the head can use: the code
  channel is collapsed in these networks (≈ 1 code per modality), which
  is itself a finding about the benchmark's training regime, recorded
  here and left to the PI.
- An untrained pathway is a random feature map of its input and is
  already linearly decodable (the `init_*` files report this); the
  positive control therefore shows that the probe can measure, not that
  training produced the information.

## Known limits

1. Linear read-out only; a non-linear probe could find more. Fixed a
   priori to keep the probe simple and auditable.
2. 512 training items per probe; mock training pools are one contiguous
   window, so the probe generalises across windows of an autocorrelated
   stream (harder, lower accuracy) — symmetric across pairs.
3. The block-length rule uses the lag autocorrelation within a probe;
   window-level variation of the label mix (ADR-0022) is only partly
   captured. With 256 items, block 16 leaves 16 independent signs per
   pair (4 with n = 64 in ADR-0022).
4. Real-ECG world: data cache absent (ADR-0022); not run.

## Results (confirmatory, preregistered)

Tag `prereg-b3-probe-v2-20261025` created 2026-10-25 19:14:09 +01:00
(commit `1d96a65`); pilot (seeds 100–101) 19:14–20:32, block lengths
committed at 20:32:53 (`fd15deb`); confirmatory grids started 20:32:58
— **after** the tag and the pilot record. Gaussian 20:32–21:26, mock
21:26–22:02, electron-server, 300/300 cells, 30/30 trios per world,
0 failure. Files: `reports/b3_probe_v2_{gaussian,mock}.json`; note
`docs/milestones/2026-10-26-b3-probe-v2-results.md`.

| world | control: vision acc / baseline / margin (≥ 0.10) | block | observed median | null mean | null P99 | p | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| Gaussian | 0.729 / 0.244 / **0.485** — pass | 1 | **0.4141** | 0.0794 | 0.0918 | **5·10⁻⁵** (0 / 20 000) | **asymmetry present** |
| mock | 0.529 / 0.247 / **0.282** — pass | 8 | **0.3320** | 0.1363 | 0.1602 | **5·10⁻⁵** (0 / 20 000) | **asymmetry present** |

Per-criterion: (1) positive control reached on both worlds (per-seed
vision margins 0.41–0.53 Gaussian, 0.21–0.35 mock); (2) grids complete;
(3) p < 0.01 on both worlds → « asymétrie présente » by the
preregistered rule.

### What carries the asymmetry (exploratory, not preregistered)

`reports/b3_probe_v2_decomposition_exploratory.json`:

- Mean accuracy **by query modality** (rows): Gaussian audio 0.691,
  vision 0.682, tactile 0.571, gravity 0.286, force 0.331; mock audio /
  vision 0.549, tactile / gravity / force 0.228 (constant zero inputs).
  **By lesioned modality** (columns): flat — 0.507–0.520 Gaussian,
  0.355–0.358 mock.
- The pair attaining the maximum always pairs a high-dimensional (or
  live) modality with gravity / force (Gaussian) or with a zeroed
  modality (mock).
- After removing each trio's additive query effect from P − Pᵀ, the
  median residual maximum is at its sign-flip null (Gaussian 0.057 vs
  null 0.058, p ≈ 0.55; mock 0.011 vs 0.089).

So the preregistered rejection of H0 is **entirely a query-modality
main effect**: gravity (3-d) and force (6-d) projections of the 32-d
latent carry less label information than the full-rank audio / vision /
tactile inputs, and the mock world zeroes three modalities. Since
P[q, m] − P[m, q] contains r_q − r_m, any difference in informativeness
between modalities violates the H0 written in the preregistration
(ADR-0022 limit 4 was wrong on this point). **No lesion-specific
asymmetry is visible**: lesioning a modality does not change what is
decodable from the others. The untrained reference reaches most of the
same margins (Gaussian vision 0.283 untrained vs 0.485 intact), and the
code channel is collapsed to a single code per modality on the intact
networks of all five confirmatory seeds.

Exploratory v1 statistic on the same new grids: Gaussian 0.156 (null
mean 0.140, p = 0.13); mock 0.266 at block 8 (null mean 0.238,
p = 0.058).

### Proposed reading (for the PI)

B-3 v2 is « asymétrie présente » by the preregistered criterion on both
worlds, but the asymmetry is the trivial one of unequal modality
informativeness built into the worlds, not an effect of the lesions.
Neither this nor its absence bears on the perceptive / proprioceptive
claim of B-3 as stated in the spec; a test of a lesion effect would need
an H0 that conditions on the query main effect (e.g. the residual of a
two-way decomposition, or P[q, m] compared with an unlesioned reference
for the same q), to be preregistered separately.

## References

ADR-0008, ADR-0009, ADR-0014-b3, ADR-0021, ADR-0022; issue #22; PI
decision of 2026-10-25 18:44.
