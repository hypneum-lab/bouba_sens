# Me6 large-n — exploratory diagnostic amendment (2026-10-31)

**Status:** exploratory. Written and pushed **before** any diagnostic
below is run. It decides nothing, changes no preregistered reading of
`docs/milestones/2026-10-31-me6-large-n-results.md`, no metric, null,
threshold or block of ADR-0026, and does not reopen B-3.
**Builds on:** ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026,
ADR-0027; tag `prereg-me6-large-n-20261031` (`b790db2`).
**Data:** electron-server
`~/scratch/bouba_sens-runs/me6-large-n-20261031/{cell,legacy}/gaussian/`,
30 trios per arm (seeds 400–404 × T1/T2 × floor/minus10/plus10),
`per_query_successes_n{64,256,1024,4096}.json` per lesion cell.

## The question

ADR-0026 expected the Gaussian rows to be compatible with the paired
null at every n; six of eight are in excess at n ≥ 256 (e.g. `cell`,
n = 4096: observed median 0.0585, null median 0.0164, p = 5·10⁻⁵).
Candidate readings:

- **A** — a dependence the item-level sign flip does not cover (item
  or cross-pair dependence): the null is too narrow.
- **B** — a real, systematic asymmetry between masking a and masking b
  (modality-specific, e.g. through the fixed encoders or input scales),
  reproducible across trios, which a larger n resolves.
- **C** (added here) — a real but idiosyncratic asymmetry: P[a, b] and
  P[b, a] are read on **two different networks** (lesion cells b and a,
  different lesion seeds), and any two networks differ by a fixed
  amount at infinite n. The item flip tests "these two networks have the
  same per-item distribution"; it cannot see the network-to-network
  variance that ADR-0026's "no structure by construction" referred to.
  C is real for the fixed networks (B-like) and a too-fine
  exchangeability unit for the population H0 (A-like).

## Seen before writing (declared)

The results note and its reports; the code; and, to learn the file
layout, the 5 × 5 accuracy matrix at n = 4096 of four trios
(`seed400_T2_floor`, `seed401_T1_plus10`, both arms). In them most
cells predict a near-constant class (the audio and vision query
vectors of a cell agree on 81–100 % of the items), with isolated
entries far from 1/4 (0.425, 0.334, 0.301, 0.192, 0.055). No
diagnostic statistic below was computed.

## Code facts relevant to A / B / C

1. Gaussian probe items are iid rows (`GaussianWorld.sample`); the probe
   forward pass is per item (`SensoryWML.step`, `CrossModalNerve.fuse`,
   no batch statistic), with per-item demodulation noise. Given the
   five networks of a trio, item-level outcomes are independent:
   item-level A is unlikely by construction.
2. The five cells of a trio share the probe items and labels (shared
   probe seed, ADR-0023), so the 10 pair differences of an item share
   entries; `paired_null` flips them independently (ADR-0022 limit).
3. Each column m is a separate network (lesion seed in `config.yaml`;
   T1 head init from that seed, T2 from the shared phase-1 checkpoint),
   so P[a, b] − P[b, a] contains a network difference → C or B.
4. `input_proj` gets only the 1e-3 bridge gradient and encoder seeds are
   fixed (1–5) in every cell; in `legacy` all probes read world seed 0.
   A modality-specific code map shared by all cells could give B,
   most plausibly in `legacy`.
5. Me6 = max |·| is invariant to sign flips of whole trios, so a
   trio- or seed-level sign-flip null **of Me6** is degenerate; coarser
   units are tested on signed differences instead (D3).

## Diagnostics on the saved outcomes (no new training)

Common to D1–D3: Gaussian world, both arms, 30 complete trios,
`discover_trios` / `pair_differences` of `scripts/me6_paired_null.py`;
d(k, p, i) ∈ {−1, 0, 1}, Δ(k, p) = mean_i d; resampling seed
**31102026**, B = 20 000, p = (#{null ≥ obs} + 1) / (B + 1). Implemented
in `scripts/me6_large_n_diagnostic.py`, committed and pushed before its
first run; electron-server, nice 19, one process.

**D1 — Is the asymmetry a fixed property of the cells? (A vs B/C)**
Files `_n4096`. Split items into even and odd indices (2 048 each, iid
halves). Statistic r = Pearson correlation of Δ_even(k, p) and
Δ_odd(k, p) over the 300 (trio, pair) points; also restricted to each
trio's argmax pair. Null: preregistered item flip (block 1) applied to
the full vector, r recomputed. Second statistic, per n ∈ {256, 1024,
4096}: overdispersion φ_n = mean over (k, p) with Σ|d| > 0 of
z² = (Σ_i d)² / Σ_i d², which has expectation 1 under the item flip;
null P99 by the same flip. Reported per arm, pair, timing, SNR.
*Points to A:* r compatible with its null, or φ_n inflated by a roughly
constant factor across n (a design effect). *Points to B/C:* r ≫ null
and φ_n − 1 growing ≈ ∝ n (× 4 per size step, fixed effect against a
null shrinking as 1/√n). Cost ≈ 5 min per arm.

**D2 — Cross-pair dependence: joint transpose flip (A)**
Files `_n{64,256,1024,4096}`. Null: one sign per (trio, item) shared by
the 10 pairs, i.e. the item's 5 × 5 outcome matrix is replaced by its
transpose with probability 1/2 (keeps the cross-pair structure of
code fact 2). Same statistic as ADR-0026 (median over trios of Me6),
block 1. *Points to A:* p moves from < 0.01 to ≥ 0.05 at n ≥ 256.
*Points away from A:* p stays < 0.01. Cost ≈ 4 min per (arm, n) under
load at n = 4096, ≈ 30 min total.

**D3 — Systematic or idiosyncratic? (B vs C)**
Files `_n4096`. Per pair p: S_p = |mean_k Δ(k, p)| over the 30 trios,
S = max_p S_p. Null: sign of Δ(k, p) flipped independently per (trio,
pair) — the trio is the exchangeable unit. Also within seed (6 trios,
exact enumeration of the 64 sign patterns per pair) for the `cell` arm,
whose world changes with the seed. Report the sign table of the argmax
pairs. *Points to B:* S p < 0.01, same sign in ≥ 25 / 30 trios for the
leading pair, stronger in `legacy` (code fact 4). *Points to C:* D1
shows a fixed effect but S compatible with its null (p ≥ 0.05), signs
balanced. Cost < 1 min.

## Diagnostic needing a new run — separate PI go required

**D4 — Fresh items and pre-lesion states.** A small script
(`scripts/me6_large_n_diagnostic.py reprobe`, committed before any run) loads saved
states and runs `query_successes` (v1, zero masking) at n = 4096, inside
`fork_rng` reseeded with the probe seed, writing
`per_query_successes_reprobe_n4096.json` to a new output root
`~/scratch/bouba_sens-runs/me6-large-n-20261031/diagnostic/reprobe/`
(no saved cell is touched). Probe seed **5 106 000 + 10·(s − 400) + trio** (s ∈ 400–404,
trio 0–5), outside every used namespace (run seeds 0–4, 100–101,
110–111, 200–204, 300–309, 400–404, 999 and their derived probe seeds).
(a) the 300 lesion states (`model.pt`, both arms): D1–D3 rerun on fresh
items — out-of-sample replication of the per-entry asymmetry.
(b) pre-lesion states: per trio, T2 = the phase-1 checkpoint of its
seed probed five times (forward RNG reseeded with probe seed + column
index, so the 5 "columns" are one network); T1 = the five untrained
initial networks built with each cell's recorded lesion seed and zero
lesion steps. Me6 and the ADR-0026 item-flip null on both.
*Reading:* excess already in (b)-T2 → a query main effect of one
network (B); excess in (b)-T1 but not (b)-T2 → network-to-network
differences (C); no excess in (b) → produced by the lesion phase.
Cost (electron-server, PAR = 4, nice 19, from the `eval` timings of
ADR-0026, 15–33 s per job): (a) 300 jobs ≈ 40 min; (b) 2 arms × (5
checkpoints + 75 initial states) ≈ 25 min; analysis ≈ 30 min; ≈ 1.5 h
under load. No KXKM, no GPU.

## Corrections made with the code (2026-10-31, before any run)

PI go for D1–D4 given 2026-10-31 ≈ 14:00 (D4 after D1–D3 are saved).
Written into this file in the commit that adds
`scripts/me6_large_n_diagnostic.py`, before its first run:

1. **Mock rows added, descriptive only.** D1–D3 are also run on the
   mock grids with the committed ADR-0026 blocks (`cell` 8/16/16/16,
   `legacy` 4/8/8/8 at n = 64/256/1024/4096): every flip acts on block
   sums; φ uses block sums (expectation 1 under the block flip); D1
   halves are the first and second half of the blocks (adjacent
   windows, not iid). The Gaussian rows remain the ones this amendment
   is about.
2. **D1 argmax r:** the argmax pair is chosen on the full vector, in the
   observed data and again in every null replicate (same selection).
   φ is also reported at n = 64.
3. **D3 within seed** is reported for both arms (descriptive).
4. **D4 details:** column index = position of the lesioned modality in
   (audio, vision, tactile, gravity, force); the (b)-T2 probe of a cell
   reseeds the forward RNG with probe seed + that index, (a) and
   (b)-T1 with the probe seed. The rebuild is checked first
   (`validate`): reprobing four saved states with their original probe
   seed must reproduce `per_query_successes_n4096.json`, and one T1 cell
   rebuilt and re-lesioned must reproduce its `model.pt`. D4 analysis =
   Me6 median with the item-flip null of ADR-0026 (block 1) but seed
   31102026, B = 20 000, plus D1 and D3; (b)-T1 and (b)-T2 have 15
   trios each per arm, (a) 30.
5. RNG streams: `default_rng([31102026, tag, n, start])`, chunks of
   100 replicates, tag 1 = D1 split, 2 = D2, 3 = D3, 4 = φ.

## What a later confirmatory test would need

A new preregistration (new tag) stating: which H0 is meant (conditional
on the networks, or over network draws); the statistic and the null
with its exchangeable unit (e.g. a hierarchical null resampling trios,
or D2's transpose flip), with its size checked on a synthetic grid of
the real shape (30 × 5 × 5 × n) under that H0 before the tag; the mock
block rule under the new null; fresh seeds outside every used
namespace, including 5 106 000–5 106 045; predictions per arm and n;
and the declaration that the 400–404 grids and every D1–D4 output are
seen data, usable only to motivate it.

## References

ADR-0021, ADR-0022, ADR-0023, ADR-0024, ADR-0026, ADR-0027;
`docs/milestones/2026-10-31-me6-large-n-results.md`;
`scripts/me6_paired_null.py`, `scripts/me6_large_n.py`,
`src/bouba_sens/loop.py` (`query_successes`), `src/bouba_sens/cli.py`
(`_write_probe_sizes`).
