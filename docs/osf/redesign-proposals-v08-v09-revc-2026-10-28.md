# Redesign proposals, rev. C — v0.8 measurement and v0.9 label / world (2026-10-28)

**Status:** DECIDED by the PI on 2026-10-28 (about 19:05; section D.4),
with v0.9's primary and secondary inverted; implemented, then piloted on
macM1 (results in the pilot report). The proposal text below is kept as
reviewed. Earlier decision (about 18:05): "pursue both options as a
proposal, then STOP for my review before any new pilot".
The drafts `amendment-v0.8-*` and `amendment-v0.9-*` only point to this
file. It follows `redesign-proposals-v08-v09-2026-10-25.md` (sections
A–C: rev. B and its pilots). It was revised after the Codex review r7
(2026-10-28).

The evidence comes from existing data only:

- the rev. B pilots (`reports/pilots/v08_revb_pilot_*.json`,
  `reports/pilots/v09_revb_pilot_main_grid.json`);
- the collapse diagnosis merged from main (ADR-0024,
  `docs/milestones/2026-10-26-code-collapse-diagnosis.md`,
  `reports/diag_code_collapse_*.json`, `reports/emission_pilot_*.json`);
- a **raw-feature feasibility check** made for this file
  (`scripts/pilots/v09_revc_label_feasibility.py` →
  `reports/pilots/v09_revc_label_feasibility.json`).
  - It trains no network and uses no seed. It fits a logistic probe on
    the raw modality features of each world, over 10 contiguous time
    folds, with 16 or 64 frames purged on each side of the test fold,
    and reports balanced accuracy.
  - It was run on macM1 (79 s) at `63ef538`.
  - Raw-feature scores bound what a linear readout of the inputs can
    reach. They do not predict what the trained network will reach.

## D.0 Prerequisite common to both campaigns (PI decision needed)

ADR-0024 (from main, merged here in `63ef538`) found that with the
default argmax emission **the network does not learn the task**:

- the argmax blocks the task gradient;
- Adam turns the 1e-6 bridge into drift;
- each modality collapses onto a few codes;
- the logit margin grows to 17–48.

This is the mechanism behind both rev. B failures: the v0.8 soft code
is constant (C.1) and the v0.9 fused code h_all is near-constant (C.2).
In the default regime, any new measurement of any representation would
measure optimiser drift, not adaptation in a learning agent. Before the
pilots below, the PI chooses one of two options:

- **(P1, recommended)** run v0.8 and v0.9 rev. C with the opt-in
  straight-through emission (`--emission straight-through`, main
  `4e1dffc`). Add a **learning gate** to each pilot. The gate follows
  Codex r7: the emission-pilot criterion (Phase-1 loss < 1.3 and a
  vision-probe margin) is **not** used. A predictor that ignores the
  input and outputs the native class proportions already reaches a
  cross-entropy of 1.051, and an upstream vision probe can stay readable
  while the emitted pathway fails. The gate is label-specific and
  measured on the task:
  - **Measure:** the held-out balanced accuracy of the **full network**
    (all modalities live, head on the fused code). It is computed on
    the held-out rows of the campaign's own label: the native scene
    label for v0.8, y for v0.9.
  - **Baseline:** any input-independent predictor scores 0.25 balanced
    accuracy.
  - **Pass:** the mean over pilot seeds is ≥ 0.40, and every seed is
    ≥ 0.30. For v0.9 the gate is checked on the plain-arm networks of
    every onset; for v0.8 on the no-lesion control of every lock.
  - **Additional v0.9 pass condition:** the full-network accuracy
    exceeds the best single-modality network query accuracy
    (`query_accuracy`) by ≥ 0.05, mean over seeds.
  - **Diagnostics only:** upstream probes (encoder h_m) are reported,
    but they do not count toward the gate.
  - **Failure:** stop and report.
- **(P2)** keep the default emission. Every result is then reported as
  *measured in a non-learning regime* (ADR-0024 wording) and is
  descriptive only.

Under P1, the rev. B fused-code readouts may recover structure. The
rev. B gates are therefore re-reported under P1 as secondary
information.

## D.1 v0.8 — what to measure

The v0.8 question is unchanged: does the representational change caused
by the lesion depend on LOCK_AFTER (Page trend over the 7 locks)? Rev. B
measured it on the fused code, downstream of the argmax. Rev. C moves
the measurement upstream of the quantisation. `SensoryWML.step`
computes `h = core(input_proj(x))`, then `pi_logits = emit_head_pi(h)`,
then the argmax.

### Option (a) — encoder representations, upstream of demodulation

- **What:** for each modality m, h_m = `core(input_proj(x_m))` (128
  dims) on the probe rows.
  - Z is the concatenation of the 5 modality blocks. Each block is
    divided by the square root of its own centred total variance, which
    must be finite and positive (D.1, gate 2).
  - d(c) = 1 − CKA_lin(Z_post(c), Z_control), as in rev. B.
  - The per-block median over the 15 cells, the Page test, the
    safeguard and protocol failure are unchanged.
  - Descriptive: the same d per modality, lesioned vs intact modalities.
- **Evidence that it is not degenerate:**
  - In the v0.9 rev. B pilot, the same h_m on the same real bridge carry
    the label. Held-out balanced accuracy, mean over 75 runs: vision
    0.986, tactile 0.943, gravity 0.765, force 0.298, audio 0.279.
  - Vision was chosen in 69 / 75 runs.
  - These matrices therefore have input-driven, between-class
    structure, unlike the soft code (total variance 8.6e-15).
  - Caveat (ADR-0024): in the default regime they move only by drift,
    which is why P1 matters.

### Option (b) — sender logits, before the softmax / argmax

- **What:** `pi_logits` per modality: 64 alphabet logits each, 320
  concatenated, with per-modality scaling as in (a). Same d, Y, Page
  test and safeguard. The emitted code is the argmax of these logits,
  so this measures the code before quantisation.
- **Evidence:** none in the rev. B data, which stored only the codes.
  The collapse diagnosis gives a reason for caution:
  - in the default regime, the margin between the winning logit and the
    next grows to 17.5 (Gaussian) and 48.1 (mock), while active codes
    fall to 1–4;
  - the centred variance of the logits is then likely concentrated in
    the margin direction of the few winning codes, and a
    variance-weighted CKA would track the margin drift.

### Primary vs secondary — recommendation

**(a) primary, (b) secondary.** Reasons:

- (a) is the only option with positive evidence of an input-driven,
  non-degenerate structure on the real bridge (v0.9 rev. B).
- (a) is the place the Phase-2 lesion training changes directly.
- (a) does not inherit the argmax/margin pathology.
- (b) is closer to the emitted code and is kept as the registered
  secondary.

If the pilot shows that (b) is well conditioned (gates below) and P1 is
chosen, the PI may swap the two before filing. No swap is allowed after
the confirmatory data.

### Gates of the rev. C pilot (before the PI accepts the design)

1. **Learning gate** (P1 only): see D.0.
2. **Non-degeneracy, checked before any scaling.**
   - Each **unnormalised** modality block of every Z_pre, Z_control and
     Z_post must be finite. Its centred total variance must be > 0 and
     ≥ 10⁻⁶ × its mean squared row norm, with that norm > 0 (an
     all-zero block fails).
   - A cell is valid only if all 5 blocks of all of its matrices pass.
   - A block is defined with ≥ 12 valid cells.
   - The design passes if ≥ 8 of 10 seeds have every lock defined.
     Failures are reported per modality.
3. **Input-driven structure.** The between-class share of the centred
   variance of the **unnormalised concatenated Z_pre** must be ≥ 0.05,
   with the median over the 7 locks of a seed taken, in ≥ 8 of 10
   seeds. This is measured on a **class-balanced** probe:
   - The rev. B probe was one contiguous window and was single-label on
     several seeds (Codex 2026-10-28), so its between-class share was 0
     by construction.
   - Rev. C uses the v0.7 class-balanced probe
     (`probe.class_balanced_probe_indices`): 8 windows of 16 frames per
     class, each window within one label run, sampled without
     replacement among the eligible windows.
   - In the real labels, class 0 is a single 180-frame run with 10
     eligible windows. Its 8 windows therefore all come from one run;
     this is reported.
4. **Drift floor.** On locks 25 and 175 of every pilot seed, run a
   second no-lesion control (control′) that differs from the control
   **only** in its Phase-2 batch / noise seeds.
   - The probe rows are fixed per block and taken from the block's
     probe seed. They are shared by Z_control, Z_control′ and every
     Z_post (the capture is decoupled from the Phase-2 seed).
   - For each seed and endpoint, the floor is F = d(control,
     control′), and the ratio is R = median over the 15 cells of
     d(post, control) / F.
   - F < 10⁻⁹ makes the block invalid (reported). There is no
     substitution.
   - **Pass:** the median of R over seeds is ≥ 2 **at each endpoint
     separately**.
   - Also reported: the control-only endpoint contrast F(175) − F(25),
     with a seed-bootstrap 90 % CI. A clearly positive value means the
     lock trend could arise without the lesion; it is shown next to the
     lesion contrast.
   - This gate answers whether the lesion signal exceeds re-training
     noise. It does **not** calibrate the guard (below).
5. **Y not degenerate** (within-seed SD > 0), and **N ≤ 40** by the
   registered rule. The N simulation uses the final metric, the
   missing-block rule of gate 2 and the guard.

**Guard scale.** Rev. B's standardised guard is kept (PI 2026-10-25,
19:47):

- "Supported" also needs the median endpoint contrast Y(s, 175) −
  Y(s, 25) > 0.5 σ̂.
- σ̂ is the UCL80 within-seed SD of the **primary** Y from the rev. C
  pilot.
- The N rule uses a total trend of 0.5 + 0.70 = 1.2 SD.
- The guard is a chosen smallest effect of interest in σ̂ units, not a
  calibrated noise bound.
- The secondary gets its own σ̂ and the same rule, descriptively.

**Eigengap / k.** Not needed for d. The rev. B secondary (principal
shift at the eigengap k) is withdrawn, because its only pilot matrices
were sub-tolerance. It comes back only if the PI asks.

**Pilot.**

- **Seeds:** 10 fresh seeds 2211–2220 (proposed). They extend the
  `v08-pilot` pool, code 8101; the confirmatory pool 2001–2044 is
  untouched, and the disjointness test is extended.
- **Captures:** all 7 locks. (a) and (b) are captured from the same
  forward pass of the same networks, plus the drift-floor controls
  (20 extra Phase-2 runs).
- **Cost:** the rev. B capture took 65 min for 70 blocks on macM1, so
  about 75–80 min. Storage is about 0.8 GB (512 × 640 and 512 × 320
  matrices instead of 512 × 4096).
- **Lock:** run under the macM1 heavy-job lock.

## D.2 v0.9 — label and world

### Why rev. B had little headroom

The data provenance explains the headroom (`scripts/fetch_studyforrest_sample.py`):

- the label is the `scenes.csv` scene index quantised to 4;
- the vision proxy is built from the same scene cut structure;
- tactile is a projected one-hot of the same scene attributes.

So two inputs are functions of the scene id that defines the label. On
the raw features (time-block CV, 16-frame purge), vision gives 0.830
and the concatenation 0.871. The margin is +0.04 for the raw
concatenation, and −0.002 (mean) for the trained-encoder control of
rev. B. Class sizes are 180 / 3480 / 4280 / 1060.

The construction leaves little room for fusion gain with this label. A
different probe or split might find some; the raw margin is only a
reference.

### Option (a) — a label that needs two modalities (two-channel label)

- **Definition.** y = 2a + b: 4 classes, so the network head is
  unchanged. The two bits come from two different real channels:
  - **a** = the narration-on flag. It is the first base feature of the
    audio proxy, which is `[on, into, density, to_next] @ proj` plus
    0.05 noise, with `proj` from `default_rng(1)`.
    - The flag is recovered exactly from the cached proxy by least
      squares: every recovered value is within 0.026 of 0 or 1, and
      the proxy residual RMS is 0.049, which is the construction
      noise.
    - So no new download is needed.
  - **b** = the sign of the force column 4: the first derivative of the
    respiratory trace, decimated to 10 Hz.
- **Why a linear label.**
  - A single modality can identify at most one bit, so ≈ 0.5 balanced
    accuracy on y. The pair identifies both, so ≈ 1.
  - y is linear in the two bits, so a linear probe on a concatenation
    can reach it. An XOR label would not be linearly readable.
- **Raw-feature feasibility of this exact label**
  (`v09_revc_label_feasibility.json`, 64-frame purge; the 16-frame
  purge gives the same values within 0.01):

  | target | audio | vision | tactile | gravity | force | concat |
  |---|---:|---:|---:|---:|---:|---:|
  | y = 2a + b | 0.510 | 0.243 | 0.307 | 0.356 | **0.548** | **0.963** |
  | a | **1.000** | 0.476 | 0.618 | 0.706 | 0.527 | 1.000 |
  | b | 0.507 | 0.499 | 0.504 | 0.505 | **0.992** | 0.977 |

  - Prevalence: a 0.305, b 0.574, φ(a, b) = 0.006.
  - Class counts: 2675 / 3578 / 1156 / 1591.
  - Raw headroom (concatenation − best single modality on y): 0.415.
- **What the check shows that matters for the design.**
  1. **Bit a leaks partly into gravity (0.71) and tactile (0.62).**
     Narration timing is correlated with the slow head-motion drift and
     with the scenes. Each bit is therefore not readable from one
     modality only. On y, however, the best single modality stays at
     0.548. So the isolation requirement is set on y, not on each bit
     (gate 2 below).
  2. **Bit b is not a respiratory phase.** Its runs last 1.8 frames on
     average: the decimated derivative changes sign almost every frame.
     Smoothing the trace over 5, 10 or 20 frames before the derivative
     leaves runs of about 2 frames, and drops its decodability from
     force to 0.66, 0.56 and 0.51. So b is a frame-level feature bit of
     the force channel. It is readable (0.99) but has no physiological
     reading. The PI may prefer a slower force bit; it would need to be
     screened the same way before any pilot.
  3. Because b switches every ~2 frames, the 16-frame label-run windows
     of the class-balanced probe do not exist for y. v0.9 uses the time
     blocks below instead.
- **Gates of the rev. C pilot:**
  1. The learning gate (D.0) on y.
  2. **Label-level isolation** on the trained encoders' h_m:
     - every single modality ≤ 0.65 held-out balanced accuracy on y;
     - the concatenation − best single ≥ 0.20, mean over seeds × onsets
       of the plain arm;
     - raw values: 0.548 and 0.415;
     - the per-bit readability is reported descriptively.
  3. **Positive control** with an explicit scope:
     - mean FG_ctrl ≥ 0.10 (twice the SESOI) over seeds × onsets of the
       **plain arm**;
     - the control is fitted without the PCA step: a logistic on the
       standardised 640-dim concatenation, L2 as in rev. B, because PCA
       can drop the useful directions (Codex r6);
     - reported per run: class recalls, convergence, and a block-level
       bootstrap CI.
  4. **Intervention sensitivity (positive treatment control).** The
     registered donor sampling is within the 4-class label
     (`decorrelation.py`, donors uniform over rows with the anchor's
     label). With y = 2a + b, every donor keeps both bits. The decor
     arm therefore only breaks the dependence between modalities
     *within* a class, and cannot remove the task evidence; a
     manipulation check can detect a change that does not matter for
     the task. The pilot therefore adds one non-registered arm in which
     the force donor is drawn **regardless of the label** (b is
     destroyed). It must lower plain bacc(h_all) by ≥ 0.10 relative to
     the joint arm. This shows that the readout responds to an
     intervention that matters. The limitation of the registered arm is
     stated in the draft: a zero joint − decor difference means "no
     sensitivity to within-class cross-modal dependence", not "no
     fusion plasticity".
  5. Plain-arm bacc(h_all) ≥ 0.35 and a zero-D share < 0.5 (as in rev.
     B).
  6. N ≤ 40. The SESOI stays 0.05 unless the pilot's FG_ctrl
     distribution argues otherwise.
- **Split** (frozen before the pilot; answers Codex r6 point 5 and r7
  point 6):
  - The 9000 frames are cut into 30 contiguous segments of 300 frames.
  - The held-out set is 3 segments chosen by a fixed-seed search, so
    that every class of y has ≥ 15 % of its rows held out. The search
    seed, the algorithm and the chosen segments are written in the
    draft.
  - 64 frames (6.4 s, more than the ±5 s density window of the audio
    proxy) are purged on each side of every held-out segment.
  - The purge applies to encoder training, probe training, the donor
    pools and the anchors.
  - The probe-training CV is grouped by the remaining segments, with
    the same 64-frame purge between folds.
  - Retained class counts and usable training windows are reported.
  - Everything is evaluated within one recording (one subject, one
    session). No claim of generalisation to other recordings is made.
- **Cost / feasibility:**
  - Code (about 1 day; no new dependency, no new data):
    - a relabelling wrapper of the existing bridge;
    - the segment split with purges, including the donor pools;
    - the control without PCA;
    - the label-breaking control arm;
    - the full-network learning gate;
    - tests.
  - The manipulation check has to be re-run, because the decorrelation
    effect depends on the label.
  - Pilot compute: about the rev. B v0.9 pilot (484 s for 75 runs),
    plus the extra arm and the manipulation check, so under 40 min on
    macM1.

### Option (b) — a different world (CC-film, `CCFilmWorld`)

- **What it is:**
  - *Big Buck Bunny* (CC-BY 3.0), with real audio (128-D log-mel) and
    real vision (16 × 16 intensity grid);
  - tactile, gravity and force are **synthetic**: RNG streams modulated
    by the audio energy-flux envelope (`extract_ccfilm_features.py`);
  - the native label is the shot-cut class: the frame-to-frame
    difference, quantile-quantised;
  - the macM1 cache has **600 frames at 2 fps** (300 s), 150 per class.
- **Raw-feature feasibility (native label).** All raw linear probes are
  near chance on these 600 frames: audio 0.223, vision 0.295, tactile
  0.267, gravity 0.270, force 0.288, concatenation 0.270 (chance 0.25).
  - The label is defined on two consecutive frames. Whether the trained
    encoders can read it from single frames is not settled by a linear
    probe on 600 frames.
  - What the check suggests is a floor, not a ceiling. Plain
    bacc(h_all) may not reach the 0.35 gate.
- **What would make it usable:**
  1. Re-extract at 10 fps with the existing
     `scripts/extract_ccfilm_features.py`: about 6,000 frames, CPU
     minutes, and a new data artefact with a MANIFEST entry.
  2. Optionally replace the native label with a two-channel label as in
     (a), for example a = audio-energy high/low and b = vision-brightness
     high/low.
     - The synthetic channels follow the energy **flux**, not the level,
       so they do not necessarily duplicate a.
     - Their readability of a would be screened as in (a).
  3. Run the same gates as (a).
- **Cost:** steps (1) and (2) take about 1–1.5 days on top of (a)'s
  code, which they reuse. The pilot is about as cheap as (a).

### Primary vs secondary — recommendation

**(a) primary, (b) secondary** (a replication world, only if (a)
passes). Reasons:

- (a) keeps the registered real bridge (real proprioception, biological
  anchoring).
- (a) has a raw headroom of 0.415 for the exact label, and needs no new
  data or world wiring.
- CC-film's native label is near chance for raw linear probes on 600
  frames. It is useful as a check that the result is not specific to
  Studyforrest, after re-extraction and with a label screened in the
  same way.

### Dropped (PI decision 2026-10-28)

The rev. B descriptive outcomes that were never implemented (the K = 8
Gumbel fused code and the held-out log-loss) are **dropped** from v0.9.

**Pilot.**

- **Seeds:** 5 fresh seeds 3206–3210 (proposed). They extend the
  `v09-pilot` pool, code 9101; the confirmatory pool 3001–3044 is
  untouched.
- **Order:** the manipulation check first, then the main grid with the
  label-breaking control arm.
- **Lock:** run under the macM1 heavy-job lock.

## D.3 Decisions requested from the PI

1. **D.0:** P1 (straight-through plus the full-network learning gate;
   recommended) or P2 (default emission, descriptive only).
2. **v0.8:**
   - (a) primary / (b) secondary as recommended, or the reverse;
   - the drift-floor gate (R ≥ 2 at each endpoint);
   - fresh seeds 2211–2220.
3. **v0.9:**
   - (a) primary / (b) secondary;
   - bit b as defined (a frame-level force bit) or a slower force bit to
     be screened first;
   - the label-level isolation gate;
   - the positive-control and label-breaking-arm criteria;
   - the segment split with 64-frame purges;
   - fresh seeds 3206–3210.
4. **Next step:** implementation then pilots, or implementation only,
   for review.

## D.4 PI decisions (2026-10-28, about 19:05) and implementation

These decisions were taken after the rev. B pilots and before any rev. C
pilot; they are recorded as post-pilot decisions of the unfiled drafts.

- **D.0: P1.** Straight-through emission, with the learning gate as in
  D.0: held-out balanced accuracy on the campaign label ≥ 0.40 (mean)
  and ≥ 0.30 (every seed); for v0.9, also ≥ +0.05 over the best
  single-modality query.
- **v0.8: adopted as proposed.**
  - (a) encoder 1 − CKA is the primary; (b) sender logits is the
    secondary;
  - the five gates, including the drift floor (R ≥ 2 at each end);
  - guard 0.5 σ̂;
  - the eigengap secondary is removed;
  - pilot seeds 2211–2220.
- **v0.9: the order is inverted. CC-film is PRIMARY**, the Studyforrest
  two-channel label y = 2a + b is SECONDARY (replication).
  - The CC-film cache is re-extracted at 10 fps with the existing
    script; the only new parameters are `--fps` and `--max-seconds`.
  - CC-film uses a two-channel label screened with the same isolation
    gates (every modality ≤ 0.65, concatenation − best ≥ 0.20), the
    positive control, the label-breaking arm and the frozen purged
    split.
  - Pilot seeds 3206–3210, CC-film first.
- **Download (PI approvals 19:14 and 19:21).**
  - The approved `.mp4` URL returns 404. The PI approved
    `…/BigBuckBunny_320x180.mp4.zip` instead: downloaded onto macM1 only
    on 2026-10-28 at 18:22 UTC and unpacked with the system `unzip`.
    - `.zip`: 64,657,225 bytes, SHA-256 `109e3ede…fceedf`;
    - `.mp4`: 64,657,027 bytes, SHA-256 `f78f3960…280508`;
    - licence: CC-BY 3.0, Blender Foundation.
  - `scripts/extract_ccfilm_features.py` verifies the `.mp4` hash before
    every extraction and refuses on mismatch.
  - **Identity check (done first):** a 2-fps / 300-s re-extraction
    reproduced all six tensor SHA-256 of the existing 2-fps cache
    exactly.
  - The 10-fps cache has 5,965 frames (596 s). Its MANIFEST, with the
    download provenance, is copied to
    `reports/pilots/ccfilm_real_10fps_MANIFEST.md`.

**Implementation.**

- **v0.8:**
  - `v08_driver.run_block(design="revc")` (straight-through networks,
    class-balanced probe rows, encoder and logit readouts, a second
    control at locks 25 / 175 with Phase-2 seed s + 500 000, learning-gate
    metrics on the stratified held-out rows);
  - `metrics/revc_shift.py` (per-block non-degeneracy before scaling,
    distance, drift floor);
  - `v08_driver.pilot_quantities_revc`.
- **v0.9:**
  - `v09_revc.py`: labels, `SegmentSplitWorld`, `LabelBreakSchedule`,
    outcome, gates;
  - `scripts/run_v09_revc.py`.
- **Learning gate:** `learning_gate.py`.
- **Seed pools:** `v08-pilot` becomes 2201–2220, `v09-pilot` 3201–3210
  (codes unchanged). The disjointness tests include the drift-floor
  seeds.

**Corrections found while implementing (stated, not silently changed).**

1. **Split.** "3 of 30 segments with every class ≥ 15 % held out" is
   infeasible: 3 / 30 of the frames cannot give every class 15 %.
   - Implemented instead: 5 of 30 segments, chosen among 2,000 random
     5-subsets (generator 20261037) as the first that maximises the
     smallest per-class held-out share.
   - Achieved shares: CC-film 0.163 (995 held-out rows, 384 purged);
     Studyforrest 0.167 (1,500 held-out, 640 purged).
2. **CC-film label** (the proposal named it only as an example):
   - a = audio energy (mean of the 128 standardised log-mel bands) above
     its median;
   - b = vision brightness (mean of the 16 × 16 grid) above its median;
   - the label-breaking arm breaks the modality that carries b (vision
     for CC-film, force for Studyforrest).
3. **Probe training** uses every 4th usable row. The best modality is
   chosen by grouped 4-fold CV over segments, purging 64 frames around
   every test row. The positive control is fitted without PCA.

**Raw screen** (no network; `reports/pilots/v09_revc_label_screen.json`,
64-frame purge):

| world | label counts | best single on y | concatenation | a read by | b read by |
|---|---|---:|---:|---|---|
| CC-film 10 fps | 1231 / 1752 / 1752 / 1230, φ = −0.17 | vision 0.536 | 0.943 | audio 0.99; the synthetic tactile/gravity/force 0.74–0.76 (audio-derived) | vision 0.98 |
| Studyforrest | 2675 / 3578 / 1156 / 1591, φ = 0.006 | force 0.548 | 0.963 | audio 1.00; gravity 0.71 | force 0.99 |

Both labels pass the raw isolation screen (≤ 0.65 and ≥ 0.20) on y. The
pilot gates apply to the trained encoders.

**Segments.** `numpy.array_split` of all frames into 30 segments:

- Studyforrest: 300 frames each; held-out segments 1, 6, 13, 16, 22.
- CC-film (5,965 frames): 198–199 frames each, not the 300 of the
  proposal; held-out segments 11, 13, 18, 19, 20.

**Codex r8 (2026-10-28) fixes, before any rev. C pilot.**

1. The learning gate is applied separately at every lock (v0.8) and at
   every onset (v0.9). Every setting must pass; nothing is averaged
   across settings.
2. The drift-floor gate needs the control of every seed at both ends.
   - A missing control fails the gate.
   - An undefined R (invalid control, Y undefined, or F < 1e-9) counts
     as R = 0 in the median and is never dropped.
3. The N rule runs only if every prerequisite gate passes. The
   secondary readout gets its own descriptive guard and N rule.
4. Gate 5 is the within-seed SD of Y over the locks (> 0 for every
   complete seed). The ANOVA σ̂ is kept for planning only.
5. The runs are staged and stop when a prerequisite fails.
   - v0.8: a learning-gate stage (Phase 1 and the no-lesion control of
     every block) runs before any capture. The capture refuses to start
     without a passed gate file.
   - v0.9: the manipulation check, then the plain arm and the learning
     gate, then the other arms.
6. New diagnostics:
   - a 16-row block-bootstrap 90 % CI of FG_ctrl;
   - per-bit readability (a and b) of every trained encoder on the plain
     arm;
   - a self-donor count for the label-breaking arm.
