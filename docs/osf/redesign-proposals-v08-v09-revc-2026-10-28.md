# Redesign proposals, rev. C — v0.8 measurement and v0.9 label / world (2026-10-28)

**Status:** PROPOSAL for PI review (PI decision 2026-10-28, about
18:05: "pursue both options as a proposal, then STOP for my review
before any new pilot"). No new pilot has been run. Nothing here is
implemented; the drafts `amendment-v0.8-*` / `amendment-v0.9-*` only
point to this file. Follows `redesign-proposals-v08-v09-2026-10-25.md`
(sections A–C, rev. B and its pilots).

Evidence comes from existing data only:

- the rev. B pilots (`reports/pilots/v08_revb_pilot_*.json`,
  `reports/pilots/v09_revb_pilot_main_grid.json`);
- the collapse diagnosis merged from main (ADR-0024,
  `docs/milestones/2026-10-26-code-collapse-diagnosis.md`,
  `reports/diag_code_collapse_*.json`, `reports/emission_pilot_*.json`);
- a **raw-feature feasibility check** made for this file
  (`reports/pilots/v09_revc_label_feasibility.json`, script
  `scripts/pilots/v09_revc_label_feasibility.py`). It trains no network
  and uses no seed: a logistic probe on the raw modality features of
  each world, 10 contiguous time folds with a 16-frame buffer either
  side of the test fold, balanced accuracy. macM1, 47 s, `e7c8a06`-era
  environment (uv `--frozen`).

## D.0 Prerequisite common to both campaigns (PI decision needed)

ADR-0024 (main, merged here in `63ef538`): with the default argmax
emission the network **does not learn the task**. The argmax blocks the
task gradient, Adam turns the 1e-6 bridge into drift, each modality
collapses onto a few codes, and the logit margin grows to 17–48. That is
the mechanism behind both rev. B failures. The v0.8 soft code is
constant (C.1) and the v0.9 fused code h_all is near-constant (C.2).

So a new measurement of **any** representation in the default regime
measures optimiser drift, not adaptation in a learning agent. Before
the pilots below, the PI has to choose one of two regimes:

- **(P1, recommended)** run v0.8 and v0.9 rev. C with the opt-in
  straight-through emission (`--emission straight-through`, main
  `4e1dffc`). The emission pilot result (ADR-0024 follow-up) shows it
  learns on the Gaussian world (loss 1.386 → 0.99, control margin 0.44).
  It did **not** pass on the Studyforrest mock (last-20 loss 1.30 ≥ 1.3).
  So add a **learning gate** to each pilot: the emission-pilot criterion
  (last-20 Phase-1 loss < 1.3 and vision control margin ≥ 0.1) on the
  real Studyforrest bridge. If it fails, stop and report.
- **(P2)** keep the default emission. Report every result as *measured
  in a non-learning regime* (ADR-0024 wording), descriptively only.

Under P1 the rev. B fused-code readouts could recover structure. The
rev. B gates are therefore re-reported under P1 as secondary
information, in addition to the measurements below.

## D.1 v0.8 — what to measure

v0.8's question stays the same: does the representational change caused
by the lesion depend on LOCK_AFTER (Page trend over the 7 locks)? Rev. B
measured it on the fused code downstream of the argmax. Rev. C moves the
measurement upstream of the quantisation. Each `SensoryWML.step` computes
`h = core(input_proj(x))`, then `pi_logits = emit_head_pi(h)`, then
argmax.

### Option (a) — encoder representations, upstream of demodulation

- **What:** for each modality m, h_m = `core(input_proj(x_m))` (128 dims)
  on the probe rows. Z is the concatenation over the 5 modalities. Each
  modality block is scaled to unit total centred variance first, so that
  no modality dominates CKA. d(c) = 1 − CKA_lin(Z_post(c), Z_control),
  as in rev. B, with the same per-block median over 15 cells, the same
  Page test and safeguard, and the same protocol failure. Descriptive:
  the same d per modality (lesioned vs intact modalities).
- **Evidence that it is not degenerate:**
  - in the v0.9 rev. B pilot, the same h_m on the same real bridge
    carry the label (held-out balanced accuracy, mean over 75 runs):
    vision 0.986, tactile 0.943, gravity 0.765, force 0.298,
    audio 0.279;
  - the probe-training CV picked vision in 69 of 75 runs.
  So these matrices have input-driven, between-class structure, unlike
  the soft code (total variance 8.6e-15). Caveat (ADR-0024): in the
  default regime they move only by drift. P1 matters here.

### Option (b) — sender logits, before the softmax / argmax

- **What:** `pi_logits` per modality (64 alphabet logits each, 320
  concatenated, per-modality scaling as in (a)). The same d, Y, Page
  test and safeguard. The code that is sent is the argmax of these
  logits, so this is the pre-quantisation code.
- **Evidence:** none in the rev. B data (only the codes were stored).
  The collapse diagnosis gives a reason for caution: in the default
  regime the margin between the winning logit and the next grows to
  17.5 (Gaussian) and 48.1 (mock) while active codes fall to 1–4. The
  centred variance of the logits is then likely concentrated in the
  margin direction of the few winning codes. A variance-weighted CKA
  would then track margin drift.

### Primary vs secondary — recommendation

**(a) primary, (b) secondary.**

- (a) is the only option with positive evidence of an input-driven,
  non-degenerate structure on the real bridge (v0.9 rev. B).
- (a) is the locus that the Phase-2 lesion training changes directly.
- (a) does not inherit the argmax/margin pathology.
- (b) is closer to the emitted code, so it is kept as the registered
  secondary. If the pilot shows (b) is well conditioned (gates below)
  and P1 is chosen, the PI may swap them before filing. No swap after
  the confirmatory data.

### Gates of the rev. C pilot (before the PI accepts the design)

1. **Learning gate** (P1 only): see D.0.
2. **Non-degeneracy:** a **relative** tolerance instead of the absolute
   10⁻¹² of rev. B: centred total variance ≥ 10⁻⁶ × mean squared row
   norm, for every Z_pre, Z_control and Z_post. At least 12 valid cells
   per block, in every block of ≥ 8 of 10 seeds.
3. **Input-driven structure:** a between-class variance share ≥ 0.05 on
   a **class-balanced** probe.
   - The rev. B probe (one contiguous window) was single-label on
     several seeds (Codex, 2026-10-28), so its between-class share was 0
     by construction.
   - Rev. C uses the v0.7 class-balanced label-run probe instead
     (`probe.class_balanced_probe_indices`, 4 classes × 8 runs × 16
     frames; class 0 has 180 frames, so it is drawn with fewer runs and
     the per-class row counts are reported).
4. **Drift floor:** on 2 locks (25 and 175) of every pilot seed, a
   second no-lesion control with a different Phase-2 noise seed.
   d(control, control′) is the floor that the lesion effect has to
   exceed. Report d(post, control) / d(control, control′) per block.
   The design is acceptable only if the median ratio is ≥ 2 for the
   primary. This answers the Codex point that rev. B could not separate
   signal from sampling noise.
5. **Y not degenerate** (within-seed SD > 0), and **N ≤ 40** by the
   registered rule.

**Guard scale:** keep rev. B's standardised guard (PI 2026-10-25,
19:47): "Supported" also needs the median endpoint contrast
Y(s, 175) − Y(s, 25) > 0.5 σ̂. σ̂ is the UCL80 within-seed SD of the
**primary** Y from the rev. C pilot, and the N rule uses a total trend
of 0.5 + 0.70 = 1.2 SD. The secondary gets its own σ̂ and the same
rule, descriptively. If gate 4 gives a floor ratio below 2, the guard
cannot be calibrated, and the outcome falls back to (b) or to dropping
v0.8 (PI).

**Eigengap / k:** not needed for d. The rev. B secondary (principal
shift at the eigengap k) is withdrawn: its only pilot matrices were
sub-tolerance. It returns only if the PI asks.

**Pilot:**

- 10 fresh seeds 2211–2220 (proposed; extend the `v08-pilot` pool, code
  8101; confirmatory 2001–2044 untouched; the disjointness test is
  extended);
- all 7 locks;
- (a) and (b) captured from the same networks in one pass (2 extra
  reads of the same forward pass, no extra training);
- plus the drift-floor controls (2 × 10 extra Phase-2 runs).
- Cost estimate from the rev. B capture (65 min for 70 blocks on
  macM1): about 75–80 min. The probe matrices are 512 × 640 and 512 ×
  320 instead of 512 × 4096, so storage is about 0.8 GB instead of
  4.1 GB.
- Under the macM1 heavy-job lock.

## D.2 v0.9 — label and world

**Why rev. B had no headroom (now traced to the data).** The provenance
of the Studyforrest bridge (`scripts/fetch_studyforrest_sample.py`)
explains the ceiling by construction:

- the label is the `scenes.csv` scene index quantised to 4;
- the vision proxy is built from the same scene cut structure;
- tactile is a one-hot of the same scene attributes, projected.

So two inputs are deterministic functions of the scene id that defines
the label. On the raw features (time-block CV), vision already gives
0.830 and the concatenation 0.871. The class sizes are 180 / 3480 /
4280 / 1060. No probe or split can create fusion headroom for this
label.

### Option (a) — a label that needs two modalities (two-channel label)

- **Definition:** y = 2a + b (4 classes, so the network head is
  unchanged), with two bits taken from two different real channels:
  - **a** from the audio proxy, the narration-on flag of
    `german_audio_description.csv` (the annotation that the audio proxy
    is built from);
  - **b** from force, inhalation vs exhalation (sign of the respiration
    first derivative, the force column 4 at 10 Hz).
  Each single modality can then identify at most one bit (≈ 0.5
  balanced accuracy on y); the pair identifies both (≈ 1). The headroom
  is built in, and y is linear in the two bits, so a linear probe on a
  concatenation can reach it (an XOR label would not).
- **Raw-feature feasibility** (`v09_revc_label_feasibility.json`; the
  bit **a** here is a stand-in, the median split of the leading
  principal component of the audio proxy, because the flag itself has
  to be recomputed from the annotation):

  | target | audio | vision | tactile | gravity | force | concat |
  |---|---:|---:|---:|---:|---:|---:|
  | y = 2a + b | 0.509 | 0.232 | 0.323 | 0.349 | **0.541** | **0.967** |
  | a | **0.998** | 0.461 | 0.639 | 0.678 | 0.534 | 0.996 |
  | b | 0.517 | 0.499 | 0.516 | 0.508 | **0.997** | 0.984 |

  - Classes are balanced (2205 / 2295 / 2300 / 2200), and φ(a, b) =
    −0.02.
  - Raw headroom (concat − best single) is 0.43, against −0.002 for
    the rev. B positive control.
  - Leakage: the stand-in **a** is partly readable from gravity (0.68)
    and tactile (0.64). It probably follows the slow narration density
    (time drift), which is why the binary flag is proposed instead.
- **Gates of the rev. C pilot:**
  1. learning gate (D.0, on the new label);
  2. bit isolation on the trained encoders' h_m: own modality ≥ 0.75
     for its bit and every other modality ≤ 0.65 for it;
  3. positive control, now with a boolean criterion: mean FG_ctrl ≥ 0.10
     (twice the SESOI). The control is fitted **without** the PCA step
     (logistic on the standardised concatenation, 640 dims, L2 as in
     rev. B), because PCA can drop the useful directions (Codex
     2026-10-28);
  4. plain-arm bacc(h_all) ≥ 0.35 and a zero-D share < 0.5 (as in rev.
     B);
  5. N ≤ 40.
  The SESOI stays 0.05 unless the pilot's FG_ctrl distribution says
  otherwise (it has to sit well below the attainable gain).
- **Split (answers Codex 2026-10-28, point 5):**
  - held-out rows = 10 contiguous time blocks, stratified so that each
    of the 4 classes has ≥ 15 % of its rows held out;
  - a 32-frame buffer (3.2 s) dropped on both sides of every held-out
    block, from both encoder training and probe training;
  - probe-training CV grouped by contiguous time blocks, not by label
    runs (the two bits switch every few seconds, so label-run blocks no
    longer apply).
- **Cost / feasibility:**
  - code: a relabelling wrapper of the existing bridge, the time-block
    split, and the control without PCA, with tests (about 1 day). No
    new dependency.
  - data: the narration flag is recomputed from the already-cited S3
    annotation. The file is fetched once, hashed and added to the data
    MANIFEST.
  - the manipulation check has to be re-run, since the decorrelation
    effect depends on the world/label.
  - pilot compute: about the rev. B v0.9 pilot (484 s for 75 runs) plus
    the manipulation check, so under 30 min on macM1.

### Option (b) — a different world (CC-film, `CCFilmWorld`)

- **What it is:**
  - *Big Buck Bunny* (CC-BY 3.0), real audio (128-D log-mel) and real
    vision (16 × 16 intensity grid);
  - **synthetic** tactile / gravity / force, built from the audio
    energy envelope;
  - native label = shot-cut classes from the frame-to-frame difference,
    quantile-quantised.
  - The cache on macM1 has **600 frames at 2 fps** (300 s), 150 per
    class.
- **Raw-feature feasibility (native label):** no raw modality decodes it
  (audio 0.223, vision 0.295, tactile 0.267, gravity 0.270, force
  0.288, concat 0.270; chance 0.25). The label depends on two
  consecutive frames, and the encoders see one. So the problem is a
  floor, not a ceiling: plain bacc(h_all) would almost surely fail the
  0.35 gate, and fusion gain would be ≈ 0 for the opposite reason to
  rev. B.
- **What would make it usable:**
  1. re-extract at 10 fps (≈ 6,000 frames, with the existing
     `scripts/extract_ccfilm_features.py`; CPU minutes, new data
     artefact + MANIFEST);
  2. replace the native label with a two-channel label as in (a), e.g.
     a = audio-energy high/low and b = vision-brightness high/low;
  3. accept that 3 of the 5 modalities are audio-derived and synthetic,
     which duplicates bit **a**. The bit-isolation gate (a.2) would
     fail for a unless tactile/gravity/force are excluded from the
     "best single modality". That is a change of the v0.9 construct.
- **Cost:** (1) + (2) about 1–1.5 days on top of (a)'s code, which it
  reuses. The pilot is about as cheap as (a).

### Primary vs secondary — recommendation

**(a) primary, (b) secondary (replication world, only if (a) passes).**

- (a) keeps the registered real bridge (real proprioception, biological
  anchoring) and builds headroom in by design: raw headroom 0.43.
- (a) needs no new world wiring and has a clean leakage test.
- The native CC-film label is undecodable from single frames, and its
  other three channels duplicate audio. CC-film is useful only with the
  same two-channel construction, as a check that the result is not
  specific to Studyforrest.

### Dropped (PI decision 2026-10-28)

The rev. B descriptive outcomes that were never implemented, the K = 8
Gumbel fused code and the held-out log-loss, are **dropped** from v0.9.

**Pilot:**

- 5 fresh seeds 3206–3210 (proposed; extend the `v09-pilot` pool, code
  9101; confirmatory 3001–3044 untouched);
- the manipulation check first, then the main grid;
- under the macM1 heavy-job lock.

## D.3 Decisions requested from the PI

1. D.0: P1 (straight-through + learning gate, recommended) or P2
   (default emission, descriptive only).
2. v0.8: (a) primary / (b) secondary as recommended, or the reverse; the
   drift-floor gate (ratio ≥ 2); fresh seeds 2211–2220.
3. v0.9: (a) primary / (b) secondary; the positive-control criterion
   (FG_ctrl ≥ 0.10, no PCA); the time-block split with 32-frame buffers;
   fresh seeds 3206–3210; fetching the narration annotation (one S3
   file).
4. Implementation then pilots, or implementation only, for review.
