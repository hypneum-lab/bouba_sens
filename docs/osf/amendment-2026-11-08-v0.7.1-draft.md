# Amendment draft — bouba_sens v0.7.1 (network-level null decides; co-primaries; positive control)

**Status: L0 DRAFT, NOT FROZEN.** Nothing here is registered. It becomes
a pre-registration only after (a) the PI closes the decisions of §6,
(b) the roadmap reading lock passes for bouba_sens v0.7.1 (scheduled
S8; `hypneum-lab/.github` `docs/ROADMAP.md`, « Lire avant de geler »),
(c) the PI creates a new L1 tag. No run, pilot, estimator fit or null
draw of this amendment has been made. L3 (OSF) is out of scope for this
PR and needs a calibrated null (§5).

**Date:** 2026-11-08. **Issue:** #23. **Drives:** PI decision 16 of
PR #66, option A (« amendement v0.7 où le nul réseau décide »), taken
2026-11-08.
**Amends (additively, no in-place edit of frozen text):**
`docs/osf/amendment-v0.7-actionsense.md` (tag
`bouba_sens/v0.7-actionsense-prereg`, commit `02c2407`), as already
corrected by `docs/osf/erratum-2026-10-31-v0.7-seeds.md`.
**Builds on:** PR #66 draft `docs/prereg/b3-network-null/prereg-draft.md`
(branch `docs/b3-network-null-draft`, HEAD `939bb1c`; ADR-0028 is on
that branch only), ADR-0022, ADR-0026 and
`docs/milestones/2026-10-31-me6-large-n-results.md`, ADR-0027, ADR-0019
(amendment 2026-11-06), PR #78, issues #77 and #85.
**Verdict vocabulary:** the 9 classes of `PROJECT-MANAGEMENT.md` §6
(confirmé, réfuté, nul (équivalence TOST), non concluant, rétracté,
amendé, en attente, non testé, non vérifiable). Nothing else is used as
a verdict.

## 0. Summary of what changes

1. For B-3 / Me6, the network-level null of PR #66 (cyclic form,
   unit-level sign flip, G independent units) is the reading that
   decides. The frozen 0.02 reading is **reported as recorded and
   decides nothing**.
2. Two co-primaries are added: label **decodability** and a **PID**
   decomposition (redundancy / unicity / synergy), each with a
   calibrated null.
3. A **positive control** bouba/kiki in a physical world is added,
   with a stated way to fail.
4. A written interpretation rule per outcome (§4).
5. Seeds 300–304 stop being the confirmatory units (consecutive seeds
   share 199 of 200 Phase-1 batch seeds, PR #66 §3.1).

Not amended: the B-1 / B-2 / B-3 thresholds as recorded, the Me1 / Me3 /
Me6 / Me7 definitions, the ActionSense extraction and world (v0.7 §2),
the frozen file contents.

## 1. Me6 / B-3: the network-level null decides

### 1.1 Reading rule

- **Deciding reading:** P1 of PR #66 (Gaussian world, `cell` arm,
  straight-through, v2 probe of 256 test items), statistic T and
  unit-level sign-flip null of PR #66 §4–§5, B = 20 000, α = 0.01,
  learning gate (PR #66 §6.3), NC1, NC2, PC (PR #66 §7) as defined
  there. This amendment does not restate or alter them.
- **Reported as recorded, deciding nothing:** the frozen rule (median
  over trios of Me6 > 0.02, strict), the ADR-0022 paired sign-flip
  null (P(null ≥ observed) < 0.05), and their v1 / v2 numbers, for
  every grid already run and every grid run under the v0.7 grids.
  Reason, from `reports/me6_large_n_{cell,legacy}_mock.json`
  (`frac_above_threshold` 0.7477 and 0.9814 at n = 4096): the 0.02 rule
  fires in three quarters or more of H0 draws on the mock world and in
  every draw at n ≤ 1024; and the ADR-0022 H0 is conditional on the
  lesioned networks it observes (PR #66 §1).
- **ActionSense:** the world is fixed (real recordings), so H0 is
  conditional on it; the network null still needs G independent units
  (PR #66 §15). Its cost (G × 30 cells per grid, on two or four grids)
  is a PI decision (§6, item 3).
- B-3 itself (Me6 > 0.02) is **not re-decided**. A confirmation of
  H1-B3net is a new, separately named result.

### 1.2 Section-by-section map (PR #66 §15)

Old text is quoted from `docs/osf/amendment-v0.7-actionsense.md`; « new »
is what this amendment substitutes **once its tag exists**. Until then
the frozen text stands.

| v0.7 § | Old text | New text |
|---|---|---|
| 4.1 grids and seeds | « Confirmatory seeds 300, 301, 302, 303, 304 (never used in the repository) »; 4 grids × 150 cells | Seeds 300–304 are **not independent units** (199 of 200 Phase-1 batch seeds shared; the « never used » claim is already corrected by the 2026-10-31 erratum). Confirmatory units are the G unit seeds Q_g = 330 000 + 200·g of PR #66 §3.1, pilot g = 0…5, confirmatory g = 20…20+G−1. Grid shape per unit: 1 Phase 1 + 30 lesion cells (+10 no-lesion jobs for NC1). Number of grids (2 or 4) per §6 item 3 |
| 4.2 exact commands | `SEEDS="300 301 302 303 304"`, `run_grid.sh`, `me6_paired_null.py` | `SEEDS` = the unit list written in the Stage-B manifest, `NO_LESION=1` jobs, `scripts/me6_network_null.py` (PR #66 §11, not yet implemented). `me6_paired_null.py` is still run, for reporting only |
| 4.3 B-3 bullet | « B-3 is reported only together with the ADR-0022 paired sign-flip null … A B-3 number without its null is not quoted » | B-3 is reported together with the **network null** (T, null median, null P99, p = (k+1)/(B+1)). The ADR-0022 numbers and the 0.02 firing rate under the network null (PR #66 row D4) are reported beside it, labelled « descriptive, does not decide » |
| 5.1 completeness | « 150 of 150 `eval_report.json` files, 75 Me7 pairs and 30 complete trios » | Per-unit completeness 30 lesion + 10 no-lesion jobs, all G units (PR #66 §12). B-1 / B-2 completeness rules unchanged |
| 5.2 median rule | « B-k PASS if the grid median is greater than or equal to its threshold … no multiple-comparison correction » | B-3 verdict from the network null only. **Inconsistency declared:** the text « greater than or equal to » contradicts the strict « > » of §4.3 and §8 item 5 (and `aggregate_grid.py`); the strict « > » is the recorded rule, the « ≥ » is a typo. B-1 / B-2 rule unchanged. Multiplicity: Holm over the family of §3.5 |
| 5.3 B-3 reading | « distinguishable from the paired null only if P(null ≥ observed) < 0.05 … Otherwise compatible with the paired null » | « Compatible with » is **removed** (never read as absence). Reading by §4 of this amendment. H0 is the absence of **cyclic** asymmetry, not symmetry of expected performances (the raw form is rejected by main effects, PR #66 D1) |
| 5.5 descriptive | per-trio Me6 values with their own null tails | Per-unit Me6 (frozen statistic) kept as descriptive row D3 / D5; per-trio null tails stay descriptive |
| 7 deviations policy | « Once tagged, this text is frozen … a new dated amendment » | Applied: this file is that dated amendment. Pointer lines are added to the v0.7 file, ADR-0027 and (when it exists) ADR-0025 (§7) |
| 8 item 3 | block-length pilot seeds 100–101, confirmatory 300–304 | Block length is not needed by the unit-level flip (PR #66 §15). Whether the 100–101 pilot ran is **not verifiable from this clone**; if it ran, it is listed as exploratory (§5.3). The unit pilot g = 0…5 replaces it for planning |
| 9 not amended | « no change to the shape of one grid … 150 cells, thresholds » | The shape of a grid per unit changes (§1.2 row 4.1). Thresholds 0.05 / 0.10 / 0.02 and the strict « > » are unchanged |
| ADR-0025 listing | reserved for the ActionSense verdicts (`docs/adr/README.md`) | The verdict ADR lists this amendment as a deviation. ADR-0025 does not exist yet: the pointer is added when it is written |

The frozen v0.7 grids, if they are ever run before this amendment is
tagged, keep their registered analysis and are marked « reported as
registered, not deciding asymmetry ».

### 1.3 B-2 inputs (issues #77, #85, PR #78)

B-2 on the Studyforrest bridge is « non mesurable » (0 of 10 seeds pass
the gate, reader accuracy 0.250 = majority; ADR-0019 amendment
2026-11-06, `E1.3c-ter`). This amendment proposes, for PI decision
(§6, item 7), the reading rules requested in #85:

1. **A single seed above threshold on a gate-failing network** (corrected
   cache, seed 2, MINE-full +0.743, `gate_pass: false`): not read.
   Reported with its gate status first; it does not enter any mean used
   for a verdict. Class: « non concluant ».
2. **Paired-null p values that are reported but not interpreted** (the
   null run recaptures the probe; per-seed B-2 differs from the main run
   by up to 0.4 bit): a p value is reported as a number only when the
   null run reproduces the main-run observed B-2 per seed within a
   tolerance fixed before the run (proposed 1e-6, same global random
   state); otherwise class « non vérifiable » for that null. No verdict
   change is proposed.

## 2. Co-primary A: label decodability

### 2.1 Why

In `GaussianWorld` (`src/bouba_sens/world/gaussian.py`) the label is a
function of z[0:2] (quadrant signs). Audio (128), vision (256) and
tactile (32) are isometric images of the 32-d latent, so each carries
the whole label. Gravity (3) and force (6) are projections onto 3- and
6-dimensional subspaces: they carry it partially (the 2026-10-24
review notes this: not « entirely » redundant). Label information is
therefore **mostly redundant across modalities**; total MI between a
code and the label cannot say what a lesion moved (B-2 is one such
total). Decodability is the operational question « can the label still
be read, and from what ».

### 2.2 Definition and estimator

- Unit, conditions, lesion cells and the no-lesion matched jobs: those
  of PR #66 §3 and §7 (NC1 jobs are the matched baseline).
- For a cell lesioned on modality m: **DEC_S = held-out accuracy of the
  v2 linear probe (`src/bouba_sens/linear_probe.py`, ADR-0023:
  multinomial logistic regression, standardised features, L2 by
  K-fold on the training pool) fitted on the concatenation of the
  frozen representations of the four surviving modalities**, minus the
  probe's own majority baseline (`ProbeFit.baseline`), on the 256 test
  items of the unit. Descriptive companion DEC_m: same, on the lesioned
  modality's representation.
- Effect: Δ DEC_S(u, c, m) = DEC_S(lesion m) − DEC_S(no-lesion matched
  job), so training drift cancels and a no-lesion network is the
  reference.
- Code gap, stated: the v2 files store item successes per query
  modality, not representations or per-item predicted classes
  (`per_query_successes_v2.json`); a probe on concatenated survivors
  and the PID inputs (§3) need a new output. Not implemented here.

### 2.3 Calibrated null

Statistic T_DEC = median over the 6 conditions of max over the 5
modalities of |mean over units of Δ DEC_S|. Null: the unit-level sign
flip of PR #66 §5 (one sign per unit, applied to all conditions and
modalities), B = 20 000, p = (k+1)/(B+1), reported with observed value,
null median and P99. Validity assumptions, size check and positive
control as PR #66 §7 (generators S0–S3, planted effect on NC1 data): a
planted drop of δ_dec accuracy points on the survivors, required
p < 0.01 at the registered G. δ_dec is a PI decision (§6, item 5).

### 2.4 Learning gate and ceiling

Gate first (hypneum-nulls rule 5). The ADR-0019 amendment gate applies
to the reader: reader accuracy ≥ majority rate + 0.10 (0.35 for a
balanced 4-class probe), plus the three P1 gate criteria of PR #66
§6.3. Failed gate: « non concluant (pas de réseau qui apprend) ».
**Ceiling rule:** if the no-lesion survivors' DEC_S is within 0.02 of its
maximum possible value (accuracy 1 − baseline), the test is « non
mesurable » (floor / ceiling effect), read as « non concluant », never
as absence.

## 3. Co-primary B: PID (redundancy / unicity / synergy)

### 3.1 Definition

Partial information decomposition of the information that two sources
carry about a target (Williams & Beer, arXiv:1004.2515, verified):
I(Y; X₁, X₂) = R + U₁ + U₂ + S, with R the redundant, U_i the unique
and S the synergistic part. Applied to multimodal data as in Liang et
al., « Quantifying & Modeling Multimodal Interactions: An Information
Decomposition Framework », arXiv:2302.12247 (verified; NeurIPS 2023 per
the lab's 2026-10-24 review, venue not rechecked here).

- Target Y = world label (4 classes).
- Sources for a cell lesioned on m: X₁ = decoded class from the lesioned
  modality's representation (probe on that modality), X₂ = decoded class
  from the survivors' concatenated representation (§2.2 probe). Both
  4-valued, so the joint P(X₁, X₂, Y) is a 4 × 4 × 4 table estimated on
  the 256 test items.
- Atoms: R, U₁ (unique to the lesioned modality), U₂ (unique to the
  survivors), S. They sum to I(Y; X₁, X₂) (plug-in), which is also
  reported.
- Operational estimand, stated: this is the PID of **decoded classes**,
  not of the representations. It is a lower-resolution quantity than the
  896-d MINE of B-2 and says nothing about information the linear probe
  cannot read.

### 3.2 Estimator

Named: the **BROJA / unique-information PID of Bertschinger, Rauh,
Olbrich, Jost & Ay** (arXiv:1311.2852, verified), exact for discrete
tables of this size (convex optimisation over the marginal-preserving
set). Sensitivity estimator for continuous high-dimensional
representations: the batch estimator of Liang et al. (arXiv:2302.12247;
code `pliang279/PID` cited in the lab's 2026-10-24 review, repository
not rechecked). **No PID code exists in this repository** (searched:
none under `src/` or `scripts/`); an implementation, or a vetted
dependency under the lab's pin-and-review policy, is a prerequisite.
PID is not unique: the atoms depend on the chosen redundancy measure
(different measures give different R / S), so the measure is part of
the registration.

### 3.3 Why PID is needed (not only decodability)

« Migration » of information after a lesion has no meaning when the
label information is redundant: the survivors already held it. What can
move is the **split**: ΔR (less redundancy), ΔU₂ (survivors gain unique
information), ΔS. A hypothesis of migration is a statement about
ΔU₂ > 0 together with ΔR < 0 (the lab's review frames it as a
redundancy-to-unicity shift). Ratios of these (the review's κ =
ΔU / (−ΔR)) have a near-zero denominator when ΔR ≈ 0: **non calculable**
(hypneum-nulls rule 10); only differences are tested.

### 3.4 Calibrated null

Effect for each atom k ∈ {R, U₂, S}: Δk(u, c, m) = k(lesion m) −
k(no-lesion matched). Statistic T_k = median over conditions of max over
modalities of |mean over units of Δk|. Null: unit-level sign flip as
§2.3 (same units, same signs structure), B = 20 000.
U₁ is not tested: with R, U₂, S and the total it is determined; it and
the total are reported. Plug-in PID is positively biased at 256 items;
the bias is common to lesion and no-lesion cells of the same unit and
cancels in Δ to first order, which is an **assumption checked by the
size check** (PR #66 §7 generators extended to a 4 × 4 × 4 table
generator with known PID, including pure-redundant, pure-unique and
XOR-synergistic tables). A label-permutation null inside each cell
(same permutation for lesion and no-lesion probes) gives the bias floor
and is reported, not used for the decision. Positive control: planted
transfer of ε table mass from R to U₂ on NC1 data, required
p < 0.01; ε is a PI decision.

### 3.5 Family, gate and prediction

- Family F = {P1, DEC_S, ΔR, ΔU₂, ΔS}, Holm at FWER 0.01 (PI decision
  4 of PR #66 is α = 0.01 for F = {P1} and Holm if a row is added by
  amendment before the Stage-B tag: this amendment is that addition,
  pending §6 item 4).
- Same learning gate and ceiling rule as §2.4, evaluated before any
  statistic is read.
- **Prediction, written before data:** in the Gaussian world, with
  three modalities carrying the full latent, DEC_S is at ceiling and
  R is large; the expected reading of DEC_S, ΔR, ΔU₂, ΔS is « non
  concluant » (ceiling) or no rejection. The co-primaries are expected
  to be informative on the physical world (§5) and on ActionSense, not
  on the Gaussian world. This is declared so a Gaussian non-rejection
  is not later presented as a finding.

## 4. Positive control: bouba/kiki in a physical world

### 4.1 What « positive » means

A **pipeline control**, not a claim about animals: a world in which a
sound-shape correspondence exists **by physics, without language**, so
that a cross-modal system with no linguistic input can in principle
recover it. The control is positive if, in that world and with a
network that passes the learning gate, the cross-modal matching score
(given the sound of an impact, pick the matching shape among a round
and a pointed one) exceeds the **calibrated** null by the registered
margin, and the same pipeline gives a null result in the decoupled
negative-control world below.

World spec (to be written; no code exists): impact sounds synthesised
from simple solid shapes (round vs pointed), with the acoustic cues
reported for the effect, spectral balance and temporal continuity
(Fort & Schwartz, Sci. Rep. 12, doi 10.1038/s41598-022-23623-w, cited by
the lab's 2026-10-24 review as the physical recipe; **not verified
here: à vérifier par le PI**). Negative control world: same sounds and
shapes with the pairing randomised (physics decoupled).

### 4.2 How it can fail (required, hypneum-nulls rules 4–6)

1. Gate failure (network does not learn): control « non concluant »,
   not passed.
2. Matching at the null in the coupled world: positive control **failed**;
   every cross-modal result of the study is then « non concluant
   (contrôle positif échoué) ».
3. Matching above the null in the decoupled world: leak or analysis
   bug; the control is invalid.
4. Tautology check: the control is not informative if the matching
   score is forced by construction (e.g. identical features used by the
   score and the generator); to be checked at implementation.

Null: label-permutation of the pairing within the held-out set, paired
with the matching score pipeline (same permutation of shapes for coupled
and decoupled worlds), B ≥ 1 000, p = (k+1)/(B+1). Effect threshold: a
PI decision (§6, item 6). The animal figures of the review (chick ≈ 0.80
vs 0.25, macaques 0.67 / 0.74) are **not** adopted as a threshold: see
§4.3.

### 4.3 Behavioural literature (what is and is not verified)

Verified by `mcp__arxiv__get_abstract` on 2026-11-08:

- Kouwenhoven, Shahrasbi & Verhoef, arXiv:2507.10013: CLIP ResNet and
  ViT « do not consistently exhibit the bouba-kiki effect » (so
  « not robust in VLMs » is supported for these two CLIP variants, not
  for VLMs in general).
- Alper & Averbuch-Elor, arXiv:2310.16781: CLIP and Stable Diffusion
  show the pattern zero-shot (opposite finding; the VLM literature is
  mixed).

**Not verified here, à vérifier par le PI** (non-arXiv; only the lab's
2026-10-24 review is the source, and that review itself records that the
Science erratum and the Proc. B article body were not read):
Loconsole, Benavides-Varela & Regolin, naive baby chicks (Science 391,
2026); the macaque preprint (bioRxiv 10.64898/2026.07.31.741992); Biffi
et al., adult dogs (Proc. R. Soc. B 293, 2026). The claim « present
without language in chick, macaque and dog » is therefore carried as
unverified.

## 5. Interpretation rule per outcome (9 classes)

Order: completeness → gate → NC1, NC2 → positive control(s) → tests.
Each row is read on the test family F of §3.5 at Holm FWER 0.01.

| Outcome | Class |
|---|---|
| Grid or unit set incomplete after the rerun rule | non concluant (grille incomplète) |
| Gate failed, or ceiling / floor | non concluant (pas de réseau qui apprend / non mesurable) |
| NC1 or NC2 failed | non concluant (contrôle négatif échoué) |
| Positive control failed (physical world) | non concluant (contrôle positif échoué) for all cross-modal claims |
| P1 rejects (p below Holm threshold), controls pass | confirmé (H1-B3net: cyclic asymmetry), with C̄ and its simultaneous CI. **Not** a statement about B-3 as frozen |
| P1 does not reject | non concluant (or non concluant (puissance) if PC failed). Never nul, never « compatible » |
| DEC_S rejects, gate and ceiling pass | confirmé (decodability changed by the lesion), direction reported |
| ΔR, ΔU₂ or ΔS rejects | confirmé for that atom only, direction reported. « Migration » is claimed only if ΔU₂ > 0 and ΔR < 0 both reject |
| Co-primary does not reject | non concluant. **nul (équivalence TOST)** only under §5.1 |
| Frozen 0.02 rule, ADR-0022 paired null | reported as recorded, no class |
| A previous B-3 « PASS » or « compatible » statement contradicted by the network null | amendé (requalified), with the pointer to this file |
| Frozen text retracted by a result | rétracté (only by a later ADR) |
| Registered, no run yet | en attente |
| Cell not run | non testé |
| Artefact absent or unrepeatable (e.g. #85 null that does not reproduce) | non vérifiable |

### 5.1 TOST, N = 95, power 0.80

- PR #66 decision 5 stays for P1: **one-sided design, no TOST**; « nul
  TOST » is not a reachable reading of P1.
- Where equivalence is wanted (co-primaries), TOST needs a **justified,
  pre-registered margin** (hypneum-nulls rule 8: no default ±0.25).
  Candidate justifications, for the PI: the smallest planted effect that
  the positive control detects at power 0.80 (then margin = that effect,
  tying equivalence to demonstrated sensitivity), or half the recorded
  threshold as in the unmerged TOST draft (B-1 ±0.025, B-2 ±0.05 bit).
  Without a decision, no TOST: non-rejection stays « non concluant ».
- **N = 95 and power 0.80** are kept where they still apply: the 95
  figure comes from the unmerged v0.7 TOST draft (branch
  `docs/osf-prereg-2026-10`, raised there to N = 150 on 2026-10-25) and
  from Welch designs of dream-of-kiki; it is a count of seeds for a
  different design. In the network-null design the sample is G units
  chosen by the PR #66 §10 rule (G ∈ {20, 30, 40, 50}, power ≥ 0.80 at
  δ_pc = 0.02, ceiling 50 h). Power 0.80 is kept as the target; N = 95
  does **not** apply to P1 and is carried only if the PI decides on a
  separate TOST for a co-primary.
- Power of the co-primaries is computed with the same simulation rule
  (upper 80 % bound of the pilot between-unit SD) before the Stage-B
  tag; if no G ≤ 50 reaches 0.80 for a co-primary, it is demoted to
  descriptive and the demotion is declared.

## 6. Decisions left to the PI

Recommended option first.

1. **Co-primary status.** (a) DEC_S and PID atoms co-primary in family F
   with P1, Holm FWER 0.01. (b) Decodability and PID descriptive for
   v0.7.1, co-primary from v0.8. (c) Only P1 confirmatory (#66 as is).
2. **PID redundancy measure and estimator.** (a) BROJA exact on decoded
   classes (§3.2). (b) Liang et al. batch estimator on the 896-d
   representations (continuous, approximate). (c) Both, (a) deciding.
3. **ActionSense cost.** (a) Network null on the two straight-through
   grids only (argmax-bridge does not learn, ADR-0024; qualification
   kept). (b) All four grids. (c) Gaussian only for v0.7.1; ActionSense
   later. State the G × 30-cell cost before choosing.
4. **Multiplicity.** (a) Holm FWER 0.01 on F = {P1, DEC_S, ΔR, ΔU₂, ΔS}.
   (b) Separate α = 0.01 per co-primary (no correction, declared). (c)
   Fixed-sequence: P1, then DEC_S, then PID.
5. **Effect sizes of the planted controls.** δ_dec (accuracy points) and
   ε (R→U₂ transfer). (a) Set from the pilot so that power is 0.80 at
   the registered G (PR #66 §10 rule). (b) Fixed a priori by the PI.
6. **Physical-world positive control.** Who writes the world spec and
   its threshold. (a) Threshold from simulation power on the decoupled
   null, not from animal figures. (b) Adopt 0.65 matching (the review's
   H9 proposal) after the PI verifies the literature of §4.3. (c)
   Postpone the control; v0.7.1 cross-modal claims then stay « non
   concluant (contrôle positif absent) ».
7. **B-2 reading rules of §1.3** (issue #85). (a) Adopt both rules. (b)
   Adopt rule 1 only. (c) Leave as in ADR-0019 amendment.
8. **Equivalence.** (a) No TOST in v0.7.1 (non-rejection = non
   concluant). (b) TOST on co-primaries with margin = smallest detected
   planted effect (§5.1). (c) Margin = half the recorded thresholds.
9. **Pointer lines.** (a) Add the pointer lines of §7 (this PR). (b)
   Keep the frozen v0.7 file byte-identical and carry the pointer only
   in the erratum folder and ADR-0027. Note: the v0.7 file hash is in
   `docs/prereg/actionsense-v0.7/freeze-manifest.md`
   (`f20be89e…`); adding a line changes the working-tree hash, not the
   tagged blob.
10. **Learning gate for the reader** (ADR-0019 amendment: majority +
    0.10) in addition to the three P1 criteria. (a) Both. (b) P1
    criteria only.

## 7. L1 → L3 procedure and exclusions

### 7.1 Procedure

1. **L0 (this PR):** draft, not frozen. The PR is reviewed by an
   independent context-free critic before merge (PM §8); only the
   researcher merges.
2. Close §6 (date and value written in a dated decisions block), then
   the roadmap reading lock for v0.7.1 (reading note, contradictor,
   three tests; S8). **No L1 tag before both.**
3. **L1:** manifest of protocol files + code SHA-256 in
   `docs/prereg/<study>/<stage>-manifest.md`; new annotated tag by the
   PI (name proposed by the PI; the lab pattern is
   `prereg-<name>-<date>`); Software Heritage save, SWHID of the
   **revision** (`swh:1:rev:…`) recorded. PR #66 uses two tags: Stage A
   (protocol, pilot plan, S0–S4, power rule) before the pilot and Stage
   B (pilot record, G, SEEDS, manifest) before the first confirmatory
   job; this amendment follows that two-stage rule.
4. Tag-before-data check: earliest run artefact later than the tagger
   date; earlier runs are excluded or declared.
5. **L2:** e-Soleau or restricted Zenodo deposit, same day (PI).
6. **L3:** OSF Registrations or Octopus (PI only, credentials), only
   with a calibrated null (size check and positive control of PR #66 §7
   and §10 passed) and after the lock. Never an OSF badge before filing.
   A confirmation commit then references the OSF id.
7. Pointer lines (this PR): at the top of
   `docs/osf/amendment-v0.7-actionsense.md` and in
   `docs/adr/0027-erratum-world-seed-train-lesion-eval.md`: « amended
   by `docs/osf/amendment-2026-11-08-v0.7.1-draft.md` (L0 draft) ».
   ADR-0025 does not exist on main (reserved for the ActionSense
   verdicts, `docs/adr/README.md`): the pointer is added when it is
   written.

### 7.2 Runs excluded or not deciding

- Any run started before the new L1 tag: excluded from the confirmatory
  set, or declared as a deviation citing the pushed lock commit.
- ActionSense grids on seeds 300–304 under v0.7 §4.1: none has run
  (erratum 2026-10-31); if any runs before the tag it is reported as
  registered and **does not decide asymmetry**.
- Seeds 300–309 Gaussian straight-through grid (2026-10-27): used to
  write predictions, never confirmatory data.
- Seeds 400–404 (ADR-0026 grids, both arms, Gaussian and mock), mock
  pilots 100–101, D1–D4 reprobe seeds 5 106 000–5 106 045, grids 0–4 and
  200–204, smoke 999, N12 seeds: seen data, not units.
- Seeds 120–121 and 500–504: free as identifiers, not independent units
  (PR #66 §3.1); not used.
- Block-length pilot (v0.7 §8 item 3, seeds 100–101): whether it ran is
  not verifiable from this clone; if it ran it is exploratory.
- The stratified B-2 runs of PR #78 (0 of 10 seeds pass the gate) and
  their paired nulls: exploratory, not confirmatory, not pooled.
- `legacy` arm, mock world and argmax-bridge emission for P1 (PR #66
  decisions 9, 10): not run as confirmatory.
- Unit seeds g = 6…19 of PR #66: never used.

## 8. What was not verified

- Code for PID, probe-on-concatenation and per-item predicted classes
  does not exist; every estimator statement is a design, not a result.
- Chick, macaque and dog results and the Fort & Schwartz recipe: not
  checked against a primary source (§4.3).
- The venue « NeurIPS 2023 » of arXiv:2302.12247 and the existence of
  `pliang279/PID`: taken from the lab review, not rechecked.
- PR #66 numbers (0.748 / 0.981, seed inventory) are quoted from that
  draft and from the results milestone; not recomputed.
- Whether the v0.7 block-length pilot ran on any host.
- The tag `bouba_sens/v0.7-actionsense-prereg` and the freeze-manifest
  hash were read from the repository text, not from the tag object.
