# External reviewer critique — bouba_sens

> Status: external review (deep-research harness), 2026-06-03. Leads to confirm.
> PARTIAL: the harness read ADR-0017/0018 + CLAUDE.md only; the actual draft
> (docs/paper/paper-v0.1-draft.md) and ADR chain 0009–0018 were not fully read.
> Target venue under review: TMLR.

## Verdict
The **self-retraction IS the genuine contribution** (a methods paper) — exemplary
critical-validation discipline. But one methodology recommendation is itself
under-derived.

## ① Novelty / grounding
- T1/T2 congenital-vs-late-acquired is **well grounded** in human neuroscience
  (Collignon 2013, Pant/Bedny 2022, Voss 2013) — but **motivational, not validating**;
  the single scalar **Me7 undercaptures connectivity**.
- **Prior-art threat: MultiBench** (arXiv 2107.07502, 15 datasets / 10 modalities /
  20 tasks) already standardizes modality robustness (noisy/missing). The SNR-degradation
  lesion protocol must justify what it adds.

## ② Soundness — the retraction is correct
- B-2 = 0.000 exactly across 5 seeds (Kraskov k-NN artefact); B-1 = 10× below threshold,
  sign-unstable; B-3 = partition exhausted on n=5 (72% ceiling). Diagnoses are sound.
- 🔴 The **"probe batch ≥128" recommendation is under-derived**: MI sample complexity
  scales with **dimensionality, not batch size** (N must be ~exponential in total
  dimensionality). "n≥128" is a heuristic, not a principled bound — the real criterion is
  dimensionality-dependent. MINE (Belghazi 2018) is consistent only asymptotically →
  supports the n=16 retraction but not the general recommendation.

## ③ Threats to validity
- The three methodology-first recommendations (n≥128, mandatory bootstrap CIs,
  null-model partition controls) are sound in spirit but **not all well-founded as stated**
  (see ②); generalizability beyond this benchmark is asserted, not shown.

## Open questions
1. Is B-2 a network result or purely the Kraskov artefact? Run MINE at n≥128.
2. n≥128 vs the actual probe dimensionality — restate the criterion dimensionally.
3. Differentiation from MultiBench (scope/breadth) for a TMLR audience.
