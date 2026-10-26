# Code-channel collapse — diagnosis (2026-10-26)

**Status:** diagnostic note, exploratory. No library default changed.
**Requested by:** PI decision 2026-10-26 (point 3), before any lesion ×
query interaction test.
**Script:** `scripts/diag_code_collapse.py`. **Data:**
`reports/diag_code_collapse_{default,default_sgd,centred,st}.json`.
**Host:** electron-server, nice 19, about 2 min per variant.

## What was measured

Phase 1 is replayed exactly as `bouba-sens train` runs it (200 steps,
batch 16, hard gating, `_seed_everything`), for seeds 0 and 1 on the
Gaussian and Studyforrest-mock worlds. Every 10 steps, on a fixed clean
batch of 256 items (global RNG saved and restored, so the trajectory is
unchanged), the script records:

- per modality: active argmax codes out of 64, code-usage entropy, top-1
  logit margin, and the offset ratio ‖mean h‖ / mean ‖h − mean h‖ of
  the hidden state;
- the gradient norm that reaches `emit_head_pi` and the encoder;
- training loss and full-network accuracy.

The averages below are over seeds and over live modalities (all five
for the Gaussian world; audio and vision for the mock, whose three other
inputs are zero).

| variant | world | active codes, step 0 → 20 → 200 | entropy, bits (0 → 200) | logit margin (0 → 200) | offset ratio (0 → 200) | ‖∇ emit_head_pi‖ | loss, first → last 20 steps |
|---|---|---|---|---|---|---:|---|
| **default** | Gaussian | 12.7 → 7.5 → **3.7** | 2.57 → 0.61 | 0.05 → **17.5** | 1.6 → 3.8 | **1.6e-6** | 1.388 → **1.385** |
| **default** | mock | 10.0 → 3.0 → **1.25** | 2.19 → 0.21 | 0.04 → **48.1** | 1.6 → 2.8 | **2.0e-6** | 1.391 → **1.390** |
| default, SGD on the sensory pathways | Gaussian | 12.7 → 12.7 → 12.7 | 2.57 → 2.57 | 0.05 → 0.05 | 1.6 → 1.6 | 1.0e-6 | 1.385 → 1.376 |
| default, SGD on the sensory pathways | mock | 10.0 → 10.0 → 10.0 | 2.19 → 2.19 | 0.04 → 0.04 | 1.6 → 1.6 | 1.9e-6 | 1.389 → 1.363 |
| centred h before the argmax | Gaussian | 12.7 → 8.9 → 3.7 | 2.57 → 0.59 | 0.05 → 22.5 | 1.6 → 2.3 | 2.1e-6 | 1.383 → 1.358 |
| **straight-through emission** | Gaussian | 12.7 → 6.1 → 3.3 | 2.57 → 0.88 | 0.05 → 20.0 | 1.6 → 2.4 | **2.3e-2** | 1.376 → **0.963** |
| **straight-through emission** | mock | 10.0 → 3.0 → 2.25 | 2.19 → 0.71 | 0.04 → 93.5 | 1.6 → 1.2 | **5.0e-3** | 1.380 → **1.276** |

Other facts:

- At initialisation, with the hidden state centred, 50 (Gaussian) and 34
  (mock) codes would be active instead of 12.7 and 10. A shared positive
  offset of the ReLU-MLP hidden state already restricts the argmax
  before training starts.
- On the default trajectory the training loss stays at ln 4 = 1.386,
  i.e. chance, throughout Phase 1 on both worlds. The network learns
  nothing about the label.

## Probable cause

1. **The argmax blocks the task gradient.** `SensoryWML.step` sends one
   code per item, `argmax(emit_head_pi(h))`. The only gradient path back
   to the sensory pathway is the bridge `1e-3 · mean(softmax(π) @
   constellation)`. That term is a single scalar per item, added at the
   same value over the whole carrier, and it carries almost no label
   information. Its gradient on `emit_head_pi` is about 1e-6.
2. **Adam turns that gradient into full-size steps.** Adam's update is
   invariant to gradient scale, so the 1e-6 bridge gradients produce
   steps of about lr = 1e-3 per parameter, in a direction that does not
   serve the task. The hidden-state offset and the logit margin then
   grow (margin 0.05 → 17–48 within about 40 steps). Every input falls
   into the same few codes, and in the mock world into a single code.
   The SGD counterfactual confirms the mechanism: with plain SGD on the
   sensory pathways (same learning rate) nothing moves, and the
   initial 10–13 codes are kept, but the network still learns nothing.
3. **Collapse is the symptom; the absence of a task gradient is the
   cause.** With a straight-through emission (forward = carrier of the
   argmax code, backward = softmax gradient), the number of active codes
   is just as small (2–3), but the codes become informative. The loss
   falls to 0.96 (Gaussian) and 1.28 (mock), and full-network accuracy
   reaches 0.54 and 0.45 on the evaluation batch. Centring h alone does
   not fix it.

## Proposed fix (not applied; defaults unchanged)

- Add an **opt-in emission mode** to `SensoryWML`, `--emission
  {argmax-bridge (default), straight-through}`. The straight-through
  mode sends the argmax code forward and back-propagates through the
  softmax over the 64 codes (the `st` variant of the script). Keeping
  the default bit-identical preserves every past result.
- Validate it on a pilot before any reading: Phase-1 loss clearly below
  ln 4, full-network accuracy and probe v2 above chance, and code usage
  reported. A code-usage entropy regulariser and longer training are
  secondary options to try if few codes limit capacity.

## Consequence for the lesion × query interaction test (PI point 2)

With the default emission the network never learns the task. Phase-1
loss stays at chance, the head is at chance with all five inputs, and
the sensory states change only through optimiser drift that carries no
label information. A lesion × query interaction measured on these
states, in grids built with the same defaults, would measure how that
drift reacts to the noise injected into one input, not cross-modal
reorganisation of a system that performs the task. **The test is
therefore uninterpretable as a test of B-3 and is not run.** Its
preregistration is drafted but neither tagged nor executed
(`docs/prereg/2026-10-26-b3-lesion-interaction.md`, separate branch). It
would become meaningful on grids trained with a fix that passes the
pilot above.
