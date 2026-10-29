# ActionSense World Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve ActionSense (S00, S02–S05) as a new real-data world `ActionSenseWorld` with the unchanged six-tensor contract, extracted reproducibly, then pre-register and run the v0.7 grids.

**Architecture:** Pure, network-free reduction functions live in `src/bouba_sens/world/actionsense_features.py` and carry all unit tests. Two thin scripts orchestrate I/O: `scripts/fetch_actionsense.py` downloads and hashes the raw files, and `scripts/extract_actionsense.py` turns them into tensors. `ActionSenseWorld` mirrors `StudyforrestRealWorld` and is registered in `cli._build_world`.

**Tech Stack:** Python 3.14, uv, torch, numpy, scipy (`signal.stft`, `signal.butter`/`sosfiltfilt`), soundfile, Pillow (all already in `uv.lock`), **h5py (new, HITL)**, system `ffmpeg`/`ffprobe` via subprocess.

**Spec:** `docs/superpowers/specs/2026-10-28-actionsense-world-design.md`

## Global Constraints

- World contract: `audio (N,128)`, `vision (N,16,16)`, `tactile (N,32)`, `gravity (N,3)`, `force (N,6)`, `labels (N,)` long with 4 classes; extra `subject.pt (N,)` long, ignored by the world contract.
- Common grid 10 Hz on the wearable clock (`time_s`, epoch seconds); frame *t* aggregates [*t*, *t* + 0.1 s).
- Subjects: exactly S00, S02, S03, S04, S05, concatenated in that order.
- Thresholds B-1 ≥ 0.05, B-2 ≥ 0.10, B-3 ≥ 0.02 and Me1/Me3/Me6/Me7 are not touched.
- Data licence CC BY-NC-SA 4.0 for raw and derived tensors, attribution DelPreto et al. 2022. Code stays Apache-2.0. `data/` stays gitignored.
- No raw frame and no waveform leaves the extraction.
- No grid runs before amendment v0.7 is committed and tagged.
- New dependency `h5py`: added only after the PI reviews the diff, pinned in `uv.lock` (hash), and recorded in the dependency audit spec.
- Zero compute on GrosMac. Unit tests run on electron-server, and pilot, extraction and grids run on macM1.
- Commits: subject ≤ 50 chars, body lines ≤ 72, English, no AI attribution, message written to a file and passed with `git commit -F`.
- Hypneum stays on GitHub: PRs on `hypneum-lab/bouba_sens`, merge only on green CI.

## Review Focus

1. **Activity rows that never close.** A `Start` row without a matching `Stop` before the next `Start` of the same label, or rows marked `Bad`. The segment is skipped and counted, and the run does not crash (Task 2).
2. **Stream gaps.** A wearable stream has a gap inside a labelled segment. Gaps ≤ 200 ms are interpolated, and frames inside longer gaps are dropped and counted (Task 2).
3. **Media outside the wearable span.** The video or audio starts after, or ends before, a labelled segment. Frames without media coverage are dropped, never padded (Task 3).
4. **Unknown activity labels.** The extraction finds a label string that is not in the family table. The label is dropped and listed in the manifest, and the extraction does not guess (Task 2).
5. **Constant channels in z-scoring.** A channel has zero variance for one subject, such as a dead taxel. Its z-scored value is 0, not NaN (Task 2).

---

### Task 1: `h5py` dependency gate

**Files:**
- Modify: `pyproject.toml` (dependencies), `uv.lock`
- The dependency inventory spec lives outside this repo, so the PR body carries a one-line inventory note: package, version and wheel hashes

- [ ] **Step 1:** On electron-server, run `uv add 'h5py>=3.12'` in a scratch clone and capture the `uv.lock` diff: version, wheel hashes and any transitive packages.
- [ ] **Step 2:** **STOP.** Post the diff in the PR (draft) and wait for explicit PI approval. Do not continue to Task 2 until the PI approves.
- [ ] **Step 3:** Commit `pyproject.toml` and `uv.lock`, with subject `build(deps): add h5py for ActionSense HDF5`.

### Task 2: Pure reduction functions

**Files:**
- Create: `src/bouba_sens/world/actionsense_features.py`
- Test: `tests/unit/test_actionsense_features.py`

**Interfaces:**
- Produces, with numpy arrays throughout:
  - `FAMILY_RULES: tuple[tuple[str, int], ...]`, made of lower-case substring to family id pairs, checked in order:
    - `("peel", 0)`, `("slic", 0)`;
    - `("spread", 1)`, `("jar", 1)`;
    - `("clean", 2)`, `("wipe", 2)`;
    - `("pour", 3)`, `("set table", 3)`, `("dishwasher", 3)`.
  - `label_family(label: str) -> int | None`
  - `activity_segments(rows: list[tuple[str, str, str]], times: np.ndarray) -> tuple[list[tuple[float, float, int]], dict[str, int]]`. Rows are `(label, "Start"|"Stop", "Good"|"Bad"|…)`, decoded as in ActionNet `parsing_data/example_parse_hdf5_file.py`. It returns `(t_start, t_end, family)` segments and a counter dict with keys `unclosed`, `bad` and `unmapped:<label>`.
  - `frame_grid(segments, hz: float = 10.0) -> tuple[np.ndarray, np.ndarray]`, which returns frame start times `(N,)` and labels `(N,)`.
  - `window_mean(t: np.ndarray, x: np.ndarray, starts: np.ndarray, width: float = 0.1, max_gap: float = 0.2) -> tuple[np.ndarray, np.ndarray]`. It returns per-frame means `(N, C)` and a boolean `valid (N,)`. A frame is valid if it has ≥ 1 sample, or if the gap between the neighbouring samples is ≤ `max_gap`, in which case it is linearly interpolated.
  - `window_rms(...)`, with the same signature as `window_mean`.
  - `pool_tactile(frames: np.ndarray) -> np.ndarray`, which maps `(N, 32, 32)` to `(N, 16)`: it averages 8×8 blocks into a 4×4 map, then flattens it. The extraction concatenates left then right into 32.
  - `emg_sectors(rms: np.ndarray) -> np.ndarray`, which maps `(N, 8)` to `(N, 3)` as the means of channels `[0:3]`, `[3:6]` and `[6:8]`.
  - `lowpass(x: np.ndarray, hz: float = 10.0, cutoff: float = 0.5) -> np.ndarray`, a 2nd-order Butterworth through `sosfiltfilt` along axis 0.
  - `zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, which returns `(z, mean, std)` and gives `z = 0` where `std < 1e-8`.
  - `check_aligned(tensors: dict[str, np.ndarray]) -> int`, which returns the common first dimension, or raises `ValueError` naming every modality and its length.

- [ ] **Step 1: Write the failing tests**

```python
def test_label_family_maps_and_rejects():
    assert label_family("Slice a cucumber") == 0
    assert label_family("Open/close a jar of almond butter") == 1
    assert label_family("Clean a plate with a towel") == 2
    assert label_family("Pour water from a pitcher into a glass") == 3
    assert label_family("Get items from refrigerator") is None

def test_activity_segments_counts_unclosed_and_bad():
    rows = [("Slice a potato", "Start", "Good"), ("Slice a potato", "Stop", "Good"),
            ("Peel a potato", "Start", "Good"),  # never stopped
            ("Pour water", "Start", "Bad"), ("Pour water", "Stop", "Bad"),
            ("Get items", "Start", "Good"), ("Get items", "Stop", "Good")]
    segs, counts = activity_segments(rows, np.arange(7.0))
    assert segs == [(0.0, 1.0, 0)]
    assert counts["unclosed"] == 1 and counts["bad"] == 1
    assert counts["unmapped:Get items"] == 1

def test_frame_grid_10hz():
    starts, labels = frame_grid([(0.0, 0.35, 2)])
    np.testing.assert_allclose(starts, [0.0, 0.1, 0.2])  # last full window ends at 0.3
    assert labels.tolist() == [2, 2, 2]

def test_window_mean_interpolates_short_gap_and_drops_long():
    t = np.array([0.0, 0.05, 0.25, 0.9]); x = np.array([[0.0], [1.0], [3.0], [9.0]])
    m, valid = window_mean(t, x, np.array([0.0, 0.1, 0.5]))
    assert valid.tolist() == [True, True, False]
    assert m[0, 0] == 0.5 and 1.0 < m[1, 0] < 3.0

def test_pool_tactile_and_emg_sectors_shapes():
    assert pool_tactile(np.ones((5, 32, 32))).shape == (5, 16)
    out = emg_sectors(np.arange(8.0)[None].repeat(2, 0))
    np.testing.assert_allclose(out[0], [1.0, 4.0, 6.5])

def test_zscore_constant_channel_is_zero():
    z, mu, sd = zscore(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert np.isfinite(z).all() and (z[:, 1] == 0).all()

def test_check_aligned_names_mismatch():
    with pytest.raises(ValueError, match=r"vision=120"):
        check_aligned({"audio": np.zeros((1800, 128)), "vision": np.zeros((120, 16, 16))})
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/test_actionsense_features.py -v` on electron-server. Expected: FAIL with an ImportError.
- [ ] **Step 3:** Implement the module with the signatures above. If the #45 close-out PR has merged, `check_aligned` reuses its length check rather than duplicating it.
- [ ] **Step 4:** Run the same command. Expected: 7 passed. Also run `uv run ruff check src tests`, `uv run ruff format --check src tests` and the mypy command of `.github/workflows/ci.yml`. Expected: clean.
- [ ] **Step 5:** Commit with subject `feat(world): add ActionSense reduction helpers`.

### Task 3: Fetch and extract scripts

**Files:**
- Create: `scripts/fetch_actionsense.py`, `scripts/extract_actionsense.py`
- Test: `tests/unit/test_extract_actionsense.py`, which uses a tiny synthetic HDF5 written with h5py plus a 1 s sine WAV, and no video

**Interfaces:**
- Consumes: everything from Task 2.
- Produces:
  - `fetch_actionsense.py --subjects S00 S02 S03 S04 S05 --out data/actionsense_raw`. It resolves the per-subject links from `https://data.csail.mit.edu/ActionNet/` (the wearable HDF5, `*_eye-tracking-video-world_frame.mp4`, the sink WAV and its timing HDF5). It `HEAD`s every file first, aborts if free disk minus total size < 20 GB, then downloads with resume and writes `MANIFEST.json` with `{url, bytes, sha256}` per file. With `--verify` it re-hashes and exits 1 on mismatch.
  - `extract_actionsense.py --raw data/actionsense_raw --out data/actionsense --subjects …`. It writes `{audio,vision,tactile,gravity,force,labels,subject}.pt` (float32 and long) and `MANIFEST.md`. The manifest holds the per-file sha256, the per-subject z-score stats, the family table with the raw label strings seen, the counts from Task 2 and Review Focus 3, the class frequencies, the majority baseline and the licence notice.
  - `extract_subject(raw_dir: Path, subject: str) -> dict[str, np.ndarray]`, importable for tests. It reads the streams `tactile-glove-{left,right}/tactile_data`, `myo-{left,right}/emg`, `myo-right/acceleration_g` and `experiment-activities/activities`.
  - Audio comes from the sink WAV, cut per frame through its chunk timestamps, and reduced to a 128-bin log-mel with a 0.1 s window. Reuse `mel_filterbank` from `scripts/extract_ccfilm_features.py` by moving it into `actionsense_features.py` or a shared helper, without duplicating it.
  - Vision takes the world-video frame nearest to each frame start. Timestamps come from `ffprobe -show_frames` against the eye-tracking stream's `time_s`. Frames are decoded with `ffmpeg` into a `tempfile.TemporaryDirectory` and reduced with the `frame_to_grid` logic (16×16 mean intensity).
  - Frames where any modality is invalid, or where the media does not cover them, are dropped and counted.

- [ ] **Step 1: Write the failing test.** `test_extract_subject_synthetic` builds an HDF5 with 2 s of tactile at 30 Hz, EMG at 200 Hz and accelerometer at 50 Hz, one `Slice` Start/Stop over [0.2, 1.2], and a WAV, and passes `vision=False`. It asserts that tactile has shape `(10, 16)`, force `(10, 6)`, gravity `(10, 3)` and audio `(10, 128)`, and that every label is 0. A second test, `test_extract_aborts_on_misalignment`, asserts that a forced length mismatch exits non-zero.
- [ ] **Step 2:** Run the test. Expected: FAIL.
- [ ] **Step 3:** Implement both scripts. Add a `vision: bool = True` parameter to `extract_subject` so the unit test can skip ffmpeg.
- [ ] **Step 4:** Run the test. Expected: PASS. Then run lint and mypy. Expected: clean.
- [ ] **Step 5:** Commit with subject `feat(scripts): fetch and extract ActionSense`.

### Task 4: `ActionSenseWorld` and CLI registration

**Files:**
- Create: `src/bouba_sens/world/actionsense.py`
- Modify: `src/bouba_sens/cli.py`. Extend `_build_world` (around lines 102–142) and the error message, and add `actionsense` to the `--world` help strings at the three `typer.Option("gaussian", …)` sites.
- Test: `tests/unit/test_actionsense_world.py`, modelled on `tests/unit/test_ccfilm_world.py`

**Interfaces:**
- Produces:
  - `ActionSenseWorld(*, seed: int = 0, data_dir: Path)` with `mode == "actionsense"`, and with `sample`, `sample_with_indices`, `gather`, `dataset_labels` and `modality_dims` identical in behaviour to `StudyforrestRealWorld`.
  - The constructor raises `ValueError` through `check_aligned` if the six tensors differ in length.
  - `_build_world("actionsense", seed)` reads `BOUBA_SENS_ACTIONSENSE_DATA`, with default `data/actionsense`. It raises `typer.BadParameter("actionsense cache missing at …; run scripts/extract_actionsense.py first")` if `audio.pt` is absent.

- [ ] **Step 1: Write the failing tests.**
  - `test_actionsense_mode`.
  - `test_actionsense_sample_shapes`, with the same asserts as the CC-film test.
  - `test_actionsense_rejects_misaligned`: `vision.pt` of 120 rows against 256 others raises `ValueError` matching `vision=120`.
  - `test_build_world_actionsense_missing_cache`: `monkeypatch` the env var to an empty tmp dir, and `_build_world` raises `typer.BadParameter`.
- [ ] **Step 2:** Run `uv run pytest tests/unit/test_actionsense_world.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the world and the CLI branch.
- [ ] **Step 4:** Run the test file. Expected: 4 passed. Then run the full `uv run pytest tests/unit tests/smoke` in the background on electron-server (about 30 min). Expected: the previous count plus the new tests, with 0 failed.
- [ ] **Step 5:** Commit with subject `feat(world): add ActionSenseWorld`. Open the PR with Tasks 1–4. Merge only on green CI and after the PI approves the h5py diff.

### Task 5: Pilot on S00, the go/no-go gate (macM1)

**Files:**
- Create: `scripts/pilot_actionsense.sh`
- Create: `docs/pilot/actionsense-s00.md`, which holds the report and the figure `docs/pilot/actionsense-s00-xcorr.png`

- [ ] **Step 1:** On macM1, in `~/hypneum-lab/actionsense/`, run `fetch_actionsense.py --subjects S00`. Record the sizes and sha256 from `MANIFEST.json`.
- [ ] **Step 2:** Run the extraction twice into `out_a/` and `out_b/`. **Check 4:** `shasum -a 256 out_a/*.pt out_b/*.pt` gives pairwise identical hashes.
- [ ] **Step 3:** **Check 1:** the extraction did not abort. **Check 2:** the manifest reports that the media spans cover the labelled segments and that frame timestamps are monotonic.
- [ ] **Step 4:** **Check 3:** on the slicing frames of `out_a`, cross-correlate the onsets of `tactile.sum(1)` with the onsets of the audio energy over ±1 s. The peak lag must be within ±200 ms. Save the figure.
- [ ] **Step 5:** **Check 5:** with `BOUBA_SENS_ACTIONSENSE_DATA=out_a`, run `train --world actionsense --steps 20`, then `lesion` T2 for 20 steps, then `eval`, twice. The four sha256 of the #35 test must be identical across the two runs.
- [ ] **Step 6:** Write `docs/pilot/actionsense-s00.md` with the five checks, PASS or FAIL each, with numbers. **STOP and report to the PI.** Any FAIL means no-go: no Task 6.
- [ ] **Step 7:** Commit with subject `docs(pilot): ActionSense S00 gate report`.

### Task 6: Amendment v0.7 (before any grid)

**Files:**
- Create: `docs/osf/amendment-v0.7-actionsense.md`, in the format of `amendment-v0.6-sprint10-17.md`. It carries a status note, the parent DOI `10.17605/OSF.IO/Q6JYN` and the previous amendment v0.6.

- [ ] **Step 1:** Write the amendment, covering:
  - the world, the subjects, the modality mapping, the four families and the 10 Hz grid;
  - the two grids, `nolock` and `LOCK_AFTER=200`, each 5 seeds × 5 modalities × T1/T2 × 3 SNR floors = 150 cells;
  - the unchanged thresholds and metrics, with B-3 reported only together with the paired null of ADR-0022;
  - that Studyforrest numbers are described, not used as a replication target;
  - that dose-response and the Sprint 11–14 blocks are out of scope;
  - the pilot report reference.
- [ ] **Step 2:** **STOP for PI sign-off** on the text.
- [ ] **Step 3:** Commit with subject `docs(osf): amendment v0.7 ActionSense`, then `git tag bouba_sens/v0.7-actionsense-prereg` and push the tag **before** Task 7.

### Task 7: Full extraction, publication, grids (macM1)

**Files:**
- Create: `docs/adr/0025-actionsense-world-verdicts.md`
- Modify: `scripts/reproduce_paper_v01.sh`, adding a `CHOOSE=v0.7_actionsense` block that runs both grids with `WORLD=actionsense`, `run_grid.sh` and the existing aggregation, written to `reports/v0.7_actionsense_{nolock,lock}_aggregate.json`

- [ ] **Step 1:** Fetch and extract all five subjects, then run `fetch_actionsense.py --verify`.
- [ ] **Step 2:** Package the seven tensors, `MANIFEST.md` and a `LICENSE-DATA` file (CC BY-NC-SA 4.0 and the attribution) as `actionsense-tensors-v0.7.tar.gz`. Attach it to a GitHub release with `gh release upload`, and put its sha256 in the ADR.
- [ ] **Step 3:** Run both grids and the paired null of ADR-0022 on macM1, in the background with nice, logging to `~/hypneum-lab/actionsense/logs/`.
- [ ] **Step 4:** Write ADR-0025 with the verdicts for B-1, B-2 and B-3 (with the null), the provenance (commit, tag, tensor sha256) and the class baseline. Commit the aggregates in `reports/`.
- [ ] **Step 5:** Commit with subject `docs(adr): ActionSense v0.7 verdicts`. Open the PR and merge it on green CI.

Paper section M5 is a separate follow-up after the PI reads ADR-0025. It is not part of this plan.
