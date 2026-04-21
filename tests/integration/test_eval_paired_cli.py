"""ADR-0007 — CLI paired-run wiring (`eval --t1-ckpt/--t2-ckpt`).

Three tests :
  (a) schema conformance of the emitted JSON vs ADR-0007:48-63
  (b) equivalence to the probe-observable Me7 within float64 epsilon
      (ADR-0007 Phase B Option 4) — paired-run CLI vs Me7 derived from
      ``me1_probe`` fields of the two cell ``eval_report.json`` files.
      Both paths invoke ``_compute_me1`` / ``query_accuracy``, so
      divergence is bounded by reload determinism. NOT tested against
      the legacy ``me1`` (primary observable, semantically distinct).
  (c) ADR-0007 Phase B — `_load_cell` is deterministic bit-exact: two
      successive invocations on the same cell directory return identical
      floats (rebuild + probe pass keyed by seed).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bouba_sens.cli import app


def _build_real_cell(
    cell_dir: Path,
    *,
    seed: int,
    timing: str,
    modality: str,
    snr: str,
    world: str = "gaussian",
    steps: int = 3,
    tmp_root: Path | None = None,
) -> float:
    """Build a cell by invoking the OFFICIAL CLI pipeline (train -> lesion -> eval).

    ADR-0007 Phase B Option 4 — the official ``eval --run`` command now
    emits BOTH ``me1`` (legacy ``me1_accuracy(report.pkl)``, primary
    paper observable) AND ``me1_probe`` (canonical probe pass from the
    reloaded ``model.pt``). The equivalence test (b) uses the
    probe-observable ``me1_probe`` on both paths -- so the |Δ| < 1e-12
    assertion is a determinism bound, not a cross-observable comparison.

    Returns the ``me1_probe`` value emitted by ``eval --run`` for this
    cell -- the probe-observable anchor compared in test (b).
    """
    runner = CliRunner()
    cell_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = tmp_root if tmp_root is not None else cell_dir.parent

    timing_up = timing.upper()

    if timing_up == "T1":
        # T1 congenital: lesion without prior pretrain checkpoint.
        result = runner.invoke(
            app,
            [
                "lesion",
                "--timing",
                "T1",
                "--modality",
                modality,
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--world",
                world,
                "--out",
                str(cell_dir),
            ],
        )
        assert result.exit_code == 0, f"lesion T1 failed: {result.stdout}"
    elif timing_up == "T2":
        # T2 late-acquired: pretrain first, then lesion from that checkpoint.
        t1_scratch = scratch_root / f"_pretrain_{cell_dir.name}"
        train_result = runner.invoke(
            app,
            [
                "train",
                "--seed",
                str(seed),
                "--steps",
                str(steps),
                "--batch-size",
                "16",
                "--world",
                world,
                "--out",
                str(t1_scratch),
            ],
        )
        assert train_result.exit_code == 0, f"train failed: {train_result.stdout}"
        result = runner.invoke(
            app,
            [
                "lesion",
                "--ckpt",
                str(t1_scratch),
                "--timing",
                "T2",
                "--modality",
                modality,
                "--steps",
                str(steps),
                "--seed",
                str(seed),
                "--world",
                world,
                "--out",
                str(cell_dir),
            ],
        )
        assert result.exit_code == 0, f"lesion T2 failed: {result.stdout}"
    else:
        raise ValueError(f"unknown timing {timing!r}")

    # Phase A artefacts must be present after lesion.
    assert (cell_dir / "model.pt").exists(), "Phase A: model.pt missing"
    assert (cell_dir / "config.yaml").exists(), "Phase A: config.yaml missing"
    assert (cell_dir / "report.pkl").exists(), "lesion: report.pkl missing"

    # Official eval --run path: emits eval_report.json with
    # me1 = me1_accuracy(report.pkl). This is the aggregator (Path A) source.
    eval_out = cell_dir / "eval_report.json"
    eval_result = runner.invoke(
        app,
        ["eval", "--run", str(cell_dir), "--out", str(eval_out)],
    )
    assert eval_result.exit_code == 0, f"eval --run failed: {eval_result.stdout}"
    assert eval_out.exists(), "eval --run did not write eval_report.json"

    # Overwrite snr in metadata.json so the aggregator regex
    # ({floor,minus10,plus10}) and _iter_cell_records pick up this cell
    # correctly. Lesion writes a numeric `snr_init`/`snr_floor` pair, not a
    # label; the label lives on the directory name, which the aggregator
    # parses itself -- but our fixture also wants metadata.snr for
    # _seed_cell-shaped callers. Keep it consistent.
    meta_path = cell_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["snr"] = snr
    meta["timing"] = timing_up
    meta_path.write_text(json.dumps(meta, indent=2))

    payload = json.loads(eval_out.read_text())
    # Option 4 — return the probe observable; the equivalence test (b)
    # compares CLI-side Me7 against an Me7 derived from me1_probe pairs.
    me1_probe = payload.get("me1_probe")
    assert me1_probe is not None, (
        "eval --run must emit me1_probe (Phase B Option 4) when model.pt "
        "+ config.yaml are present; got null"
    )
    return float(me1_probe)


def _seed_cell(
    cell_dir: Path, *, me1: float, seed: int, timing: str, modality: str, snr: str
) -> None:
    """Fabricate a minimal cell directory matching the Sprint 4+ format.

    Writes `eval_report.json` (consumed by the aggregator) + `metadata.json`
    (source of truth for seed / modality / timing / snr).
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "eval_report.json").write_text(json.dumps({"me1": me1}))
    (cell_dir / "metadata.json").write_text(
        json.dumps(
            {
                "phase": "lesion",
                "seed": seed,
                "timing": timing,
                "modality": modality,
                "snr": snr,
            }
        )
    )


def test_eval_paired_cli_emits_expected_schema(tmp_path: Path) -> None:
    """(a) ADR-0007 §Output schema — CLI JSON payload shape."""
    t1_dir = tmp_path / "seed0_T1_vision_plus10"
    t2_dir = tmp_path / "seed1_T2_vision_plus10"
    _seed_cell(t1_dir, me1=0.71, seed=0, timing="T1", modality="vision", snr="plus10")
    _seed_cell(t2_dir, me1=0.68, seed=1, timing="T2", modality="vision", snr="plus10")

    out_path = tmp_path / "paired.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "--t1-ckpt",
            str(t1_dir),
            "--t2-ckpt",
            str(t2_dir),
            "--modality",
            "vision",
            "--snr",
            "10.0",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads(out_path.read_text())

    # Top-level keys per ADR-0007:48-63.
    assert set(payload.keys()) == {"pair", "me1_t1", "me1_t2", "me7", "timestamp"}

    # pair.{seed_t1, seed_t2, cell_id, modality, snr}
    assert set(payload["pair"].keys()) == {"seed_t1", "seed_t2", "cell_id", "modality", "snr"}
    assert payload["pair"]["seed_t1"] == 0
    assert payload["pair"]["seed_t2"] == 1
    assert payload["pair"]["modality"] == "vision"
    assert payload["pair"]["snr"] == 10.0
    # cell_id comes from the cell dir names; ADR says nullable for ad-hoc.
    assert payload["pair"]["cell_id"] is not None

    # Numeric contract.
    assert payload["me1_t1"] == 0.71
    assert payload["me1_t2"] == 0.68
    assert abs(payload["me7"] - (0.71 - 0.68)) < 1e-12

    # Timestamp is ISO 8601 string.
    assert isinstance(payload["timestamp"], str)
    assert "T" in payload["timestamp"]


def test_cli_me7_equivalence_to_probe_aggregator(tmp_path: Path) -> None:
    """(b) ADR-0007 Phase B Option 4 — |me7_cli - me7_probe| < 1e-12.

    The paired-run CLI (``eval --t1-ckpt --t2-ckpt``) and the standalone
    ``eval --run`` command both invoke the same canonical probe pass
    (``_compute_me1`` → ``query_accuracy("audio", seed+777)``). This
    test wires the two paths and asserts they converge on the same
    Me7 float:

      * Path A (probe-aggregator): ``eval --run`` emits ``me1_probe``
        for each cell; we subtract them to obtain Me7.
      * Path B (CLI paired-run): ``eval --t1-ckpt --t2-ckpt`` loads both
        cells and emits Me7 directly.

    Equivalence is bounded by reload determinism only — both paths are
    functionally identical. The legacy ``me1`` field is NOT subject to
    this equivalence (it is a different observable: Phase 2 adaptation
    tail, not probe pass).
    """
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    t1_dir = grid_root / "seed0_T1_audio_floor"
    t2_dir = grid_root / "seed0_T2_audio_floor"
    # _build_real_cell returns me1_probe (Option 4 -- probe observable).
    me1_probe_t1 = _build_real_cell(
        t1_dir, seed=0, timing="T1", modality="audio", snr="floor", steps=3
    )
    me1_probe_t2 = _build_real_cell(
        t2_dir, seed=1, timing="T2", modality="audio", snr="floor", steps=3
    )

    # Path A — Me7 derived from the two me1_probe fields.
    me7_probe = me1_probe_t1 - me1_probe_t2

    # Path B — CLI paired-run.
    out_path = tmp_path / "paired.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "--t1-ckpt",
            str(t1_dir),
            "--t2-ckpt",
            str(t2_dir),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    me7_cli = json.loads(out_path.read_text())["me7"]

    # Strict equivalence by construction: both paths route through the
    # same ``_compute_me1`` function. Any residual delta is reload-
    # determinism noise, which torch.save/load + deterministic probe
    # seeding should drive to exactly zero.
    assert abs(me7_cli - me7_probe) < 1e-12, (
        f"me7_cli={me7_cli}, me7_probe={me7_probe}, |Δ|={abs(me7_cli - me7_probe)}"
    )


def test_load_cell_is_deterministic(tmp_path: Path) -> None:
    """(c) ADR-0007 Phase B — ``_load_cell`` is deterministic bit-exact.

    Two successive invocations on the same cell directory must return
    identical ``(me1, seed, cell_name)`` triples within float64 epsilon.
    The rebuild path goes through ``_build_model_from_config`` →
    ``torch.load`` → ``load_state_dict`` → ``_compute_me1`` (which
    invokes ``query_accuracy("audio", seed+777)``); this chain is
    expected to be deterministic given a fixed ``(config.yaml,
    model.pt, seed)`` triplet.

    Reformulated from the prior "canonical probe-pass" test: comparing
    ``_load_cell`` output against an ``eval_report.json`` ``me1`` field
    would cross the observable boundary (adaptation tail vs. probe
    pass), which is ADR-0007 Phase B Option 4's precise distinction.
    Determinism is the invariant ``_load_cell`` actually needs to
    guarantee for the paired-run CLI to have stable Me7 output.
    """
    from bouba_sens.cli import _load_cell

    cell_dir = tmp_path / "seed0_T1_audio_floor"
    _build_real_cell(cell_dir, seed=0, timing="T1", modality="audio", snr="floor", steps=3)

    me1_a, seed_a, name_a = _load_cell(cell_dir)
    me1_b, seed_b, name_b = _load_cell(cell_dir)

    # Contract: seed tracked from config.yaml, cell_name from directory.
    assert seed_a == 0 == seed_b
    assert name_a == cell_dir.name == name_b
    # Me1 value range.
    assert 0.0 <= me1_a <= 1.0
    # Determinism: two successive invocations agree bit-exact (modulo
    # float64 epsilon slack).
    assert abs(me1_a - me1_b) < 1e-12, (
        f"_load_cell non-deterministic: me1_a={me1_a}, me1_b={me1_b}, |Δ|={abs(me1_a - me1_b)}"
    )
