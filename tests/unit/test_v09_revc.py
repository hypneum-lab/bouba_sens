"""v0.9 rev. C (PI decision 2026-10-28): labels, purged split, label-break arm, outcome."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from bouba_sens import v09_revc as rc
from bouba_sens.decorrelation import DecorrelationWindow
from bouba_sens.seeds import BY_NAME

pytestmark = pytest.mark.usefixtures("deterministic_torch")

N = 3000


def _write(d: Path, *, studyforrest: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(N)
    on = ((t // 37) % 3 == 0).astype(np.float32)
    if studyforrest:
        base = np.stack([on, rng.random(N), rng.random(N), rng.random(N)], 1).astype(np.float32)
        proj = np.random.default_rng(1).standard_normal((4, 128)).astype(np.float32)
        audio = base @ proj + 0.05 * rng.standard_normal((N, 128)).astype(np.float32)
    else:
        audio = rng.standard_normal((N, 128)).astype(np.float32) + on[:, None]
    vision = rng.random((N, 16, 16)).astype(np.float32)
    vision += ((t // 11) % 2)[:, None, None].astype(np.float32)
    force = rng.standard_normal((N, 6)).astype(np.float32)
    for name, arr in (("audio", audio), ("vision", vision), ("force", force),
                      ("tactile", rng.standard_normal((N, 32)).astype(np.float32)),
                      ("gravity", rng.standard_normal((N, 3)).astype(np.float32))):  # fmt: skip
        torch.save(torch.from_numpy(arr), d / f"{name}.pt")
    torch.save(torch.zeros(N, dtype=torch.long), d / "labels.pt")
    return on


def test_studyforrest_label_recovers_the_narration_flag(tmp_path: Path) -> None:
    on = _write(tmp_path, studyforrest=True)
    w = rc.SegmentSplitWorld("studyforrest", tmp_path)
    y = w.gather(torch.arange(N)).label.numpy()
    assert np.array_equal(y // 2, on.astype(int))
    force4 = torch.load(tmp_path / "force.pt")[:, 4].numpy()
    assert np.array_equal(y % 2, (force4 > 0).astype(int))
    assert w.label_info["a_max_rounding_dev"] < 0.1


def test_ccfilm_label_is_energy_by_brightness(tmp_path: Path) -> None:
    _write(tmp_path, studyforrest=False)
    x = {m: torch.load(tmp_path / f"{m}.pt").reshape(N, -1).double().numpy()
         for m in ("audio", "vision", "tactile", "gravity", "force")}  # fmt: skip
    y, info = rc.two_channel_labels("ccfilm", x)
    e, b = x["audio"].mean(1), x["vision"].mean(1)
    assert np.array_equal(y, 2 * (e > np.median(e)) + (b > np.median(b)))
    assert sum(info["counts"]) == N
    with pytest.raises(ValueError):
        rc.two_channel_labels("xor", x)


def test_split_is_frozen_purged_and_respected(tmp_path: Path) -> None:
    _write(tmp_path, studyforrest=False)
    a = rc.SegmentSplitWorld("ccfilm", tmp_path)
    b = rc.SegmentSplitWorld("ccfilm", tmp_path)
    assert a.held_segments == b.held_segments and len(a.held_segments) == rc.N_HELD
    held = set(a.probe_sample_with_indices(a.n_probe, 0)[1].tolist())
    labels = a.dataset_labels
    excluded = set(torch.nonzero(labels < 0).flatten().tolist())
    assert held <= excluded and a.n_purged == len(excluded - held)
    held_arr = np.array(sorted(held))
    for s in range(20):
        _, idx = a.sample_with_indices(16, s)
        assert not set(idx.tolist()) & excluded
        assert np.min(np.abs(idx.numpy()[:, None] - held_arr[None, :])) > rc.PURGE
    assert a.describe()["min_class_heldout_share"] == a.min_class_heldout_share
    with pytest.raises(ValueError, match="held-out probe"):
        a.probe_sample_with_indices(10, 0)


def test_purged_folds_keep_training_rows_away_from_test_rows() -> None:
    rows = np.arange(0, 3000, 4)
    groups = rows // 100
    for tr, te in rc._purged_group_folds(rows, groups):
        assert len(tr) and len(te)
        d = np.abs(rows[tr][:, None] - rows[te][None, :]).min()
        assert d > rc.PURGE
        assert not set(groups[tr]) & set(groups[te])


def test_label_break_replaces_only_its_modality_inside_the_window(tmp_path: Path) -> None:
    _write(tmp_path, studyforrest=False)
    w = rc.SegmentSplitWorld("ccfilm", tmp_path)
    sch = rc.LabelBreakSchedule(DecorrelationWindow(5, 3), seed=3206, world=w,
                                modality="vision", seed_namespace=9101)  # fmt: skip
    out = sch.sample(w, batch_size=16, seed=42, step=6)
    base = w.sample(16, 42)
    assert torch.equal(out.audio, base.audio) and torch.equal(out.label, base.label)
    assert not torch.equal(out.vision, base.vision)
    outside = sch.sample(w, batch_size=16, seed=42, step=1)
    assert torch.equal(outside.vision, base.vision)


def test_run_cell_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rc, "ONSETS", (0, 3))
    _write(tmp_path, studyforrest=False)
    w = rc.SegmentSplitWorld("ccfilm", tmp_path)
    ns = BY_NAME["v09-pilot"].code
    cells = []
    for s in (3206, 3207):
        for o in rc.ONSETS:
            for arm in rc.ARMS:
                cells.append(rc.run_cell_revc(w, s, o, arm, seed_namespace=ns, width=1,
                                              recovery=1, batch=8))  # fmt: skip
    again = rc.run_cell_revc(w, 3206, 0, "decor", seed_namespace=ns, width=1, recovery=1, batch=8)
    first = next(c for c in cells if (c["seed"], c["onset"], c["arm"]) == (3206, 0, "decor"))
    assert again["lin"] == first["lin"] and again["task"] == first["task"]
    out = rc.summarise(cells, [3206, 3207])
    assert set(out["gates"]) == {"gate1_learning", "gate2_isolation", "gate3_positive_control",
                                 "gate4_label_break", "gate5_headroom"}  # fmt: skip
    assert np.array(out["D"]).shape == (2, len(rc.ONSETS))
    assert len(out["label_break_drops"]) == 2 * len(rc.ONSETS)
    assert "sigma_plan" in out["variance_components"]


def test_extraction_refuses_a_film_with_another_hash(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "extract_ccfilm_features.py"
    spec = importlib.util.spec_from_file_location("extract_ccfilm", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fake = tmp_path / "film.mp4"
    fake.write_bytes(b"not the film")
    with pytest.raises(SystemExit, match="refusing"):
        mod.verify_film(fake)
    assert len(mod.FILM_SHA256) == 64
    assert hashlib.sha256(b"not the film").hexdigest() != mod.FILM_SHA256


def test_plain_cells_carry_diagnostics_and_gate_needs_all_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rc, "ONSETS", (0,))
    _write(tmp_path, studyforrest=False)
    w = rc.SegmentSplitWorld("ccfilm", tmp_path)
    ns = BY_NAME["v09-pilot"].code
    plain = rc.run_cell_revc(w, 3206, 0, "plain", seed_namespace=ns, width=1, recovery=1, batch=8)
    decor = rc.run_cell_revc(w, 3206, 0, "decor", seed_namespace=ns, width=1, recovery=1, batch=8)
    assert set(plain["lin"]["bit_bacc"]["audio"]) == {"a", "b"} and decor["lin"]["bit_bacc"] is None
    lo, hi = plain["lin"]["fg_control_ci90"]
    assert lo <= plain["lin"]["fg_control"] + 1e-9 or hi >= plain["lin"]["fg_control"]
    with pytest.raises(ValueError, match="missing"):
        rc.plain_learning_gate([plain], [3206, 3207])
    assert rc.plain_learning_gate([plain], [3206])["per_setting"].keys() == {"0"}


def test_label_break_counts_self_donors(tmp_path: Path) -> None:
    _write(tmp_path, studyforrest=False)
    w = rc.SegmentSplitWorld("ccfilm", tmp_path)
    sch = rc.LabelBreakSchedule(DecorrelationWindow(0, 10_000), seed=3206, world=w,
                                modality="vision", seed_namespace=9101)  # fmt: skip
    sch.pool = torch.tensor([int(w.sample_with_indices(16, 5)[1][0])])  # single-row pool
    sch.sample(w, batch_size=16, seed=5, step=0)
    assert sch.n_self_donor == 1


def test_derive_gives_no_n_for_a_stopped_or_failed_revc_run(tmp_path: Path) -> None:
    import json

    path = Path(__file__).resolve().parents[2] / "scripts" / "pilots" / "derive_v09_pilot.py"
    spec = importlib.util.spec_from_file_location("derive_v09", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for rec in ({"design": "rev. C", "stopped": "learning gate failed: stop and report"},
                {"design": "rev. C", "all_gates_passed": False,
                 "variance_components": {"sigma_plan": 0.1}}):  # fmt: skip
        src, dst = tmp_path / "in.json", tmp_path / "out.json"
        src.write_text(json.dumps(rec))
        mod.main(src, dst)
        assert json.loads(dst.read_text())["N"] is None


def test_cell_records_the_loss_curve_and_honours_the_recovery(tmp_path: Path) -> None:
    _write(tmp_path, studyforrest=False)
    w = rc.SegmentSplitWorld("ccfilm", tmp_path)
    ns = BY_NAME["v09-pilot"].code
    c = rc.run_cell_revc(w, 3211, 0, "plain", seed_namespace=ns, width=1, recovery=120, batch=8)
    assert c["steps"] == 121 and len(c["loss_curve"]) == 3  # bins of 50: 50, 50, 21
    assert rc.RECOVERY_REVD == 1750 and rc.PILOT_SEEDS_REVD == (3211, 3212, 3213, 3214, 3215)
    assert set(rc.PILOT_SEEDS_REVD) <= set(BY_NAME["v09-pilot"].seeds)
