"""v0.9 decorrelation-window scheduler, CKA estimator and manipulation gate.

Spec: docs/osf/amendment-v0.9-decorrelation-window-2026-10-23.md (§3).
Synthetic data only; nothing here runs an experiment.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from bouba_sens.decorrelation import (
    CHECK_N_ANCHORS,
    DecorrelationSchedule,
    DecorrelationWindow,
    DonorIndex,
    build_arm_sample,
    check_generator_seed,
    cka_statistic,
    fusion_gain,
    linear_cka,
    manipulation_gate,
    run_check_draw,
    training_generator_seed,
    within_class_cka,
)
from bouba_sens.loop import AdaptationLoop
from bouba_sens.sensory import MODALITIES
from bouba_sens.world.base import WorldSample
from bouba_sens.world.studyforrest import StudyforrestRealWorld

DIMS = {"audio": 8, "vision": 12, "tactile": 6, "gravity": 3, "force": 6}


class SyntheticIndexedWorld:
    """Rows share a per-row latent g within a class, so modalities correlate."""

    def __init__(self, n: int = 600, n_classes: int = 3, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self._labels = torch.arange(n) % n_classes
        g = torch.randn(n, 1, generator=gen)
        self._data: dict[str, torch.Tensor] = {}
        for m in MODALITIES:
            d = DIMS[m]
            w = torch.randn(1, d, generator=gen)
            means = 3.0 * torch.randn(n_classes, d, generator=gen)
            noise = 0.3 * torch.randn(n, d, generator=gen)
            self._data[m] = means[self._labels] + g * w + noise
        self._n = n

    @property
    def dataset_labels(self) -> torch.Tensor:
        return self._labels

    def sample_with_indices(self, batch_size: int, seed: int) -> tuple[WorldSample, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        start = int(torch.randint(0, self._n - batch_size, (1,), generator=gen).item())
        idx = torch.arange(start, start + batch_size) % self._n
        return self.gather(idx), idx

    def gather(self, idx: torch.Tensor) -> WorldSample:
        return WorldSample(
            z=torch.zeros(len(idx), 1),
            audio=self._data["audio"][idx],
            vision=self._data["vision"][idx],
            tactile=self._data["tactile"][idx],
            gravity=self._data["gravity"][idx],
            force=self._data["force"][idx],
            label=self._labels[idx],
        )

    def sample(self, batch_size: int, seed: int) -> WorldSample:
        return self.sample_with_indices(batch_size, seed)[0]


def _anchors(world: SyntheticIndexedWorld, b: int = 16, seed: int = 1) -> torch.Tensor:
    return world.sample_with_indices(b, seed)[1]


# --- donor rule (§3.1) -------------------------------------------------


def test_generator_seed_formulas() -> None:
    assert training_generator_seed(7, 12) == 10**9 + 1000 * 7 + 12
    assert check_generator_seed(7, 12) == 10**9 + 7 * 10**8 + 1000 * 7 + 12


def test_donor_never_self_and_same_label() -> None:
    world = SyntheticIndexedWorld()
    index = DonorIndex(world.dataset_labels)
    for t in range(60):
        anchors = _anchors(world, 16, seed=t)
        donors, n_self = index.draw(anchors, training_generator_seed(3101, t))
        assert donors.shape == (5, 16)
        assert n_self == 0
        assert not (donors == anchors.unsqueeze(0)).any()
        assert (world.dataset_labels[donors] == world.dataset_labels[anchors]).all()


def test_donor_drawn_from_whole_dataset_uniformly() -> None:
    world = SyntheticIndexedWorld(n=60, n_classes=3)  # 20 rows per class
    index = DonorIndex(world.dataset_labels)
    anchors = torch.zeros(4000, dtype=torch.long)  # same anchor row, many draws
    donors, _ = index.draw(anchors, 123)
    seen = torch.unique(donors[0])
    expected = (world.dataset_labels == world.dataset_labels[0]).nonzero().flatten()
    expected = expected[expected != 0]
    assert set(seen.tolist()) == set(expected.tolist())  # all 19, never row 0
    counts = torch.bincount(donors[0], minlength=60)[expected].float()
    assert counts.min() > 0.6 * counts.mean()  # roughly uniform, no class weighting


def test_singleton_class_is_own_donor_and_counted() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    index = DonorIndex(labels)
    anchors = torch.tensor([3, 3, 0])
    donors, n_self = index.draw(anchors, 5)
    assert (donors[:, :2] == 3).all()
    assert n_self == 5 * 2  # two singleton anchors x five modality draws
    assert (donors[:, 2] != 0).all()


def test_per_step_determinism_from_seed() -> None:
    world = SyntheticIndexedWorld()
    index = DonorIndex(world.dataset_labels)
    anchors = _anchors(world)
    a, _ = index.draw(anchors, training_generator_seed(1, 4))
    b, _ = index.draw(anchors, training_generator_seed(1, 4))
    c, _ = index.draw(anchors, training_generator_seed(1, 5))
    d, _ = index.draw(anchors, training_generator_seed(2, 4))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a, d)


def test_schedule_is_stateless_across_call_order() -> None:
    world = SyntheticIndexedWorld()
    sch = DecorrelationSchedule(DecorrelationWindow(0, 50), "decor", seed=9, world=world)
    first = sch.sample(world, batch_size=16, seed=100, step=7)
    sch.sample(world, batch_size=16, seed=101, step=8)
    again = sch.sample(world, batch_size=16, seed=100, step=7)
    for m in MODALITIES:
        assert torch.equal(getattr(first, m), getattr(again, m))


# --- arms (§3.2) -------------------------------------------------------


def test_joint_shares_audio_donor_and_decor_is_independent() -> None:
    world = SyntheticIndexedWorld()
    index = DonorIndex(world.dataset_labels)
    anchors = _anchors(world)
    donors, _ = index.draw(anchors, training_generator_seed(1, 0))
    joint = build_arm_sample(world, anchors, donors, "joint")
    decor = build_arm_sample(world, anchors, donors, "decor")
    audio_donor = world.gather(donors[0])
    for m in MODALITIES:  # joint: every modality from the audio donor
        assert torch.equal(getattr(joint, m), getattr(audio_donor, m))
    assert torch.equal(decor.audio, joint.audio)  # shared audio donor
    for k, m in enumerate(MODALITIES):
        assert torch.equal(getattr(decor, m), getattr(world.gather(donors[k]), m))
    assert not torch.equal(decor.vision, joint.vision)
    for s in (joint, decor):  # labels untouched
        assert torch.equal(s.label, world.gather(anchors).label)


def test_plain_arm_is_identity_everywhere() -> None:
    world = SyntheticIndexedWorld()
    sch = DecorrelationSchedule(DecorrelationWindow(0, 50), "plain", seed=1, world=world)
    for t in (0, 10, 49, 60):
        got = sch.sample(world, batch_size=16, seed=t, step=t)
        ref = world.sample(16, t)
        for m in MODALITIES:
            assert torch.equal(getattr(got, m), getattr(ref, m))


# --- window boundaries (§3.2, §4) ---------------------------------------


@pytest.mark.parametrize("onset", [0, 50, 100, 150, 200])
def test_window_boundaries_exact(onset: int) -> None:
    w = DecorrelationWindow(onset, 50)
    assert not w.active(onset - 1)
    assert w.active(onset)
    assert w.active(onset + 49)
    assert not w.active(onset + 50)
    assert sum(w.active(t) for t in range(0, 600)) == 50


def test_identity_outside_window_and_change_inside() -> None:
    world = SyntheticIndexedWorld()
    sch = DecorrelationSchedule(DecorrelationWindow(100, 50), "decor", seed=1, world=world)
    for t in (0, 99, 150, 400):
        got = sch.sample(world, batch_size=16, seed=t, step=t)
        ref = world.sample(16, t)
        for m in MODALITIES:
            assert torch.equal(getattr(got, m), getattr(ref, m))
    inside = sch.sample(world, batch_size=16, seed=100, step=100)
    assert not torch.equal(inside.audio, world.sample(16, 100).audio)


def test_window_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="onset"):
        DecorrelationWindow(-1, 50)
    with pytest.raises(ValueError, match="width"):
        DecorrelationWindow(0, 0)
    with pytest.raises(ValueError, match="arm"):
        DecorrelationSchedule(
            DecorrelationWindow(0, 50), "bogus", seed=1, world=SyntheticIndexedWorld()
        )


# --- CKA estimator (§3.3) ----------------------------------------------

_XA = torch.tensor([[1.0, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float64)
_XB = torch.tensor([[1.0, 0], [-1, 0], [0, 0], [0, 0]], dtype=torch.float64)


def test_linear_cka_hand_computed() -> None:
    # XaT Xa = diag(2,2) (F = sqrt 8); XbT Xb = diag(2,0) (F = 2);
    # XbT Xa = diag(2,0) (F^2 = 4)  ->  4 / (sqrt 8 * 2) = 1/sqrt 2.
    assert linear_cka(_XA, _XB) == pytest.approx(2**-0.5, abs=1e-12)


def test_linear_cka_centres_columns() -> None:
    shifted = _XA + torch.tensor([5.0, -3.0], dtype=torch.float64)
    assert linear_cka(shifted, _XB + 7.0) == pytest.approx(2**-0.5, abs=1e-12)


def test_linear_cka_identity_and_scale_invariance() -> None:
    x = torch.randn(40, 5, dtype=torch.float64)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-12)
    assert linear_cka(x, 3.7 * x) == pytest.approx(1.0, abs=1e-12)


def test_zero_denominator_returns_none() -> None:
    x = torch.randn(20, 4, dtype=torch.float64)
    const = torch.full((20, 3), 2.5, dtype=torch.float64)
    assert linear_cka(x, const) is None
    assert linear_cka(torch.zeros(20, 3, dtype=torch.float64), x) is None


def _tile(x: torch.Tensor, k: int) -> torch.Tensor:
    return x.repeat(k, 1)


def test_within_class_cka_weighted_mean_and_min_rows() -> None:
    # class 0: 12 rows, CKA 1 ; class 1: 20 rows, CKA 1/sqrt 2 ; class 2: 4 rows (< 10, ignored)
    xa = torch.cat([_tile(_XA, 3), _tile(_XA, 5), _XA])
    xb = torch.cat([_tile(_XA, 3), _tile(_XB, 5), _XA * 0 + torch.randn(4, 2)])
    labels = torch.cat([torch.zeros(12), torch.ones(20), 2 * torch.ones(4)]).long()
    got = within_class_cka(xa, xb, labels)
    assert got == pytest.approx((12 * 1.0 + 20 * 2**-0.5) / 32, abs=1e-12)


def test_within_class_cka_skips_zero_denominator_classes() -> None:
    xa = torch.cat([_tile(_XA, 3), _tile(_XA, 3)])
    xb = torch.cat([_tile(_XA, 3), torch.ones(12, 2, dtype=torch.float64)])  # class 1 constant
    labels = torch.cat([torch.zeros(12), torch.ones(12)]).long()
    assert within_class_cka(xa, xb, labels) == pytest.approx(1.0, abs=1e-12)
    xb_all_const = torch.ones(24, 2, dtype=torch.float64)
    assert within_class_cka(xa, xb_all_const, labels) is None


def test_cka_statistic_identical_features_is_one_and_none_when_invalid() -> None:
    n = 40
    base = torch.randn(n, 6, dtype=torch.float64)
    labels = torch.arange(n) % 2
    mods = {m: base.clone() for m in MODALITIES}
    assert cka_statistic(mods, labels) == pytest.approx(1.0, abs=1e-12)
    const = {m: torch.ones(n, 3, dtype=torch.float64) for m in MODALITIES}
    assert cka_statistic(const, labels) is None


# --- joint preserves pairing, decor breaks it ----------------------------


def test_cka_drops_for_decor_but_not_joint() -> None:
    world = SyntheticIndexedWorld(n=1200, n_classes=3)
    kj, kd = [], []
    for b in range(10):
        draw = run_check_draw(world, train_seed=3101, draw=b)
        assert draw.k_joint is not None and draw.k_decor is not None
        kj.append(draw.k_joint)
        kd.append(draw.k_decor)
    kj_t, kd_t = torch.tensor(kj), torch.tensor(kd)
    assert kj_t.mean() > 0.5
    assert kd_t.mean() < 0.2
    assert (kj_t - kd_t).min() > 0.2


def test_check_draw_uses_512_anchors_and_is_deterministic() -> None:
    assert CHECK_N_ANCHORS == 512
    world = SyntheticIndexedWorld(n=1200)
    a = run_check_draw(world, train_seed=1, draw=3)
    b = run_check_draw(world, train_seed=1, draw=3)
    assert a == b
    assert a.n_anchors == 512
    assert a.k_original is not None


# --- permutation gate (§3.3) --------------------------------------------


def test_gate_passes_on_real_difference() -> None:
    g = torch.Generator().manual_seed(0)
    kd = 0.1 + 0.02 * torch.randn(5, 200, generator=g)
    kj = kd + 0.3 + 0.02 * torch.randn(5, 200, generator=g)
    res = manipulation_gate(kj, kd, n_perm=2000)
    assert res.passed
    assert res.delta_k == pytest.approx(0.3, abs=0.01)
    assert res.p_value == pytest.approx(1 / 2001)
    assert res.k_joint == pytest.approx(float(kj.mean()))


def test_gate_fails_under_null_and_formula_for_p() -> None:
    g = torch.Generator().manual_seed(1)
    kj = 0.3 + 0.05 * torch.randn(5, 200, generator=g)
    kd = 0.3 + 0.05 * torch.randn(5, 200, generator=g)
    res = manipulation_gate(kj, kd, n_perm=2000)
    assert not res.passed
    assert res.p_value >= 0.05
    zero = manipulation_gate(torch.ones(2, 3), torch.ones(2, 3), n_perm=100)
    assert zero.p_value == 1.0  # (1 + 100) / (1 + 100)


def test_gate_is_reproducible_and_fails_on_invalid_draw() -> None:
    kj = torch.rand(3, 20)
    kd = torch.rand(3, 20) - 0.5
    a = manipulation_gate(kj, kd, n_perm=500)
    b = manipulation_gate(kj, kd, n_perm=500)
    assert a == b
    bad = kd.clone()
    bad[0, 0] = float("nan")  # a draw with no valid (pair, class)
    res = manipulation_gate(kj, bad, n_perm=500)
    assert not res.passed
    assert "valid" in res.reason


# --- world exposes dataset indices (§3.1) --------------------------------


@pytest.fixture
def real_dir(tmp_path: Path) -> Path:
    n = 512
    torch.save(torch.randn(n, 128), tmp_path / "audio.pt")
    torch.save(torch.randn(n, 16, 16), tmp_path / "vision.pt")
    torch.save(torch.randn(n, 32), tmp_path / "tactile.pt")
    torch.save(torch.randn(n, 3), tmp_path / "gravity.pt")
    torch.save(torch.randn(n, 6), tmp_path / "force.pt")
    torch.save(torch.randint(0, 4, (n,), dtype=torch.long), tmp_path / "labels.pt")
    return tmp_path


def test_real_world_exposes_indices(real_dir: Path) -> None:
    world = StudyforrestRealWorld(seed=0, data_dir=real_dir)
    sample, idx = world.sample_with_indices(16, 42)
    ref = world.sample(16, 42)
    assert torch.equal(sample.audio, ref.audio)
    assert torch.equal(sample.label, ref.label)
    assert torch.equal(world.dataset_labels[idx], sample.label)
    assert torch.equal(world.gather(idx).force, sample.force)
    assert world.dataset_labels.shape == (512,)


# --- loop hook and fusion gain -------------------------------------------


def test_pretrain_hook_default_unchanged_and_window_applied(real_dir: Path) -> None:
    from tests.unit.test_loop import _build_loop

    world = StudyforrestRealWorld(seed=0, data_dir=real_dir)
    seen: list[torch.Tensor] = []

    loop = _build_loop(n_classes=4)
    loop.world = world
    orig = loop._forward

    def spy(sample: WorldSample) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(sample.audio.clone())
        return orig(sample)

    loop._forward = spy  # type: ignore[method-assign]
    sch = DecorrelationSchedule(DecorrelationWindow(2, 2), "decor", seed=5, world=world)
    loop.pretrain(steps=6, batch_size=16, seed=0, decorrelation=sch)
    for t in (0, 1, 4, 5):
        assert torch.equal(seen[t], world.sample(16, t).audio)
    for t in (2, 3):
        assert not torch.equal(seen[t], world.sample(16, t).audio)


def test_fusion_gain_is_all_minus_best_single() -> None:
    class StubLoop:
        world = SyntheticIndexedWorld()

        def _forward(self, sample: WorldSample) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.nn.functional.one_hot(sample.label, 3).float()
            return torch.tensor(0.0), logits  # perfect, acc_all = 1

        def query_accuracy(self, m: str, *, batch_size: int, seed: int) -> float:
            assert batch_size == 512
            assert seed == 11 + 2 * 10**6
            return {"audio": 0.5, "vision": 0.7}.get(m, 0.4)

    res = fusion_gain(StubLoop(), train_seed=11)  # type: ignore[arg-type]
    assert res.acc_all == 1.0
    assert res.fusion_gain == pytest.approx(0.3)


def test_cka_large_offset_is_translation_invariant() -> None:
    col = torch.tensor([1e12 - 1, 1e12 + 1] * 6, dtype=torch.float64).unsqueeze(1)
    assert linear_cka(col, col.clone()) == pytest.approx(1.0, abs=1e-12)
    x = torch.randn(30, 3, dtype=torch.float64)
    assert linear_cka(x + 1e12, x * 2 - 1e12) == pytest.approx(1.0, abs=1e-6)
    assert linear_cka(col, torch.full((12, 1), 1e12, dtype=torch.float64)) is None


def test_gate_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        manipulation_gate(torch.ones(2, 3), torch.ones(3, 2), n_perm=10)


# Reachable objects that cannot be compared bit-for-bit; listed, never silently skipped.
_NOT_COMPARABLE: dict[str, str] = {
    "function": "code object; no mutable training state",
    "builtin_function_or_method": "code object; no mutable training state",
    "method": "bound method; state lives on its owner, which is flattened separately",
    "module": "imported python module; not owned state",
    "type": "class object; not owned state",
    "Path": "filesystem path value; no mutable state",
}


def _flatten_state(
    obj: object,
    prefix: str,
    out: dict[str, object],
    seen: dict[int, object],
    skipped: dict[str, str],
) -> None:
    """Collect every tensor/array/number/RNG state reachable from ``obj``."""
    if id(obj) in seen:
        return
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj.detach().clone()
        return
    if isinstance(obj, np.ndarray):
        out[prefix] = torch.from_numpy(obj.copy())
        return
    if isinstance(obj, bool | int | float | str | bytes | type(None) | np.generic):
        out[prefix] = obj
        return
    seen[id(obj)] = obj  # keep alive: transient state_dict() ids must not be reused
    if isinstance(obj, torch.Generator):
        out[prefix + ".torch_generator"] = obj.get_state()
    elif isinstance(obj, np.random.Generator):
        _flatten_state(obj.bit_generator.state, prefix + ".np_generator", out, seen, skipped)
    elif isinstance(obj, np.random.RandomState):
        _flatten_state(obj.get_state(), prefix + ".np_randomstate", out, seen, skipped)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_state(v, f"{prefix}[{k!r}]", out, seen, skipped)
    elif isinstance(obj, list | tuple | set | frozenset):
        items = sorted(obj, key=repr) if isinstance(obj, set | frozenset) else obj
        for k, v in enumerate(items):
            _flatten_state(v, f"{prefix}[{k}]", out, seen, skipped)
    elif isinstance(obj, torch.optim.Optimizer):
        _flatten_state(obj.state_dict(), f"{prefix}.optimizer", out, seen, skipped)
    elif isinstance(obj, torch.nn.Module):
        for k, v in obj.state_dict().items():  # parameters + buffers
            out[f"{prefix}.state_dict.{k}"] = v.detach().clone()
        for k, v in vars(obj).items():  # plain attrs set via object.__setattr__
            if not k.startswith("_") or k in ("_step", "_t", "_count"):
                _flatten_state(v, f"{prefix}.{k}", out, seen, skipped)
    elif hasattr(obj, "__dict__") and type(obj).__name__ not in _NOT_COMPARABLE:
        for k, v in vars(obj).items():
            _flatten_state(v, f"{prefix}.{k}", out, seen, skipped)
    else:
        skipped[prefix] = type(obj).__name__


def _full_loop_state(loop: object) -> dict[str, object]:
    """Everything the loop AND its world own, plus every process-global RNG."""
    out: dict[str, object] = {}
    seen: dict[int, object] = {}
    skipped: dict[str, str] = {}
    for name, val in vars(loop).items():
        _flatten_state(val, name, out, seen, skipped)
    unexplained = {k: t for k, t in skipped.items() if t not in _NOT_COMPARABLE}
    assert not unexplained, f"uncomparable state without a documented reason: {unexplained}"
    out["<global torch RNG>"] = torch.get_rng_state()
    out["<global numpy RNG>"] = _flatten_np_global()
    out["<python random>"] = random.getstate()
    if torch.cuda.is_available():
        out["<cuda RNG>"] = torch.cuda.get_rng_state_all()  # pragma: no cover
    if torch.backends.mps.is_available():
        out["<mps RNG>"] = torch.mps.get_rng_state()  # pragma: no cover
    return out


def _flatten_np_global() -> object:
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return (kind, keys.tobytes(), pos, has_gauss, cached)


def _seed_all(n: int) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def _assert_identical(a: dict[str, object], b: dict[str, object]) -> None:
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
    for k in a:
        va, vb = a[k], b[k]
        same = torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb
        assert same, f"first differing entry: {k}"


class _StatefulWorld(StudyforrestRealWorld):
    """Real world plus hidden mutable state, so extra world draws become observable."""

    def __init__(self, *, data_dir: Path) -> None:
        super().__init__(seed=0, data_dir=data_dir)
        self.calls = 0
        self.cursor = torch.zeros(1)
        self.gen = torch.Generator().manual_seed(7)
        self.np_gen = np.random.default_rng(7)
        self.np_rs = np.random.RandomState(7)

    def sample_with_indices(self, batch_size: int, seed: int):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.cursor += 1
        torch.rand(1, generator=self.gen)
        self.np_gen.random()
        self.np_rs.rand()
        return super().sample_with_indices(batch_size, seed)


def _main_replica_vs_loop(
    real_dir: Path, inject: Callable[[AdaptationLoop], None] | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    """Run pretrain() (no decorrelation) and the origin/main replica; return both states.

    ``inject`` perturbs the replica side only (extra RNG/world draw per step).
    """
    from tests.unit.test_loop import _build_loop

    torch.manual_seed(0)
    a = _build_loop(n_classes=4)
    torch.manual_seed(0)
    b = _build_loop(n_classes=4)
    a.world = _StatefulWorld(data_dir=real_dir)
    b.world = _StatefulWorld(data_dir=real_dir)  # own instance
    _assert_identical(_full_loop_state(a), _full_loop_state(b))  # same start
    start_keys = _full_loop_state(a).keys()
    for owned in ("world", "mux", "nerve", "head", "sensories", "opt"):
        assert any(k.startswith(owned) for k in start_keys), owned
    seen_a: list[torch.Tensor] = []
    orig = a._forward

    def spy(sample: WorldSample) -> tuple[torch.Tensor, torch.Tensor]:
        seen_a.append(sample.audio.clone())
        return orig(sample)

    a._forward = spy  # type: ignore[method-assign]
    _seed_all(1)
    a.pretrain(steps=4, batch_size=16, seed=3)
    a._forward = orig  # type: ignore[method-assign]
    state_a = _full_loop_state(a)  # process-global RNGs must be read right after the run
    _seed_all(1)
    # Replica of AdaptationLoop.pretrain as on origin/main f99f2dc938fa
    # (src/bouba_sens/loop.py lines 118-128: the mux_step/nerve_step lookup
    # and the `for step in range(steps)` body, before the Checkpoint return).
    mux_step = getattr(b.mux, "step", None)
    nerve_step = getattr(b.nerve, "step", None)
    for step in range(4):
        sample = b.world.sample(batch_size=16, seed=3 + step)
        assert torch.equal(seen_a[step], sample.audio), f"batch differs at step {step}"
        if inject is not None:
            inject(b)
        loss, _ = b._forward(sample)
        b.opt.zero_grad()
        loss.backward()
        b.opt.step()
        if callable(mux_step):
            mux_step()
        if callable(nerve_step):
            nerve_step()
    assert len(seen_a) == 4
    return state_a, _full_loop_state(b)


def test_pretrain_without_decorrelation_matches_main_behaviour(
    real_dir: Path, deterministic_torch: None
) -> None:
    state_a, state_b = _main_replica_vs_loop(real_dir)
    _assert_identical(state_a, state_b)


def _extra_world_sample(b: AdaptationLoop) -> None:
    b.world.sample(batch_size=16, seed=0)


def _extra_numpy_draw(_: AdaptationLoop) -> None:
    np.random.rand()


def _extra_python_draw(_: AdaptationLoop) -> None:
    random.random()


def _extra_torch_draw(_: AdaptationLoop) -> None:
    torch.rand(1)


@pytest.mark.parametrize(
    "inject",
    [_extra_world_sample, _extra_numpy_draw, _extra_python_draw, _extra_torch_draw],
    ids=lambda f: f.__name__,
)
def test_main_equivalence_check_catches_injected_draws(
    real_dir: Path, deterministic_torch: None, inject: Callable[[AdaptationLoop], None]
) -> None:
    state_a, state_b = _main_replica_vs_loop(real_dir, inject)
    with pytest.raises(AssertionError, match="first differing entry"):
        _assert_identical(state_a, state_b)
