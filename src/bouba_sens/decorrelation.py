"""v0.9 decorrelation-window scheduler, CKA estimator and manipulation gate.

Implements section 3 of the draft amendment
``docs/osf/amendment-v0.9-decorrelation-window-2026-10-23.md``:

* donor resampling (3.1): same label, whole dataset, never the sample
  itself, seeded per step, uniform within class;
* arms (3.2): ``decor`` (independent donor per modality), ``joint``
  (all modalities from the audio donor), ``plain`` (no resampling);
* manipulation check (3.3): within-class linear CKA, matched joint vs
  decor draws, label-swap permutation null;
* fusion-gain probe (3.4).

The training loop touches this module through one optional argument of
``AdaptationLoop.pretrain``; everything else lives here. Nothing in this
module runs an experiment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

import torch
from torch import Tensor

from bouba_sens.sensory import MODALITIES
from bouba_sens.world.base import WorldSample

if TYPE_CHECKING:
    from bouba_sens.loop import AdaptationLoop

Arm = Literal["plain", "joint", "decor"]
ARMS: tuple[str, ...] = ("plain", "joint", "decor")

CHECK_N_WINDOWS = 32
CHECK_WINDOW = 16
CHECK_N_ANCHORS = CHECK_N_WINDOWS * CHECK_WINDOW  # 512
CHECK_N_DRAWS = 200
CKA_MIN_CLASS_ROWS = 10
GATE_N_PERM = 10_000
GATE_SEED = 20_261_023
GATE_ALPHA = 0.05
PROBE_ROWS = 512
V09_PROBE_FRACTION = 0.10  # last 10 % of frames held out (3.4, PI 2026-10-24)
RECON_WEIGHT = 1.0  # mitigation arm (3.5, PI 2026-10-24): fixed, not tuned
RECON_EPS = 1e-6


class IndexedWorld(Protocol):
    """World that exposes dataset row indices (spec 3.1)."""

    @property
    def dataset_labels(self) -> Tensor: ...

    def sample_with_indices(self, batch_size: int, seed: int) -> tuple[WorldSample, Tensor]: ...

    def gather(self, idx: Tensor) -> WorldSample: ...


# --- seeds ---------------------------------------------------------------


def training_generator_seed(train_seed: int, step: int) -> int:
    """Generator seed for the donor draws at Phase-1 step ``step`` (3.1)."""
    return 10**9 + 1000 * train_seed + step


def check_generator_seed(train_seed: int, draw: int) -> int:
    """Generator seed for manipulation-check draw ``draw`` (3.3)."""
    return 10**9 + 7 * 10**8 + 1000 * train_seed + draw


# --- window ----------------------------------------------------------------


@dataclass(frozen=True)
class DecorrelationWindow:
    """Half-open Phase-1 step interval [onset, onset + width)."""

    onset: int
    width: int = 50

    def __post_init__(self) -> None:
        if self.onset < 0:
            raise ValueError(f"onset must be >= 0, got {self.onset}")
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")

    def active(self, step: int) -> bool:
        return self.onset <= step < self.onset + self.width


# --- donors ----------------------------------------------------------------


class DonorIndex:
    """Per-class row lists over the WHOLE dataset, for donor draws (3.1)."""

    def __init__(self, labels: Tensor) -> None:
        labels = labels.detach().cpu().long()
        _, codes = torch.unique(labels, return_inverse=True)
        order = torch.argsort(codes, stable=True)
        counts = torch.bincount(codes)
        offsets = torch.cumsum(counts, 0) - counts
        position = torch.empty_like(codes)
        position[order] = torch.arange(len(codes)) - offsets[codes[order]]
        self._codes = codes
        self._rows = order  # rows grouped by class, ascending within class
        self._counts = counts
        self._offsets = offsets
        self._position = position  # rank of each row inside its class
        self.n_rows = len(labels)

    def draw(self, anchors: Tensor, generator_seed: int) -> tuple[Tensor, int]:
        """One donor per (modality, anchor): shape (5, B), plus self-donor count.

        Donor j is uniform over {j : label_j = label_k, j != k} for anchor
        row k. Modalities are drawn in the fixed order of ``MODALITIES``
        from a generator freshly seeded with ``generator_seed``. A class
        with a single row makes the anchor its own donor; those draws are
        counted and returned.
        """
        anchors = anchors.detach().cpu().long()
        gen = torch.Generator().manual_seed(generator_seed)
        cls = self._codes[anchors]
        n_c = self._counts[cls]
        pos = self._position[anchors]
        single = n_c == 1
        out = []
        n_self = 0
        for _ in MODALITIES:
            u = torch.rand(len(anchors), generator=gen, dtype=torch.float64)
            r = torch.floor(u * (n_c - 1).double()).long()
            r = torch.where(r >= pos, r + 1, r)  # skip the anchor's own rank
            r = torch.where(single, pos, r)
            out.append(self._rows[self._offsets[cls] + r])
            n_self += int(single.sum().item())
        return torch.stack(out), n_self


def build_arm_sample(
    world: IndexedWorld, anchors: Tensor, donors: Tensor, arm: Arm | str
) -> WorldSample:
    """Resampled batch for ``arm``; labels stay those of the anchors."""
    if arm not in ("joint", "decor"):
        raise ValueError(f"build_arm_sample handles 'joint'/'decor', got {arm!r}")
    base = world.gather(anchors)
    fields: dict[str, Tensor] = {}
    for k, m in enumerate(MODALITIES):
        src = donors[0] if arm == "joint" else donors[k]
        fields[m] = getattr(world.gather(src), m)
    return replace(base, **fields)


# --- scheduler (training hook) -----------------------------------------------


class DecorrelationSchedule:
    """Phase-1 batch source for one (arm, window, training seed)."""

    def __init__(
        self,
        window: DecorrelationWindow,
        arm: Arm | str,
        *,
        seed: int,
        world: IndexedWorld,
        seed_namespace: int | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        self.window = window
        self.arm = arm
        self.seed = seed
        self.index = DonorIndex(world.dataset_labels)
        self.n_self_donor = 0  # logged, not expected (3.1)
        # PI decision 2026-10-25 (19:47): namespaced donor seeds per campaign.
        self.seed_namespace = seed_namespace

    def donor_seed(self, step: int) -> int:
        if self.seed_namespace is None:
            return training_generator_seed(self.seed, step)
        from bouba_sens.seeds import STREAMS, ns_seed

        return ns_seed(self.seed_namespace, self.seed, STREAMS["donor"], step)

    def sample(self, world: IndexedWorld, *, batch_size: int, seed: int, step: int) -> WorldSample:
        """The batch for Phase-1 ``step``; identical to ``world.sample`` outside the window."""
        if self.arm == "plain" or not self.window.active(step):
            return world.sample(batch_size, seed)  # type: ignore[attr-defined, no-any-return]
        _, anchors = world.sample_with_indices(batch_size, seed)
        donors, n_self = self.index.draw(anchors, self.donor_seed(step))
        self.n_self_donor += n_self
        return build_arm_sample(world, anchors, donors, self.arm)


# --- mitigation arm: cross-sensor reconstruction (3.5) ------------------------


class CrossSensorReconstruction(torch.nn.Module):
    """Auxiliary cross-sensor reconstruction objective for Phase 1 (v0.9, 3.5).

    Kleinman, Achille & Soatto (CVPR 2023) report that architectures
    trained with cross-sensor reconstruction objectives are markedly more
    resilient to critical periods. Here, for every modality m, a linear
    decoder maps the mean of the other four modalities' encoder
    representations (``core(input_proj(x))``, ``(B, d_hidden)``) to m's
    input, standardised within the batch per feature; the loss is the
    mean squared error averaged over features and modalities, added to
    the classification loss with weight ``weight``.

    Decoders are initialised from their own generator, so building the
    module never touches the global torch RNG; with ``weight = 0`` the
    training trajectory is bit-identical to no objective.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, tuple[int, ...]],
        *,
        d_hidden: int = 128,
        weight: float = RECON_WEIGHT,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        gen = torch.Generator().manual_seed(seed)
        bound = 1.0 / math.sqrt(d_hidden)
        global_state = torch.get_rng_state()  # nn.Linear init draws from it
        decoders: dict[str, torch.nn.Module] = {}
        for m in MODALITIES:
            out = int(math.prod(modality_dims[m]))
            lin = torch.nn.Linear(d_hidden, out)
            with torch.no_grad():
                lin.weight.copy_((torch.rand(out, d_hidden, generator=gen) * 2 - 1) * bound)
                lin.bias.copy_((torch.rand(out, generator=gen) * 2 - 1) * bound)
            decoders[m] = lin
        torch.set_rng_state(global_state)
        self.decoders = torch.nn.ModuleDict(decoders)

    def forward(self, loop: AdaptationLoop, sample: WorldSample) -> Tensor:
        reps = {
            m: loop.sensories[m].core(loop.sensories[m].input_proj(getattr(sample, m)))
            for m in MODALITIES
        }
        terms = []
        for m in MODALITIES:
            others = torch.stack([reps[o] for o in MODALITIES if o != m]).mean(0)
            x = getattr(sample, m).reshape(len(sample.label), -1)
            target = (x - x.mean(0)) / (x.std(0, unbiased=False) + RECON_EPS)
            terms.append(((self.decoders[m](others) - target) ** 2).mean())
        return torch.stack(terms).mean()


# --- within-class linear CKA (3.3) -----------------------------------------------


def _centre(x: Tensor) -> Tensor | None:
    """Column-centre ``x``; ``None`` if every column is exactly constant.

    Exact constancy is detected by subtracting the first row (translation
    invariant, no magnitude-relative cutoff); the shifted block is then
    mean-centred, which keeps large offsets numerically harmless.
    """
    shifted = x - x[:1]
    if not bool((shifted != 0).any()):
        return None
    return shifted - shifted.mean(dim=0, keepdim=True)


def linear_cka(xa: Tensor, xb: Tensor) -> float | None:
    """Linear CKA, biased HSIC: ||Xb^T Xa||_F^2 / (||Xa^T Xa||_F ||Xb^T Xb||_F).

    Columns are centred first. Returns ``None`` only for the zero
    CENTRED denominator case (a block constant within the class).
    """
    xa = xa.double().reshape(len(xa), -1)
    xb = xb.double().reshape(len(xb), -1)
    ca, cb = _centre(xa), _centre(xb)
    if ca is None or cb is None:
        return None
    num = float((cb.T @ ca).norm() ** 2)
    den = float((ca.T @ ca).norm() * (cb.T @ cb).norm())
    if den <= 0.0:
        return None
    return num / den


def within_class_cka(
    xa: Tensor, xb: Tensor, labels: Tensor, *, min_rows: int = CKA_MIN_CLASS_ROWS
) -> float | None:
    """n_c-weighted mean of per-class CKA over classes with n_c >= min_rows.

    Classes with a zero denominator are skipped; ``None`` if none remain.
    """
    total = 0.0
    weight = 0
    for c in torch.unique(labels):
        mask = labels == c
        n_c = int(mask.sum().item())
        if n_c < min_rows:
            continue
        val = linear_cka(xa[mask], xb[mask])
        if val is None:
            continue
        total += n_c * val
        weight += n_c
    return None if weight == 0 else total / weight


def cka_statistic(features: dict[str, Tensor], labels: Tensor) -> float | None:
    """K: mean over the 10 modality pairs of the within-class CKA (pairs without value dropped)."""
    vals: list[float] = []
    for i, a in enumerate(MODALITIES):
        for b in MODALITIES[i + 1 :]:
            v = within_class_cka(features[a], features[b], labels)
            if v is not None:
                vals.append(v)
    return None if not vals else sum(vals) / len(vals)


# --- manipulation check (3.3) --------------------------------------------------------


@dataclass(frozen=True)
class CheckDraw:
    """One draw b of the check on one seed: K for joint, decor and the raw anchors."""

    k_joint: float | None
    k_decor: float | None
    k_original: float | None
    n_anchors: int
    n_self_donor: int


def _features(sample: WorldSample) -> dict[str, Tensor]:
    return {m: getattr(sample, m).double().reshape(len(sample.label), -1) for m in MODALITIES}


def _check_seed(train_seed: int, legacy: int, stream: str, index: int, ns: int | None) -> int:
    if ns is None:
        return legacy
    from bouba_sens.seeds import STREAMS, ns_seed

    return ns_seed(ns, train_seed, STREAMS[stream], index)


def check_anchors(
    world: IndexedWorld, train_seed: int, seed_namespace: int | None = None
) -> tuple[WorldSample, Tensor]:
    """512 anchors: 32 contiguous windows of 16 (``seed = s + 3e6 + r``, or namespaced)."""
    parts = [
        world.sample_with_indices(
            CHECK_WINDOW,
            _check_seed(train_seed, train_seed + 3 * 10**6 + r, "check_anchor", r, seed_namespace),
        )
        for r in range(CHECK_N_WINDOWS)
    ]
    idx = torch.cat([p[1] for p in parts])
    return world.gather(idx), idx


def run_check_draw(
    world: IndexedWorld, *, train_seed: int, draw: int, seed_namespace: int | None = None
) -> CheckDraw:
    """One matched joint/decor draw on the pilot seed's anchors."""
    original, anchors = check_anchors(world, train_seed, seed_namespace)
    index = DonorIndex(world.dataset_labels)
    gen_seed = _check_seed(
        train_seed, check_generator_seed(train_seed, draw), "check_draw", draw, seed_namespace
    )
    donors, n_self = index.draw(anchors, gen_seed)
    joint = build_arm_sample(world, anchors, donors, "joint")
    decor = build_arm_sample(world, anchors, donors, "decor")
    labels = original.label
    return CheckDraw(
        k_joint=cka_statistic(_features(joint), labels),
        k_decor=cka_statistic(_features(decor), labels),
        k_original=cka_statistic(_features(original), labels),
        n_anchors=len(anchors),
        n_self_donor=n_self,
    )


@dataclass(frozen=True)
class GateResult:
    passed: bool
    delta_k: float
    p_value: float
    k_joint: float
    k_decor: float
    n_perm: int
    reason: str


def manipulation_gate(
    k_joint: Tensor, k_decor: Tensor, *, n_perm: int = GATE_N_PERM, seed: int = GATE_SEED
) -> GateResult:
    """Label-swap permutation gate on Delta_K = mean(K_joint - K_decor).

    Inputs have shape (n_seeds, n_draws). Joint/decor labels are swapped
    independently within each (seed, draw) pair (a sign flip of the
    difference). One-sided p = (1 + #{perm >= obs}) / (1 + n_perm); the
    gate passes if p < 0.05. Any non-finite entry (a draw with no valid
    (pair, class)) fails the check.
    """
    if k_joint.shape != k_decor.shape:
        raise ValueError(f"shape mismatch: {tuple(k_joint.shape)} vs {tuple(k_decor.shape)}")
    kj = k_joint.double().flatten()
    kd = k_decor.double().flatten()
    if not (torch.isfinite(kj).all() and torch.isfinite(kd).all()):
        return GateResult(
            False, math.nan, math.nan, math.nan, math.nan, n_perm,
            "a draw had no valid (pair, class): manipulation check failed",
        )  # fmt: skip
    diff = kj - kd
    obs = float(diff.mean())
    gen = torch.Generator().manual_seed(seed)
    ge = 0
    done = 0
    chunk = max(1, 4_000_000 // max(1, len(diff)))
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = torch.randint(0, 2, (m, len(diff)), generator=gen, dtype=torch.int8)
        perm = ((2 * signs - 1).double() * diff).mean(dim=1)
        ge += int((perm >= obs - 1e-12).sum().item())
        done += m
    p = (1 + ge) / (1 + n_perm)
    passed = p < GATE_ALPHA
    return GateResult(
        passed, obs, p, float(kj.mean()), float(kd.mean()), n_perm,
        "ok" if passed else "p >= 0.05: uninformative in this world",
    )  # fmt: skip


def run_manipulation_check(
    world: IndexedWorld,
    seeds: list[int],
    *,
    n_draws: int = CHECK_N_DRAWS,
    n_perm: int = GATE_N_PERM,
    seed_namespace: int | None = None,
) -> tuple[GateResult, float | None]:
    """Full gate over pilot ``seeds``; also returns mean CKA of the original anchors (reported)."""
    kj = torch.full((len(seeds), n_draws), math.nan, dtype=torch.float64)
    kd = kj.clone()
    originals: list[float] = []
    for i, s in enumerate(seeds):
        for b in range(1, n_draws + 1):
            d = run_check_draw(world, train_seed=s, draw=b, seed_namespace=seed_namespace)
            if d.k_joint is not None:
                kj[i, b - 1] = d.k_joint
            if d.k_decor is not None:
                kd[i, b - 1] = d.k_decor
            if d.k_original is not None:
                originals.append(d.k_original)
    k_orig = sum(originals) / len(originals) if originals else None
    return manipulation_gate(kj, kd, n_perm=n_perm), k_orig


# --- fusion gain (3.4) --------------------------------------------------------------------


@dataclass(frozen=True)
class FusionGain:
    acc_all: float
    acc_by_modality: Mapping[str, float]
    fusion_gain: float


def _accuracy(loop: AdaptationLoop, sample: WorldSample) -> float:
    with torch.no_grad():
        _, logits = loop._forward(sample)
    return float((logits.argmax(-1) == sample.label).float().mean().item())


def fusion_gain(loop: AdaptationLoop, *, train_seed: int) -> FusionGain:
    """FG = acc_all - max_m acc_m on the clean probe ``seed = s + 2e6`` (512 rows).

    When the world holds out probe frames (``n_probe`` > 0, v0.9
    section 3.4) the probe window is drawn from those frames and every
    single-modality accuracy is computed on that same window, with the
    other four modalities zeroed as in ``AdaptationLoop.query_accuracy``.
    Otherwise the legacy path is used unchanged.
    """
    probe_seed = train_seed + 2 * 10**6
    if getattr(loop.world, "n_probe", 0):
        sample, _ = loop.world.probe_sample_with_indices(PROBE_ROWS, probe_seed)  # type: ignore[attr-defined]
        acc_all = _accuracy(loop, sample)
        held: dict[str, float] = {}
        for m in MODALITIES:
            masked = replace(
                sample,
                **{
                    k: (getattr(sample, k) if k == m else torch.zeros_like(getattr(sample, k)))
                    for k in MODALITIES
                },
            )
            held[m] = _accuracy(loop, masked)
        return FusionGain(acc_all, held, acc_all - max(held.values()))
    sample = loop.world.sample(batch_size=PROBE_ROWS, seed=probe_seed)
    with torch.no_grad():
        _, logits = loop._forward(sample)
        acc_all = float((logits.argmax(-1) == sample.label).float().mean().item())
    per_m: dict[str, float] = {
        m: loop.query_accuracy(m, batch_size=PROBE_ROWS, seed=probe_seed) for m in MODALITIES
    }
    return FusionGain(acc_all, per_m, acc_all - max(per_m.values()))
