"""Known-signal calibration harness of amendment v0.7 (section 4.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import calibrate_mi_known_signal as cal  # type: ignore[import-not-found]  # noqa: E402


@pytest.fixture(scope="module")
def design() -> cal.Design:
    return cal.make_design(mc_draws=20_000)


def test_design_is_fixed_and_well_formed(design: cal.Design) -> None:
    again = cal.make_design(mc_draws=20_000)
    np.testing.assert_array_equal(design.means, again.means)
    np.testing.assert_array_equal(design.embed, again.embed)
    np.testing.assert_allclose(np.linalg.norm(design.means, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(design.embed.T @ design.embed, np.eye(cal.D_SIGNAL), atol=1e-12)
    assert design.embed.shape == (cal.D_AMBIENT, cal.D_SIGNAL)


def test_true_mi_limits(design: cal.Design) -> None:
    assert abs(cal.true_mi_bits(0.0, design)) < 1e-12
    assert abs(cal.true_mi_bits(200.0, design) - 2.0) < 1e-3  # log2(4 classes)
    vals = [cal.true_mi_bits(a, design) for a in (0.5, 1.0, 2.0, 4.0)]
    assert vals == sorted(vals)


def test_true_mi_matches_closed_form_two_point_case() -> None:
    """2 opposite unit means, 1-D projection: compare with a 1-D integral."""
    rng = np.random.default_rng(0)
    design = cal.Design(
        means=np.array([[1.0] + [0.0] * 11, [-1.0] + [0.0] * 11] * 2),
        embed=np.linalg.qr(rng.normal(size=(cal.D_AMBIENT, cal.D_SIGNAL)))[0],
        mc_noise=rng.normal(size=(400_000, cal.D_SIGNAL)),
        mc_labels=rng.integers(0, 4, size=400_000),
    )
    a, s = 1.3, np.sqrt(1.0 + cal.NOISE_SD**2)
    xs = np.linspace(-15, 15, 200_001)
    p1 = np.exp(-0.5 * ((xs - a) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    p2 = np.exp(-0.5 * ((xs + a) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    pm = 0.5 * (p1 + p2)
    exact = 0.5 * np.trapezoid(p1 * np.log2(p1 / pm) + p2 * np.log2(p2 / pm), xs)
    assert abs(cal.true_mi_bits(a, design) - exact) < 0.01


@pytest.mark.parametrize("target", [0.05, 0.25, 1.0, 1.2])
def test_solve_scale_hits_target(design: cal.Design, target: float) -> None:
    a = cal.solve_scale(target, design)
    assert abs(cal.true_mi_bits(a, design) - target) < 1e-4


def test_sample_codes_shape_and_balance(design: cal.Design) -> None:
    z, y = cal.sample_codes(1.0, design, np.random.default_rng(1))
    assert z.shape == (cal.N_ROWS, cal.D_AMBIENT)
    assert np.bincount(y).tolist() == [cal.N_ROWS // cal.N_CLASSES] * cal.N_CLASSES


def test_ls_line_exact() -> None:
    assert cal.ls_line([0, 1, 2], [1, 3, 5]) == pytest.approx((2.0, 1.0))


def test_calibrate_table_structure_with_oracle(design: cal.Design) -> None:
    """An estimator that returns the projected-code oracle has slope ~1."""

    def oracle(z: np.ndarray, y: np.ndarray) -> float:
        # class-mean separation along the signal subspace, monotone in a
        u = z @ design.embed
        return float(np.linalg.norm(u[y == 0].mean(0) - u[y == 1].mean(0)))

    out = cal.calibrate(
        {"oracle": oracle},
        baselines=(0.5,),
        deltas=(-0.1, 0.0, 0.1),
        replicates=3,
        design=design,
    )
    cells = out["cells"]
    assert len(cells) == 3
    assert {c["label"] for c in cells} == {"known-null", "injected"}
    assert {"pre_mean", "pre_sd", "delta_bias", "true_delta_bits"} <= set(cells[0])
    null = next(c for c in cells if c["delta_bits"] == 0.0)
    assert abs(null["true_delta_bits"]) < 1e-12
    assert out["slope_intercept"]["oracle"]["0.50"]["slope"] > 0
