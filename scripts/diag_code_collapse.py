"""Diagnose the code-channel collapse of the sensory pathways (exploratory).

ADR-0023 found that each SensoryWML emits a single argmax code for almost
every input after the grid's training schedule. This script replays Phase 1
(``bouba-sens train`` defaults: 200 steps, batch 16, hard gating) for a few
seeds and worlds and records, every ``--every`` steps, on a fixed clean
evaluation batch of 256 items:

- per modality: number of active codes (distinct argmax over the 64-code
  alphabet), code-usage entropy (bits), the same two quantities for the
  hidden state centred on its batch mean (counterfactual: what the argmax
  would do without the shared offset), the offset ratio
  ``||mean_b h|| / mean_b ||h - mean_b h||``, the top-1 logit margin, and
  the fraction of the carrier demodulated back to the emitted code;
- per modality: the gradient norm reaching ``emit_head_pi`` and the encoder
  on the current training step (the learning signal that can move the code);
- global: training loss, training accuracy, accuracy of the full network on
  the evaluation batch.

Evaluation saves and restores the global torch RNG, so the training
trajectory is the one ``bouba-sens train`` produces for the same seed.

With ``--variant`` the same schedule is replayed with two
counterfactual emission rules, monkeypatched in this process only (library
defaults unchanged):

- ``centred``: argmax of ``emit_head_pi(h - mean_b h)``;
- ``st``: straight-through estimator: forward = one-hot of the argmax,
  backward = gradient of the softmax (replaces the 1e-3 bridge).

Usage:
    uv run python scripts/diag_code_collapse.py --seeds 0 1 --worlds gaussian studyforrest \\
        --variant default --out reports/diag_code_collapse_default.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from bouba_sens.cli import _build_world, _seed_everything
from bouba_sens.encoders import (
    AudioEncoder,
    ForceEncoder,
    GravityEncoder,
    TactileEncoder,
    VisionEncoder,
)
from bouba_sens.head import IntegrationHead
from bouba_sens.loop import AdaptationLoop
from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import MODALITIES, Modality, SensoryWML

EVAL_SEED = 777_000
EVAL_SIZE = 256


def _entropy_bits(codes: Tensor, n: int) -> float:
    counts = torch.bincount(codes.flatten(), minlength=n).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def _build(world_name: str, seed: int) -> AdaptationLoop:
    """Same construction as ``bouba-sens train`` (cli.train)."""
    from track_p.multiplexer import GammaThetaMultiplexer

    _seed_everything(seed)
    world = _build_world(world_name, seed)
    mux = GammaThetaMultiplexer(seed=seed)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(0, "audio", AudioEncoder(), mux, seed=seed + 1),
        "vision": SensoryWML(1, "vision", VisionEncoder(), mux, seed=seed + 2),
        "tactile": SensoryWML(2, "tactile", TactileEncoder(), mux, seed=seed + 3),
        "gravity": SensoryWML(3, "gravity", GravityEncoder(), mux, seed=seed + 4),
        "force": SensoryWML(4, "force", ForceEncoder(), mux, seed=seed + 5),
    }
    nerve = CrossModalNerve(mux, seed=seed, transducer_gating="hard", gumbel_tau=1.0)
    head = IntegrationHead(n_classes=4)
    return AdaptationLoop(world, mux, sensories, nerve, head)


def _install_variant(variant: str) -> None:
    """Monkeypatch ``SensoryWML.step`` in this process (counterfactual only)."""
    if variant == "default":
        return

    def step(self: SensoryWML, x: Tensor) -> Tensor:
        h = self.core(self.input_proj(x))
        if variant == "centred":
            h = h - h.mean(0, keepdim=True)
        logits = self.emit_head_pi(h)
        k = self.mux.cfg.symbols_per_theta
        if variant == "centred":
            codes = logits.argmax(-1).unsqueeze(-1).expand(-1, k)
            carrier: Tensor = self.mux.forward(codes)
            soft_iq = logits.softmax(-1) @ self.mux.constellation
            return carrier + soft_iq.mean(-1, keepdim=True).expand_as(carrier) * 1e-3
        # Straight-through: forward = carrier of the argmax code, backward =
        # gradient of the softmax over the 64 codes (replaces the 1e-3 bridge).
        soft = logits.softmax(-1)
        hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        st = hard + soft - soft.detach()  # (B, A), value = one-hot
        alphabet = torch.arange(soft.shape[-1]).unsqueeze(-1).expand(-1, k)
        table = self.mux.forward(alphabet)  # (A, T) carrier of every code
        return st @ table

    SensoryWML.step = step  # type: ignore[method-assign]


def _evaluate(loop: AdaptationLoop, sample: Any) -> dict[str, Any]:
    rng = torch.get_rng_state()
    out: dict[str, Any] = {}
    n_alpha = loop.mux.cfg.alphabet_size
    with torch.no_grad():
        for m in MODALITIES:
            s = loop.sensories[m]
            h = s.core(s.input_proj(getattr(sample, m)))
            logits = s.emit_head_pi(h)
            codes = logits.argmax(-1)
            hc = h - h.mean(0, keepdim=True)
            codes_c = s.emit_head_pi(hc).argmax(-1)
            top2 = logits.topk(2, dim=-1).values
            offset = float(h.mean(0).norm() / (hc.norm(dim=-1).mean() + 1e-12))
            carrier = s.step(getattr(sample, m))
            demod = loop.mux.demodulate(carrier, hard=True)[:, 0]
            out[m] = {
                "active_codes": int(codes.unique().numel()),
                "usage_entropy_bits": _entropy_bits(codes, n_alpha),
                "active_codes_centred": int(codes_c.unique().numel()),
                "usage_entropy_bits_centred": _entropy_bits(codes_c, n_alpha),
                "offset_ratio": offset,
                "top1_margin": float((top2[:, 0] - top2[:, 1]).mean()),
                "demod_agreement": float((demod == codes).float().mean()),
            }
        _, logits = loop._forward(sample)
        out["full_input_accuracy"] = float((logits.argmax(-1) == sample.label).float().mean())
    torch.set_rng_state(rng)
    return out


def replay(
    world_name: str, seed: int, *, steps: int, batch: int, every: int, sensory_sgd: bool = False
) -> dict[str, Any]:
    loop = _build(world_name, seed)
    if sensory_sgd:
        # Counterfactual: plain SGD (same lr) on the sensory pathways, Adam on
        # the rest, to test whether Adam's scale invariance turns the tiny
        # bridge gradient into full-size updates.
        sensory_ids = {id(p) for s in loop.sensories.values() for p in s.parameters()}
        rest = [p for g in loop.opt.param_groups for p in g["params"] if id(p) not in sensory_ids]
        sens = [p for g in loop.opt.param_groups for p in g["params"] if id(p) in sensory_ids]
        adam, sgd = torch.optim.Adam(rest, lr=1e-3), torch.optim.SGD(sens, lr=1e-3)

        class _Both:
            def zero_grad(self) -> None:
                adam.zero_grad()
                sgd.zero_grad()

            def step(self) -> None:
                adam.step()
                sgd.step()

        loop.opt = _Both()  # type: ignore[assignment]
    eval_sample = loop.world.sample(batch_size=EVAL_SIZE, seed=EVAL_SEED)
    trace: list[dict[str, Any]] = [{"step": 0, **_evaluate(loop, eval_sample)}]
    losses: list[float] = []
    accs: list[float] = []
    grads: list[dict[str, dict[str, float]]] = []
    for step in range(steps):
        sample = loop.world.sample(batch_size=batch, seed=seed + step)
        loss, logits = loop._forward(sample)
        loop.opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        grads.append(
            {
                m: {
                    "emit_head_pi": float(
                        loop.sensories[m].emit_head_pi.weight.grad.norm()
                        if loop.sensories[m].emit_head_pi.weight.grad is not None
                        else 0.0
                    ),
                    "encoder": float(
                        math.sqrt(
                            sum(
                                float(p.grad.norm()) ** 2
                                for p in loop.sensories[m].input_proj.parameters()
                                if p.grad is not None
                            )
                        )
                    ),
                }
                for m in MODALITIES
            }
        )
        loop.opt.step()
        loop.mux.step()
        loop.nerve.step()
        losses.append(float(loss.detach()))
        accs.append(float((logits.argmax(-1) == sample.label).float().mean()))
        if (step + 1) % every == 0 or step == steps - 1:
            trace.append({"step": step + 1, **_evaluate(loop, eval_sample)})
    head_grad = {
        m: float(sum(g[m]["emit_head_pi"] for g in grads) / len(grads)) for m in MODALITIES
    }
    enc_grad = {m: float(sum(g[m]["encoder"] for g in grads) / len(grads)) for m in MODALITIES}
    return {
        "world": world_name,
        "seed": seed,
        "steps": steps,
        "batch": batch,
        "loss_curve": losses,
        "train_accuracy_curve": accs,
        "mean_grad_norm_emit_head_pi": head_grad,
        "mean_grad_norm_encoder": enc_grad,
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--worlds", nargs="+", default=["gaussian", "studyforrest"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--every", type=int, default=20)
    parser.add_argument("--variant", default="default", choices=["default", "centred", "st"])
    parser.add_argument(
        "--sensory-sgd", action="store_true", help="SGD instead of Adam on the sensory pathways"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _install_variant(args.variant)
    t0 = time.perf_counter()
    runs = [
        replay(
            w, s, steps=args.steps, batch=args.batch, every=args.every, sensory_sgd=args.sensory_sgd
        )
        for w in args.worlds
        for s in args.seeds
    ]
    result = {
        "schema": "bouba_sens.diag_code_collapse.v1",
        "variant": args.variant,
        "sensory_sgd": args.sensory_sgd,
        "eval": {"seed": EVAL_SEED, "size": EVAL_SIZE},
        "runs": runs,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    for r in runs:
        first, last = r["trace"][0], r["trace"][-1]
        print(
            r["world"],
            r["seed"],
            args.variant + ("+sgd" if args.sensory_sgd else ""),
            "codes",
            {m: (first[m]["active_codes"], last[m]["active_codes"]) for m in MODALITIES},
            "centred",
            {m: last[m]["active_codes_centred"] for m in MODALITIES},
            "offset",
            {m: round(last[m]["offset_ratio"], 2) for m in MODALITIES},
            "acc",
            round(last["full_input_accuracy"], 3),
            "loss",
            round(sum(r["loss_curve"][-20:]) / 20, 3),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
