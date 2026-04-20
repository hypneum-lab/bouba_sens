"""OQ1 spike — shared vs local codebook on a toy 2-modality task.

Runs a miniature controlled experiment to decide whether the 64-code alphabet
should be shared across SensoryWMLs (v0.1 default) or whether per-WML codebooks
plus a CodebookAligner give measurably better adaptation.

Usage:
    uv run python scripts/spikes/oq1_codebook.py --mode shared
    uv run python scripts/spikes/oq1_codebook.py --mode local
    uv run python scripts/spikes/oq1_codebook.py --mode both --report out/oq1_results.json

Execution policy:
    This script must be run on Studio (M3 Ultra 512GB) per the project's
    compute-routing directive — GrosMac (M5 dev machine) is reserved for
    lightweight orchestration. After execution, update docs/adr/0001-codebook-sharing.md
    with the empirical numbers from out/oq1_results.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim

Mode = Literal["shared", "local", "both"]


@dataclass
class SpikeResult:
    mode: str
    final_accuracy: float
    final_loss: float
    steps: int
    seed: int


class ToyTwoModalityTask:
    """Binary classification on two coherent modalities derived from a shared latent."""

    def __init__(self, d_z: int = 8, noise: float = 0.1) -> None:
        self.d_z = d_z
        self.noise = noise
        self.W_a = torch.randn(d_z, 16)
        self.W_v = torch.randn(d_z, 16)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randn(batch_size, self.d_z)
        audio = z @ self.W_a + self.noise * torch.randn(batch_size, 16)
        vision = z @ self.W_v + self.noise * torch.randn(batch_size, 16)
        label = (z[:, 0] > 0).long()
        return audio, vision, label


class SharedCodebookModel(nn.Module):
    """Both modalities project into a shared 64-code alphabet, fused by mean."""

    def __init__(self, k_codes: int = 64) -> None:
        super().__init__()
        self.audio_enc = nn.Linear(16, k_codes)
        self.vision_enc = nn.Linear(16, k_codes)
        self.head = nn.Linear(k_codes, 2)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.audio_enc(audio), dim=-1)
        v = torch.softmax(self.vision_enc(vision), dim=-1)
        fused = (a + v) / 2.0
        return self.head(fused)


class LocalCodebookModel(nn.Module):
    """Each modality has its own 64-code alphabet; an aligner learns to match."""

    def __init__(self, k_codes: int = 64) -> None:
        super().__init__()
        self.audio_enc = nn.Linear(16, k_codes)
        self.vision_enc = nn.Linear(16, k_codes)
        self.aligner = nn.Linear(2 * k_codes, k_codes)
        self.head = nn.Linear(k_codes, 2)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.audio_enc(audio), dim=-1)
        v = torch.softmax(self.vision_enc(vision), dim=-1)
        fused = self.aligner(torch.cat([a, v], dim=-1))
        return self.head(fused)


def train(model: nn.Module, task: ToyTwoModalityTask, steps: int, seed: int) -> SpikeResult:
    torch.manual_seed(seed)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(steps):
        audio, vision, label = task.sample(batch_size=128)
        logits = model(audio, vision)
        loss = nn.functional.cross_entropy(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    audio, vision, label = task.sample(batch_size=1024)
    with torch.no_grad():
        preds = model(audio, vision).argmax(dim=-1)
        acc = (preds == label).float().mean().item()
    return SpikeResult(
        mode=type(model).__name__,
        final_accuracy=acc,
        final_loss=losses[-1],
        steps=steps,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shared", "local", "both"], default="both")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--report", type=Path, default=Path("out/oq1_results.json"))
    args = parser.parse_args()

    task = ToyTwoModalityTask()
    all_results: list[SpikeResult] = []

    for seed in range(args.seeds):
        if args.mode in ("shared", "both"):
            all_results.append(train(SharedCodebookModel(), task, args.steps, seed))
        if args.mode in ("local", "both"):
            all_results.append(train(LocalCodebookModel(), task, args.steps, seed))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps([asdict(r) for r in all_results], indent=2))

    grouped: dict[str, list[float]] = {}
    for r in all_results:
        grouped.setdefault(r.mode, []).append(r.final_accuracy)
    print("\n=== OQ1 Spike Results ===")
    for mode, accs in grouped.items():
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1))
        print(f"{mode:<24s}  acc = {mean:.4f} +/- {std:.4f}  (n={len(accs)})")


if __name__ == "__main__":
    main()
