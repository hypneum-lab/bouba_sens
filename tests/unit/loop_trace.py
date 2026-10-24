"""Deterministic trace of the DEFAULT AdaptationLoop path (regression guard).

NOT comparable (explicitly excluded, with reason):
* callables (lesion schedule closure, bound methods): identity is not state;
  only their type name is hashed.
* the per-call ``torch.Generator`` created and discarded inside
  ``world.sample``: unreachable after the call; its effect is covered by the
  batch contents (code hashes) and the batch (size, seed) log.
* CUDA RNG: no CUDA device on the reference host (hashed when available).
* hash values on a different torch version / CPU architecture: float
  kernels may differ, so the test only enforces them where both match.

Used by test_loop_default_regression.py and, to generate the golden file,
against a checkout of origin/main (before the v0.8 probe/control options).
"""

from __future__ import annotations

import hashlib
import json
import platform

import torch
from track_p.multiplexer import GammaThetaMultiplexer

from bouba_sens.encoders import (
    AudioEncoder,
    ForceEncoder,
    GravityEncoder,
    TactileEncoder,
    VisionEncoder,
)
from bouba_sens.head import IntegrationHead
from bouba_sens.lesion import LesionSpec, m2_snr_schedule
from bouba_sens.loop import AdaptationLoop
from bouba_sens.nerve import CrossModalNerve
from bouba_sens.sensory import Modality, SensoryWML
from bouba_sens.world.gaussian import GaussianWorld


class RecordingWorld(GaussianWorld):
    def __init__(self) -> None:
        super().__init__(seed=0)
        self.calls: list[tuple[int, int]] = []

    def sample(self, batch_size: int, seed: int):  # type: ignore[no-untyped-def]
        self.calls.append((batch_size, seed))
        return super().sample(batch_size=batch_size, seed=seed)


PROVENANCE = {
    "reference_commit": "f99f2dc938fafa4584ff37aaad8f2d3a27bc024a",  # origin/main
    "reference_code": "src/bouba_sens/loop.py:110-259 (pretrain, restore, lesion_phase)",
    "generated_by": "python tests/unit/loop_trace.py on a checkout of that commit",
}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return _sha_bytes(str((t.dtype, tuple(t.shape))).encode() + t.numpy().tobytes())


def _walk(name: str, obj: object, out: dict[str, str], seen: set[int], depth: int = 0) -> None:
    """Hash everything reachable from ``obj`` (tensors, generators, containers, attributes).

    Not hashable by value, hence recorded only as a type tag: callables
    (bound methods, lesion schedules) and modules' non-state objects.
    """
    import numpy as np

    if isinstance(obj, bool | int | float | str | bytes) or obj is None:
        out[name] = _sha_bytes(repr(obj).encode())
        return
    if id(obj) in seen or depth > 6:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out[name] = _sha_tensor(obj)
    elif isinstance(obj, torch.Generator):
        out[name] = _sha_bytes(obj.get_state().numpy().tobytes())
    elif isinstance(obj, np.random.Generator):
        out[name] = _sha_bytes(repr(obj.bit_generator.state).encode())
    elif isinstance(obj, np.random.RandomState):
        out[name] = _sha_bytes(repr(obj.get_state()).encode())
    elif isinstance(obj, np.ndarray):
        out[name] = _sha_bytes(str((obj.dtype, obj.shape)).encode() + obj.tobytes())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(f"{name}[{k!r}]", v, out, seen, depth + 1)
    elif isinstance(obj, list | tuple | set | frozenset):
        items = sorted(obj, key=repr) if isinstance(obj, set | frozenset) else obj
        for i, v in enumerate(items):
            _walk(f"{name}[{i}]", v, out, seen, depth + 1)
    elif callable(obj) and not isinstance(obj, torch.nn.Module):
        out[name] = _sha_bytes(type(obj).__qualname__.encode())
    else:
        if isinstance(obj, torch.nn.Module):
            for k, v in obj.state_dict().items():
                out[f"{name}.state.{k}"] = _sha_tensor(v)
            for k, v in obj.named_buffers():
                out[f"{name}.buffer.{k}"] = _sha_tensor(v)
            for k, v in obj.named_parameters():
                out[f"{name}.param.{k}"] = _sha_tensor(v)
                out[f"{name}.requires_grad.{k}"] = _sha_bytes(repr(v.requires_grad).encode())
                out[f"{name}.grad.{k}"] = (
                    _sha_tensor(v.grad) if v.grad is not None else _sha_bytes(b"grad=None")
                )
            out[f"{name}.training"] = _sha_bytes(repr(obj.training).encode())
        for k, v in sorted(vars(obj).items()):
            if k.startswith("_") and isinstance(obj, torch.nn.Module) and k in _MODULE_INTERNALS:
                continue
            _walk(f"{name}.attr.{k}", v, out, seen, depth + 1)


_MODULE_INTERNALS = {
    "_parameters", "_buffers", "_modules", "_non_persistent_buffers_set", "training",
    "_backward_hooks", "_backward_pre_hooks", "_forward_hooks", "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called", "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks", "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks", "_is_full_backward_hook",
}  # fmt: skip


def loop_entries(loop: AdaptationLoop) -> dict[str, str]:
    """Hash the COMPLETE loop state (incl. the world), enumerated from the loop object."""
    out: dict[str, str] = {}
    for attr, value in sorted(vars(loop).items()):
        if isinstance(value, torch.optim.Optimizer):
            sd = value.state_dict()
            out[f"{attr}.param_groups"] = _sha_bytes(
                json.dumps(sd["param_groups"], sort_keys=True, default=str).encode()
            )
            for pid, st in sorted(sd["state"].items()):
                for k, v in sorted(st.items()):
                    out[f"{attr}.state[{pid}].{k}"] = (
                        _sha_tensor(v)
                        if isinstance(v, torch.Tensor)
                        else _sha_bytes(str(v).encode())
                    )
        else:
            _walk(attr, value, out, set())
    return out


def _rng_entries() -> dict[str, str]:
    import random

    import numpy as np
    import torch.mps

    dev: dict[str, str] = {}
    if torch.cuda.is_available():
        dev["rng.cuda"] = _sha_bytes(
            b"".join(t.numpy().tobytes() for t in torch.cuda.get_rng_state_all())
        )
    if torch.backends.mps.is_available():
        dev["rng.mps"] = _sha_bytes(torch.mps.get_rng_state().numpy().tobytes())
    return {
        **dev,
        "rng.torch": _sha_bytes(torch.get_rng_state().numpy().tobytes()),
        "rng.numpy": _sha_bytes(repr(np.random.get_state()).encode()),
        "rng.python": _sha_bytes(repr(random.getstate()).encode()),
    }


def trace() -> dict[str, object]:
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    world = RecordingWorld()
    mux = GammaThetaMultiplexer(seed=0)
    sensories: dict[Modality, SensoryWML] = {
        "audio": SensoryWML(id=0, modality="audio", input_proj=AudioEncoder(), mux=mux, seed=1),
        "vision": SensoryWML(id=1, modality="vision", input_proj=VisionEncoder(), mux=mux, seed=2),
        "tactile": SensoryWML(
            id=2, modality="tactile", input_proj=TactileEncoder(), mux=mux, seed=3
        ),
        "gravity": SensoryWML(
            id=3, modality="gravity", input_proj=GravityEncoder(), mux=mux, seed=4
        ),
        "force": SensoryWML(id=4, modality="force", input_proj=ForceEncoder(), mux=mux, seed=5),
    }
    nerve = CrossModalNerve(mux, seed=0)
    head = IntegrationHead(n_classes=4)
    loop = AdaptationLoop(world, mux, sensories, nerve, head, lr=1e-3)

    torch.manual_seed(1)
    loop.pretrain(steps=5, batch_size=4, seed=0)
    n_pretrain = len(world.calls)
    entries = {f"after_pretrain.{k}": v for k, v in loop_entries(loop).items()}
    spec = LesionSpec(modality="audio", mode="M2", timing="T2", schedule=m2_snr_schedule)
    report = loop.lesion_phase(spec, steps=6, batch_size=4, seed=50, stats_every=2)
    assert report.pre_lesion_codes is not None and report.post_lesion_codes is not None
    entries.update({f"after_lesion.{k}": v for k, v in loop_entries(loop).items()})
    entries.update(_rng_entries())
    entries["report.pre_lesion_codes"] = _sha_tensor(report.pre_lesion_codes)
    entries["report.post_lesion_codes"] = _sha_tensor(report.post_lesion_codes)
    entries["report.probe_labels"] = _sha_tensor(report.probe_labels)  # type: ignore[arg-type]
    for fld in ("loss_curve", "accuracy_curve", "codebook_entropy_trajectory"):
        entries[f"report.{fld}"] = _sha_bytes(json.dumps(getattr(report, fld)).encode())
    entries["report.lesion_events"] = _sha_bytes(repr(report.lesion_events).encode())
    entries["report.transducer_activation_trajectory"] = _sha_bytes(
        json.dumps(report.transducer_activation_trajectory).encode()
    )
    entries["report.gate_trajectory"] = _sha_bytes(
        json.dumps(report.gate_trajectory, sort_keys=True).encode()
    )
    return {
        "provenance": PROVENANCE,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "pretrain_calls": world.calls[:n_pretrain],
        "lesion_calls": world.calls[n_pretrain:],
        "loss_curve": report.loss_curve,
        "entries": entries,
    }


if __name__ == "__main__":
    print(json.dumps(trace(), indent=1))
