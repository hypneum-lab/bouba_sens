"""Modality-specific input_proj encoders for SensoryWML (spec §3.2, Task 1.7).

Each encoder maps the native tensor shape of its modality to a common
`(B, d_hidden=128)` latent. All constructors save+restore
`torch.get_rng_state()` around nn layer creation so they never pollute
the global RNG (MlpWML convention).
"""

from bouba_sens.encoders.audio import AudioEncoder
from bouba_sens.encoders.force import ForceEncoder
from bouba_sens.encoders.gravity import GravityEncoder
from bouba_sens.encoders.tactile import TactileEncoder
from bouba_sens.encoders.vision import VisionEncoder

__all__ = [
    "AudioEncoder",
    "ForceEncoder",
    "GravityEncoder",
    "TactileEncoder",
    "VisionEncoder",
]
