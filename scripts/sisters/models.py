"""Public model definitions and native binary export/import for Sisters of Battle."""
from __future__ import annotations

from pathlib import Path
import sys

_LIB = Path(__file__).resolve().parent / "_lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

from train_sisters_recurrent_world import World  # noqa: E402
from wp18_scaled_model import (  # noqa: E402
    PHASE_MAGIC,
    SCALED_MAGIC,
    ScaledRecurrentActor,
    ScaledRecurrentValues,
    serialize_scaled_actor,
)
from wp22_attn_model import (  # noqa: E402
    ATTN_MAGIC,
    AttnRecurrentActor,
    AttnRecurrentValues,
    export_two_phase_attn,
    motion_history_4lag,
    serialize_attn_actor,
)

__all__ = [
    "ATTN_MAGIC",
    "PHASE_MAGIC",
    "SCALED_MAGIC",
    "AttnRecurrentActor",
    "AttnRecurrentValues",
    "ScaledRecurrentActor",
    "ScaledRecurrentValues",
    "World",
    "export_two_phase_attn",
    "motion_history_4lag",
    "serialize_attn_actor",
    "serialize_scaled_actor",
]
