from __future__ import annotations

from lgdo import lh5

from ._version import version as __version__
from .build_hit import build_hit

__all__ = [
    "__version__",
    "build_hit",
]

lh5.settings.DEFAULT_HDF5_SETTINGS = {"shuffle": True, "compression": "lzf"}
