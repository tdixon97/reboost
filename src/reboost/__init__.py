from __future__ import annotations

from lgdo import lh5

from . import build_hit, core, iterator, math, shape
from ._version import version as __version__

__all__ = [
    "__version__",
    "build_glm",
    "build_hit",
    "build_hit",
    "build_tcm",
    "core",
    "iterator",
    "math",
    "optmap",
    "shape",
]

lh5.settings.DEFAULT_HDF5_SETTINGS = {"shuffle": True, "compression": "lzf"}
