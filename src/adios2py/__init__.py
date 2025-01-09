"""
Copyright (c) 2025 Kai Germaschewski. All rights reserved.

adios2py: A pythonic interface to the ADIOS2 I/O library
"""

from __future__ import annotations

from . import util
from ._version import version as __version__
from .file import File
from .group import Group

__all__ = [
    "File",
    "Group",
    "__version__",
    "util",
]
