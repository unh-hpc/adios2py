"""
Copyright (c) 2025 Kai Germaschewski. All rights reserved.

adios2py: A pythonic interface to the ADIOS2 I/O library
"""

from __future__ import annotations

from . import util
from ._version import version as __version__
from .array_proxy import ArrayProxy
from .attrs_proxy import AttrsProxy
from .file import File
from .group import Group
from .step import Step
from .steps_proxy import StepsProxy

__all__ = [
    "ArrayProxy",
    "AttrsProxy",
    "File",
    "Group",
    "Step",
    "StepsProxy",
    "__version__",
    "util",
]
