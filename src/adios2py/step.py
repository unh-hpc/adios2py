from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from adios2py.file import File


class Step:
    def __init__(self, File: File) -> None:
        self._file = File

    def read(self, name: str) -> np.ndarray[Any, Any]:
        return self._file.read(name)
