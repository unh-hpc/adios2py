from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from adios2py.file import File


class Step:
    def __init__(self, file: File, step: int | None = None) -> None:
        """Represents a step in an ADIOS2 file."""
        self._file = file
        if step is None:
            step = file.current_step()
        self._step = step

    def read(self, name: str) -> np.ndarray[Any, Any]:
        if self._file._mode == "rra":
            return self._file._read(name, step_selection=(self._step, 1))

        assert self._step == self._file.current_step()
        return self._file._read(name)

    def write(self, name: str, data: np.ndarray[Any, Any]) -> None:
        self._file._write(name, data)
