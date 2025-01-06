from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from adios2py.file import File


class Step(Mapping[str, np.ndarray[Any, Any]]):
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

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()

    def __getitem__(self, name: str) -> np.ndarray[Any, Any]:
        if self._file._mode not in ("r", "rra"):
            msg = f"Cannot read variables in mode {self._file._mode}."
            raise ValueError(msg)
        return self.read(name)

    def _keys(self) -> set[str]:
        return self._file.io.AvailableVariables()  # type: ignore[no-any-return]
