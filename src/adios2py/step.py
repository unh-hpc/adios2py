from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from adios2py.file import File

from adios2py import util
from adios2py.array_proxy import ArrayProxy


class Step(Mapping[str, ArrayProxy]):
    """
    Represents a step in an ADIOS2 file.
    """

    def __init__(self, file: File, step: int | None = None) -> None:
        self._file = file
        if step is None:
            step = file.current_step()
        self._step = step

    def _write(self, name: str, data: ArrayLike) -> None:
        self._file._write(name, data)  # pylint: disable=W0212

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()

    def __getitem__(self, name: str) -> ArrayProxy:
        var = self._file.io.InquireVariable(name)
        if not var:
            msg = f"Variable {name} not found."
            raise KeyError(msg)
        dtype = np.dtype(util.adios2_to_dtype(var.Type()))
        shape = tuple(var.Shape())
        return ArrayProxy(self._file, self._step, name, dtype, shape)

    def __setitem__(self, name: str, data: ArrayLike) -> None:
        self._write(name, data)

    def _keys(self) -> set[str]:
        return self._file.io.AvailableVariables()  # type: ignore[no-any-return]
