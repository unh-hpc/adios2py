from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from adios2py import util
from adios2py.array_proxy import ArrayProxy

if TYPE_CHECKING:
    from adios2py.file import File


class Group(Mapping[str, ArrayProxy]):
    def __init__(self, file: File, step: int | None = None) -> None:
        self._file = file
        self._step = step

    def __len__(self) -> int:
        return len(self._file._available_variables())

    def __iter__(self) -> Iterator[str]:
        yield from self._file._available_variables().keys()

    def __getitem__(self, name: str) -> ArrayProxy:
        """Read a variable from the file."""
        var = self._file.io.InquireVariable(name)
        if not var:
            msg = f"Variable {name} not found."
            raise KeyError(msg)
        dtype = np.dtype(util.adios2_to_dtype(var.Type()))
        if self._step is None:  # represent all steps as additional dimension
            shape = (self._file._steps(), *var.Shape())
            return ArrayProxy(
                self._file, step=slice(None), name=name, dtype=dtype, shape=shape
            )

        shape = tuple(var.Shape())
        return ArrayProxy(self._file, self._step, name, dtype, shape)
