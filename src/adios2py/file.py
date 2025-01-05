from __future__ import annotations

import os
from typing import Any

import adios2.bindings as adios2bindings  # type: ignore[import-untyped]
import numpy as np

from adios2py import util


class File:
    """
    Represents an ADIOS2 File or Stream.
    """

    _io: adios2bindings.IO | None = None
    _engine: adios2bindings.Engine | None = None
    _mode: str = ""

    def __init__(self, filename: os.PathLike[Any] | str, mode: str = "rra") -> None:
        """Open the file in the specified mode."""
        filename = os.fspath(filename)
        self._mode = mode
        self._adios = adios2bindings.ADIOS()
        self._io_name = "io-adios2py"
        self._io = self._adios.DeclareIO(self._io_name)
        self._engine = self._io.Open(os.fspath(filename), util.openmode_to_adios2(mode))

    def __bool__(self) -> bool:
        """Returns True if the file is open."""
        return self._engine is not None and self._io is not None

    @property
    def io(self) -> adios2bindings.IO:
        """Returns the underlying IO object."""
        assert self  # is_open
        return self._io

    @property
    def engine(self) -> adios2bindings.Engine:
        """Returns the underlying Engine object."""
        assert self  # is_open
        return self._engine

    def close(self) -> None:
        """Close the file."""
        if not self:
            msg = "File is not open"
            raise ValueError(msg)
        self.engine.Close()
        self._adios.RemoveIO(self._io_name)
        self._engine = None
        self._io = None

    def __del__(self) -> None:
        """Close the file when the object is deleted."""
        if self:
            self.close()

    def __repr__(self) -> str:
        if not self:
            return "adios2py.File(closed)"

        filename = self.engine.Name()
        return f"adios2py.File({filename=}, mode={self._mode})"

    def __enter__(self) -> File:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def read(self, name: str) -> np.ndarray[Any, Any]:
        """Read a variable from the file."""
        var = self.io.InquireVariable(name)
        if not var:
            msg = f"Variable '{name}' not found"
            raise ValueError(msg)
        dtype = util.adios2_to_dtype(var.Type())
        shape = tuple(var.Shape())
        data = np.empty(shape, dtype=dtype)
        self.engine.Get(var, data, adios2bindings.Mode.Sync)
        return data

    def write(self, name: str, data: np.ndarray[Any, Any]) -> None:
        var = self.io.DefineVariable(
            name, data, data.shape, [0] * data.ndim, data.shape
        )
        self.engine.Put(var, np.asarray(data), adios2bindings.Mode.Sync)
