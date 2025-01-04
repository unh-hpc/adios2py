from __future__ import annotations

import os
from typing import Any

import adios2.bindings as adios2bindings  # type: ignore[import-untyped]


class File:
    """
    Represents an ADIOS2 File or Stream.
    """

    def __init__(self, filename: os.PathLike[Any] | str, mode: str = "rra") -> None:
        """Open the file in the specified mode."""
        filename = os.fspath(filename)
        self._adios = adios2bindings.ADIOS()
        self._io_name = "io-adios2py"
        self._io = self._adios.DeclareIO(self._io_name)
        self._engine = self._io.Open(os.fspath(filename), _mode_to_adios2[mode])

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


_mode_to_adios2 = {
    "r": adios2bindings.Mode.Read,
    "rra": adios2bindings.Mode.ReadRandomAccess,
    "w": adios2bindings.Mode.Write,
}
