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
        self._io = self._adios.DeclareIO("io-test1")
        self._engine = self._io.Open(os.fspath(filename), _mode_to_adios2[mode])


_mode_to_adios2 = {
    "r": adios2bindings.Mode.Read,
    "rra": adios2bindings.Mode.ReadRandomAccess,
    "w": adios2bindings.Mode.Write,
}
