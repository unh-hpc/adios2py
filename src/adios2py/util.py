from __future__ import annotations

import adios2.bindings as adios2bindings  # type: ignore[import-untyped]

_mode_to_adios2 = {
    "r": adios2bindings.Mode.Read,
    "rra": adios2bindings.Mode.ReadRandomAccess,
    "w": adios2bindings.Mode.Write,
}


def openmode_to_adios2(mode: str) -> adios2bindings.Mode:
    return _mode_to_adios2[mode]
