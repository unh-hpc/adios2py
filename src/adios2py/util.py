from __future__ import annotations

from typing import Any

import adios2.bindings as adios2bindings  # type: ignore[import-untyped]
import numpy as np


def adios2_to_dtype(type_str: str) -> np.dtype[Any]:
    if type_str == "char":
        type_str = "int8" if adios2bindings.is_char_signed else "uint8"
    elif type_str.endswith("_t"):
        type_str = type_str[:-2]

    return np.dtype(type_str)
