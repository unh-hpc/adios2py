from __future__ import annotations

import os
from typing import Any

import adios2.bindings as ab  # type: ignore[import-untyped]
import numpy as np
import pytest


@pytest.fixture
def test1_file(tmp_path):
    """Creates a simple adios2 test1.bp

    with an int and an array of floats"""

    filename = tmp_path / "test1.bp"
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.Write)

    test_int = np.array(99)
    var = io.DefineVariable("test_int", test_int)
    engine.Put(var, test_int, launch=ab.Mode.Sync)

    test_floats = np.asarray(np.arange(5.0))
    var = io.DefineVariable(
        "test_floats", test_floats, test_floats.shape, [0], test_floats.shape
    )
    engine.Put(var, test_floats, launch=ab.Mode.Sync)

    test_char2d = np.asarray(np.arange(12), dtype="int8").reshape(3, 4)
    var = io.DefineVariable(
        "test_char2d", test_char2d, test_char2d.shape, [0, 0], test_char2d.shape
    )
    engine.Put(var, test_char2d, launch=ab.Mode.Sync)

    engine.Close()

    return filename


def _adios2_to_dtype(type_str: str) -> np.dtype[Any]:
    if type_str.endswith("_t"):
        type_str = type_str[:-2]

    return np.dtype(type_str)


def check_test1_file_lowlevel(filename: os.PathLike[Any] | str) -> None:
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.ReadRandomAccess)
    assert engine

    var = io.InquireVariable("test_int")
    assert var
    dtype = _adios2_to_dtype(var.Type())
    assert dtype == np.int64
    size = var.Count()
    assert size == []
    shape = var.Shape()
    assert shape == []
    shape = tuple(shape)
    test_int = np.empty(shape, dtype=dtype)
    engine.Get(var, test_int)
    assert test_int == 99

    var = io.InquireVariable("test_floats")
    assert var
    dtype = _adios2_to_dtype(var.Type())
    assert dtype == np.float64
    size = var.Count()
    assert size == [5]
    shape = var.Shape()
    assert shape == [5]
    shape = tuple(shape)
    test_floats = np.empty(shape, dtype=dtype)
    engine.Get(var, test_floats)
    assert np.all(test_floats == np.arange(5.0))

    var = io.InquireVariable("test_char2d")
    assert var
    dtype = _adios2_to_dtype(var.Type())
    assert dtype == np.int8
    size = var.Count()
    assert size == [3, 4]
    shape = var.Shape()
    assert shape == [3, 4]
    shape = tuple(shape)
    test_char2d = np.empty(shape, dtype=dtype)
    engine.Get(var, test_char2d)
    assert np.all(test_char2d == np.arange(12).reshape(3, 4))


def test_write_test1_file(test1_file):
    """Checks that test1_file is properly written.

    And at the same time, that it can be read using the low-level API.
    """
    assert test1_file
    check_test1_file_lowlevel(test1_file)
