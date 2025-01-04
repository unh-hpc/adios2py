from __future__ import annotations

import os

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

    test_floats = np.asarray(np.arange(5))
    var = io.DefineVariable(
        "test_floats", test_floats, test_floats.shape, [0], test_floats.shape
    )
    engine.Put(var, test_floats, launch=ab.Mode.Sync)

    engine.Close()

    return filename


def test_write_test1_file(test1_file):
    """Checks that test1_file is properly written.

    And at the same time, that it can be read using the low-level API.
    """
    assert test1_file
