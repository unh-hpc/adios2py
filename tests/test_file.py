from __future__ import annotations

import os

import adios2.bindings as ab  # type: ignore[import-untyped]
import numpy as np


def test_write_file(tmp_path):
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(tmp_path / "test1.bp"), ab.Mode.Write)

    test_int = np.array(99)
    var = io.DefineVariable("test_int", test_int)
    engine.Put(var, test_int, launch=ab.Mode.Sync)

    test_floats = np.asarray(np.arange(5))
    var = io.DefineVariable(
        "test_floats", test_floats, test_floats.shape, [0], test_floats.shape
    )
    engine.Put(var, test_floats, launch=ab.Mode.Sync)

    engine.Close()
