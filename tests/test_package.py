from __future__ import annotations

import importlib.metadata

import adios2py as m


def test_version():
    assert importlib.metadata.version("adios2py") == m.__version__
