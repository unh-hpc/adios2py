from __future__ import annotations

import os
from typing import Any

import adios2.bindings as ab  # type: ignore[import-untyped]
import numpy as np
import pytest

import adios2py

sample_data = {
    "test_int_0d": np.array(99),
    "test_float_1d": np.arange(5.0),
    "test_int8_2d": np.arange(12, dtype="int8").reshape(3, 4),
    "test_uint8_1d": np.arange(4, dtype="uint8"),
}


@pytest.fixture
def test1_file(tmp_path):
    """Fixture returning the filename of a simple adios2 test file containing sample_data."""

    filename = tmp_path / "test1.bp"
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.Write)

    for name, data in sample_data.items():
        var = io.DefineVariable(name, data, data.shape, [0] * data.ndim, data.shape)
        engine.Put(var, np.asarray(data), ab.Mode.Sync)

    engine.Close()

    return filename


def check_test1_file_lowlevel(filename: os.PathLike[Any] | str) -> None:
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.ReadRandomAccess)
    assert engine

    for name, ref_data in sample_data.items():
        var = io.InquireVariable(name)
        assert var
        dtype = adios2py.util.adios2_to_dtype(var.Type())
        assert dtype == ref_data.dtype
        shape = tuple(var.Shape())
        assert shape == ref_data.shape
        data = np.empty(shape, dtype=dtype)
        engine.Get(var, data, ab.Mode.Sync)
        assert np.all(data == ref_data)

    engine.Close()


@pytest.mark.xfail
def test_adios2_1(tmp_path):
    filename = tmp_path / "test.bp"
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.Write)
    data = np.array(99)
    var = io.DefineVariable("test", data, data.shape, [0] * data.ndim, data.shape)
    engine.Put(var, np.asarray(data), ab.Mode.Sync)
    engine.Close()

    import subprocess

    proc = subprocess.run(
        ["bpls", "-dt", filename], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0
    assert "99" in str(proc.stdout)

    proc = subprocess.run(
        ["bpls", "-dDt", filename], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0
    assert "99" in str(proc.stdout)


@pytest.mark.xfail
def test_adios2_2(tmp_path):
    filename = tmp_path / "test.bp"
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.Write)
    data = np.array(99, dtype=np.uint8)
    var = io.DefineVariable("test", data, data.shape, [0] * data.ndim, data.shape)
    engine.Put(var, np.asarray(data), ab.Mode.Sync)
    engine.Close()

    import subprocess

    proc = subprocess.run(
        ["bpls", "-dt", filename], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0
    assert "char" in str(proc.stdout)


@pytest.mark.skip
def test_adios2_3(tmp_path):
    filename = tmp_path / "test.bp"
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test1")
    engine = io.Open(os.fspath(filename), ab.Mode.Write)
    data = np.array(99, dtype=np.uint8)
    var = io.DefineVariable("test", data, data.shape, [0] * data.ndim, data.shape)
    engine.Put(var, np.asarray(data), ab.Mode.Sync)
    engine.Close()
    ad.RemoveAllIOs()

    assert var.Name()


def test_write_test1_file_lowlevel(test1_file):
    """Checks that test1_file is properly written.

    And at the same time, that it can be read using the low-level API.
    """
    assert test1_file
    check_test1_file_lowlevel(test1_file)


def test_File_open(test1_file):
    adios2py.File(test1_file)


def test_File_bool(test1_file):
    file = adios2py.File(test1_file)
    assert bool(file)


def test_File_close(test1_file):
    file = adios2py.File(test1_file)
    assert file
    file.close()
    assert "closed" in repr(file)
    assert not file
    with pytest.raises(ValueError, match="File is not open"):
        file.close()


def test_File_del(test1_file):
    file = adios2py.File(test1_file)
    assert file
    del file


def test_File_repr(test1_file):
    file = adios2py.File(test1_file)
    assert "adios2py.File" in repr(file)


def test_File_with(test1_file):
    with adios2py.File(test1_file) as file:
        assert file


def test_File_read(test1_file):
    file = adios2py.File(test1_file)
    for name, ref_data in sample_data.items():
        data = file._read(name)
        assert data.dtype == ref_data.dtype
        assert data.shape == ref_data.shape
        assert np.all(data == ref_data)
        with pytest.raises(ValueError, match="not found"):
            file._read("not_there")


def test_write_test1_file(tmp_path):
    filename = tmp_path / "test1.bp"
    with adios2py.File(filename, "w") as file:  # noqa: SIM117
        with file.steps.next() as step:
            for name, data in sample_data.items():
                step.write(name, data)

    check_test1_file_lowlevel(filename)


# Tests for a file with multiple steps


@pytest.fixture
def test2_file(tmp_path):
    """Fixture returning the filename of a adios2 test file containing two steps."""

    filename = tmp_path / "test2.bp"
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test2")
    engine = io.Open(os.fspath(filename), ab.Mode.Write)

    for step in range(2):
        status = engine.BeginStep()
        assert status == ab.StepStatus.OK
        for name, data in sample_data.items():
            var = io.InquireVariable(name)
            if not var:
                var = io.DefineVariable(
                    name, data, data.shape, [0] * data.ndim, data.shape
                )
            engine.Put(var, np.asarray(data + step), ab.Mode.Sync)
        engine.EndStep()

    engine.Close()

    return filename


def check_test2_file_lowlevel(filename: os.PathLike[Any] | str) -> None:
    ad = ab.ADIOS()
    io = ad.DeclareIO("io-test2")
    engine = io.Open(os.fspath(filename), ab.Mode.Read)
    assert engine

    for step in range(2):
        status = engine.BeginStep()
        assert status == ab.StepStatus.OK
        for name, ref_data in sample_data.items():
            var = io.InquireVariable(name)
            assert var
            dtype = adios2py.util.adios2_to_dtype(var.Type())
            assert dtype == ref_data.dtype
            shape = tuple(var.Shape())
            assert shape == ref_data.shape
            data = np.empty(shape, dtype=dtype)
            engine.Get(var, data, ab.Mode.Sync)
            assert np.all(data == ref_data + step)
        engine.EndStep()

    engine.Close()


def test_write_test2_file_lowlevel(test2_file):
    """Checks that test2_file is properly written.

    And at the same time, that it can be read using the low-level API.
    """
    assert test2_file
    check_test2_file_lowlevel(test2_file)


def test_write_test2_file(tmp_path):
    filename = tmp_path / "test2.bp"
    with adios2py.File(filename, "w") as file:
        for n in range(2):
            with file.steps.next() as step:
                for name, data in sample_data.items():
                    step.write(name, data + n)

    check_test2_file_lowlevel(filename)


def test_test2_read(test2_file):
    file = adios2py.File(test2_file, "r")
    for n in range(2):
        file._begin_step()
        for name, ref_data in sample_data.items():
            data = file._read(name)
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)
        file._end_step()


def test_test2_read_rra(test2_file):
    file = adios2py.File(test2_file, "rra")
    for n in range(2):
        step = file.steps[n]
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_File_next_step(test2_file, mode):
    file = adios2py.File(test2_file, mode)
    for n in range(2):
        with file.steps.next() as step:
            for name, ref_data in sample_data.items():
                data = step[name]
                assert data.dtype == ref_data.dtype
                assert data.shape == ref_data.shape
                assert np.all(data == ref_data + n)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_File_iter(test2_file, mode):
    file = adios2py.File(test2_file, mode=mode)

    for n, step in enumerate(file.steps):
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_File_iter_break(test2_file, mode):
    """Test that breaking out of the iteration works, and iteration can be resumed."""
    file = adios2py.File(test2_file, mode=mode)

    for n, step in enumerate(file.steps):
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)
        break

    assert not file.in_step()

    for n, step in enumerate(file.steps):
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n + 1)
        break


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_File_num_steps(test2_file, mode):
    file = adios2py.File(test2_file, mode=mode)
    assert len(file.steps) == 2


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_Step_Mapping(test2_file, mode):
    file = adios2py.File(test2_file, mode=mode)
    for step in file.steps:
        assert len(step) == len(sample_data)
        assert set(step) == set(sample_data)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_Step_getitem(test2_file, mode):
    file = adios2py.File(test2_file, mode=mode)
    for n, step in enumerate(file.steps):
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.ndim == ref_data.ndim
            assert data.size == data.size
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)

            if data.ndim == 0:
                msg = r"len\(\) of unsized object"
                with pytest.raises(TypeError, match=msg):
                    len(data)
                with pytest.raises(TypeError, match=msg):
                    len(ref_data)
            else:
                assert len(data) == len(ref_data)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_Step_stale(test2_file, mode):
    file = adios2py.File(test2_file, mode=mode)
    steps = list(file.steps)

    for n, step in enumerate(steps):
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            if mode == "r":
                with pytest.raises(ValueError, match="non-current step"):
                    assert np.all(data == ref_data + n)
            else:
                assert np.all(data == ref_data + n)


@pytest.mark.parametrize("mode", ["r", "rra"])
def test_Array_stale(test2_file, mode):
    file = adios2py.File(test2_file, mode=mode)
    steps = [dict(step.items()) for step in file.steps]

    for n, step in enumerate(steps):
        for name, ref_data in sample_data.items():
            data = step[name]
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            if mode == "r":
                with pytest.raises(ValueError, match="non-current step"):
                    assert np.all(data == ref_data + n)
            else:
                assert np.all(data == ref_data + n)


def test_File_getitem(test2_file):
    file = adios2py.File(test2_file, mode="rra")
    for n in range(len(file.steps)):
        for name, ref_data in sample_data.items():
            data = file[name][n, ...]
            assert data.ndim == ref_data.ndim
            assert data.size == data.size
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)


def test_File_getitem_r(test2_file):
    file = adios2py.File(test2_file, mode="r")
    for n, _ in enumerate(file.steps):
        for name, ref_data in sample_data.items():
            data = file[name][n, ...]
            assert data.ndim == ref_data.ndim
            assert data.size == data.size
            assert data.dtype == ref_data.dtype
            assert data.shape == ref_data.shape
            assert np.all(data == ref_data + n)


def test_File_getitem_r_out_of_order(test2_file):
    file = adios2py.File(test2_file, mode="r")
    for n, _ in enumerate(file.steps):
        for name, _ in sample_data.items():
            with pytest.raises(ValueError, match="non-current step"):
                file[name][n + 1, ...]


def test_File_getitem_slice(test2_file):
    file = adios2py.File(test2_file, mode="rra")
    for n in range(len(file.steps)):
        data = file["test_float_1d"][n, 1:3]
        ref_data = sample_data["test_float_1d"][1:3]
        assert data.ndim == ref_data.ndim
        assert data.size == data.size
        assert data.dtype == ref_data.dtype
        assert data.shape == ref_data.shape
        assert np.all(data == ref_data + n)


def test_File_getitem_time_slice(test2_file):
    file = adios2py.File(test2_file, mode="rra")
    ref_data = sample_data["test_float_1d"][np.newaxis, 1:3]
    data = file["test_float_1d"][0:1, 1:3]
    assert data.ndim == ref_data.ndim
    assert data.size == data.size
    assert data.dtype == ref_data.dtype
    assert data.shape == ref_data.shape
    assert np.all(data == ref_data)


def test_File_getitem_time_slice2(test2_file):
    file = adios2py.File(test2_file, mode="rra")
    ref_data = np.stack(
        (sample_data["test_float_1d"], sample_data["test_float_1d"] + 1), axis=0
    )
    ref_data = ref_data[0:2, 1:3]
    data = file["test_float_1d"][0:2, 1:3]
    assert data.ndim == ref_data.ndim
    assert data.size == data.size
    assert data.dtype == ref_data.dtype
    assert data.shape == ref_data.shape
    assert np.all(data == ref_data)


def test_File_getitem_time_all(test2_file):
    file = adios2py.File(test2_file, mode="rra")
    ref_data = np.stack(
        (sample_data["test_float_1d"], sample_data["test_float_1d"] + 1), axis=0
    )
    data = file["test_float_1d"][:]
    ref_data = ref_data[:]
    assert data.dtype == ref_data.dtype
    assert data.shape == ref_data.shape
    assert np.all(data == ref_data)


def test_File_getitem_as_array(test2_file):
    file = adios2py.File(test2_file, mode="rra")
    ref_data = np.stack(
        (sample_data["test_float_1d"], sample_data["test_float_1d"] + 1), axis=0
    )
    data = file["test_float_1d"]
    print("data", data)
    assert data.shape == ref_data.shape
    assert data.dtype == ref_data.dtype
    assert np.all(data == ref_data)
