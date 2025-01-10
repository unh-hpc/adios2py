from __future__ import annotations

import itertools
import operator
import os
from collections.abc import Mapping
from types import EllipsisType
from typing import Any, SupportsIndex

import adios2.bindings as adios2bindings  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from adios2py import util
from adios2py.group import Group
from adios2py.steps_proxy import StepsProxy


class File(Group):
    """
    Represents an ADIOS2 File or Stream.
    """

    _io: adios2bindings.IO | None = None
    _engine: adios2bindings.Engine | None = None
    _mode: str = ""
    _current_step: int = -1
    _in_step: bool = False

    def __init__(
        self,
        filename: os.PathLike[Any] | str,
        mode: str = "rra",
        parameters: dict[str, str] | None = None,
        engine_type: str | None = None,
    ) -> None:
        """Open the file in the specified mode."""
        self._filename = filename
        self._mode = mode
        self._adios = adios2bindings.ADIOS()
        self._io_name = "io-adios2py"
        self._io = self._adios.DeclareIO(self._io_name)
        if parameters is not None:
            # when used as xarray backend, CachingFileManager needs to pass
            # something hashable, so convert back to dict
            self._io.SetParameters(dict(parameters))
        if engine_type is not None:
            self._io.SetEngine(engine_type)
        self._engine = self._io.Open(os.fspath(filename), util.openmode_to_adios2(mode))
        super().__init__(self)

    def __bool__(self) -> bool:
        """Returns True if the file is open."""
        return self._engine is not None and self._io is not None

    @property
    def filename(self) -> os.PathLike[Any] | str:
        return self._filename

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

    @property
    def parameters(self) -> Mapping[str, str]:
        return self.io.Parameters()  # type: ignore[no-any-return]

    @property
    def engine_type(self) -> str:
        return self.io.EngineType()  # type: ignore[no-any-return]

    def _read(
        self, name: str, index: tuple[SupportsIndex | slice | EllipsisType | None, ...]
    ) -> NDArray[Any]:
        """Read a variable from the file."""
        var = self.io.InquireVariable(name)
        if not var:
            msg = f"Variable '{name}' not found"
            raise ValueError(msg)

        dtype = util.adios2_to_dtype(var.Type())

        var_shape = (self._steps(), *var.Shape())
        args = list(index)
        sel: list[tuple[int, int]] = []  # list of (start, count)
        arr_shape: list[int] = []

        n_ellipses = args.count(Ellipsis)
        if n_ellipses > 1:
            msg = "an index can only have a single ellipsis ('...')"
            raise IndexError(msg)
        if n_ellipses == 1:  # replace ellipses by corresponding number of full slices
            i = args.index(Ellipsis)
            args[i : i + 1] = [slice(None)] * (len(var_shape) - len(args) + 1)

        assert len(args) <= len(var_shape)
        for arg, length in itertools.zip_longest(args, var_shape):
            if arg is None:
                # if too fewer slices/indices were passed, pad with full slices
                sel.append((0, length))
                arr_shape.append(length)
            elif isinstance(arg, slice):
                start, stop, step = arg.indices(length)
                assert start < stop
                assert step == 1
                sel.append((start, stop - start))
                arr_shape.append(stop - start)
            elif isinstance(arg, SupportsIndex):
                idx = operator.index(arg)
                if idx < 0:
                    idx += length
                sel.append((idx, 1))
            else:
                raise NotImplementedError()

        sel_start, sel_count = zip(*sel, strict=False)
        if self._mode == "rra":
            var.SetStepSelection(sel[0])
        else:  # streaming mode  # noqa: PLR5501
            if not self.in_step() or sel[0] != (self.current_step(), 1):
                msg = "Trying to access non-current step in streaming mode"
                raise IndexError(msg)

        if len(sel) > 1:
            var.SetSelection((sel_start[1:], sel_count[1:]))

        data = np.empty(arr_shape, dtype=dtype)
        self.engine.Get(var, data, adios2bindings.Mode.Sync)

        return data

    def _write(self, name: str, data: ArrayLike) -> None:
        """Write a variable to the file."""
        if not self.in_step():
            msg = "Data needs to be written inside an active step."
            raise ValueError(msg)

        data = np.asarray(data)
        if data.ndim != 0:
            data = np.ascontiguousarray(data)

        var = self.io.InquireVariable(name)
        if not var:
            var = self.io.DefineVariable(
                name,
                data,
                data.shape,
                [0] * data.ndim,
                data.shape,
                isConstantDims=True,
            )
        # don't allow for changing variable shape
        assert tuple(var.Shape()) == data.shape
        self.engine.Put(var, data, adios2bindings.Mode.Sync)

    def _available_variables(self) -> Mapping[str, Any]:
        return self.io.AvailableVariables()  # type: ignore[no-any-return]

    def current_step(self) -> int:
        assert self.in_step()
        return self._current_step

    def in_step(self) -> bool:
        return self._in_step

    def _begin_step(self) -> None:
        assert not self.in_step()
        if self._mode == "rra":
            if self._current_step >= self._steps() - 1:
                raise EOFError()
            self._current_step = self._current_step + 1
        else:
            status = self.engine.BeginStep()
            if status == adios2bindings.StepStatus.EndOfStream:
                msg = "End of stream"
                raise EOFError(msg)
            if status != adios2bindings.StepStatus.OK:
                msg = f"BeginStep failed with status {status}"
                raise RuntimeError(msg)
            self._current_step = self.engine.CurrentStep()
        self._in_step = True

    def _end_step(self) -> None:
        assert self.in_step()
        if self._mode != "rra":
            self.engine.EndStep()
        self._in_step = False

    def _steps(self) -> int:
        return self.engine.Steps()  # type: ignore[no-any-return]

    def _write_attribute(
        self, name: str, data: ArrayLike, variable: str | None = None
    ) -> None:
        try:
            attr_data = np.asarray(self._read_attribute(name, variable))
            is_floating = bool(
                np.issubdtype(attr_data.dtype, np.floating)
                or np.issubdtype(attr_data.dtype, np.complexfloating)
            )
            if np.array_equal(attr_data, data, equal_nan=is_floating):
                return  # attribute already exists with same value
        except KeyError:
            pass

        variable = "" if variable is None else variable
        if isinstance(data, (str, list)):
            self.io.DefineAttribute(name, data, variable)
        else:
            self.io.DefineAttribute(name, np.asarray(data), variable)

    def _read_attribute(self, name: str, variable: str | None = None) -> ArrayLike:
        attr_name = f"{variable}/{name}" if variable is not None else name
        attr = self.io.InquireAttribute(attr_name)
        if not attr:
            msg = "Attribute not found"
            raise KeyError(msg)

        data = attr.DataString() if attr.Type() == "string" else attr.Data()
        return data[0] if attr.SingleValue() else data  # type: ignore[no-any-return]

    @property
    def steps(self) -> StepsProxy:
        return StepsProxy(self)
