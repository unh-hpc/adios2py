from __future__ import annotations

import contextlib
import operator
import os
from typing import Any, Generator, Iterable, Iterator, SupportsIndex

import adios2.bindings as adios2bindings  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from adios2py import util
from adios2py.array_proxy import ArrayProxy
from adios2py.step import Step


class File:
    """
    Represents an ADIOS2 File or Stream.
    """

    _io: adios2bindings.IO | None = None
    _engine: adios2bindings.Engine | None = None
    _mode: str = ""
    _current_step: int = -1
    _in_step: bool = False

    def __init__(self, filename: os.PathLike[Any] | str, mode: str = "rra") -> None:
        """Open the file in the specified mode."""
        filename = os.fspath(filename)
        self._mode = mode
        self._adios = adios2bindings.ADIOS()
        self._io_name = "io-adios2py"
        self._io = self._adios.DeclareIO(self._io_name)
        self._engine = self._io.Open(os.fspath(filename), util.openmode_to_adios2(mode))

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

    def _read(
        self, name: str, step: SupportsIndex | slice | None = None
    ) -> NDArray[Any]:
        """Read a variable from the file."""
        var = self.io.InquireVariable(name)
        if not var:
            msg = f"Variable '{name}' not found"
            raise ValueError(msg)

        dtype = util.adios2_to_dtype(var.Type())
        shape = tuple(var.Shape())

        if step is not None:
            if isinstance(step, SupportsIndex):
                step_selection = (operator.index(step), 1)
            elif isinstance(step, slice):
                start, stop, _step = step.indices(self._steps())
                if _step != 1:
                    raise NotImplementedError()
                step_selection = (start, stop - start)
                shape = (stop - start, *shape)
            else:
                raise NotImplementedError()

            if self._mode == "rra":
                var.SetStepSelection(step_selection)
            else:  # streaming mode  # noqa: PLR5501
                if not self.in_step() or step_selection != (self.current_step(), 1):
                    msg = "Trying to access non-current step in streaming mode"
                    raise ValueError(msg)

        data = np.empty(shape, dtype=dtype)
        self.engine.Get(var, data, adios2bindings.Mode.Sync)
        return data

    def _write(self, name: str, data: ArrayLike) -> None:
        """Write a variable to the file."""
        if not self.in_step():
            msg = "Data needs to be written inside an active step."
            raise ValueError(msg)

        data = np.asarray(data)
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

    def __getitem__(self, name: str) -> ArrayProxy:
        """Read a variable from the file."""
        if self._mode not in ("r", "rra"):
            msg = f"Cannot read variables in mode {self._mode}."
            raise ValueError(msg)
        var = self.io.InquireVariable(name)
        if not var:
            msg = f"Variable {name} not found."
            raise KeyError(msg)
        dtype = np.dtype(util.adios2_to_dtype(var.Type()))
        shape = (self._steps, *var.Shape())
        return ArrayProxy(self, step=slice(None), name=name, dtype=dtype, shape=shape)

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

    @property
    def steps(self) -> StepsProxy:
        return StepsProxy(self)


class StepsProxy(Iterable[Step]):
    def __init__(self, File: File) -> None:
        self._file = File

    def __iter__(self) -> Iterator[Step]:
        try:
            while True:
                try:
                    self._file._begin_step()
                except EOFError:
                    break
                yield Step(self._file)
                self._file._end_step()
        finally:
            if self._file.in_step():
                self._file._end_step()

    @contextlib.contextmanager
    def next(self) -> Generator[Step]:
        self._file._begin_step()
        yield Step(self._file)
        self._file._end_step()

    def __getitem__(self, index: int) -> Step:
        if self._file._mode != "rra":
            msg = "Selecting steps by index is only supported in 'rra' mode"
            raise IndexError(msg)

        if index < 0 or index >= len(self):
            msg = "Index out of range"
            raise IndexError(msg)

        return Step(self._file, index)

    def __len__(self) -> int:
        return self._file._steps()
