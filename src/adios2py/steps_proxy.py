from __future__ import annotations

import contextlib
from collections.abc import Generator, Iterable, Iterator
from typing import TYPE_CHECKING

from adios2py.step import Step

if TYPE_CHECKING:
    from adios2py.file import File


class StepsProxy(Iterable[Step]):
    """
    Implements File.steps to provide an iterable interface to Steps.
    """

    def __init__(self, file: File) -> None:
        self._file = file

    def __iter__(self) -> Iterator[Step]:
        try:
            while True:
                try:
                    self._file._begin_step()
                except EOFError:
                    break
                yield Step(self._file, self._file.current_step())
                self._file._end_step()
        finally:
            if self._file.in_step():
                self._file._end_step()

    @contextlib.contextmanager
    def next(self) -> Generator[Step]:
        self._file._begin_step()  # pylint: disable=W0212
        yield Step(self._file, self._file.current_step())
        self._file._end_step()  # pylint: disable=W0212

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
