from __future__ import annotations

from numpy.typing import ArrayLike

from adios2py.group import Group


class Step(Group):
    """
    Represents a step in an ADIOS2 file.
    """

    def _write(self, name: str, data: ArrayLike) -> None:
        self._file._write(name, data)  # pylint: disable=W0212

    def __setitem__(self, name: str, data: ArrayLike) -> None:
        self._write(name, data)

    def step(self) -> int:
        assert isinstance(self._step, int)
        return self._step
