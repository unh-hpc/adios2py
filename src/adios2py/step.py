from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from adios2py.file import File

from adios2py.group import Group


class Step(Group):
    """
    Represents a step in an ADIOS2 file.
    """

    _step: int

    def __init__(self, file: File, step: int | None = None) -> None:
        self._file = file
        if step is None:
            step = file.current_step()
        assert isinstance(step, int)
        self._step = step
        super().__init__(self._file, step)

    def _write(self, name: str, data: ArrayLike) -> None:
        self._file._write(name, data)  # pylint: disable=W0212

    def __setitem__(self, name: str, data: ArrayLike) -> None:
        self._write(name, data)

    def step(self) -> int:
        return self._step
