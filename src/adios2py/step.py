from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import EllipsisType as ellipsis
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from adios2py.file import File


class ArrayProxy:
    def __init__(self, step: Step, name: str) -> None:
        self._step = step
        self._name = name

    def __repr__(self) -> str:
        return f"ArrayProxy({self._name})"

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.__array__().dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__array__().shape

    def __getitem__(
        self,
        key: (
            None
            | slice
            | ellipsis
            | SupportsIndex
            | tuple[None | slice | ellipsis | SupportsIndex, ...]
        ),
    ) -> NDArray[Any]:
        return self.__array__()[key]

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        data = self._step.read(self._name)
        dtype = data.dtype if dtype is None else dtype
        return data.astype(dtype)


class Step(Mapping[str, ArrayProxy]):
    def __init__(self, file: File, step: int | None = None) -> None:
        """Represents a step in an ADIOS2 file."""
        self._file = file
        if step is None:
            step = file.current_step()
        self._step = step

    def read(self, name: str) -> NDArray[Any]:
        if self._file._mode == "rra":
            return self._file._read(name, step_selection=(self._step, 1))

        assert self._step == self._file.current_step()
        return self._file._read(name)

    def write(self, name: str, data: NDArray[Any]) -> None:
        self._file._write(name, data)

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        yield from self._keys()

    def __getitem__(self, name: str) -> ArrayProxy:
        if self._file._mode not in ("r", "rra"):
            msg = f"Cannot read variables in mode {self._file._mode}."
            raise ValueError(msg)
        return ArrayProxy(self, name)

    def _keys(self) -> set[str]:
        return self._file.io.AvailableVariables()  # type: ignore[no-any-return]
