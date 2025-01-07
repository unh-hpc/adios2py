from __future__ import annotations

import operator
from types import EllipsisType as ellipsis
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from adios2py.file import File


class ArrayProxy:
    def __init__(
        self,
        file: File,
        step: int | None,
        name: str,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
    ) -> None:
        self._file = file
        self._step = step
        self._name = name
        self._dtype = dtype
        self._shape = shape

    def __repr__(self) -> str:
        return f"ArrayProxy({self._name})"

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        if self.ndim == 0:
            msg = "len() of unsized object"
            raise TypeError(msg)

        return self.shape[0]

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
        if self._step is None:
            assert isinstance(key, tuple)
            assert len(key) > 0
            step, *rem = key
            assert isinstance(step, SupportsIndex)
            data = self._file._read(
                self._name, step_selection=(operator.index(step), 1)
            )

            return data[np.newaxis, ...][(0, *rem)]

        return self.__array__()[key]

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        assert self._step is not None
        data = self._file._read(self._name, step_selection=(self._step, 1))

        return data.astype(dtype)
