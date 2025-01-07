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
        self._step = slice(step, step + 1) if step is not None else None
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
        if not isinstance(key, tuple):
            key = (key,)

        if self._step is not None:
            key = (self._step, *key)

        return self._getitem(key)

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        assert isinstance(self._step, slice)
        data = self._getitem((self._step,))
        return data.astype(dtype)

    def _getitem(
        self,
        key: tuple[None | slice | ellipsis | SupportsIndex, ...],
    ) -> NDArray[Any]:
        assert isinstance(key, tuple)
        assert len(key) > 0
        step, *rem = key

        if isinstance(step, SupportsIndex):
            data = self._file._read(
                self._name, step_selection=(operator.index(step), 1)
            )
        elif isinstance(step, slice):
            step_selection = (step.start, step.stop - step.start)
            data = self._file._read(self._name, step_selection=step_selection)
        else:
            raise NotImplementedError()

        if not rem:
            return data
        return data[tuple(rem)]
