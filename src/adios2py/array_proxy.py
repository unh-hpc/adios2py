from __future__ import annotations

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
        step: SupportsIndex | slice,
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
        return f"ArrayProxy(name={self._name}, shape={self.shape}, dtype={self.dtype})"

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

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        data = self[(self._step,)]
        return data.astype(dtype)

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
        if key in ((), (Ellipsis,)):
            key = (slice(None), Ellipsis)

        step, *rem_list = key
        rem = tuple(rem_list)

        assert isinstance(step, SupportsIndex | slice)

        if not isinstance(step, SupportsIndex):
            rem = (slice(None), *rem)

        data = self._file._read(self._name, step=step)

        return data[rem] if rem else data

        raise NotImplementedError()
