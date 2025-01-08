from __future__ import annotations

from types import EllipsisType as ellipsis
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from adios2py.attrs_proxy import AttrsProxy

if TYPE_CHECKING:
    from adios2py.file import File


class ArrayProxy:
    """
    Represents an array-like interface to an ADIOS2 variable.

    Actual loading of the data happens lazily as the data is accessed via slicing or
    .__array__().
    """

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
        return f"ArrayProxy(name={self._name}, shape={self.shape}, dtype={self.dtype} step={self._step})"

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
        return self[...].astype(dtype)

    def __getitem__(
        self,
        index: (
            None
            | slice
            | ellipsis
            | SupportsIndex
            | tuple[None | slice | ellipsis | SupportsIndex, ...]
        ),
    ) -> NDArray[Any]:
        if self._file._mode not in ("r", "rra"):
            msg = f"Cannot read variables in mode {self._file._mode}."
            raise ValueError(msg)

        if not isinstance(index, tuple):
            index = (index,)

        if isinstance(self._step, SupportsIndex):
            index = (self._step, *index)
        elif self._step != slice(None):
            raise NotImplementedError

        if index in ((), (Ellipsis,)):
            index = (slice(None), Ellipsis)

        return self._file._read(self._name, index)

    @property
    def attrs(self) -> AttrsProxy:
        return AttrsProxy(self._file, variable=self._name)
