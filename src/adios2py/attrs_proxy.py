from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Mapping

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from adios2py.file import File


class AttrsProxy(Mapping[str, Any]):
    def __init__(self, file: File, variable: str | None = None) -> None:
        self._file = file
        self._variable = variable
        self._keys = file.io.AvailableAttributes().keys()
        if variable is None:
            self._keys = {key for key in self._keys if "/" not in key}
        else:
            pfx = f"{variable}/"
            self._keys = {
                key.removeprefix(pfx) for key in self._keys if key.startswith(pfx)
            }

    def __getitem__(self, name: str) -> Any:
        return self._file._read_attribute(name, self._variable)

    def __setitem__(self, name: str, data: ArrayLike) -> None:
        self._file._write_attribute(name, data, self._variable)
        self._keys.add(name)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        yield from self._keys
