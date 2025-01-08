from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Mapping

if TYPE_CHECKING:
    from adios2py.file import File


class AttrsProxy(Mapping[str, Any]):
    def __init__(self, file: File) -> None:
        self._file = file
        self._keys = file.io.AvailableAttributes().keys()

    def __getitem__(self, name: str) -> Any:
        return self._file._read_attribute(name)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        yield from self._keys
