from __future__ import annotations

from collections.abc import Mapping

from numpy.typing import ArrayLike


class Group(Mapping[str, ArrayLike]):
    pass
