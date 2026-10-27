from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PayloadTooLarge(ValueError):
    """Payload does not fit the chosen format. Encoders raise this instead of truncating."""


@dataclass(frozen=True)
class MarkerEllipse:
    f1: int  # index into nodes; f2 is the next point after f1: nodes[f1 + 1], or the goal
    s: int  # length of the replaced sub-path, rounded up, in grid units (>= 1)


@dataclass(frozen=True)
class MarkerPayload:
    nodes: np.ndarray  # (n, 2) int64 grid indices; start first; never the goal
    ellipse: MarkerEllipse | None = None

    def __post_init__(self) -> None:  # check / copy / make read-only
        arr = np.asarray(self.nodes)

        # check shape and dtype of nodes
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 2:
            raise ValueError("Nodes array is the wrong shape")

        n = arr.shape[0]

        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("Nodes array dtype is not an integer type")

        # ensure all values in nodes non-negative
        if np.any(arr < 0):
            raise ValueError("One or more values in nodes is negative")

        # if ellipse: check f1 index is in range and s >= 1
        if self.ellipse is not None:
            if self.ellipse.f1 < 0 or self.ellipse.f1 >= n:
                raise ValueError("Focus f1 index not in range of nodes")
            if self.ellipse.s < 1:
                raise ValueError("Ellipse value s is too small (s < 1)")

        object.__setattr__(self, "nodes", np.array(arr, dtype=np.int64))
        self.nodes.flags.writeable = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarkerPayload):
            return NotImplemented
        return np.array_equal(self.nodes, other.nodes) and self.ellipse == other.ellipse


class Codec(Protocol):
    def encode(self, payload: MarkerPayload) -> str: ...  # raises PayloadTooLarge, no truncation
    def decode(self, text: str) -> MarkerPayload: ...
    def fits(self, payload: MarkerPayload) -> bool: ...
