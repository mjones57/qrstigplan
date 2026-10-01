from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class Reason(Enum):
    """Stores reason failure mode"""

    TOO_FEW_NODES = auto()  # < 2 nodes
    WRONG_START = auto()  # path[0] farther than tol from start
    WRONG_GOAL = auto()  # path[-1] is farther than tol from goal
    SEGMENT_BLOCKED = auto()  # some segment is not in free space


@dataclass(frozen=True)
class Validation:
    """Object tracking validity of a path, reporting reason if invalid"""

    reason: Reason | None = None  # None means valid
    bad_segment: int | None = None  # i means path[i]->path[i+1]

    @property
    def ok(self) -> bool:
        return self.reason is None

    def __bool__(self) -> bool:
        return self.reason is None

    def __post_init__(self):
        if self.reason == Reason.SEGMENT_BLOCKED and self.bad_segment is None:
            raise ValueError("Blocked segment: Expected a value for bad_segment but got None")
        if self.reason != Reason.SEGMENT_BLOCKED and self.bad_segment is not None:
            raise ValueError("Bad segment returned while reason is None")


def validate_path(path, env, start, goal, tol=1e-9) -> Validation:
    """Validate a given path, returning Validation object containing result"""

    # convert input to np array
    arr = np.asarray(path, dtype=float)

    # Ensure array shape is (N, 2), raise error otherwise
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError("Path array either wrong shape or contains non-finite value")

    # Check if path too short
    if len(arr) < 2:
        return Validation(reason=Reason.TOO_FEW_NODES)

    # Ensure start point within tol of nominal start

    if np.linalg.norm(arr[0] - start) > tol:
        return Validation(reason=Reason.WRONG_START)

    # Ensure end point within tol of nominal goal
    if np.linalg.norm(arr[-1] - goal) > tol:
        return Validation(reason=Reason.WRONG_GOAL)

    # Check for blocked segment along path
    for i in range(len(arr) - 1):
        free_segm = env.segment_is_free(arr[i], arr[i + 1])
        if not free_segm:
            return Validation(reason=Reason.SEGMENT_BLOCKED, bad_segment=i)

    return Validation()
