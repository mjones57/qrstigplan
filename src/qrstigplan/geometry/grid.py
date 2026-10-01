import math

import numpy as np


def snap_shift_bound(resolution: float) -> float:
    """Compute max distance a point moves when snapping to grid resolution"""
    _check_resolution(resolution)
    delta = resolution * math.sqrt(2) / 2
    return delta


def snap_to_grid(path, resolution: float = 1.0) -> np.ndarray:
    """
    Rounds each point to nearest grid index (ties to even),
    each point moves at most snap_shift_bound(resolution)

    Precondition: snapping preserves the start/goal only if they lie on the grid
    (integer multiples of resolution). Same holds for map bounds
    """

    _check_resolution(resolution)

    # convert input to np array
    arr = np.asarray(path, dtype=float)

    # Ensure array shape is (N, 2) with finite vals, raise error otherwise
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError("Path array either wrong shape or contains non-finite value")

    rounded_path = np.rint(arr / resolution).astype(np.int64)

    return rounded_path


def grid_to_world(indices, resolution: float = 1.0) -> np.ndarray:
    """Reverts indices from snap_to_grid back to float resolution"""

    # Check inputs for errors
    _check_resolution(resolution)

    arr = np.asarray(indices)

    # Ensure array shape is (N, 2) of integers, raise error otherwise
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Indices array is wrong shape or dtype, expected (N, 2), dtype=np.integer")

    # Convert snapped path back to world coords
    return arr * float(resolution)


def _check_resolution(resolution: float) -> None:
    """Raise error for invalid resolution input"""
    if not resolution > 0.0 or not math.isfinite(resolution):
        raise ValueError("Resolution must be positive and finite")
