import math
from dataclasses import dataclass, field

import numpy as np


# Ellipse class to represent ellipses
@dataclass(frozen=True)
class Ellipse:
    f1: tuple[float, float]
    f2: tuple[float, float]
    s: float  # sum of length of path segments between foci
    center: tuple[float, float] = field(init=False)
    a: float = field(init=False)
    b: float = field(init=False)
    c: float = field(init=False)
    theta: float = field(init=False)  # angle of the major axis (radians)
    area: float = field(init=False)

    def __post_init__(self) -> None:

        # Convert f1, f2 strictly to tuple[float, float], s to float
        x, y = np.asarray(self.f1, dtype=float)
        object.__setattr__(self, "f1", (float(x), float(y)))

        x, y = np.asarray(self.f2, dtype=float)
        object.__setattr__(self, "f2", (float(x), float(y)))

        object.__setattr__(self, "s", float(self.s))

        # Check if all values finite
        if not all(math.isfinite(x) for x in self.f1):
            raise ValueError("Focus f1 contains a value that is not finite.")

        if not all(math.isfinite(x) for x in self.f2):
            raise ValueError("Focus f2 contains a value that is not finite.")

        if not math.isfinite(self.s):
            raise ValueError("Ellipse parameter s is not finite.")

        # Enforce s >= 2c property
        dist = math.hypot(self.f2[0] - self.f1[0], self.f2[1] - self.f1[1]) / 2
        object.__setattr__(self, "c", dist)
        if self.s <= 2 * self.c:
            raise ValueError("Value s is too small (s <= 2c)")

        # Compute the rest of the ellipse values
        object.__setattr__(self, "a", self.s / 2)
        object.__setattr__(self, "b", math.sqrt(self.a**2 - self.c**2))
        cen = (0.5 * (self.f1[0] + self.f2[0]), 0.5 * (self.f1[1] + self.f2[1]))
        object.__setattr__(self, "center", cen)
        th = np.arctan2(self.f2[1] - self.f1[1], self.f2[0] - self.f1[0])
        object.__setattr__(self, "theta", th)
        object.__setattr__(self, "area", np.pi * self.a * self.b)

    def contains(self, points, tol: float = 1e-9) -> np.ndarray:
        """
        Given array of points with shape (2,) or (n,2)
        Return True where pt is inside or on the ellipse
        """
        pts = np.asarray(points, dtype=float)
        dist1 = np.linalg.norm(pts - self.f1, axis=-1)
        dist2 = np.linalg.norm(pts - self.f2, axis=-1)
        return np.asarray(dist1 + dist2 <= self.s + tol)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample n points directly inside the ellipse"""

        # Sample unit disk
        phi = rng.uniform(0.0, 2 * np.pi, n)
        r = np.sqrt(rng.random(n))

        # Stretch to match ellipse
        x = self.a * r * np.cos(phi)
        y = self.b * r * np.sin(phi)

        # Rotate and shift
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        px = x * cos_t - y * sin_t + self.center[0]
        py = x * sin_t + y * cos_t + self.center[1]

        return np.column_stack((px, py))
