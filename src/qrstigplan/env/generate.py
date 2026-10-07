from dataclasses import dataclass

import numpy as np
import shapely
from shapely import MultiPoint, Polygon

from qrstigplan.env.environment import Environment
from qrstigplan.env.visibility import Solvability, solvability

BOUNDS = (0.0, 0.0, 50.0, 50.0)


@dataclass(frozen=True)
class ObstacleShape:
    n_points: int = 4  # random corners before hull
    max_area_ratio: float = 0.1  # hull area < max_area_ratio * map area
    max_side: float = 20.0  # every side <= max_side


THESIS_SHAPE = ObstacleShape()  # the values in gen_obstacle.py


def random_obstacle(rng, bounds=BOUNDS, shape=THESIS_SHAPE, max_tries=10_000) -> Polygon:
    """Generates one random obstacle polygon"""
    # Bounds must be whole numbers because corners are ints
    for v in bounds:
        if not float(v).is_integer():
            raise ValueError("Bounds must be whole numbers")

    # Loop to generate obstacle until its valid
    map_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    for _ in range(max_tries):
        points = rng.integers(bounds[:2], bounds[2:], size=(shape.n_points, 2), endpoint=True)
        hull = MultiPoint(points).convex_hull
        if not isinstance(hull, Polygon):
            continue

        if hull.area >= shape.max_area_ratio * map_area:
            continue

        coords = shapely.get_coordinates(hull)
        steps = np.diff(coords, axis=0)
        sides = np.linalg.norm(steps, axis=1)
        if (sides > shape.max_side).any():
            continue
        return hull
    raise RuntimeError("Max tries reached")


@dataclass(frozen=True)
class Generated:
    env: Environment
    blocked_obstacles: int  # obstacles redrawn because of blocked start or goal
    disconnected_maps: int  # maps redrawn because start and goal disconnected


def generate_environment(
    rng,
    start,
    goal,
    *,
    bounds=BOUNDS,
    n_obstacles=8,
    inflation=0.0,
    shape=THESIS_SHAPE,
    max_maps=1_000,
) -> Generated:
    """Draw solvable map according to documented guidelines/requirements"""
    # Raise error if start and goal out of bounds
    empty = Environment(bounds, ())
    if not empty.point_is_free(start):
        raise ValueError("start not within bounds")
    if not empty.point_is_free(goal):
        raise ValueError("goal is not within bounds")

    blocked = 0  # obstacles redrawn because they cover start/goal
    disconnected = 0  # maps redrawn because start and goal disconnected

    # Try to generate a map (capped at max_maps)
    for _ in range(max_maps):
        obstacles = []
        while len(obstacles) < n_obstacles:
            obstacle = random_obstacle(rng, bounds, shape)
            one = Environment(bounds, obstacles=(obstacle,), inflation=inflation)
            if one.point_is_free(start) and one.point_is_free(goal):
                obstacles.append(obstacle)
            else:
                blocked += 1
        env = Environment(bounds, obstacles=tuple(obstacles), inflation=inflation)
        if solvability(env, start, goal) is Solvability.SOLVABLE:
            return Generated(env, blocked_obstacles=blocked, disconnected_maps=disconnected)
        disconnected += 1

    raise RuntimeError("Max tries exceeded, all maps disconnected")
