"""Tests for env/generate.py: obstacles in same style as thesis, and solvable maps."""

import random

import numpy as np
import pytest
import shapely
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely import Polygon

from qrstigplan.env.generate import (
    BOUNDS,
    ObstacleShape,
    generate_environment,
    random_obstacle,
)
from qrstigplan.env.visibility import Solvability, solvability

START, GOAL = (5.0, 5.0), (45.0, 45.0)
seeds = st.integers(0, 2**32 - 1)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def obstacle_coords(env) -> list[list[list[float]]]:
    return [shapely.get_coordinates(o).tolist() for o in env.obstacles]


@given(seed=seeds)
def test_obstacle_follows_thesis(seed):
    poly = random_obstacle(rng(seed), BOUNDS)
    coords = shapely.get_coordinates(poly)  # closed ring
    assert isinstance(poly, Polygon)
    assert len(coords) in (4, 5)  # triangle or quadrilateral
    assert poly.equals(poly.convex_hull)
    assert np.array_equal(coords, np.round(coords))
    assert shapely.box(*BOUNDS).covers(poly)
    assert poly.area < 0.1 * 50 * 50
    assert (np.linalg.norm(np.diff(coords, axis=0), axis=1) <= 20).all()


def test_corners_reach_map_edges():
    # Original thesis version used random.randint, which is inclusive
    # rng.integers excludes upper bound unless endpoint=True
    gen = rng(0)
    coords = np.vstack([shapely.get_coordinates(random_obstacle(gen, BOUNDS)) for _ in range(500)])
    assert coords.min() == 0
    assert coords.max() == 50


def test_area_limit():
    small = (0.0, 0.0, 20.0, 20.0)
    gen = rng(1)
    for _ in range(200):
        poly = random_obstacle(gen, small)
        assert poly.area < 0.1 * 20 * 20
        assert shapely.box(*small).covers(poly)


def test_fractional_bounds_refused():
    with pytest.raises(ValueError):
        random_obstacle(rng(0), (0.0, 0.0, 50.5, 50.0))


def test_impossible_shape_raises_instead_of_loop():
    shape = ObstacleShape(max_side=0.5)  # int corners always at least 1 apart
    with pytest.raises(RuntimeError):
        random_obstacle(rng(0), BOUNDS, shape, max_tries=100)


@settings(max_examples=30, deadline=None)  # no deadline: redrawn map can take > 200ms
@given(seed=seeds)
def test_generated_maps_always_solvable(seed):
    result = generate_environment(rng(seed), START, GOAL)
    assert solvability(result.env, START, GOAL) is Solvability.SOLVABLE
    assert len(result.env.obstacles) == 8
    assert result.env.bounds == BOUNDS


def test_inflation_respected():
    result = generate_environment(rng(3), START, GOAL, inflation=0.71)
    assert result.env.inflation == 0.71
    assert solvability(result.env, START, GOAL) is Solvability.SOLVABLE


def test_same_seed_same_map():
    a = generate_environment(rng(7), START, GOAL)
    b = generate_environment(rng(7), START, GOAL)
    assert obstacle_coords(a.env) == obstacle_coords(b.env)
    assert (a.blocked_obstacles, a.disconnected_maps) == (b.blocked_obstacles, b.disconnected_maps)


def test_different_seeds_different_maps():
    a = generate_environment(rng(7), START, GOAL)
    b = generate_environment(rng(8), START, GOAL)
    assert obstacle_coords(a.env) != obstacle_coords(b.env)


def test_global_rand_state_not_used():
    random.seed(1)
    np.random.seed(1)  # noqa: NPY002
    a = generate_environment(rng(7), START, GOAL)
    random.seed(2)
    np.random.seed(2)  # noqa: NPY002
    b = generate_environment(rng(7), START, GOAL)
    assert obstacle_coords(a.env) == obstacle_coords(b.env)


def test_rejections_are_counted():
    # 16 obstacles per map makes other kinds of rejection common enough to count
    results = [generate_environment(rng(s), START, GOAL, n_obstacles=16) for s in range(40)]
    assert sum(r.blocked_obstacles for r in results) > 0
    assert sum(r.disconnected_maps for r in results) > 0


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        pytest.param((-1.0, 5.0), GOAL, id="start-outside"),
        pytest.param(START, (45.0, 51.0), id="goal-outside"),
    ],
)
def test_start_or_goal_outside_map_refused(start, goal):
    with pytest.raises(ValueError):
        generate_environment(rng(0), start, goal)


def test_gives_up_after_max_maps():
    with pytest.raises(RuntimeError):
        generate_environment(rng(0), START, GOAL, max_maps=0)
