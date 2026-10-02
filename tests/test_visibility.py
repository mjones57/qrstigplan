import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from shapely import Point, Polygon, box

from qrstigplan.env.environment import Environment
from qrstigplan.env.visibility import optimal_path
from qrstigplan.planning.validity import validate_path

BOUNDS = (0.0, 0.0, 50.0, 50.0)
SQUARE = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
WALL_WITH_GAP = (box(0, 24, 24, 26), box(26, 24, 50, 26))  # gap at x = [24 26]
RING = (box(35, 35, 45, 36), box(35, 44, 45, 45), box(35, 35, 36, 45), box(44, 35, 45, 45))
START, GOAL = (5.0, 5.0), (45.0, 45.0)


def path_length(path) -> float:
    """Sum of segment lengths of an (N, 2) path"""
    return float(np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0), axis=1).sum())


def test_blocked_is_the_inflated_union():
    env = Environment(BOUNDS, (SQUARE,), inflation=1.0)
    assert env.blocked.contains(Point(25, 25))
    assert env.blocked.contains(Point(30.5, 25))
    assert not env.blocked.contains(Point(32, 25))


@pytest.mark.parametrize(
    ("obstacles", "start", "goal", "expected"),
    [
        pytest.param((), START, GOAL, 40 * math.sqrt(2), id="empty-map"),
        pytest.param(
            (SQUARE,), (10, 25), (40, 25), 10 + 2 * math.hypot(10, 5), id="around-square-side"
        ),
        pytest.param((SQUARE,), START, GOAL, 2 * math.hypot(15, 25), id="around-square-diagonal"),
        pytest.param(WALL_WITH_GAP, (25, 5), (25, 45), 40.0, id="through-gap"),
    ],
)
def test_known_optimum(obstacles, start, goal, expected):
    env = Environment(BOUNDS, obstacles)
    result = optimal_path(env, start, goal)
    assert result  # solvable
    assert result.length == pytest.approx(expected, abs=1e-4)
    assert result.path is not None


@pytest.mark.parametrize(
    ("obstacles", "inflation", "start", "goal"),
    [
        pytest.param((SQUARE,), 0.0, (25, 25), GOAL, id="start-inside-obstacle"),
        pytest.param((SQUARE,), 0.0, START, (25, 25), id="goal-inside-obstacle"),
        pytest.param(WALL_WITH_GAP, 1.5, (25, 5), (25, 45), id="gap-closed-by-inflation"),
        pytest.param(RING, 0.0, START, (40, 40), id="goal-walled-in"),
    ],
)
def test_unsolvable(obstacles, inflation, start, goal):
    env = Environment(BOUNDS, obstacles, inflation=inflation)
    result = optimal_path(env, start, goal)
    assert not result
    assert result.length == math.inf
    assert result.path is None


def test_optimal_path_is_valid_with_length_cstar(env):
    result = optimal_path(env, START, GOAL)
    assert validate_path(result.path, env, START, GOAL)
    assert path_length(result.path) == pytest.approx(result.length)


def test_inflation_lengthens_the_optimum():
    thin = optimal_path(Environment(BOUNDS, (SQUARE,)), (10, 25), (40, 25))
    fat_env = Environment(BOUNDS, (SQUARE,), inflation=1.0)
    fat = optimal_path(fat_env, (10, 25), (40, 25))
    assert fat.length > thin.length
    assert validate_path(fat.path, fat_env, (10, 25), (40, 25))


points = st.tuples(st.floats(0, 50), st.floats(0, 50))


@given(s=points, g=points)
def test_never_shorter_than_straight_line(env, s, g):
    assume(env.point_is_free(s) and env.point_is_free(g) and s != g)
    result = optimal_path(env, s, g)
    straight = math.dist(s, g)
    if env.segment_is_free(s, g):
        assert result.length == pytest.approx(straight)
    assert validate_path(result.path, env, s, g)


@given(waypoints=st.lists(points, min_size=1, max_size=3))
def test_no_valid_path_beats_cstar(env, waypoints):
    path = [START, *waypoints, GOAL]
    assume(validate_path(path, env, START, GOAL))
    assert path_length(path) >= optimal_path(env, START, GOAL).length - 1e-9
