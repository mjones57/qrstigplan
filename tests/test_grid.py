import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qrstigplan.env.environment import Environment
from qrstigplan.geometry.grid import grid_to_world, snap_shift_bound, snap_to_grid
from qrstigplan.planning.validity import validate_path

START = (5.0, 5.0)
GOAL = (45.0, 45.0)
RESOLUTION = 2.0


coords = st.floats(min_value=-100, max_value=100)
shapes = st.tuples(st.integers(1, 20), st.just(2))  # shape (N, 2)
paths = arrays(np.float64, shapes, elements=coords)
resolutions = st.floats(min_value=0.01, max_value=10)
near_square = st.tuples(st.floats(15, 35), st.floats(15, 35))
waypoint = st.tuples(st.floats(0, 50), st.floats(0, 50))


@pytest.mark.parametrize(
    ("path", "resolution", "expected"),
    [
        pytest.param([(0.4, 0.6)], 1, [(0, 1)], id="nearest"),
        pytest.param([(2.5, 0.5)], 1, [(2, 0)], id="ties-to-even"),
        pytest.param([(-1.5, -0.4)], 1, [(-2, 0)], id="negative"),
        pytest.param([(0.3, 1.2)], 0.5, [(1, 2)], id="half-res"),
        pytest.param([(5, 5), (45, 45)], 1, [(5, 5), (45, 45)], id="already-on-grid"),
    ],
)
def test_snap_examples(path, resolution, expected):
    got = snap_to_grid(path, resolution)
    assert np.array_equal(got, expected)
    assert np.issubdtype(got.dtype, np.integer)
    assert np.shape(got) == (len(path), 2)


@pytest.mark.parametrize(
    "resolution",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
    ],
)
def test_bad_resolution_raises(resolution):
    with pytest.raises(ValueError):
        snap_to_grid([(1.0, 1.0)], resolution)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param([(1.0, float("nan"))], id="nan-coordinate"),
        pytest.param([(1.0, 2.0, 3.0)], id="three-columns"),
    ],
)
def test_bad_path_raises(path):
    with pytest.raises(ValueError):
        snap_to_grid(path, 1.0)


@pytest.mark.parametrize(
    "indices",
    [
        pytest.param([[1.7, 2.0]], id="non-integer"),
        pytest.param([[1, 2, 3]], id="three-columns"),
    ],
)
def test_bad_indices_raises(indices):
    with pytest.raises(ValueError):
        grid_to_world(indices, 1.0)


@given(path=paths, resolution=resolutions)
def test_snapping_moves_each_pt_at_most_delta(path, resolution):
    indices = snap_to_grid(path, resolution)
    moved = grid_to_world(indices, resolution)
    distances = np.linalg.norm(moved - path, axis=1)
    assert (distances <= snap_shift_bound(resolution)).all()


indices = arrays(np.int64, shapes, elements=st.integers(-10_000, 10_000))


@given(idx=indices, resolution=resolutions)
def test_grid_round_trip(idx, resolution):
    world = grid_to_world(idx, resolution)
    assert np.array_equal(snap_to_grid(world, resolution), idx)


@given(p=near_square, q=near_square)
def test_rounding_guarantee(env, safe_env, p, q):
    """Clearance > delta before snapping --> collision free after"""
    assume(safe_env.segment_is_free(p, q))  # keep only segments with clearance > delta
    snapped = grid_to_world(snap_to_grid([p, q], 1.0), 1.0)
    assert env.segment_is_free(snapped[0], snapped[1])


@pytest.mark.xfail(reason="precondition: see snap_to_grid docstring")
@given(waypoints=st.lists(waypoint, min_size=1, max_size=3))
def test_snapped_path_stays_valid(env, waypoints):
    """Whole-path version of the guarantee at resolution 2"""
    safe = Environment(env.bounds, env.obstacles, inflation=snap_shift_bound(RESOLUTION))
    path = [START, *waypoints, GOAL]
    assume(validate_path(path, safe, START, GOAL))  # keep paths w/ clearance > delta
    snapped = grid_to_world(snap_to_grid(path, RESOLUTION), RESOLUTION)
    result = validate_path(snapped, env, START, GOAL)
    assert result, result.reason
