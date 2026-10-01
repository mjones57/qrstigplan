import pytest

from qrstigplan.env.environment import Environment


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        pytest.param((10, 35), (40, 35), True, id="clear"),
        pytest.param((10, 25), (40, 25), False, id="through"),
        pytest.param((22, 22), (28, 28), False, id="entirely-inside"),
        pytest.param((10, 30), (40, 30), False, id="grazes-edge"),
        pytest.param((25, 35), (35, 25), False, id="grazes-corner"),
        pytest.param((45, 45), (55, 45), False, id="leaves-map"),
        pytest.param((0, 0), (0, 10), True, id="runs-along-map-edge"),
        pytest.param((0, 0), (25, 25), False, id="crosses-inside"),
    ],
)
def test_segment_is_free(env, p, q, expected):
    assert env.segment_is_free(p, q) == expected


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        pytest.param((5, 5), True, id="open"),
        pytest.param((25, 25), False, id="inside"),
        pytest.param((-1, 5), False, id="outside-map"),
        pytest.param((0, 25), True, id="on-map-boundary"),
    ],
)
def test_point_is_free(env, p, expected):
    assert env.point_is_free(p) == expected


def test_inflation_blocks_near_misses(env):
    env1 = env
    env2 = Environment(env.bounds, env.obstacles, inflation=1.0)
    assert env1.segment_is_free((10, 30.5), (40, 30.5))
    assert not env2.segment_is_free((10, 30.5), (40, 30.5))
