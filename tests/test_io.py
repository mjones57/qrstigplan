import json

import numpy as np
import pytest
import shapely
from hypothesis import given
from hypothesis import strategies as st
from shapely import Polygon, box

from qrstigplan.env.environment import Environment
from qrstigplan.env.io import load_environments, save_environments

BOUNDS = (0.0, 0.0, 50.0, 50.0)
TRIANGLE = Polygon([(21, 48), (4, 49), (14, 45)])  # env 0 obs 0 from thesis


def assert_same(a: Environment, b: Environment) -> None:
    """Same bounds, inflation, and obstacle coords"""
    assert a.bounds == b.bounds
    assert a.inflation == b.inflation
    assert len(a.obstacles) == len(b.obstacles)
    for p, q in zip(a.obstacles, b.obstacles, strict=True):
        assert np.array_equal(shapely.get_coordinates(p), shapely.get_coordinates(q))


def test_round_trip(tmp_path):
    envs = [
        Environment(BOUNDS, (TRIANGLE, box(20, 20, 30, 30))),
        Environment(BOUNDS, (), inflation=0.71),
    ]
    file = tmp_path / "envs.json"
    save_environments(file, envs)
    loaded = load_environments(file)
    assert len(loaded) == 2
    for a, b in zip(envs, loaded, strict=True):
        assert_same(a, b)


def test_json_file_w_format_and_version(tmp_path):
    file = tmp_path / "envs.json"
    save_environments(file, [Environment(BOUNDS, (TRIANGLE,))])
    doc = json.loads(file.read_text(encoding="utf-8"))
    assert doc["format"] == "qrstigplan.environments"
    assert doc["version"] == 1
    assert len(doc["environments"]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("version", 99, id="future-version"),
        pytest.param("format", "something-else", id="wrong-format"),
    ],
)
def test_refuses_files_it_does_not_understand(tmp_path, field, value):
    file = tmp_path / "envs.json"
    save_environments(file, [Environment(BOUNDS, (TRIANGLE,))])
    doc = json.loads(file.read_text(encoding="utf-8"))
    doc[field] = value
    file.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_environments(file)


def test_obstacle_with_hole_refused(tmp_path):
    donut = Polygon(
        [(10, 10), (30, 10), (30, 30), (10, 30)], holes=[[(15, 15), (25, 15), (25, 25), (15, 25)]]
    )
    with pytest.raises(ValueError):
        save_environments(tmp_path / "envs.json", [Environment(BOUNDS, (donut,))])


corner = st.tuples(st.floats(0, 45), st.floats(0, 45))
size = st.tuples(st.floats(0.01, 5), st.floats(0.01, 5))
obstacle = st.builds(lambda c, s: box(c[0], c[1], c[0] + s[0], c[1] + s[1]), corner, size)
environment = st.builds(
    lambda obs, infl: Environment(BOUNDS, tuple(obs), inflation=infl),
    st.lists(obstacle, max_size=4),
    st.floats(0, 2),
)


@given(envs=st.lists(environment, min_size=1, max_size=3))
def test_round_trip_exact(tmp_path_factory, envs):
    file = tmp_path_factory.mktemp("rt") / "envs.json"
    save_environments(file, envs)
    for a, b in zip(envs, load_environments(file), strict=True):
        assert_same(a, b)
