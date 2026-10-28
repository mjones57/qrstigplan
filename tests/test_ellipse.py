"""Tests for geometry/ellipse.py: the region robot 2 searches."""

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from qrstigplan.geometry.ellipse import Ellipse

# Simple ellipse: the foci are 10 units apart, s = 26, so c = 5, a = 13, b = 12.
E = Ellipse((0, 0), (6, 8), 26)


def test_derived_ellipse_values():
    assert (E.c, E.a, E.b) == pytest.approx((5, 13, 12))
    assert E.center == pytest.approx((3, 4))
    assert E.theta == pytest.approx(math.atan2(8, 6))
    assert E.area == pytest.approx(math.pi * 13 * 12)


@pytest.mark.parametrize("s", [10, 9], ids=["flat", "too-short"])
def test_impossible_ellipse_refused(s):
    # s = 10 is a line segment, but s = 9 is no ellipse at all (s < 2c)
    with pytest.raises(ValueError):
        Ellipse((0, 0), (6, 8), s)


def test_contains():
    points = np.array(
        [
            [0, 0],  # f1
            [6, 8],  # f2
            [10.8, 14.4],  # on boundary of major axis end
            [-6.6, 11.2],  # on boundary of minor axis end
            [11.4, 15.2],  # just outside major axis
            [-7.4, 11.8],  # just outside minor axis
        ]
    )
    expected = np.array([True, True, True, True, False, False])
    np.testing.assert_array_equal(E.contains(points), expected)


def test_samples_are_inside_and_reproducible():
    samples = E.sample(np.random.default_rng(1), 1_000)
    assert samples.shape == (1_000, 2)
    assert E.contains(samples).all()
    np.testing.assert_array_equal(samples, E.sample(np.random.default_rng(1), 1_000))


def test_samples_uniformity():
    # Used scale factor k to create smaller inner ellipse (has k^2 of the area)
    # As such, uniform samples must have have about k^2 points inside as well
    k = 0.5
    samples = E.sample(np.random.default_rng(2), 20_000)
    center = np.array(E.center)
    in_inner = E.contains(center + (samples - center) / k)
    assert in_inner.mean() == pytest.approx(k**2, abs=0.015)
    # The samples must also be spread evenly around the center, not on one side.
    assert samples.mean(axis=0) == pytest.approx(center, abs=0.3)


# ---------- Testing thesis lemma property ----------
# A path from f1 to f2 with length L lies inside the ellipse with s = ceil(L).
# The corresponding ellipse must contain the full path if satisfied.

coords = st.floats(0, 50)
points = st.tuples(coords, coords)


@given(f1=points, f2=points, middle=st.lists(points, max_size=4))
def test_a_path_no_longer_than_s_is_inside(f1, f2, middle):
    path = np.array([f1, *middle, f2])
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    s = math.ceil(length)  # robot 1 rounds s up before it writes the marker
    assume(s > math.dist(f1, f2))
    assert Ellipse(f1, f2, s).contains(path).all()
