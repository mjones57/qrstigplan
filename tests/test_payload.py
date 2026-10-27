"""
Uses and tests codec/payload.py:
Tests what the marker (QR) holds, prior to any formatting/encoding.
"""

import dataclasses

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qrstigplan.codec.payload import MarkerEllipse, MarkerPayload, PayloadTooLarge

NODES = np.array([[5, 5], [19, 4], [33, 8]])


def test_nodes_become_int64_array():
    p = MarkerPayload(NODES)
    assert isinstance(p.nodes, np.ndarray)
    assert p.nodes.dtype == np.int64
    assert p.nodes.shape == (3, 2)
    assert p.ellipse is None


def test_nodes_read_only():
    p = MarkerPayload(NODES)
    with pytest.raises(ValueError):  # "assignment destination is read-only"
        p.nodes[0, 0] = 99


def test_nodes_are_a_copy():
    source = NODES.copy()
    p = MarkerPayload(source)
    source[0, 0] = 99
    assert p.nodes[0, 0] == 5


def test_payload_is_frozen():
    p = MarkerPayload(NODES)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.ellipse = MarkerEllipse(0, 10)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("a", "b", "equal"),
    [
        pytest.param(MarkerPayload(NODES), MarkerPayload(NODES), True, id="same"),
        pytest.param(MarkerPayload(NODES), MarkerPayload(NODES[:2]), False, id="fewer-nodes"),
        pytest.param(
            MarkerPayload(NODES),
            MarkerPayload(np.array([[5, 5], [19, 4], [33, 9]])),
            False,
            id="one-value",
        ),
        pytest.param(
            MarkerPayload(NODES, MarkerEllipse(1, 18)),
            MarkerPayload(NODES, MarkerEllipse(1, 19)),
            False,
            id="different-s",
        ),
        pytest.param(
            MarkerPayload(NODES),
            MarkerPayload(NODES, MarkerEllipse(1, 18)),
            False,
            id="ellipse-or-not",
        ),
    ],
)
def test_equality_compares_values(a, b, equal):
    assert (a == b) is equal
    assert (a != b) is not equal


def test_payload_never_equal_to_something_else():
    assert MarkerPayload(NODES) != "5 5 0 19 4 0 33 8 0"


@pytest.mark.parametrize(
    ("nodes", "ellipse"),
    [
        pytest.param(np.empty((0, 2), dtype=np.int64), None, id="no-nodes"),
        pytest.param(np.array([5, 5]), None, id="one-dimensional"),
        pytest.param(np.array([[5, 5, 0]]), None, id="three-columns"),
        pytest.param(np.array([[5.0, 5.0]]), None, id="floats"),
        pytest.param(np.array([[5, -1]]), None, id="negative"),
        pytest.param(NODES, MarkerEllipse(3, 18), id="f1-past-the-end"),
        pytest.param(NODES, MarkerEllipse(-1, 18), id="f1-negative"),
        pytest.param(NODES, MarkerEllipse(1, 0), id="s-zero"),
    ],
)
def test_invalid_payloads_refused(nodes, ellipse):
    with pytest.raises(ValueError):
        MarkerPayload(nodes, ellipse)


def test_payload_too_large_raises_value_error():
    assert issubclass(PayloadTooLarge, ValueError)


node_lists = st.lists(st.tuples(st.integers(0, 50), st.integers(0, 50)), min_size=1, max_size=12)


@given(nodes=node_lists, s=st.integers(1, 200), data=st.data())
def test_rebuilt_payload_is_equal(nodes, s, data):
    f1 = data.draw(st.integers(0, len(nodes) - 1))
    p = MarkerPayload(np.array(nodes), MarkerEllipse(f1, s))
    assert MarkerPayload(np.array(p.nodes.tolist()), MarkerEllipse(f1, s)) == p
