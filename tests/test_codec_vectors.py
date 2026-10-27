"""
Uses docs/codec_vectors.json. Contains fixed paths/vectors that any encoding method must reproduce

These tests only check the file itself, not encoding/decoding accuracy.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from qrstigplan.codec.payload import MarkerEllipse, MarkerPayload
from qrstigplan.codec.qr import fits_version1_l

VECTORS = Path(__file__).resolve().parents[1] / "docs" / "codec_vectors.json"
LEGACY_SYNTAX = re.compile(r"\d+( \d+)*")  # integers, delimited by whitespace


@pytest.fixture(scope="module")
def doc():
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def payload(vector) -> MarkerPayload:
    ellipse = vector["ellipse"]
    return MarkerPayload(
        np.array(vector["nodes"]), None if ellipse is None else MarkerEllipse(**ellipse)
    )


def test_header(doc):
    assert doc["format"] == "qrstigplan.codec-vectors"
    assert doc["version"] == 1
    assert (doc["grid_max"], doc["legacy_limit"], doc["packed_limit"]) == (50, 25, 41)


def test_names_are_unique(doc):
    names = [v["name"] for v in doc["vectors"]]
    assert len(names) == len(set(names))


def test_every_vector_is_valid_payload(doc):
    for v in doc["vectors"]:
        p = payload(v)
        assert (p.nodes <= doc["grid_max"]).all(), v["name"]


def test_every_ellipse_possible(doc):
    # path length s must be longer than dist between foci
    for v in doc["vectors"]:
        if v["ellipse"] is not None:
            points = [*v["nodes"], doc["goal"]]
            f1 = v["ellipse"]["f1"]
            assert v["ellipse"]["s"] > math.dist(points[f1], points[f1 + 1]), v["name"]


def test_legacy_strings(doc):
    for v in doc["vectors"]:
        if v["legacy"] is not None:
            assert LEGACY_SYNTAX.fullmatch(v["legacy"]), v["name"]
            assert len(v["legacy"]) <= doc["legacy_limit"], v["name"]
            assert fits_version1_l(v["legacy"]), v["name"]


def test_packed_strings(doc):
    for v in doc["vectors"]:
        if v["packed"] is not None:
            assert v["packed"].isdigit(), v["name"]
            assert v["packed"].startswith("1"), v["name"]  # format version 1
            assert len(v["packed"]) <= doc["packed_limit"], v["name"]
            assert fits_version1_l(v["packed"]), v["name"]


def test_the_vectors_cover_edge_cases(doc):
    vectors = doc["vectors"]
    assert any(v["legacy"] is not None and len(v["legacy"]) == 25 for v in vectors)
    assert any(v["legacy"] is None and v["packed"] is not None for v in vectors)
    assert any(v["legacy"] is None and v["packed"] is None for v in vectors)
    assert any(v["ellipse"] and v["ellipse"]["f1"] == 0 for v in vectors)
    assert any(v["ellipse"] and v["ellipse"]["f1"] == len(v["nodes"]) - 1 for v in vectors)
