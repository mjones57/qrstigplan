"""Tests for codec/legacy.py: Format A, the text format of the thesis (codec_spec.md section 5)."""

import json
from pathlib import Path

import numpy as np
import pytest

from qrstigplan.codec.legacy import LegacyCodec
from qrstigplan.codec.payload import MarkerEllipse, MarkerPayload, PayloadTooLarge

# Get data from thesis
ROOT = Path(__file__).resolve().parents[1]
VECTORS = json.loads((ROOT / "docs" / "codec_vectors.json").read_text(encoding="utf-8"))
THESIS = json.loads((ROOT / "data" / "thesis_markers.json").read_text(encoding="utf-8"))
CODEC = LegacyCodec()


# Turn given vector into MarkerPayload/MarkerEllipse
def payload(item) -> MarkerPayload:
    ellipse = item["ellipse"]
    return MarkerPayload(
        np.array(item["nodes"]), None if ellipse is None else MarkerEllipse(**ellipse)
    )


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["name"])
def test_vectors(vector):
    p = payload(vector)
    if vector["legacy"] is None:
        assert CODEC.fits(p) is False
        with pytest.raises(PayloadTooLarge):
            CODEC.encode(p)
    else:
        assert CODEC.fits(p) is True
        assert CODEC.encode(p) == vector["legacy"]
        assert CODEC.decode(vector["legacy"]) == p


def test_all_thesis_markers():
    # Every unique marker that the original code wrote, with the
    # corresponding payload the original decoder read
    for marker in THESIS["markers"]:
        p = payload(marker)
        assert CODEC.decode(marker["text"]) == p, marker["text"]
        assert CODEC.encode(p) == marker["text"], marker["text"]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),  # to check start-blocked maps
        pytest.param("5 5", id="bare-pair"),  # original decoder reads this as a node
        pytest.param("5 5 0 7", id="dangling-number"),  # original decoder truncates the 7
        pytest.param("5 5 9 19 4 3 3 4", id="two-ellipses"),
        pytest.param("5  5 0", id="two-spaces"),
        pytest.param("05 5 0", id="leading-zero"),
    ],
)
def test_incorrectly_formatted_text_refused(text):
    with pytest.raises(ValueError):
        CODEC.decode(text)
