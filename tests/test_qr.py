"""
Uses and tests codec/qr.py:
Can a string fit inside a Version 1-L QR code?
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qrstigplan.codec.qr import fits_version1_l

LEGACY_25 = "5 5 0 9 4 0 24 6 0 33 8 0"  # alphanumeric mode


@pytest.mark.parametrize(
    ("text", "fits"),
    [
        pytest.param("1" * 41, True, id="numeric-41"),
        pytest.param("1" * 42, False, id="numeric-42"),
        pytest.param(LEGACY_25, True, id="alphanumeric-25"),
        pytest.param(LEGACY_25 + "1", False, id="alphanumeric-26"),
        pytest.param("a" * 17, True, id="byte-17"),  # byte mode
        pytest.param("a" * 18, False, id="byte-18"),
    ],
)
def test_version1_l_capacity(text, fits):
    assert fits_version1_l(text) is fits


@given(text=st.text(alphabet="0123456789", min_size=1, max_size=60))
def test_digit_strings_fit_up_to_41(text):
    assert fits_version1_l(text) is (len(text) <= 41)
