import pytest
from shapely import Polygon

from qrstigplan.env.environment import Environment

BOUNDS = (0.0, 0.0, 50.0, 50.0)
SQUARE = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])


@pytest.fixture
def env():
    return Environment(bounds=BOUNDS, obstacles=(SQUARE,), inflation=0.0)
