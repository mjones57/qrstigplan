import pytest
from shapely import Polygon

from qrstigplan.env.environment import Environment
from qrstigplan.geometry.grid import snap_shift_bound

BOUNDS = (0.0, 0.0, 50.0, 50.0)
SQUARE = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])


@pytest.fixture(scope="session")
def env():
    return Environment(bounds=BOUNDS, obstacles=(SQUARE,), inflation=0.0)


@pytest.fixture(scope="session")
def safe_env():
    return Environment(bounds=BOUNDS, obstacles=(SQUARE,), inflation=snap_shift_bound(1.0))
