import pytest
from shapely import box

from qrstigplan.env.environment import Environment
from qrstigplan.env.visibility import Solvability, solvability

BOUNDS = (0.0, 0.0, 50.0, 50.0)
SQUARE = box(20, 20, 30, 30)
RING = (box(35, 35, 45, 36), box(35, 44, 45, 45), box(35, 35, 36, 45), box(44, 35, 45, 45))
CORNER_CONTACT = (box(0, 20, 25, 30), box(25, 30, 50, 40))  # two blocks touch at (25, 30)


@pytest.mark.parametrize(
    ("obstacles", "start", "goal", "expected"),
    [
        pytest.param((SQUARE,), (5, 5), (45, 45), Solvability.SOLVABLE, id="solvable"),
        pytest.param((SQUARE,), (25, 25), (45, 45), Solvability.START_BLOCKED, id="start-blocked"),
        pytest.param((SQUARE,), (5, 5), (25, 25), Solvability.GOAL_BLOCKED, id="goal-blocked"),
        pytest.param((SQUARE,), (25, 25), (25, 25), Solvability.START_BLOCKED, id="both-blocked"),
        pytest.param(RING, (5, 5), (40, 40), Solvability.DISCONNECTED, id="goal-walled-in"),
        pytest.param(
            CORNER_CONTACT, (5, 5), (45, 45), Solvability.DISCONNECTED, id="point-contact"
        ),
    ],
)
def test_solvability(obstacles, start, goal, expected):
    assert solvability(Environment(BOUNDS, obstacles), start, goal) is expected
