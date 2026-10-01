import pytest

from qrstigplan.planning.validity import Reason, validate_path

START = (5.0, 5.0)
GOAL = (45.0, 45.0)


@pytest.mark.parametrize(
    ("path", "reason", "bad_segment"),
    [
        pytest.param([(5, 5), (5, 45), (45, 45)], None, None, id="valid-around"),
        pytest.param([(5, 5), (45, 45)], Reason.SEGMENT_BLOCKED, 0, id="straight-through"),
        pytest.param(
            [(5, 5), (15, 25), (35, 25), (45, 45)],
            Reason.SEGMENT_BLOCKED,
            1,
            id="nodes-free-segment-blocked",
        ),
        pytest.param(
            [(5, 5), (15, 15), (45, 45)], Reason.SEGMENT_BLOCKED, 1, id="last-segment-blocked"
        ),
        pytest.param(
            [(5, 5), (-2, 25), (5, 45), (45, 45)], Reason.SEGMENT_BLOCKED, 0, id="leaves-map"
        ),
        pytest.param([(5, 5), (5, 45), (40, 45)], Reason.WRONG_GOAL, None, id="ends-short"),
        pytest.param([(6, 5), (5, 45), (45, 45)], Reason.WRONG_START, None, id="wrong-start"),
        pytest.param([(5, 5)], Reason.TOO_FEW_NODES, None, id="single-node"),
        pytest.param([(5, 5), (5, 5), (5, 45), (45, 45)], None, None, id="repeated-node"),
        pytest.param(
            [(6, 5), (15, 15), (45, 45)], Reason.WRONG_START, None, id="wrong-start-and-blocked"
        ),
    ],
)
def test_validate_path(env, path, reason, bad_segment):
    result = validate_path(path, env, START, GOAL)
    assert result.reason == reason
    assert result.bad_segment == bad_segment
    assert result.ok == (reason is None)


def test_invalid_result_is_false(env):
    assert not validate_path([(5, 5), (45, 45)], env, START, GOAL)


def test_nan_node_raises(env):
    with pytest.raises(ValueError, match="finite"):
        validate_path([(5, 5), (float("nan"), 25), (40, 45)], env, START, GOAL)
