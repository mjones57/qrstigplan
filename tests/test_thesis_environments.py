"""Re-checking original 100 environments from thesis"""

from pathlib import Path

import pytest

from qrstigplan.env.io import load_environments
from qrstigplan.env.visibility import Solvability, optimal_path, solvability

DATA = Path(__file__).resolve().parents[1] / "data" / "thesis_environments.json"
START, GOAL = (5.0, 5.0), (45.0, 45.0)

START_BLOCKED = {17, 27, 33, 43, 53, 56, 81}
GOAL_BLOCKED = {9, 55, 73, 76, 87, 89}

DISCONNECTED = {2, 34, 77}


@pytest.fixture(scope="module")
def envs():
    return load_environments(DATA)


def test_100_envs_W_8_obs(envs):
    assert len(envs) == 100
    assert all(len(env.obstacles) == 8 for env in envs)


def test_unsolvable_environments_with_reason(envs):
    status = {n: solvability(env, START, GOAL) for n, env in enumerate(envs)}
    assert {n for n, s in status.items() if s is Solvability.START_BLOCKED} == START_BLOCKED
    assert {n for n, s in status.items() if s is Solvability.GOAL_BLOCKED} == GOAL_BLOCKED
    assert {n for n, s in status.items() if s is Solvability.DISCONNECTED} == DISCONNECTED
    assert sum(s is Solvability.SOLVABLE for s in status.values()) == 84


def test_cstar_of_env0(envs):
    assert optimal_path(envs[0], START, GOAL).length == pytest.approx(69.548, abs=1e-3)
