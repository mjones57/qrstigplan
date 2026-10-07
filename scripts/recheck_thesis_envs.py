"""
Re-check the thesis environments and write it to docs/thesis_environments.csv

Run from the repo root: python scripts/recheck_thesis_envs.py
"""

import csv
from pathlib import Path

from qrstigplan.env.io import load_environments
from qrstigplan.env.visibility import Solvability, optimal_path, solvability

DATA = Path("data") / "thesis_environments.json"
OUT = Path("docs") / "thesis_environments.csv"
START, GOAL = (5.0, 5.0), (45.0, 45.0)


def main() -> None:
    envs = load_environments(DATA)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "solvability", "c_star"])  # header
        for n, env in enumerate(envs):
            status = solvability(env, START, GOAL)
            if status is Solvability.SOLVABLE:
                cstar = optimal_path(env, START, GOAL).length
            else:
                cstar = ""
            writer.writerow([n, status.name, cstar])
    print(f"wrote {len(envs)} rows to {OUT}")


if __name__ == "__main__":
    main()
