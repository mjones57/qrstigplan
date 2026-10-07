"""
One-time script to convert the original pickled environments to json.
Converted files will go to data/thesis_environments.json

Run this script from repo root: python scripts/convert_thesis_envs.py
"""

import pickle
from pathlib import Path

from shapely import Polygon

from qrstigplan.env.environment import Environment
from qrstigplan.env.io import save_environments

LEGACY = Path.home() / "Documents" / "GitHub" / "Miles-Thesis" / "Environments"
OUT = Path("data") / "thesis_environments.json"
BOUNDS = (0.0, 0.0, 50.0, 50.0)  # bounds from thesis


def main() -> None:
    environments = []
    for n in range(100):
        with open(LEGACY / f"environment_polygon_{n}.pickle", "rb") as f:
            vertex_lists = pickle.load(f)
        environments.append(Environment(BOUNDS, tuple(Polygon(v) for v in vertex_lists)))
    OUT.parent.mkdir(exist_ok=True)
    save_environments(OUT, environments)
    print(f"wrote {len(environments)} environments to {OUT}")


if __name__ == "__main__":
    main()
