import json
from pathlib import Path

from shapely import Polygon

from qrstigplan.env.environment import Environment

FORMAT = "qrstigplan.environments"
VERSION = 1


def _from_dict(item) -> Environment:
    """Takes one environment dict from the file --> Environment"""

    # each val to float, then tuple
    xmin, ymin, xmax, ymax = (float(v) for v in item["bounds"])
    bounds = (xmin, ymin, xmax, ymax)

    inflation = float(item["inflation"])

    # tuple of Polygon(ring), one for each ring in item["obstacles"]
    obstacles = tuple(Polygon(ring) for ring in item["obstacles"])
    return Environment(bounds, obstacles, inflation=inflation)


def _to_dict(env):
    """Takes Environment -> one environment dict for the file"""
    rings = []
    for poly in env.obstacles:
        if len(poly.interiors) > 0:  # refuse holes
            raise ValueError("One of the obstacles contains a hole")
        coords = list(poly.exterior.coords)[:-1]  # exterior ring (w/o close point)
        rings.append(coords)
    return {"bounds": env.bounds, "inflation": env.inflation, "obstacles": rings}


def save_environments(path, environments) -> None:
    items = [_to_dict(env) for env in environments]
    doc = {"format": FORMAT, "version": VERSION, "environments": items}
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load_environments(path) -> list[Environment]:
    """Load environment(s) from json file, return list of Environment objects"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))

    # check format, version -> ValueError if incorrect
    if doc.get("format") != FORMAT:
        raise ValueError(f"Incorrect format {doc.get('format')}, expected {FORMAT}")
    if doc.get("version") != VERSION:
        raise ValueError(f"Incorrect version {doc.get('version')}, expected {VERSION}")

    return [_from_dict(item) for item in doc["environments"]]
