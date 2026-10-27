# qrstigplan

**Stigmergic multi-robot motion planning with QR-encoded paths.**

`qrstigplan` is a Python library for motion planning where robots share what they learned through the environment, while communication is limited. A robot that solves a planning problem leaves a small QR code at the goal. A later robot reads the code and uses it to plan faster.

It is a clean rebuild of the code behind my M.S. thesis:

> Miles Jones. *Stigmergy-Based Multi-Agent Motion Planning Using QR Codes.* M.S. thesis, University of Maryland, 2025.

> **Status: early development (`0.1.0.dev0`).** The environment and validation layers are done. The planners, the marker codecs and the experiment pipeline are in progress. The API can change before v0.1.0.

## The idea

1. **Robot 1** plans a path with RRT# on a 2D map with polygon obstacles.
2. It shortens the path by line of sight and snaps it to a grid.
3. If the path is too long for one small QR code (Version 1, error correction L), it replaces part of the path with an **ellipse**. The ellipse is guaranteed to contain the part that it replaces.
4. It encodes the result and leaves the QR code at the goal.
5. **Robot 2** decodes the marker and plans only inside the ellipse. It then joins the new section to the stored parts of the path.

The QR code acts as a small, long-lasting memory for planning experience, tied to one location in the environment. The ellipse compresses a solved problem into a region that is known to already contain a solution.

## Design principles

- **A success is always a valid path.** Every path that the library reports as a success has passed `validate_path`. If a path fails, the result tells you why and which segment is at fault.
- **Every environment is solvable before it is used.** An exact visibility-graph check computes the optimal path length c\* for each map. The generator never returns a map that cannot be solved (unless loosened and reconfigured to do so).
- **The marker format is a written spec.** The encoding has fixed test vectors, so that other implementations can be checked against the same strings.
- **Reproducible.** All randomness goes through seeded NumPy `Generator`s. Data is stored as versioned JSON, not as pickles.
- **No ROS dependency.** The interfaces follow OMPL's style, so the core can be ported to C++ and ROS 2 later.

## Install

`qrstigplan` is not on PyPI yet. Install it from source, with Python 3.12 or newer:

```sh
git clone https://github.com/mjones57/qrstigplan.git
cd qrstigplan
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from qrstigplan.env.generate import generate_environment
from qrstigplan.env.io import load_environments
from qrstigplan.env.visibility import optimal_path
from qrstigplan.planning.validity import validate_path

start, goal = (5.0, 5.0), (45.0, 45.0)

# Draw a random map that is guaranteed to be solvable
rng = np.random.default_rng(seed=7)
env = generate_environment(rng, start, goal).env

# The shortest collision-free path, from the visibility graph
best = optimal_path(env, start, goal)
print(f"c* = {best.length:.2f}")  # c* = 67.71

# Every path is checked before it counts as a success
print(validate_path(best.path, env, start, goal))  # Validation(reason=None, bad_segment=None)

# A straight line through an obstacle is refused, with the reason
print(validate_path([start, goal], env, start, goal))
# Validation(reason=<Reason.SEGMENT_BLOCKED: 4>, bad_segment=0)

# The 100 environments from the thesis experiments
thesis = load_environments("data/thesis_environments.json")
```

## What is in the package

| Module | Contents | Status |
|---|---|---|
| `env.environment` | `Environment`: map bounds, obstacles, inflation, and fast collision queries | Done |
| `env.visibility` | Exact solvability check and optimal path length c\* | Done |
| `env.generate` | Seeded generator for random maps that are always solvable | Done |
| `env.io` | Save and load environments as versioned JSON | Done |
| `geometry.grid` | Grid snapping with a proven bound on how far each point moves | Done |
| `planning.validity` | `validate_path`, with a failure reason and the bad segment | Done |
| `codec` | Marker payload and QR codecs: the thesis text format and a denser packed format | In progress |
| `geometry` | Ellipse geometry and direct sampling inside an ellipse | Planned |
| `planning` | Planner interface, RRT and RRT# | Planned |
| `stigmergy` | The robot 1 → marker → robot 2 pipeline | Planned |
| `experiments`, `analysis` | Experiment runner, data format and paired statistics | Planned |

## Roadmap

- **v0.1.0:** reproduce the thesis experiments with this library, and document every difference from the published results in `REPRODUCTION.md`.

## Development

```sh
python -m pytest      # unit tests and Hypothesis property tests
ruff check .          # lint
ruff format .         # format
```

## License

MIT. See [`LICENSE`](LICENSE).
