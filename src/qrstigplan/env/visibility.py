import itertools
import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class OptimalPath:
    """"""

    length: float  # c*; math.inf if no path exists
    path: np.ndarray | None  # (K, 2) float nodes start->goal; None if unsolvable

    def __bool__(self) -> bool:
        """Return True if solvable"""
        return self.path is not None


def optimal_path(env, start, goal, eps: float = 1e-6) -> OptimalPath:
    """
    Compute sparse visibility graph and use Dijkstra to compute optimal path.
    If start or goal isn't free or Dijkstra returns infinite distance -> unsolvable
    """

    # check start and goal for bad input
    if len(start) != 2 or not np.isfinite(np.asarray(start)).all():
        raise ValueError("start input must be finite 2-vector")
    if len(goal) != 2 or not np.isfinite(np.asarray(goal)).all():
        raise ValueError("goal input must be finite 2-vector")

    if not env.point_is_free(start) or not env.point_is_free(goal):
        return UNSOLVABLE

    # buffer obstacle geometry by epsilon
    inflated_obs = env.blocked.buffer(eps, join_style="mitre")

    # get corner nodes of the buffered obstacle geometry
    corners = shapely.get_coordinates(inflated_obs)
    corners = np.unique(corners, axis=0)  # Remove duplicates

    # init stack array with start and goal
    nodes = np.vstack((start, goal, corners), dtype=float)
    n = len(nodes)

    # check if each segment free, apply weight = length if free
    rows, cols, weights = [], [], []  # edge between node rows[k]<->cols[k] has length weights[k]
    for i, j in itertools.combinations(range(n), 2):
        if env.segment_is_free(nodes[i], nodes[j]):
            rows.append(i)
            cols.append(j)
            weights.append(math.dist(nodes[i], nodes[j]))

    # build sparse graph and run dijkstra
    graph = coo_array((weights, (rows, cols)), shape=(n, n))
    dist, pred = dijkstra(graph, directed=False, indices=0, return_predecessors=True)
    # dist[k]: shortest distance from node 0 to node k (inf = unreachable)
    # pred[k]: node before k on that path (-9999 if unreachable)

    # check if unsolvable
    if math.isinf(dist[1]):
        return UNSOLVABLE

    # use pred to follow back to node 0
    k = 1
    indices = [k]  # init with goal index
    while k != 0:
        k = pred[k]
        indices.append(k)

    indices.reverse()  # reverse so path is start idx -> goal idx
    path = nodes[indices]  # index the nodes array for actual path
    return OptimalPath(length=float(dist[1]), path=path)


UNSOLVABLE = OptimalPath(math.inf, None)  # UNSOLVABLE
