from dataclasses import dataclass, field

import shapely
from shapely.geometry import LineString, Point, Polygon


@dataclass(frozen=True)
class Environment:
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    obstacles: tuple[Polygon, ...]
    inflation: float = 0.0
    # and other private derived fields, init=False
    _box: Polygon = field(init=False, repr=False)
    _union: shapely.Geometry = field(init=False, repr=False)  # Polygon | MultiPolygon |
    # GeometryCollection

    def __post_init__(self):
        object.__setattr__(self, "_box", shapely.box(*self.bounds))
        object.__setattr__(
            self,
            "_union",
            shapely.union_all([o.buffer(self.inflation) for o in self.obstacles]),
        )
        shapely.prepare(self._box)
        shapely.prepare(self._union)

    def segment_is_free(self, p, q) -> bool:
        """Return True if segment p->q stays inside the map and touches no obstacles"""
        line = LineString([p, q])
        if self._union.intersects(line) or not self._box.covers(line):
            return False
        return True

    def point_is_free(self, p) -> bool:
        """Return True if point p stays inside the map and touches no obstacles"""
        if not self._box.covers(Point(p)) or self._union.intersects(Point(p)):
            return False
        return True
