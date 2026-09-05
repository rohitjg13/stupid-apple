"""Zone lookup and heatmap cells on a precomputed raster.

No shapely on the board: zones are rasterised once at HEATMAP_CELL_M and every
later lookup is an array index. SHARED.md §6.
"""
from __future__ import annotations

import numpy as np

HEATMAP_CELL_M = 0.25
PAD_M = 1.0                  # keep a margin so points just off the floor still index


class ZoneMap:
    def __init__(self, zones, cell_m: float = HEATMAP_CELL_M):
        import cv2                       # rasterise once at startup; not per frame

        self.cell_m = cell_m
        self.ids = [z["id"] for z in zones]
        polys = [np.asarray(z["polygon"], dtype=float) for z in zones]
        allp = np.vstack(polys)
        self.origin = allp.min(axis=0) - PAD_M
        extent = allp.max(axis=0) + PAD_M - self.origin
        self.shape = tuple(int(np.ceil(e / cell_m)) + 1 for e in extent)   # (nx, ny)

        # 0 = no zone, i+1 = zones[i]. Cell g covers [g, g+1) in cell units, so its
        # centre sits at g+0.5 -- shift the polygon by half a cell or every zone
        # lands half a cell off. `shift` gives fillPoly 1/16-cell subpixel accuracy.
        SHIFT = 4
        grid = np.zeros((self.shape[1], self.shape[0]), dtype=np.int16)
        for i, poly in enumerate(polys):
            cells = ((poly - self.origin) / cell_m - 0.5) * (1 << SHIFT)
            cv2.fillPoly(grid, [np.round(cells).astype(np.int32)], i + 1, shift=SHIFT)
        self.grid = grid

    def _cell(self, X, Y):
        gx = int(np.floor((X - self.origin[0]) / self.cell_m))
        gy = int(np.floor((Y - self.origin[1]) / self.cell_m))
        return gx, gy

    def zone_of(self, X, Y):
        gx, gy = self._cell(X, Y)
        if not (0 <= gx < self.shape[0] and 0 <= gy < self.shape[1]):
            return None
        v = int(self.grid[gy, gx])
        return self.ids[v - 1] if v else None

    def zone_of_many(self, pts):
        return [self.zone_of(X, Y) for X, Y in np.atleast_2d(pts)]

    def heatmap_index(self, X, Y):
        """Absolute 0.25 m cell, independent of the raster origin, so the
        dashboard's tiles mean the same thing across configs."""
        return (int(np.floor(X / HEATMAP_CELL_M)), int(np.floor(Y / HEATMAP_CELL_M)))
