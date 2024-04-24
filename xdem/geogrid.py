from typing import Any, Iterator, Literal
import itertools

import math
import numpy as np
import rasterio as rio
import pandas as pd
import geopandas as gpd

# Those two functions only require GeoPandas/Shapely
from geoutils.projtools import _get_footprint_projected, _get_bounds_projected

class GeoGrid:
    """Georeferenced grid class."""

    def __init__(self, transform: rio.transform.Affine, shape: tuple[int, int], crs: rio.crs.CRS | None):

        self._transform = transform
        self._shape = shape
        self._crs = crs

    @property
    def transform(self) -> rio.transform.Affine:
        return self._transform

    @property
    def crs(self) -> rio.crs.CRS:
        return self._crs

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def res(self) -> tuple[int, int]:
        return self.transform[0], abs(self.transform[4])

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        return rio.coords.BoundingBox(*rio.transform.array_bounds(self.height, self.width, self.transform))

    def bounds_projected(self, crs: rio.crs.CRS = None) -> rio.coords.BoundingBox:
        if crs is None:
            crs = self.crs
        return _get_bounds_projected(bounds=self.bounds, in_crs=self.crs, out_crs=crs)

    @property
    def footprint(self) -> gpd.GeoDataFrame:
        return _get_footprint_projected(self.bounds, in_crs=self.crs, out_crs=self.crs, densify_points=100)

    def footprint_projected(self, crs: rio.crs.CRS = None):
        if crs is None:
            crs = self.crs
        return _get_footprint_projected(self.bounds, in_crs=self.crs, out_crs=crs, densify_points=100)

    def shift(self,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "pixel"):
        """Shift geogrid, not inplace."""

        if distance_unit not in["georeferenced", "pixel"]:
            raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

        # Get transform
        dx, b, xmin, d, dy, ymax = list(self.transform)[:6]

        # Convert pixel offsets to georeferenced units
        if distance_unit == "pixel":
            xoff *= self.res[0]
            yoff *= self.res[1]

        shifted_transform = rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)

        return GeoGrid(transform=shifted_transform, crs=self.crs, shape=self.shape)


def _get_block_ids_per_chunk(chunks: tuple[tuple[int, ...], tuple[int, ...]]) -> list[dict[str, int]]:
    """Get location of chunks based on array shape and list of chunk sizes."""

    # Get number of chunks
    num_chunks = (len(chunks[0]), len(chunks[1]))

    # Get robust list of chunk locations (using what is done in block_id of dask.array.map_blocks)
    # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
    from dask.utils import cached_cumsum
    starts = [cached_cumsum(c, initial_zero=True) for c in chunks]
    nb_blocks = num_chunks[0] * num_chunks[1]
    ixi, iyi = np.unravel_index(np.arange(nb_blocks), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [{"num_block": i, "ys": starts[0][ixi[i]],  "xs": starts[1][iyi[i]],
                  "ye": starts[0][ixi[i] + 1], "xe":  starts[1][iyi[i] + 1]}
                 for i in range(nb_blocks)]

    return block_ids

class ChunkedGeoGrid:
    """Chunked georeferenced grid class."""

    def __init__(self, grid: GeoGrid, chunks: tuple[tuple[int, ...], tuple[int, ...]]):

        self._grid = grid
        self._chunks = chunks

    @property
    def grid(self) -> GeoGrid:
        return self._grid

    @property
    def chunks(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._chunks

    def get_block_locations(self) -> list[dict[str, int]]:
        """Get block locations in 2D: xstart, xend, ystart, yend."""
        return _get_block_ids_per_chunk(self._chunks)

    def get_blocks_as_geogrids(self) -> list[GeoGrid]:
        """Get blocks as geogrids with updated transform/shape."""

        block_ids = self.get_block_locations()

        list_geogrids = []
        for bid in block_ids:
            # We get the block size
            block_shape = (bid["ye"] - bid["ys"], bid["xe"] - bid["xs"])
            # Build a temporary geogrid with the same transform as the full grid
            geogrid_tmp = GeoGrid(transform=self.grid.transform, crs=self.grid.crs, shape=block_shape)
            # And shift it to the right location (X is positive in index direction, Y is negative)
            geogrid_block = geogrid_tmp.shift(xoff=bid["xs"], yoff=-bid["ys"])
            list_geogrids.append(geogrid_block)

        return list_geogrids

    def get_block_footprints(self, crs: rio.crs.CRS = None) -> gpd.GeoDataFrame:
        """Get block projected footprints as a single geodataframe."""

        geogrids = self.get_blocks_as_geogrids()
        footprints = [gg.footprint_projected(crs=crs) if crs is not None else gg.footprint for gg in geogrids]

        return pd.concat(footprints)