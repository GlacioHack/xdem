# Copyright (c) 2024 xDEM developers
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block-wise co-registration processing class to run a step in segmented parts of the grid."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Literal

import geopandas as gpd
import geoutils as gu
import numpy as np
import pyransac3d as pyrsc
import rasterio as rio
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.raster import Mask, RasterType
from geoutils.raster.distributed_computing import (
    MultiprocConfig,
    map_multiproc_collect,
    map_overlap_multiproc_save,
)
from geoutils.raster.tiling import compute_tiling
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg import Coreg, CoregPipeline
from xdem.coreg.base import Coreg, CoregPipeline, CoregType
import xdem


class BlockwiseCoreg(Coreg):
    """
    A processing class of choice is run on a subdivision of the raster. When later applying the step
    the optimal warping is interpolated based on X/Y/Z shifts from the coreg algorithm at the grid points.
    """

    def __init__(
        self,
        step: Coreg | CoregPipeline,
        tile_size: int = 300,
        apply_z_correction: bool = True,
        output_path: str = "",
    ) -> None:
        """
        Instantiate a blockwise processing object.

        :param step: An instantiated co-registration step object to fit in the subdivided DEMs.
        :param tile_size: Size of chunks in pixels.
        :param apply_z_correction: Boolean to toggle whether the Z-offset correction is applied or not (default True).
        """
        if isinstance(step, type):
            raise ValueError(
                "The 'step' argument must be an instantiated Coreg subclass. "
                "Hint: write e.g. ICP() instead of ICP"
            )
        self.procstep = step
        self.tile_size = tile_size
        self.apply_z_correction = apply_z_correction
        self.output_path = output_path
        self.x_coords = []
        self.y_coords = []
        self.shifts_x = []
        self.shifts_y = []
        self.shifts_z = []

        tile_size = 500
        self.mp_config = MultiprocConfig(tile_size)

        super().__init__()

    @staticmethod
    def coreg_wrapper(
        ref_dem_tiled, tba_dem, coreg_method, inlier_mask
    ) -> Coreg | CoregPipeline:
        """
        Wrapper function to apply Nuth & Kääb coregistration on a tile pair.
        """
        coreg_method = coreg_method.copy()
        tba_dem_tiled = tba_dem.crop(ref_dem_tiled)
        return coreg_method.fit(ref_dem_tiled, tba_dem_tiled, inlier_mask)

    def preprocess(self, ref: RasterType, sec: RasterType) -> RasterType | None:
        """
        Reproject the secondary elevation dataset to match the reference dataset.

        :param ref: Reference elevation data (used as target for reprojection).
        :param sec: Secondary elevation data to be reprojected.
        :return: Reprojected secondary elevation dataset.
        """

        mp_config = MultiprocConfig(
            chunk_size=self.tile_size, outfile=self.output_path + "SEC_reprojected.tif"
        )
        reproject_dem = sec.reproject(
            ref=ref, resampling="cubic", multiproc_config=mp_config, silent=True
        )

        return reproject_dem

    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> CoregType:

        if isinstance(reference_elev, gpd.GeoDataFrame) and isinstance(
            to_be_aligned_elev, gpd.GeoDataFrame
        ):
            raise NotImplementedError(
                "Blockwise coregistration does not yet support two elevation point cloud inputs."
            )

        self.reproject_dem = to_be_aligned_elev.reproject(
            ref=reference_elev, multiproc_config=self.mp_config, silent=True
        )

        self.meta["inputs"] = self.procstep.meta["inputs"]

        outputs_coreg = map_multiproc_collect(
            self.coreg_wrapper,
            reference_elev,
            self.mp_config,
            self.reproject_dem,
            self.procstep,
            inlier_mask,
            return_tile=True,
        )

        shape_tiling_grid = compute_tiling(
            self.tile_size, reference_elev.shape, to_be_aligned_elev.shape
        ).shape
        rows_cols = list(
            itertools.product(range(shape_tiling_grid[0]), range(shape_tiling_grid[1]))
        )

        for idx, (coreg, tile_coords) in enumerate(outputs_coreg):
            tile_str = f"{rows_cols[idx][0]}_{rows_cols[idx][1]}"
            try:
                shift_x = coreg.meta["outputs"]["affine"]["shift_x"]
                shift_y = coreg.meta["outputs"]["affine"]["shift_y"]
                shift_z = coreg.meta["outputs"]["affine"]["shift_z"]
                self.meta["outputs"][tile_str] = {
                    "shift_x": shift_y,
                    "shift_y": shift_y,
                    "shift_z": shift_z,
                }

                self.shifts_x.append(shift_x)
                self.shifts_y.append(shift_y)
                self.shifts_z.append(shift_z)

            except KeyError:
                self.meta["outputs"][tile_str] = {
                    "shift_x": np.nan,
                    "shift_y": np.nan,
                    "shift_z": np.nan,
                }

            x, y = (
                tile_coords[2] + self.tile_size / 2,
                tile_coords[0] + self.tile_size / 2,
            ) * self.reproject_dem.transform

            self.x_coords.append(x)
            self.y_coords.append(y)

        self.x_coords, self.y_coords, self.shifts_x, self.shifts_y = map(
            np.array, (self.x_coords, self.y_coords, self.shifts_x, self.shifts_y)
        )

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @staticmethod
    def ransac(
        x_coords: NDArrayf,
        y_coords: NDArrayf,
        shifts: NDArrayf,
        threshold: float = 1e-7,
        min_inliers: int = 10,
    ) -> tuple[float, float, float]:
        # Create 3D point clouds
        points = np.squeeze(np.dstack([x_coords, y_coords, shifts]))
        points = points[~np.isnan(points).any(axis=1), :]

        plane = pyrsc.Plane()
        try:
            (a, b, c, d), _ = plane.fit(points, thresh=threshold, minPoints=min_inliers)
        except ValueError:
            raise ValueError(
                "Not enough inliers, please increase the size of your tiles. "
            )

        # Convert plane ax + by + cz + d = 0 to z = f(x, y)
        # z = -(a*x + b*y + d) / c

        return -a / c, -b / c, -d / c

    def wrapper_apply_epc(
        self,
        tba_dem_tile: RasterType,
        coeff_x: tuple[float, float, float],
        coeff_y: tuple[float, float, float],
        coeff_z: tuple[float, float, float],
        resampling: str | rio.warp.Resampling = "linear",
    ):
        # To pointcloud
        epc = tba_dem_tile.to_pointcloud(data_column_name="z").ds
        # Unpack coefficients
        a_x, b_x, d_x = coeff_x
        a_y, b_y, d_y = coeff_y
        a_z, b_z, d_z = coeff_z

        # Extract x, y, z from the point cloud
        x = epc.geometry.x.values
        y = epc.geometry.y.values
        z = epc["z"].values

        # Compute modeled shift fields
        shift_x = a_x * x + b_x * y + d_x
        shift_y = a_y * x + b_y * y + d_y
        shift_z = a_z * x + b_z * y + d_z

        # Apply shifts to the coordinates
        x_new = x + shift_x
        y_new = y + shift_y
        z_new = z + shift_z

        trans_epc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x_new, y_new, crs=epc.crs),
            data={"z": z_new if self.apply_z_correction else z},
        )

        # To raster
        new_dem = _grid_pointcloud(
            trans_epc,
            grid_coords=tba_dem_tile.coords(grid=False),
            data_column_name="z",
            resampling=resampling,
        )
        applied_dem_tile = gu.Raster.from_array(
            new_dem, tba_dem_tile.transform, tba_dem_tile.crs, tba_dem_tile.nodata
        )
        return applied_dem_tile

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """
        Apply the coregistration transformation to an elevation array using a ransac filter.

        :param elev: Elevation data as a NumPy array.
        :param transform: Affine transform associated with the input elevation data.
        :param crs: Coordinate reference system of the elevation data.
        :param bias_vars: Optional dictionary of bias variables to apply (not used here).
        :param kwargs: Additional keyword arguments (ignored in this implementation).
        :return: Tuple of dummy values (0, 0), as output is saved directly to disk.
        """

        coeff_x = self.ransac(self.x_coords, self.y_coords, self.shifts_x)
        coeff_y = self.ransac(self.x_coords, self.y_coords, self.shifts_y)
        coeff_z = self.ransac(self.x_coords, self.y_coords, self.shifts_z)

        apply_config = MultiprocConfig(
            chunk_size=self.tile_size,
            outfile=self.output_path + "aligned_dem.tif",
        )

        _ = map_overlap_multiproc_save(
            self.wrapper_apply_epc,
            self.reproject_dem,
            apply_config,
            coeff_x,
            coeff_y,
            coeff_z,
            depth=10,
        )

        transform = xdem.DEM(self.output_path + "aligned_dem.tif").transform
        logging.warning(
            f"No DEM returned, but saved at {self.output_path}_aligned_dem.tif"
        )

        return np.empty([2, 2]), transform
