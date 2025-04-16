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
import math
import warnings
from pathlib import Path

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
from xdem.coreg.affine import NuthKaab
from xdem.coreg.base import Coreg, CoregPipeline


class BlockwiseCoreg:
    """
    A processing class of choice is run on a subdivision of the raster. When later applying the step
    the optimal warping is interpolated based on X/Y/Z shifts from the coreg algorithm at the grid points.
    """

    def __init__(
        self,
        step: Coreg | CoregPipeline,
        mp_config: MultiprocConfig,
    ) -> None:
        """
        Instantiate a blockwise processing object.

        :param step: An instantiated co-registration step object to fit in the subdivided DEMs.
        :param block_size: Size of chunks in pixels.
        """
        if isinstance(step, type):
            raise ValueError(
                "The 'step' argument must be an instantiated Coreg subclass. " "Hint: write e.g. ICP() instead of ICP"
            )
        self.procstep = step
        self.block_size = mp_config.chunk_size
        if isinstance(step, NuthKaab):
            self.apply_z_correction = step.vertical_shift  # type: ignore
        self.mp_config = mp_config

        self.output_path_reproject = Path(mp_config.outfile).parent / "reprojected_dem.tif"
        self.output_path_aligned = Path(mp_config.outfile).parent / "aligned_dem.tif"

        self.meta = {"inputs": {}, "outputs": {}}

        self.reproject_dem = None

    @staticmethod
    def _coreg_wrapper(
        ref_dem_tiled: RasterType,
        tba_dem: RasterType,
        coreg_method: Coreg | CoregPipeline,
        inlier_mask: RasterType | None = None,
    ) -> Coreg | CoregPipeline:
        """
        Wrapper function to apply Nuth & Kääb coregistration on a tile pair.
        """
        coreg_method = coreg_method.copy()
        tba_dem_tiled = tba_dem.crop(ref_dem_tiled)
        if inlier_mask:
            inlier_mask = inlier_mask.crop(ref_dem_tiled)
        return coreg_method.fit(ref_dem_tiled, tba_dem_tiled, inlier_mask)

    def fit(
        self: BlockwiseCoreg,
        reference_elev: NDArrayf | MArrayf | RasterType,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType,
        inlier_mask: NDArrayb | Mask | None = None,
    ) -> None:

        self.mp_config.outfile = self.output_path_reproject

        self.reproject_dem = to_be_aligned_elev.reproject(  # type: ignore
            ref=reference_elev, multiproc_config=self.mp_config, silent=True
        )

        logging.warning(f"No reprojected DEM returned, but saved at {self.output_path_reproject}")

        self.meta["inputs"] = self.procstep.meta["inputs"]  # type: ignore

        outputs_coreg = map_multiproc_collect(
            self._coreg_wrapper,
            reference_elev,
            self.mp_config,
            self.reproject_dem,
            self.procstep,
            inlier_mask,
            return_tile=True,
        )

        shape_tiling_grid = compute_tiling(self.block_size, reference_elev.shape, to_be_aligned_elev.shape).shape
        rows_cols = list(itertools.product(range(shape_tiling_grid[0]), range(shape_tiling_grid[1])))

        self.x_coords = []
        self.y_coords = []
        self.shifts_x = []
        self.shifts_y = []
        self.shifts_z = []

        for idx, (coreg, tile_coords) in enumerate(outputs_coreg):
            try:
                shift_x = coreg.meta["outputs"]["affine"]["shift_x"]
                shift_y = coreg.meta["outputs"]["affine"]["shift_y"]
                shift_z = coreg.meta["outputs"]["affine"]["shift_z"]

            except KeyError:
                continue

            x, y = (
                tile_coords[2] + self.block_size / 2,
                tile_coords[0] + self.block_size / 2,
            ) * self.reproject_dem.transform

            self.x_coords.append(x)
            self.y_coords.append(y)

            self.shifts_x.append(shift_x)
            self.shifts_y.append(shift_y)
            self.shifts_z.append(shift_z)

            tile_str = f"{rows_cols[idx][0]}_{rows_cols[idx][1]}"
            self.meta["outputs"][tile_str] = {  # type: ignore
                "shift_x": shift_x,
                "shift_y": shift_y,
                "shift_z": shift_z,
            }

        self.x_coords, self.y_coords, self.shifts_x, self.shifts_y, self.shifts_z = map(  # type: ignore
            np.array, (self.x_coords, self.y_coords, self.shifts_x, self.shifts_y, self.shifts_z)
        )

    @staticmethod
    def _ransac(
        x_coords: NDArrayf,
        y_coords: NDArrayf,
        shifts: NDArrayf,
        threshold: float = 0.01,
        min_inliers: int = 10,
        max_iterations: int = 2000,
    ) -> tuple[float, float, float]:

        # Create 3D point clouds
        points = np.squeeze(np.dstack([x_coords, y_coords, shifts]))
        points = points[~np.isnan(points).any(axis=1), :]

        plane = pyrsc.Plane()
        (a, b, c, d), _ = plane.fit(points, thresh=threshold, minPoints=min_inliers, maxIteration=max_iterations)

        # Convert plane ax + by + cz + d = 0 to z = f(x, y)
        # z = -(a*x + b*y + d) / c

        return -a / c, -b / c, -d / c

    def _wrapper_apply_epc(
        self,
        tba_dem_tile: RasterType,
        coeff_x: tuple[float, float, float],
        coeff_y: tuple[float, float, float],
        coeff_z: tuple[float, float, float],
        resampling: str | rio.warp.Resampling = "linear",
    ) -> RasterType:
        """
        Applies a geometric transformation to a raster using specific coefficients for the X, Y, and Z axes.

        :param tba_dem_tile: Input DEM raster to be transformed.
        :param coeff_x: Transformation coefficients for the X axis in the form (a, b, c).
        :param coeff_y: Transformation coefficients for the Y axis in the form (a, b, c).
        :param coeff_z: Transformation coefficients for the Z axis in the form (a, b, c).
        :param resampling: Resampling method to use during transformation. Default is "linear".
        :return: Transformed DEM raster with the applied coefficients.
        """
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

        with warnings.catch_warnings():
            # CRS mismatch between the CRS of left geometries and the CRS of right geometries.
            warnings.filterwarnings("ignore", category=UserWarning)
            # To raster
            new_dem = _grid_pointcloud(
                trans_epc,
                grid_coords=tba_dem_tile.coords(grid=False),
                data_column_name="z",
                resampling=resampling,
            )

        applied_dem_tile = gu.Raster.from_array(new_dem, tba_dem_tile.transform, tba_dem_tile.crs, tba_dem_tile.nodata)
        return applied_dem_tile

    def apply(
        self,
        threshold_ransac: float = 0.01,
        min_inliers_ransac: int = 10,
        max_iterations_ransac: int = 2000,
    ) -> RasterType:
        """
        Apply the coregistration transformation to an elevation array using a ransac filter.
        """

        coeff_x = self._ransac(
            self.x_coords,  # type: ignore
            self.y_coords,  # type: ignore
            self.shifts_x,  # type: ignore
            threshold_ransac,
            min_inliers_ransac,
            max_iterations_ransac,
        )
        coeff_y = self._ransac(
            self.x_coords,  # type: ignore
            self.y_coords,  # type: ignore
            self.shifts_y,  # type: ignore
            threshold_ransac,
            min_inliers_ransac,
            max_iterations_ransac,
        )

        if self.apply_z_correction:
            coeff_z = self._ransac(
                self.x_coords,  # type: ignore
                self.y_coords,  # type: ignore
                self.shifts_z,  # type: ignore
                threshold_ransac,
                min_inliers_ransac,
                max_iterations_ransac,
            )
        else:
            coeff_z = (0, 0, 0)

        self.mp_config.outfile = self.output_path_aligned

        # be careful with depth value if Out of Memory
        depth = np.max([np.max(self.shifts_x), np.max(self.shifts_y)])

        aligned_dem = map_overlap_multiproc_save(
            self._wrapper_apply_epc,
            self.reproject_dem,
            self.mp_config,
            coeff_x,
            coeff_y,
            coeff_z,
            depth=math.ceil(depth),
        )

        return aligned_dem
