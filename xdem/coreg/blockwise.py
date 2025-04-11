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

""" Block-wise co-registration processing class to run a step in segmented parts of the grid."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Literal

import geopandas as gpd
import numpy as np
import pyransac3d as pyrsc
import rasterio as rio
from geoutils.raster import Mask, RasterType
from geoutils.raster.distributed_computing import (
    MultiprocConfig,
    map_multiproc_collect,
    map_overlap_multiproc_save,
)
from geoutils.raster.tiling import compute_tiling

import xdem
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg import NuthKaab
from xdem.coreg.base import Coreg, CoregPipeline, CoregType


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
                "The 'step' argument must be an instantiated Coreg subclass. " "Hint: write e.g. ICP() instead of ICP"
            )
        self.procstep = step
        self.tile_size = tile_size
        self.apply_z_correction = apply_z_correction
        self.output_path = output_path
        self.new_meta = {}

        super().__init__()

    @staticmethod
    def coreg_wrapper(ref_tile: RasterType, sec_raster: RasterType, inlier_mask: NDArrayb | Mask | None = None) -> list:
        """
        Wrapper function to apply Nuth & Kääb coregistration on a tile pair.

        :param ref_tile: Reference DEM tile.
        :param sec_raster: Secondary raster object from which a tile is cropped to match the reference tile.
        :param inlier_mask: Optional inlier mask to restrict fitting to certain areas.
        :return: List of shifts [shift_x, shift_y, shift_z], or NaNs if fitting fails.
        """

        sec_tile = sec_raster.crop(ref_tile)
        try:
            nuth_kaab = NuthKaab()
            nuth_kaab.fit(ref_tile, sec_tile, inlier_mask=inlier_mask)
            shift = nuth_kaab.meta["outputs"]["affine"]
            x_y_z = [shift["shift_x"], shift["shift_y"], shift["shift_z"]]
        except ValueError:
            x_y_z = [np.nan, np.nan, np.nan]

        return x_y_z

    def preprocess(self, ref: RasterType, sec: RasterType) -> RasterType | None:
        """
        Reproject the secondary elevation dataset to match the reference dataset.

        :param ref: Reference elevation data (used as target for reprojection).
        :param sec: Secondary elevation data to be reprojected.
        :return: Reprojected secondary elevation dataset.
        """

        mp_config = MultiprocConfig(chunk_size=self.tile_size, outfile=self.output_path + "SEC_reprojected.tif")
        result = sec.reproject(ref=ref, resampling="cubic", multiproc_config=mp_config, silent=True)

        return result

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

        if isinstance(reference_elev, gpd.GeoDataFrame) and isinstance(to_be_aligned_elev, gpd.GeoDataFrame):
            raise NotImplementedError("Blockwise coregistration does not yet support two elevation point cloud inputs.")

        # Define inlier mask if None, before indexing subdivided array in process function below
        if inlier_mask is None:
            mask = np.ones(to_be_aligned_elev.shape, dtype=bool)
        else:
            mask = inlier_mask

        outputs_reproj = self.preprocess(reference_elev, to_be_aligned_elev)
        if outputs_reproj is None:
            self.dem_to_be_aligned = xdem.DEM(self.output_path + "SEC_reprojected.tif")

        else:
            self.dem_to_be_aligned = to_be_aligned_elev

        tiling_grid = compute_tiling(self.tile_size, reference_elev.shape, to_be_aligned_elev.shape)
        shape_tiling_grid = tiling_grid.shape

        config_multiproc = MultiprocConfig(chunk_size=self.tile_size)
        res_multiproc = map_multiproc_collect(
            self.coreg_wrapper, reference_elev.filename, config_multiproc, to_be_aligned_elev, mask, return_tile=True
        )

        self.res_coreg = [element[0] for element in res_multiproc]
        tiles_coords = [element[1] for element in res_multiproc]

        rows_cols = list(itertools.product(range(shape_tiling_grid[0]), range(shape_tiling_grid[1])))
        resolution = reference_elev.res
        res_correg = np.empty((shape_tiling_grid[0], shape_tiling_grid[1], 3))
        res_positions = np.empty((shape_tiling_grid[0], shape_tiling_grid[1], 2))

        for _, (coreg_res, tile, (row, col)) in enumerate(zip(self.res_coreg, tiles_coords, rows_cols)):
            center_position = ((tile[0] + tile[1]) / 2, (tile[2] + tile[3]) / 2)
            res_correg[row, col] = np.array([coreg_res[0] / resolution[0], coreg_res[1] / resolution[1], coreg_res[2]])
            res_positions[row, col] = center_position
            self.new_meta[str((row, col))] = {"shift_x": coreg_res[0], "shift_y": coreg_res[1], "shift_z": coreg_res[2]}

        self.res_coreg_for_apply = {"positions": res_positions, "coreg": res_correg}

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @staticmethod
    def resample_wrapper(tile_dem: RasterType, coeff_x_grid: list[float], coeff_y_grid: list[float]) -> RasterType:
        """
        Apply a geometric translation to a DEM tile based on polynomial coefficients.

        :param tile_dem: A DEM tile object with resolution information and a translate method.
        :param coeff_x_grid: Tuple of 3 coefficients (A1, B1, C1) for computing x-axis shifts.
        :param coeff_y_grid: Tuple of 3 coefficients (A2, B2, C2) for computing y-axis shifts.
        :return: The translated DEM tile (modified in place).
        """

        A1, B1, C1 = coeff_x_grid
        A2, B2, C2 = coeff_y_grid
        x_shifts = A1 * tile_dem.res[0] + B1 * tile_dem.res[1] + C1
        y_shifts = A2 * tile_dem.res[0] + B2 * tile_dem.res[1] + C2

        tile_dem.translate(x_shifts, -y_shifts, distance_unit="pixel", inplace=True)

        return tile_dem

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

        coeff_x_grid, coeff_y_grid = self.generate_ransac_filter(self.res_coreg_for_apply)

        config_multiproc = MultiprocConfig(chunk_size=self.tile_size, outfile=self.output_path + "aligned_DEM.tif")
        map_overlap_multiproc_save(
            self.resample_wrapper,
            self.dem_to_be_aligned.filename,
            config_multiproc,
            coeff_x_grid,
            coeff_y_grid,
        )

        return 0, 0

    @staticmethod
    def filter_ransac(
        coreg: NDArrayb, positions: NDArrayb, thresh: float = 0.05, minPoints: int = 10, maxIteration: int = 100
    ) -> list[float]:
        """
        Apply RANSAC filtering to coregistration data to fit a plane model.

        This function uses the RANSAC algorithm to fit a plane to the coregistration data,
        identifying inliers and outliers based on the specified threshold. It returns the
        fitted plane parameters and the adjusted coregistration array.

        :param coreg: A 2D array of coregistration values
        :param positions: A 3D array where the first two dimensions represent row and column indices
        :param thresh: Threshold distance for a point to be considered an inlier, defaults to 0.05
        :param minPoints: Minimum number of points required to fit a plane, defaults to 10
        :param maxIteration: Maximum number of iterations for the RANSAC algorithm, defaults to 100

        :return: A tuple containing:
                 - [A, B, C]: The coefficients of the fitted plane equation Ax + By + C = z
        """
        rows, cols = positions[:, :, 0], positions[:, :, 1]
        rows, cols, arr = rows.flatten(), cols.flatten(), coreg.flatten()

        points = np.squeeze(np.dstack([rows, cols, arr]))
        points = points[~np.isnan(points).any(axis=1), :]

        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(points, thresh=thresh, minPoints=minPoints, maxIteration=maxIteration)
        A = -best_eq[0] / best_eq[2]
        B = -best_eq[1] / best_eq[2]
        C = -best_eq[3] / best_eq[2]

        return [A, B, C]

    def generate_ransac_filter(self, correg: dict) -> tuple[list, list]:
        """
        Generate a correction grid for coregistration data.

        This method generates a correction grid for the coregistration data using RANSAC filtering
        to fit plane models to the x and y shifts. The grid is generated over the specified DEM shape
        with a given step size.

        :param correg: A dictionary containing coregistration data with keys 'coreg' and 'positions'
        :param step: The step size for generating the grid
        :param dem_shape: A tuple representing the shape of the DEM (rows, columns)

        :return: A dictionary containing:
                 - grid: A stacked grid of row and column corrections
                 - step: The step size used for generating the grid
        """

        correg_shift = correg["coreg"]
        positions = correg["positions"]

        percent_not_nan = np.sum(~np.isnan(np.array(correg_shift))) / np.size(np.array(correg_shift))

        nb_points_min = int((correg_shift.shape[0] * correg_shift.shape[1]) * 0.85 * percent_not_nan)
        max_iter = 2000

        thresh_x = (np.nanpercentile(correg_shift[:, :, 0], 90) - np.nanpercentile(correg_shift[:, :, 0], 10)) / 3
        thresh_y = (np.nanpercentile(correg_shift[:, :, 1], 90) - np.nanpercentile(correg_shift[:, :, 1], 10)) / 3

        coef_ransac_x = self.filter_ransac(
            correg_shift[:, :, 0],
            positions,
            thresh=thresh_x,
            minPoints=nb_points_min,
            maxIteration=max_iter,
        )
        coef_ransac_y = self.filter_ransac(
            correg_shift[:, :, 1],
            positions,
            thresh=thresh_y,
            minPoints=nb_points_min,
            maxIteration=max_iter,
        )

        return coef_ransac_x, coef_ransac_y
