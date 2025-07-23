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
import os
import warnings
from pathlib import Path

import geopandas as gpd
import geoutils as gu
import numpy as np
import rasterio as rio
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.raster import Mask, RasterType
from geoutils.raster.array import get_array_and_mask
from geoutils.raster.distributed_computing import (
    MultiprocConfig,
    map_multiproc_collect,
    map_overlap_multiproc_save,
)
from geoutils.raster.tiling import compute_tiling

try:
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    _has_sklearn = True
except ImportError:
    _has_sklearn = False

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
        mp_config: MultiprocConfig | None = None,
        block_size_fit: int = 500,
        block_size_apply: int = 500,
        parent_path: str = None,
    ) -> None:
        """
        Instantiate a blockwise processing object for performing coregistration on subdivided DEM tiles.

        :param step: An instantiated coregistration method or pipeline to apply on each tile.
        :param mp_config: Configuration object for multiprocessing
        :param block_size_fit: Size of tiles to process per coregistration step in fit step.
        :param block_size_apply: Size of tiles to process per coregistration step in apply step.
        :param parent_path: Parent path for output files.
        """

        if (mp_config is not None) and (parent_path is not None):
            raise ValueError("Only one of the parameters 'mp_config' or 'parent_path' may be specified.")
        if (mp_config is None) and (parent_path is None):
            raise ValueError("Exactly one of the parameters 'mp_config' or 'parent_path' must be provided.")

        if isinstance(step, type):
            raise ValueError(
                "The 'step' argument must be an instantiated Coreg subclass. " "Hint: write e.g. ICP() instead of ICP"
            )

        self.procstep = step
        self.block_size_fit = block_size_fit
        # NB: in case of memory peak reduce block_size_apply
        self.block_size_apply = block_size_apply

        if isinstance(step, NuthKaab):
            self.apply_z_correction = step.vertical_shift  # type: ignore

        if mp_config is not None:
            self.mp_config = mp_config
            self.mp_config.chunk_size = block_size_fit
            self.parent_path = Path(mp_config.outfile).parent
        else:
            self.mp_config = MultiprocConfig(chunk_size=self.block_size_fit, outfile="aligned_dem.tif")
            self.parent_path = Path(parent_path)  # type: ignore

        os.makedirs(self.parent_path, exist_ok=True)

        self.output_path_reproject = self.parent_path / "reprojected_dem.tif"
        self.output_path_aligned = self.parent_path / "aligned_dem.tif"

        self.meta = {"inputs": {}, "outputs": {}}
        self.shape_tiling_grid = (0, 0, 0)

        self.reproject_dem = None

    @staticmethod
    def _coreg_wrapper(
        ref_dem_tiled: RasterType,
        tba_dem: RasterType,
        coreg_method: Coreg | CoregPipeline,
        inlier_mask: RasterType | None = None,
    ) -> Coreg | CoregPipeline:
        """
         Wrapper function to apply a coregistration method (e.g., Nuth & Kääb) on a pair of DEM tiles.

        :param ref_dem_tiled: Reference DEM tile to align to.
        :param tba_dem: DEM tile to be aligned.
        :param coreg_method: Coregistration method or pipeline to apply.
        :param inlier_mask: Optional mask indicating valid data points to consider during coregistration.
        :return: The coregistration method or pipeline with updated transformation parameters.
        """
        coreg_method = coreg_method.copy()
        tba_dem_tiled = tba_dem.crop(ref_dem_tiled)

        _, ref_mask = get_array_and_mask(ref_dem_tiled)
        _, sec_mask = get_array_and_mask(tba_dem_tiled)

        if np.all(ref_mask) or np.all(sec_mask):
            coreg_method.meta["outputs"] = {"affine": {"shift_x": np.nan}}
            coreg_method.meta["outputs"] = {"affine": {"shift_y": np.nan}}
            coreg_method.meta["outputs"] = {"affine": {"shift_z": np.nan}}
            return coreg_method
        else:
            if inlier_mask:
                inlier_mask = inlier_mask.crop(ref_dem_tiled)
            try:
                return coreg_method.fit(ref_dem_tiled, tba_dem_tiled, inlier_mask)
            except (ValueError, TypeError) as e:
                logging.error(f"Failed to fit coregistration. Reason: {e}")
                coreg_method.meta["outputs"] = {"affine": {"shift_x": np.nan}}
                coreg_method.meta["outputs"] = {"affine": {"shift_y": np.nan}}
                coreg_method.meta["outputs"] = {"affine": {"shift_z": np.nan}}
                return coreg_method

    def fit(
        self: BlockwiseCoreg,
        reference_elev: NDArrayf | MArrayf | RasterType,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType,
        inlier_mask: NDArrayb | Mask | None = None,
    ) -> None:
        """
        Fit the coregistration model by estimating transformation parameters
        between the reference and target elevation data.

        :param reference_elev: Reference elevation data to align to.
        :param to_be_aligned_elev: Elevation data to be aligned (transformed).
        :param inlier_mask: Optional boolean mask indicating valid data points to use in the fitting.
        :return: None. Updates internal model parameters.
        """

        self.mp_config.outfile = self.output_path_reproject

        self.reproject_dem = to_be_aligned_elev.reproject(  # type: ignore
            ref=reference_elev, multiproc_config=self.mp_config, silent=True
        )

        logging.info(f"No reprojected DEM returned, but saved at {self.output_path_reproject}")

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

        self.shape_tiling_grid = compute_tiling(
            self.block_size_fit, reference_elev.shape, to_be_aligned_elev.shape
        ).shape

        rows_cols = list(itertools.product(range(self.shape_tiling_grid[0]), range(self.shape_tiling_grid[1])))

        self.x_coords = []  # type: ignore
        self.y_coords = []  # type: ignore
        self.shifts_x = []  # type: ignore
        self.shifts_y = []  # type: ignore
        self.shifts_z = []  # type: ignore

        for idx, (coreg, tile_coords) in enumerate(outputs_coreg):
            try:
                shift_x = coreg.meta["outputs"]["affine"]["shift_x"]
                shift_y = coreg.meta["outputs"]["affine"]["shift_y"]
                shift_z = coreg.meta["outputs"]["affine"]["shift_z"]

            except KeyError:
                continue

            x, y = (
                tile_coords[2] + self.block_size_fit / 2,
                tile_coords[0] + self.block_size_fit / 2,
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
        max_iterations: int = 2000,
    ) -> tuple[float, float, float]:
        """
        Estimate a geometric transformation using the RANSAC algorithm.
        warning : it can fail

        :param x_coords: 1D array of x coordinates.
        :param y_coords: 1D array of y coordinates.
        :param shifts: 1D array of observed shifts (errors) at the corresponding (x, y) positions.
        :param threshold: Maximum allowed deviation to consider a point as an inlier.
        :param max_iterations: Maximum number of iterations to run the RANSAC algorithm.
        :return: Estimated transformation coefficients (a, b, c) such as shift = a * x + b * y + c.
        """
        if not _has_sklearn:
            raise ValueError("Optional dependency needed. Install 'scikit-learn'.")

        # Stack and squeeze
        points = np.dstack([x_coords, y_coords, shifts])
        points = np.squeeze(points)

        # If only one point, squeeze() gives (3,) -> reshape to (1, 3)
        if points.ndim == 1:
            if points.shape[0] != 3:
                raise ValueError(f"Unexpected point shape: {points.shape}")
            points = points.reshape(1, 3)

        # Remove NaNs safely
        points = points[~np.isnan(points).any(axis=1)]

        if points.size == 0:
            raise ValueError("No valid points after removing NaNs.")

        # 1D: Variation on only 1 dimension
        if np.allclose(points[:, 1], points[0, 1]):
            # 1D variation on x
            a, c = np.polyfit(points[:, 0], points[:, 2], 1)
            b = 0.0
        elif np.allclose(points[:, 0], points[0, 0]):
            # 1D variation on y
            b, c = np.polyfit(points[:, 1], points[:, 2], 1)
            a = 0.0
        else:
            # 2D: fit RANSAC
            X = points[:, :2]
            y = points[:, 2]
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=threshold,
                max_trials=max_iterations,
            )
            ransac.fit(X, y)
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_

        return a, b, c

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
        max_iterations_ransac: int = 2000,
    ) -> RasterType:
        """
        Apply the coregistration transformation to an elevation array using a ransac filter.

        :param threshold_ransac: Maximum distance threshold to consider a point as an inlier.
        :param max_iterations_ransac: Maximum number of RANSAC iterations to perform.
        :return: The transformed elevation raster.
        """

        coeff_x = self._ransac(
            self.x_coords,  # type: ignore
            self.y_coords,  # type: ignore
            self.shifts_x,  # type: ignore
            threshold_ransac,
            max_iterations_ransac,
        )
        coeff_y = self._ransac(
            self.x_coords,  # type: ignore
            self.y_coords,  # type: ignore
            self.shifts_y,  # type: ignore
            threshold_ransac,
            max_iterations_ransac,
        )
        if self.apply_z_correction:
            coeff_z = self._ransac(
                self.x_coords,  # type: ignore
                self.y_coords,  # type: ignore
                self.shifts_z,  # type: ignore
                threshold_ransac,
                max_iterations_ransac,
            )
        else:
            coeff_z = (0, 0, 0)

        self.mp_config.outfile = self.output_path_aligned
        self.mp_config.chunk_size = self.block_size_apply

        # be careful with depth value if Out of Memory
        depth = max(np.abs(self.shifts_x).max(), np.abs(self.shifts_y).max())

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
