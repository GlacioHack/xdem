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

# /!\ code based on CARS, cf licence resample_dem function (Apache V2.0)

""""""
import copy
import itertools
import math
from typing import Any, List, Literal, Tuple

import numpy as np
import pyransac3d as pyrsc
import rasterio as rio
import resample as cresample
from geoutils.raster import Mask, RasterType
# from geoutils.raster.distributed_computing import (
#     MultiprocConfig,
#     load_raster_tile,
#     map_overlap_multiproc,
# )
from geoutils.raster.tiling import compute_tiling
from rasterio.windows import Window

import xdem
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg.base import Coreg, CoregType


class ScaledBlockwiseCoreg(Coreg):
    """
    ScaledBlockwiseCoreg co-registration processing class to run a step in segmented parts of the grid with
    Nuth and Kaab by saving on disk the tiles.
    """

    def __init__(
        self,
        tile_size: int,
        reference_dem: RasterType,
        to_be_aligned_dem: RasterType,
        output_directory: str,
    ):
        """
        Instantiate a scaled-blockwise processing object.

        :param tile_size: pixel size dimension of tiles
        :param reference_dem: Reference DEM object
        :param to_be_aligned_dem: to_be_aligned_dem DEM object
        :param output_directory: path for saving results
        """
        # self.tile_size = tile_size
        self.reference_dem = reference_dem
        self.to_be_aligned_dem = to_be_aligned_dem
        self.output_directory = output_directory
        super().__init__()

        # Define config for multiprocessing:
        self.config_multiproc = MultiprocConfig(chunk_size=tile_size, outfile=output_directory)

        self.tiling_grid = compute_tiling(tile_size, self.reference_dem.shape, self.to_be_aligned_dem.shape)
        self.res_coreg = None
        # TODO : save coreg results as meta
        self.stats_diff_before = {}
        self.stats_diff_after = {}

        # TODO : preprocess inputs
        # reprojection

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

        resultats = map_overlap_multiproc(
            self.coreg_wrapper,
            reference_elev,
            self.config_multiproc,
            self.to_be_aligned_dem,
            return_tiles=True,
        )
        rows_cols = list(itertools.product(range(self.tiling_grid.shape[0]), range(self.tiling_grid.shape[1])))

        resolution = reference_elev.res
        shape_correg = (self.tiling_grid.shape[0], self.tiling_grid.shape[1], 3)
        shape_positions = (self.tiling_grid.shape[0], self.tiling_grid.shape[1], 2)
        res_correg = np.empty(shape_correg)
        res_positions = np.empty(shape_positions)

        for idx, res in enumerate(resultats):
            coreg_res, tile = res
            center_position = (tile[0] + tile[1]) / 2, (tile[2] + tile[3]) / 2
            row, col = rows_cols[idx]
            res_correg[row, col, :] = np.array(
                [coreg_res[0] / resolution[0], coreg_res[1] / resolution[1], coreg_res[2]]
            )
            res_positions[row, col, :] = center_position

        self.res_coreg = {"positions": res_positions, "coreg": res_correg}

        np.savetxt(self.output_directory + "/res_coreg.csv",
                   res_correg.reshape(-1, 3),
                   delimiter=",",
                   header="X,Y,Z")
        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @staticmethod
    def coreg_wrapper(ref_tile, sec_raster):
        sec_tile = sec_raster.crop(ref_tile)
        try:
            nuth_kaab = xdem.coreg.NuthKaab()
            nuth_kaab.fit(ref_tile, sec_tile)
            shift = nuth_kaab.meta["outputs"]["affine"]
            x_y_z = [shift["shift_x"], shift["shift_y"], shift["shift_z"]]
        except ValueError:
            x_y_z = [np.nan, np.nan, np.nan]

        return x_y_z

    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]:
        ...

        correction_grid = self.generate_correction_grid(self.res_coreg, 30, self.to_be_aligned_dem.shape)

        meta = self.reference_dem.profile
        meta.update(dtype=rio.float32, count=1, compress="lzw")

        with rio.open(self.output_directory + "/aligned_dem.tif", "w", **meta) as dst:
            for row in range(self.tiling_grid.shape[0]):
                for col in range(self.tiling_grid.shape[1]):

                    tile = self.tiling_grid[row, col, :]
                    row_min, row_max, col_min, col_max = tile
                    row_min, row_max, col_min, col_max = (
                        int(row_min),
                        int(row_max),
                        int(col_min),
                        int(col_max),
                    )

                    dem_sec_aligned = load_raster_tile(self.to_be_aligned_dem, tile)

                    arr_sec = self.resample_dem(
                        int(tile[0]),
                        int(tile[1]),
                        int(tile[2]),
                        int(tile[3]),
                        correction_grid,
                        self.to_be_aligned_dem.filename,
                    )
                    dem_sec_aligned.data = arr_sec

                    window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
                    dst.write(arr_sec, 1, window=window)
        return 0

    @staticmethod
    def filter_ransac(coreg, positions, thresh=0.05, minPoints=10, maxIteration=100) -> Tuple[np.ndarray, List[float]]:
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
                 - new_arr: The adjusted coregistration array based on the fitted plane
                 - [A, B, C]: The coefficients of the fitted plane equation Ax + By + C = z
        """
        in_shape = coreg.shape
        rows, cols = positions[:, :, 0], positions[:, :, 1]
        rows, cols, arr = rows.flatten(), cols.flatten(), coreg.flatten()

        points = np.squeeze(np.dstack([rows, cols, arr]))
        points = points[~np.isnan(points).any(axis=1), :]

        plane = pyrsc.Plane()
        best_eq, best_inliers = plane.fit(points, thresh=thresh, minPoints=minPoints, maxIteration=maxIteration)
        A = -best_eq[0] / best_eq[2]
        B = -best_eq[1] / best_eq[2]
        C = -best_eq[3] / best_eq[2]

        new_arr = rows * A + cols * B + C

        new_arr = np.reshape(new_arr, in_shape)
        return new_arr, [A, B, C]

    @staticmethod
    def generate_grids_from_ransac(raster_shape, ransac_coefs, step=30):
        """
        Generate position grids based on RANSAC coefficients.

        This function creates a grid of values using the RANSAC coefficients, which represent
        the parameters of a fitted plane. The grid is generated over the specified raster shape
        with a given step size.

        :param raster_shape: A tuple representing the shape of the raster (rows, columns)
        :param ransac_coefs: A list of coefficients [A, B, C] representing the plane equation Ax + By + C = z
        :param step: The step size for generating the grid, defaults to 30

        :return: A tuple containing:
                 - grid: The generated grid of values based on the RANSAC coefficients
                 - row_gridvalues: The grid values for the row indices
                 - col_gridvalues: The grid values for the column indices
        """

        row_min = 0
        row_max = raster_shape[0]
        col_min = 0
        col_max = raster_shape[1]

        row_range = np.arange(start=row_min, stop=row_max + step, step=step)
        col_range = np.arange(start=col_min, stop=col_max + step, step=step)

        (
            col_gridvalues,
            row_gridvalues,
        ) = np.meshgrid(
            col_range,
            row_range,
        )

        grid = ransac_coefs[0] * row_gridvalues + ransac_coefs[1] * col_gridvalues + ransac_coefs[2]

        return grid, row_gridvalues, col_gridvalues

    def generate_correction_grid(self, correg, step, dem_shape):
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

        x_2d_ransac, coef_ransac_x = self.filter_ransac(
            correg_shift[:, :, 0],
            positions,
            thresh=thresh_x,
            minPoints=nb_points_min,
            maxIteration=max_iter,
        )
        y_2d_ransac, coef_ransac_y = self.filter_ransac(
            correg_shift[:, :, 1],
            positions,
            thresh=thresh_y,
            minPoints=nb_points_min,
            maxIteration=max_iter,
        )

        row_grid, row_gridvalues, col_gridvalues = self.generate_grids_from_ransac(dem_shape, coef_ransac_x, step=step)
        col_grid, _, _ = self.generate_grids_from_ransac(dem_shape, coef_ransac_y, step=step)

        row_grid += row_gridvalues
        col_grid += col_gridvalues

        position_grid = {"grid": np.stack([row_grid, col_grid], axis=0), "step": step}

        return position_grid

    def resample_dem(self, row_min, row_max, col_min, col_max, position_grid, dem_path) -> np.ndarray:
        """
        Resample a DEM within a specified region using a position grid.

        This method reads a DEM file and resamples it within the specified row and column bounds,
        using a position grid to determine the oversampling factor. The resampling is performed
        using bicubic interpolation.

        :param row_min: Minimum row index of the region to resample
        :param row_max: Maximum row index of the region to resample
        :param col_min: Minimum column index of the region to resample
        :param col_max: Maximum column index of the region to resample
        :param position_grid: A dictionary containing the position grid and oversampling step
        :param dem_path: The file path to the DEM to be resampled

        :return: A resampled block of the DEM as a NumPy array
        """
        with rio.open(dem_path) as img_reader:
            block_region = [row_min, col_min, row_max, col_max]

            oversampling = position_grid["step"]
            grid = position_grid["grid"]

            # Convert resampled region to grid region with oversampling
            grid_region = [
                math.floor(block_region[0] / oversampling),
                math.floor(block_region[1] / oversampling),
                math.ceil(block_region[2] / oversampling),
                math.ceil(block_region[3] / oversampling),
            ]
            grid_region[0::2] = list(np.clip(grid_region[0::2], 0, grid.shape[1]))
            grid_region[1::2] = list(np.clip(grid_region[1::2], 0, grid.shape[2]))

            grid_as_array = copy.copy(
                grid[
                    :,
                    grid_region[0] : grid_region[2] + 1,
                    grid_region[1] : grid_region[3] + 1,
                ]
            )

            # get needed source bounding box
            left = math.floor(np.amin(grid_as_array[1, ...]))
            right = math.ceil(np.amax(grid_as_array[1, ...]))
            top = math.floor(np.amin(grid_as_array[0, ...]))
            bottom = math.ceil(np.amax(grid_as_array[0, ...]))

            # filter margin for bicubic = 2
            filter_margin = 2
            top -= filter_margin
            bottom += filter_margin
            left -= filter_margin
            right += filter_margin

            left, right = list(np.clip([left, right], 0, self.to_be_aligned_dem.shape[1]))
            top, bottom = list(np.clip([top, bottom], 0, self.to_be_aligned_dem.shape[0]))

            img_window = Window.from_slices([top, bottom], [left, right])

            # round window
            img_window = img_window.round_offsets()
            img_window = img_window.round_lengths()

            # Compute offset
            col_offset = min(left, right)
            row_offset = min(top, bottom)

            # Get dem data
            img_as_array = img_reader.read(window=img_window)
            img_as_array = np.swapaxes(img_as_array, 1, 2)

            # shift grid regarding the img extraction
            grid_as_array[1, ...] -= col_offset
            grid_as_array[0, ...] -= row_offset

            block_resamp = cresample.grid(
                img_as_array,
                grid_as_array,
                oversampling,
                interpolator="bicubic",
                nodata=0,
            ).astype(np.float32)

            # extract exact region

            out_region = oversampling * np.array(grid_region)
            ext_region = block_region - out_region

            block_resamp = block_resamp[
                0,
                ext_region[0] : ext_region[2] - 1,
                ext_region[1] : ext_region[3] - 1,
            ]

        return block_resamp
