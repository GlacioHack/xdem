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
import pyransac3d as pyrsc
import concurrent.futures
import inspect
import logging
import warnings
from typing import Any, Literal
from xdem.coreg import NuthKaab
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import skimage
from geoutils.raster import Mask, RasterType, subdivide_array
from geoutils.raster.array import get_array_and_mask
from geoutils.raster.georeferencing import _bounds, _res
from geoutils.raster.tiling import compute_tiling
from tqdm import tqdm
from geoutils.raster.distributed_computing import (
    MultiprocConfig,
    map_multiproc_collect,
    map_overlap_multiproc_save,
)
from geoutils.raster.geotransformations import (
    _get_target_georeferenced_grid,
    _user_input_reproject,
)
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregDict,
    CoregPipeline,
    CoregType,
    _preprocess_coreg_fit,
    _apply_matrix_pts,
)
from geoutils.interface.gridding import _grid_pointcloud
import itertools
import geoutils as gu
#from geoutils.raster import MultiprocConfig, ClusterGenerator
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
        output_path: str = None,
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

        super().__init__()

    @staticmethod
    def coreg_wrapper(ref_tile, sec_raster, inlier_mask=None):
        # TODO : commentaire explicatif ici
        sec_tile = sec_raster.crop(ref_tile)
        try:
            nuth_kaab = NuthKaab()
            nuth_kaab.fit(ref_tile, sec_tile, inlier_mask=inlier_mask)
            shift = nuth_kaab.meta["outputs"]["affine"]
            x_y_z = [shift["shift_x"], shift["shift_y"], shift["shift_z"]]
        except ValueError:
            x_y_z = [np.nan, np.nan, np.nan]

        return x_y_z

    def preprocess(self, ref, sec):
        """

        """

        mp_config = MultiprocConfig(chunk_size=self.tile_size, outfile=self.output_path + "/SEC_reprojected.tif")
        sec.reproject(
            ref=ref,
            resampling="cubic",
            multiproc_config=mp_config)

        return self.output_path + "/SEC_reprojected.tif"


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

        # # Define inlier mask if None, before indexing subdivided array in process function below
        # if inlier_mask is None:
        #     mask = np.ones(tba_dem.shape, dtype=bool)
        # else:
        #     mask = inlier_mask

     #   path_reprojected = self.preprocess(reference_elev, to_be_aligned_elev)
        path_reprojected = "/home/adebardo/development/Qualif/outputs_test/1990_reproject_origin.tif"

        self.dem_to_be_aligned = xdem.DEM(path_reprojected)

        tiling_grid = compute_tiling(self.tile_size, reference_elev.shape, to_be_aligned_elev.shape)
        shape_tiling_grid = tiling_grid.shape

        config_multiproc = MultiprocConfig(chunk_size=self.tile_size)
        res_multiproc = map_multiproc_collect(
            self.coreg_wrapper, reference_elev.filename, config_multiproc, to_be_aligned_elev, return_tile=True
        )

        self.res_coreg = [element[0] for element in res_multiproc]
        tiles_coords = [element[1] for element in res_multiproc]

        rows_cols = list(itertools.product(range(shape_tiling_grid[0]), range(shape_tiling_grid[1])))
        resolution = reference_elev.res
        res_correg = np.empty((shape_tiling_grid[0], shape_tiling_grid[1], 3))
        res_positions = np.empty((shape_tiling_grid[0], shape_tiling_grid[1], 2))

        self.new_meta = {}

        for idx, (coreg_res, tile, (row, col)) in enumerate(zip(self.res_coreg, tiles_coords, rows_cols)):
            center_position = ((tile[0] + tile[1]) / 2, (tile[2] + tile[3]) / 2)
            res_correg[row, col] = np.array([coreg_res[0] / resolution[0], coreg_res[1] / resolution[1], coreg_res[2]])
            res_correg[row, col] = np.array([coreg_res[0], coreg_res[1], coreg_res[2]])
            res_positions[row, col] = center_position
            self.new_meta[str((row, col))] = {"shift_x": coreg_res[0], "shift_y": coreg_res[1], "shift_z": coreg_res[2]}

        self.res_coreg_for_apply = {"positions": res_positions, "coreg": res_correg}

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    def apply_ransac_shift(self, dem_array, transform, positions):
        """
        Applique un décalage basé sur les résultats RANSAC à chaque pixel du MNT.
        """
        # Appliquer le modèle RANSAC pour les décalages en X et Y
        thresh_x = (np.nanpercentile(dem_array[:, :, 0], 90) - np.nanpercentile(dem_array[:, :, 0], 10)) / 3
        thresh_y = (np.nanpercentile(dem_array[:, :, 1], 90) - np.nanpercentile(dem_array[:, :, 1], 10)) / 3

        x_2d_ransac, coef_ransac_x = self.filter_ransac(dem_array[:, :, 0], positions, thresh=thresh_x)
        y_2d_ransac, coef_ransac_y = self.filter_ransac(dem_array[:, :, 1], positions, thresh=thresh_y)

        # Générer les nouvelles grilles de décalage
        row_grid, row_gridvalues, col_gridvalues = self.generate_grids_from_ransac(dem_array.shape, coef_ransac_x)
        col_grid, _, _ = self.generate_grids_from_ransac(dem_array.shape, coef_ransac_y)

        row_grid += row_gridvalues
        col_grid += col_gridvalues

        return row_grid, col_grid

    def resample_wrapper(self, tile_dem, coeff_x_grid, coeff_y_grid, dst_raster, resampling="linear"):
        """ """
        dst_raster = dst_raster.crop(tile_dem)
     #   Converts the array to point cloud, removing the NaNs
        epc = tile_dem.to_pointcloud(data_column_name="z", skip_nodata=True).ds
        # Get shifts for each point
        x = epc.geometry.x.values
        y = epc.geometry.y.values
        A1, B1, C1 = coeff_x_grid
        A2, B2, C2 = coeff_y_grid
        x_shifts = A1 * x + B1**y + C1
        y_shifts = A2 * x + B2**y + C2

        # Transform X/Y with X/Y shifts in-place
        epc.geometry = gpd.points_from_xy(x=x + x_shifts, y=y + y_shifts)
        # Grid point cloud back to a DEM
        new_tile_dem = _grid_pointcloud(
            epc, grid_coords=dst_raster.coords(grid=False), data_column_name="z", resampling=resampling
        )

        rast = gu.Raster.from_array(
            data=new_tile_dem, transform=dst_raster.transform, crs=dst_raster.crs, nodata=dst_raster.nodata
        )





       # return rast

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # coeff_x_grid, coeff_y_grid = self.generate_correction_grid(
        #     self.res_coreg_for_apply, 30, self.dem_to_be_aligned.shape
        # )

        correction_grid = self.generate_correction_grid(self.res_coreg, 30, self.to_be_aligned_dem.shape)

        # # Retrieve transform and grid_size
        # transform, grid_size = _get_target_georeferenced_grid(
        #     self.dem_to_be_aligned, crs=crs, res=self.dem_to_be_aligned.res
        # )
        # width, height = grid_size

        dst_raster = gu.Raster(self.output_path + "aligned_DEM.tif")

        # # Open file on disk to write tile by tile
        # with rio.open(
        #     self.output_path + "aligned_DEM.tif",
        #     "w",
        #     driver="GTiff",
        #     height=height,
        #     width=width,
        #     count=self.dem_to_be_aligned.count,
        #     dtype=self.dem_to_be_aligned.dtype,
        #     crs=crs,
        #     transform=transform,
        #     nodata=self.dem_to_be_aligned.nodata,
        # ):
        #     pass

        config_multiproc = MultiprocConfig(chunk_size=self.tile_size, outfile=self.output_path + "aligned_DEM.tif")
        map_overlap_multiproc_save(
            self.resample_wrapper,
            self.dem_to_be_aligned.filename,
            config_multiproc,
            coeff_x_grid,
            coeff_y_grid,
            dst_raster,
        )

        return 0, 0



        # if np.count_nonzero(np.isfinite(elev)) == 0:
        #     return elev, transform
        #
        # # Other option than resample=True is not implemented for this case
        # if "resample" in kwargs and kwargs["resample"] is not True:
        #     raise NotImplementedError("Option `resample=False` not supported for coreg method BlockwiseCoreg.")
        #
        # points = self.to_points()
        # # Check for NaN values across both the old and new positions for each point
        # mask = ~np.isnan(points).any(axis=(1, 2))
        #
        # # Filter out points where there are no NaN values
        # points = points[mask]
        #
        # bounds = _bounds(transform=transform, shape=elev.shape)
        # resolution = _res(transform)
        #
        # representative_height = np.nanmean(elev)
        # edges_source_arr = np.array(
        #     [
        #         [bounds.left + resolution[0] / 2, bounds.top - resolution[1] / 2, representative_height],
        #         [bounds.right - resolution[0] / 2, bounds.top - resolution[1] / 2, representative_height],
        #         [bounds.left + resolution[0] / 2, bounds.bottom + resolution[1] / 2, representative_height],
        #         [bounds.right - resolution[0] / 2, bounds.bottom + resolution[1] / 2, representative_height],
        #     ]
        # )
        # edges_source = gpd.GeoDataFrame(
        #     geometry=gpd.points_from_xy(x=edges_source_arr[:, 0], y=edges_source_arr[:, 1], crs=None),
        #     data={"z": edges_source_arr[:, 2]},
        # )
        #
        # edges_dest = self.apply(edges_source)
        # edges_dest_arr = np.array(
        #     [edges_dest.geometry.x.values, edges_dest.geometry.y.values, edges_dest["z"].values]
        # ).T
        # edges = np.dstack((edges_source_arr, edges_dest_arr))
        #
        # all_points = np.append(points, edges, axis=0)
        #
        # warped_dem = warp_dem(
        #     dem=elev,
        #     transform=transform,
        #     source_coords=all_points[:, :, 1],
        #     destination_coords=all_points[:, :, 0],
        #     resampling="linear",
        #     apply_z_correction=self.apply_z_correction,
        # )
        #
        # return warped_dem, transform

    #     def _apply_pts(
    #         self, elev: gpd.GeoDataFrame, z_name: str = "z", bias_vars: dict[str, NDArrayf] | None = None, **kwargs: Any
    #     ) -> gpd.GeoDataFrame:
    #         """Apply the scaling model to a set of points."""
    #         points = self.to_points()
    #
    #         # Check for NaN values across both the old and new positions for each point
    #         mask = ~np.isnan(points).any(axis=(1, 2))
    #
    #         # Filter out points where there are no NaN values
    #         points = points[mask]
    #
    #         new_coords = np.array([elev.geometry.x.values, elev.geometry.y.values, elev["z"].values]).T
    #
    #         for dim in range(0, 3):
    #             with warnings.catch_warnings():
    #                 # ZeroDivisionErrors may happen when the transformation is empty (which is fine)
    #                 warnings.filterwarnings("ignore", message="ZeroDivisionError")
    #                 model = scipy.interpolate.Rbf(
    #                     points[:, 0, 0],
    #                     points[:, 1, 0],
    #                     points[:, dim, 1] - points[:, dim, 0],
    #                     function="linear",
    #                 )
    #
    #             new_coords[:, dim] += model(elev.geometry.x.values, elev.geometry.y.values)
    #
    #         gdf_new_coords = gpd.GeoDataFrame(
    #             geometry=gpd.points_from_xy(x=new_coords[:, 0], y=new_coords[:, 1], crs=None), data={"z": new_coords[:, 2]}
    #         )
    #
    #         return gdf_new_coords
    #
    #
    # def warp_dem(
    #     dem: NDArrayf,
    #     transform: rio.transform.Affine,
    #     source_coords: NDArrayf,
    #     destination_coords: NDArrayf,
    #     resampling: str = "cubic",
    #     trim_border: bool = True,
    #     dilate_mask: bool = True,
    #     apply_z_correction: bool = True,
    # ) -> NDArrayf:
    #     """
    #     (22/08/24: Method currently used only for blockwise coregistration)
    #     Warp a DEM using a set of source-destination 2D or 3D coordinates.
    #
    #     :param dem: The DEM to warp. Allowed shapes are (1, row, col) or (row, col)
    #     :param transform: The Affine transform of the DEM.
    #     :param source_coords: The source 2D or 3D points. must be X/Y/(Z) coords of shape (N, 2) or (N, 3).
    #     :param destination_coords: The destination 2D or 3D points. Must have the exact same shape as 'source_coords'
    #     :param resampling: The resampling order to use. Choices: ['nearest', 'linear', 'cubic'].
    #     :param trim_border: Remove values outside of the interpolation regime (True) or leave them unmodified (False).
    #     :param dilate_mask: Dilate the nan mask to exclude edge pixels that could be wrong.
    #     :param apply_z_correction: Boolean to toggle whether the Z-offset correction is applied or not (default True).
    #
    #     :raises ValueError: If the inputs are poorly formatted.
    #     :raises AssertionError: For unexpected outputs.
    #
    #     :returns: A warped DEM with the same shape as the input.
    #     """
    #     if source_coords.shape != destination_coords.shape:
    #         raise ValueError(
    #             f"Incompatible shapes: source_coords '({source_coords.shape})' and "
    #             f"destination_coords '({destination_coords.shape})' shapes must be the same"
    #         )
    #     if (len(source_coords.shape) > 2) or (source_coords.shape[1] < 2) or (source_coords.shape[1] > 3):
    #         raise ValueError(
    #             "Invalid coordinate shape. Expected 2D or 3D coordinates of shape (N, 2) or (N, 3). "
    #             f"Got '{source_coords.shape}'"
    #         )
    #     allowed_resampling_strs = ["nearest", "linear", "cubic"]
    #     if resampling not in allowed_resampling_strs:
    #         raise ValueError(f"Resampling type '{resampling}' not understood. Choices: {allowed_resampling_strs}")
    #
    #     dem_arr, dem_mask = get_array_and_mask(dem)
    #
    #     bounds = _bounds(transform=transform, shape=dem_arr.shape)
    #
    #     no_horizontal = np.sum(np.linalg.norm(destination_coords[:, :2] - source_coords[:, :2], axis=1)) < 1e-6
    #     no_vertical = source_coords.shape[1] > 2 and np.sum(np.abs(destination_coords[:, 2] - source_coords[:, 2])) < 1e-6
    #
    #     if no_horizontal and no_vertical:
    #         warnings.warn("No difference between source and destination coordinates. Returning self.")
    #         return dem
    #
    #     source_coords_scaled = source_coords.copy()
    #     destination_coords_scaled = destination_coords.copy()
    #     # Scale the coordinates to index-space
    #     for coords in (source_coords_scaled, destination_coords_scaled):
    #         coords[:, 0] = dem_arr.shape[1] * (coords[:, 0] - bounds.left) / (bounds.right - bounds.left)
    #         coords[:, 1] = dem_arr.shape[0] * (1 - (coords[:, 1] - bounds.bottom) / (bounds.top - bounds.bottom))
    #
    #     # Generate a grid of x and y index coordinates.
    #     grid_y, grid_x = np.mgrid[0 : dem_arr.shape[0], 0 : dem_arr.shape[1]]
    #
    #     if no_horizontal:
    #         warped = dem_arr.copy()
    #     else:
    #         # Interpolate the sparse source-destination points to a grid.
    #         # (row, col, 0) represents the destination y-coordinates of the pixels.
    #         # (row, col, 1) represents the destination x-coordinates of the pixels.
    #         new_indices = scipy.interpolate.griddata(
    #             source_coords_scaled[:, [1, 0]],
    #             destination_coords_scaled[:, [1, 0]],  # Coordinates should be in y/x (not x/y) for some reason..
    #             (grid_y, grid_x),
    #             method="linear",
    #         )
    #
    #         # If the border should not be trimmed, just assign the original indices to the missing values.
    #         if not trim_border:
    #             missing_ys = np.isnan(new_indices[:, :, 0])
    #             missing_xs = np.isnan(new_indices[:, :, 1])
    #             new_indices[:, :, 0][missing_ys] = grid_y[missing_ys]
    #             new_indices[:, :, 1][missing_xs] = grid_x[missing_xs]
    #
    #         order = {"nearest": 0, "linear": 1, "cubic": 3}
    #
    #         with warnings.catch_warnings():
    #             # A skimage warning that will hopefully be fixed soon. (2021-06-08)
    #             warnings.filterwarnings("ignore", message="Passing `np.nan` to mean no clipping in np.clip")
    #             warped = skimage.transform.warp(
    #                 image=np.where(dem_mask, np.nan, dem_arr),
    #                 inverse_map=np.moveaxis(new_indices, 2, 0),
    #                 output_shape=dem_arr.shape,
    #                 preserve_range=True,
    #                 order=order[resampling],
    #                 cval=np.nan,
    #             )
    #             new_mask = (
    #                 skimage.transform.warp(
    #                     image=dem_mask, inverse_map=np.moveaxis(new_indices, 2, 0), output_shape=dem_arr.shape, cval=False
    #                 )
    #                 > 0
    #             )
    #
    #         if dilate_mask:
    #             new_mask = scipy.ndimage.binary_dilation(new_mask, iterations=order[resampling]).astype(new_mask.dtype)
    #
    #         warped[new_mask] = np.nan
    #
    #     # Apply the Z-correction if apply_z_correction is True and if the coordinates are 3D (N, 3)
    #     if not no_vertical and apply_z_correction:
    #         grid_offsets = scipy.interpolate.griddata(
    #             points=destination_coords_scaled[:, :2],
    #             values=source_coords_scaled[:, 2] - destination_coords_scaled[:, 2],
    #             xi=(grid_x, grid_y),
    #             method=resampling,
    #             fill_value=np.nan,
    #         )
    #         if not trim_border:
    #             grid_offsets[np.isnan(grid_offsets)] = np.nanmean(grid_offsets)
    #
    #         warped += grid_offsets
    #
    #     assert not np.all(np.isnan(warped)), "All-NaN output."
    #
    #     return warped.reshape(dem.shape)

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
