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

import concurrent.futures
import inspect
import logging
import warnings
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import skimage
from geoutils.raster import RasterMask, RasterType, subdivide_array
from geoutils.raster.array import get_array_and_mask
from geoutils.raster.georeferencing import _bounds, _res
from tqdm import tqdm

from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregDict,
    CoregPipeline,
    CoregType,
    _preprocess_coreg_fit,
)


class BlockwiseCoreg(Coreg):
    """
    A processing class of choice is run on an arbitrary subdivision of the raster. When later applying the step
    the optimal warping is interpolated based on X/Y/Z shifts from the coreg algorithm at the grid points.

    For instance: a subdivision of 4 triggers a division of the DEM in four equally sized parts. These parts are then
    processed separately, with 4 .fit() results. If the subdivision is not divisible by the raster shape,
    subdivision is made as good as possible to have approximately equal pixel counts.
    """

    def __init__(
        self,
        step: Coreg | CoregPipeline,
        subdivision: int,
        success_threshold: float = 0.8,
        n_threads: int | None = None,
        warn_failures: bool = False,
        apply_z_correction: bool = True,
    ) -> None:
        """
        Instantiate a blockwise processing object.

        :param step: An instantiated co-registration step object to fit in the subdivided DEMs.
        :param subdivision: The number of chunks to divide the DEMs in. E.g. 4 means four different transforms.
        :param success_threshold: Raise an error if fewer chunks than the fraction failed for any reason.
        :param n_threads: The maximum amount of threads to use. Default=auto
        :param warn_failures: Trigger or ignore warnings for each exception/warning in each block.
        :param apply_z_correction: Boolean to toggle whether the Z-offset correction is applied or not (default True).
        """
        if isinstance(step, type):
            raise ValueError(
                "The 'step' argument must be an instantiated Coreg subclass. " "Hint: write e.g. ICP() instead of ICP"
            )
        self.procstep = step
        self.subdivision = subdivision
        self.success_threshold = success_threshold
        self.n_threads = n_threads
        self.warn_failures = warn_failures
        self.apply_z_correction = apply_z_correction

        super().__init__()

        self._meta: CoregDict = {"step_meta": []}
        self._groups: NDArrayf = np.array([])

    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType,
        inlier_mask: NDArrayb | RasterMask | None = None,
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

        # Check if subsample arguments are different from their default value for any of the coreg steps:
        # get default value in argument spec and "subsample" stored in meta, and compare both are consistent
        if not isinstance(self.procstep, CoregPipeline):
            steps = [self.procstep]
        else:
            steps = list(self.procstep.pipeline)
        argspec = [inspect.getfullargspec(s.__class__) for s in steps]
        sub_meta = [s._meta["inputs"]["random"]["subsample"] for s in steps]
        sub_is_default = [
            argspec[i].defaults[argspec[i].args.index("subsample") - 1] == sub_meta[i]  # type: ignore
            for i in range(len(argspec))
        ]
        if subsample is not None and not all(sub_is_default):
            warnings.warn(
                "Subsample argument passed to fit() will override non-default subsample values defined in the"
                " step within the blockwise method. To silence this warning: only define 'subsample' in "
                "either fit(subsample=...) or instantiation e.g., VerticalShift(subsample=...)."
            )

        # Pre-process the inputs, by reprojecting and subsampling, without any subsampling (done in each step)
        ref_dem, tba_dem, inlier_mask, transform, crs, area_or_point = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

        # Define inlier mask if None, before indexing subdivided array in process function below
        if inlier_mask is None:
            mask = np.ones(tba_dem.shape, dtype=bool)
        else:
            mask = inlier_mask

        self._groups = self.subdivide_array(tba_dem.shape if isinstance(tba_dem, np.ndarray) else ref_dem.shape)

        indices = np.unique(self._groups)

        progress_bar = tqdm(
            total=indices.size, desc="Processing chunks", disable=logging.getLogger().getEffectiveLevel() > logging.INFO
        )

        def process(i: int) -> dict[str, Any] | BaseException | None:
            """
            Process a chunk in a thread-safe way.

            :returns:
                * If it succeeds: A dictionary of the fitting metadata.
                * If it fails: The associated exception.
                * If the block is empty: None
            """
            group_mask = self._groups == i

            # Find the corresponding slice of the inlier_mask to subset the data
            rows, cols = np.where(group_mask)
            arrayslice = np.s_[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

            # Copy a subset of the two DEMs, the mask, the coreg instance, and make a new subset transform
            ref_subset = ref_dem[arrayslice].copy()
            tba_subset = tba_dem[arrayslice].copy()

            if any(np.all(~np.isfinite(dem)) for dem in (ref_subset, tba_subset)):
                return None
            mask_subset = mask[arrayslice].copy()
            west, top = rio.transform.xy(transform, min(rows), min(cols), offset="ul")
            transform_subset = rio.transform.from_origin(west, top, transform.a, -transform.e)  # type: ignore
            procstep = self.procstep.copy()

            # Try to run the coregistration. If it fails for any reason, skip it and save the exception.
            try:
                procstep.fit(
                    reference_elev=ref_subset,
                    to_be_aligned_elev=tba_subset,
                    transform=transform_subset,
                    inlier_mask=mask_subset,
                    bias_vars=bias_vars,
                    weights=weights,
                    crs=crs,
                    area_or_point=area_or_point,
                    z_name=z_name,
                    subsample=subsample,
                    random_state=random_state,
                )
                nmad, median = procstep.error(
                    reference_elev=ref_subset,
                    to_be_aligned_elev=tba_subset,
                    error_type=["nmad", "median"],
                    inlier_mask=mask_subset,
                    transform=transform_subset,
                    crs=crs,
                )
            except Exception as exception:
                return exception

            meta: dict[str, Any] = {
                "i": i,
                "transform": transform_subset,
                "inlier_count": np.count_nonzero(mask_subset & np.isfinite(ref_subset) & np.isfinite(tba_subset)),
                "nmad": nmad,
                "median": median,
            }
            # Find the center of the inliers.
            inlier_positions = np.argwhere(mask_subset)
            mid_row = np.mean(inlier_positions[:, 0]).astype(int)
            mid_col = np.mean(inlier_positions[:, 1]).astype(int)

            # Find the indices of all finites within the mask
            finites = np.argwhere(np.isfinite(tba_subset) & mask_subset)
            # Calculate the distance between the approximate center and all finite indices
            distances = np.linalg.norm(finites - np.array([mid_row, mid_col]), axis=1)
            # Find the index representing the closest finite value to the center.
            closest = np.argwhere(distances == distances.min())

            # Assign the closest finite value as the representative point
            representative_row, representative_col = finites[closest][0][0]
            meta["representative_x"], meta["representative_y"] = rio.transform.xy(
                transform_subset, representative_row, representative_col
            )

            repr_val = ref_subset[representative_row, representative_col]
            if ~np.isfinite(repr_val):
                repr_val = 0
            meta["representative_val"] = repr_val

            # If the coreg is a pipeline, copy its metadatas to the output meta
            if hasattr(procstep, "pipeline"):
                meta["pipeline"] = [step.meta.copy() for step in procstep.pipeline]

            # Copy all current metadata (except for the already existing keys like "i", "min_row", etc, and the
            # "coreg_meta" key)
            # This can then be iteratively restored when the apply function should be called.
            meta.update(
                {key: value for key, value in procstep.meta.items() if key not in ["step_meta"] + list(meta.keys())}
            )

            progress_bar.update()

            return meta.copy()

        # Catch warnings; only show them if
        exceptions: list[BaseException | warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("default")
            with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                results = executor.map(process, indices)

            exceptions += list(caught_warnings)

        empty_blocks = 0
        for result in results:
            if isinstance(result, BaseException):
                exceptions.append(result)
            elif result is None:
                empty_blocks += 1
                continue
            else:
                self._meta["step_meta"].append(result)

        progress_bar.close()

        # Stop if the success rate was below the threshold
        if ((len(self._meta["step_meta"]) + empty_blocks) / self.subdivision) <= self.success_threshold:
            raise ValueError(
                f"Fitting failed for {len(exceptions)} chunks:\n"
                + "\n".join(map(str, exceptions[:5]))
                + f"\n... and {len(exceptions) - 5} more"
                if len(exceptions) > 5
                else ""
            )

        if self.warn_failures:
            for exception in exceptions:
                warnings.warn(str(exception))

        # Set the _fit_called parameters (only identical copies of self.coreg have actually been called)
        self.procstep._fit_called = True
        if isinstance(self.procstep, CoregPipeline):
            for step in self.procstep.pipeline:
                step._fit_called = True

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    def _restore_metadata(self, meta: CoregDict) -> None:
        """
        Given some metadata, set it in the right place.

        :param meta: A metadata file to update self._meta
        """
        self.procstep._meta.update(meta)

        if isinstance(self.procstep, CoregPipeline) and "pipeline" in meta:
            for i, step in enumerate(self.procstep.pipeline):
                step._meta.update(meta["pipeline"][i])

    def to_points(self) -> NDArrayf:
        """
        Convert the blockwise coregistration matrices to 3D (source -> destination) points.

        The returned shape is (N, 3, 2) where the dimensions represent:
            0. The point index where N is equal to the amount of subdivisions.
            1. The X/Y/Z coordinate of the point.
            2. The old/new position of the point.

        To acquire the first point's original position: points[0, :, 0]
        To acquire the first point's new position: points[0, :, 1]
        To acquire the first point's Z difference: points[0, 2, 1] - points[0, 2, 0]

        :returns: An array of 3D source -> destination points.
        """
        if len(self._meta["step_meta"]) == 0:
            raise AssertionError("No coreg results exist. Has '.fit()' been called?")
        points = np.empty(shape=(0, 3, 2))

        for i in range(self.subdivision):
            # Try to restore the metadata for this chunk (if it succeeded)
            chunk_meta = next((meta for meta in self._meta["step_meta"] if meta["i"] == i), None)

            if chunk_meta is not None:
                # Successful chunk: Retrieve the representative X, Y, Z coordinates
                self._restore_metadata(chunk_meta)
                x_coord, y_coord = chunk_meta["representative_x"], chunk_meta["representative_y"]
                repr_val = chunk_meta["representative_val"]
            else:
                # Failed chunk: Calculate the approximate center using the group's bounds
                rows, cols = np.where(self._groups == i)
                center_row = (rows.min() + rows.max()) // 2
                center_col = (cols.min() + cols.max()) // 2

                transform = self._meta["step_meta"][0]["transform"]  # Assuming all chunks share a transform
                x_coord, y_coord = rio.transform.xy(transform, center_row, center_col)
                repr_val = np.nan  # No valid Z value for failed chunks

            # Old position based on the calculated or retrieved coordinates
            old_pos_arr = np.reshape([x_coord, y_coord, repr_val], (1, 3))
            old_position = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x=old_pos_arr[:, 0], y=old_pos_arr[:, 1], crs=None),
                data={"z": old_pos_arr[:, 2]},
            )

            if chunk_meta is not None:
                # Successful chunk: Apply the transformation
                new_position = self.procstep.apply(old_position)
                new_pos_arr = np.reshape(
                    [new_position.geometry.x.values, new_position.geometry.y.values, new_position["z"].values], (1, 3)
                )
            else:
                # Failed chunk: Keep the new position the same as the old position (no transformation)
                new_pos_arr = old_pos_arr.copy()

            # Append the result
            points = np.append(points, np.dstack((old_pos_arr, new_pos_arr)), axis=0)

        return points

    def stats(self) -> pd.DataFrame:
        """
        Return statistics for each chunk in the blockwise coregistration.

            * center_{x,y,z}: The center coordinate of the chunk in georeferenced units.
            * {x,y,z}_off: The calculated offset in georeferenced units.
            * inlier_count: The number of pixels that were inliers in the chunk.
            * nmad: The NMAD of elevation differences (robust dispersion) after coregistration.
            * median: The median of elevation differences (vertical shift) after coregistration.

        :raises ValueError: If no coregistration results exist yet.

        :returns: A dataframe of statistics for each chunk.
        If a chunk fails (not present in `chunk_meta`), the statistics will be returned as `NaN`.
        """
        points = self.to_points()

        chunk_meta = {meta["i"]: meta for meta in self.meta["step_meta"]}

        statistics: list[dict[str, Any]] = []
        for i in range(points.shape[0]):
            if i not in chunk_meta:
                # For missing chunks, return NaN for all stats
                statistics.append(
                    {
                        "center_x": points[i, 0, 0],
                        "center_y": points[i, 1, 0],
                        "center_z": points[i, 2, 0],
                        "x_off": np.nan,
                        "y_off": np.nan,
                        "z_off": np.nan,
                        "inlier_count": np.nan,
                        "nmad": np.nan,
                        "median": np.nan,
                    }
                )
            else:
                statistics.append(
                    {
                        "center_x": points[i, 0, 0],
                        "center_y": points[i, 1, 0],
                        "center_z": points[i, 2, 0],
                        "x_off": points[i, 0, 1] - points[i, 0, 0],
                        "y_off": points[i, 1, 1] - points[i, 1, 0],
                        "z_off": points[i, 2, 1] - points[i, 2, 0],
                        "inlier_count": chunk_meta[i]["inlier_count"],
                        "nmad": chunk_meta[i]["nmad"],
                        "median": chunk_meta[i]["median"],
                    }
                )

        stats_df = pd.DataFrame(statistics)
        stats_df.index.name = "chunk"

        return stats_df

    def subdivide_array(self, shape: tuple[int, ...]) -> NDArrayf:
        """
        Return the grid subdivision for a given DEM shape.

        :param shape: The shape of the input DEM.

        :returns: An array of shape 'shape' with 'self.subdivision' unique indices.
        """
        if len(shape) == 3 and shape[0] == 1:  # Account for (1, row, col) shapes
            shape = (shape[1], shape[2])
        return subdivide_array(shape, count=self.subdivision)

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if np.count_nonzero(np.isfinite(elev)) == 0:
            return elev, transform

        # Other option than resample=True is not implemented for this case
        if "resample" in kwargs and kwargs["resample"] is not True:
            raise NotImplementedError("Option `resample=False` not supported for coreg method BlockwiseCoreg.")

        points = self.to_points()
        # Check for NaN values across both the old and new positions for each point
        mask = ~np.isnan(points).any(axis=(1, 2))

        # Filter out points where there are no NaN values
        points = points[mask]

        bounds = _bounds(transform=transform, shape=elev.shape)
        resolution = _res(transform)

        representative_height = np.nanmean(elev)
        edges_source_arr = np.array(
            [
                [bounds.left + resolution[0] / 2, bounds.top - resolution[1] / 2, representative_height],
                [bounds.right - resolution[0] / 2, bounds.top - resolution[1] / 2, representative_height],
                [bounds.left + resolution[0] / 2, bounds.bottom + resolution[1] / 2, representative_height],
                [bounds.right - resolution[0] / 2, bounds.bottom + resolution[1] / 2, representative_height],
            ]
        )
        edges_source = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=edges_source_arr[:, 0], y=edges_source_arr[:, 1], crs=None),
            data={"z": edges_source_arr[:, 2]},
        )

        edges_dest = self.apply(edges_source)
        edges_dest_arr = np.array(
            [edges_dest.geometry.x.values, edges_dest.geometry.y.values, edges_dest["z"].values]
        ).T
        edges = np.dstack((edges_source_arr, edges_dest_arr))

        all_points = np.append(points, edges, axis=0)

        warped_dem = warp_dem(
            dem=elev,
            transform=transform,
            source_coords=all_points[:, :, 1],
            destination_coords=all_points[:, :, 0],
            resampling="linear",
            apply_z_correction=self.apply_z_correction,
        )

        return warped_dem, transform

    def _apply_pts(
        self, elev: gpd.GeoDataFrame, z_name: str = "z", bias_vars: dict[str, NDArrayf] | None = None, **kwargs: Any
    ) -> gpd.GeoDataFrame:
        """Apply the scaling model to a set of points."""
        points = self.to_points()

        # Check for NaN values across both the old and new positions for each point
        mask = ~np.isnan(points).any(axis=(1, 2))

        # Filter out points where there are no NaN values
        points = points[mask]

        new_coords = np.array([elev.geometry.x.values, elev.geometry.y.values, elev["z"].values]).T

        for dim in range(0, 3):
            with warnings.catch_warnings():
                # ZeroDivisionErrors may happen when the transformation is empty (which is fine)
                warnings.filterwarnings("ignore", message="ZeroDivisionError")
                model = scipy.interpolate.Rbf(
                    points[:, 0, 0],
                    points[:, 1, 0],
                    points[:, dim, 1] - points[:, dim, 0],
                    function="linear",
                )

            new_coords[:, dim] += model(elev.geometry.x.values, elev.geometry.y.values)

        gdf_new_coords = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=new_coords[:, 0], y=new_coords[:, 1], crs=None), data={"z": new_coords[:, 2]}
        )

        return gdf_new_coords


def warp_dem(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    source_coords: NDArrayf,
    destination_coords: NDArrayf,
    resampling: str = "cubic",
    trim_border: bool = True,
    dilate_mask: bool = True,
    apply_z_correction: bool = True,
) -> NDArrayf:
    """
    (22/08/24: Method currently used only for blockwise coregistration)
    Warp a DEM using a set of source-destination 2D or 3D coordinates.

    :param dem: The DEM to warp. Allowed shapes are (1, row, col) or (row, col)
    :param transform: The Affine transform of the DEM.
    :param source_coords: The source 2D or 3D points. must be X/Y/(Z) coords of shape (N, 2) or (N, 3).
    :param destination_coords: The destination 2D or 3D points. Must have the exact same shape as 'source_coords'
    :param resampling: The resampling order to use. Choices: ['nearest', 'linear', 'cubic'].
    :param trim_border: Remove values outside of the interpolation regime (True) or leave them unmodified (False).
    :param dilate_mask: Dilate the nan mask to exclude edge pixels that could be wrong.
    :param apply_z_correction: Boolean to toggle whether the Z-offset correction is applied or not (default True).

    :raises ValueError: If the inputs are poorly formatted.
    :raises AssertionError: For unexpected outputs.

    :returns: A warped DEM with the same shape as the input.
    """
    if source_coords.shape != destination_coords.shape:
        raise ValueError(
            f"Incompatible shapes: source_coords '({source_coords.shape})' and "
            f"destination_coords '({destination_coords.shape})' shapes must be the same"
        )
    if (len(source_coords.shape) > 2) or (source_coords.shape[1] < 2) or (source_coords.shape[1] > 3):
        raise ValueError(
            "Invalid coordinate shape. Expected 2D or 3D coordinates of shape (N, 2) or (N, 3). "
            f"Got '{source_coords.shape}'"
        )
    allowed_resampling_strs = ["nearest", "linear", "cubic"]
    if resampling not in allowed_resampling_strs:
        raise ValueError(f"Resampling type '{resampling}' not understood. Choices: {allowed_resampling_strs}")

    dem_arr, dem_mask = get_array_and_mask(dem)

    bounds = _bounds(transform=transform, shape=dem_arr.shape)

    no_horizontal = np.sum(np.linalg.norm(destination_coords[:, :2] - source_coords[:, :2], axis=1)) < 1e-6
    no_vertical = source_coords.shape[1] > 2 and np.sum(np.abs(destination_coords[:, 2] - source_coords[:, 2])) < 1e-6

    if no_horizontal and no_vertical:
        warnings.warn("No difference between source and destination coordinates. Returning self.")
        return dem

    source_coords_scaled = source_coords.copy()
    destination_coords_scaled = destination_coords.copy()
    # Scale the coordinates to index-space
    for coords in (source_coords_scaled, destination_coords_scaled):
        coords[:, 0] = dem_arr.shape[1] * (coords[:, 0] - bounds.left) / (bounds.right - bounds.left)
        coords[:, 1] = dem_arr.shape[0] * (1 - (coords[:, 1] - bounds.bottom) / (bounds.top - bounds.bottom))

    # Generate a grid of x and y index coordinates.
    grid_y, grid_x = np.mgrid[0 : dem_arr.shape[0], 0 : dem_arr.shape[1]]

    if no_horizontal:
        warped = dem_arr.copy()
    else:
        # Interpolate the sparse source-destination points to a grid.
        # (row, col, 0) represents the destination y-coordinates of the pixels.
        # (row, col, 1) represents the destination x-coordinates of the pixels.
        new_indices = scipy.interpolate.griddata(
            source_coords_scaled[:, [1, 0]],
            destination_coords_scaled[:, [1, 0]],  # Coordinates should be in y/x (not x/y) for some reason..
            (grid_y, grid_x),
            method="linear",
        )

        # If the border should not be trimmed, just assign the original indices to the missing values.
        if not trim_border:
            missing_ys = np.isnan(new_indices[:, :, 0])
            missing_xs = np.isnan(new_indices[:, :, 1])
            new_indices[:, :, 0][missing_ys] = grid_y[missing_ys]
            new_indices[:, :, 1][missing_xs] = grid_x[missing_xs]

        order = {"nearest": 0, "linear": 1, "cubic": 3}

        with warnings.catch_warnings():
            # A skimage warning that will hopefully be fixed soon. (2021-06-08)
            warnings.filterwarnings("ignore", message="Passing `np.nan` to mean no clipping in np.clip")
            warped = skimage.transform.warp(
                image=np.where(dem_mask, np.nan, dem_arr),
                inverse_map=np.moveaxis(new_indices, 2, 0),
                output_shape=dem_arr.shape,
                preserve_range=True,
                order=order[resampling],
                cval=np.nan,
            )
            new_mask = (
                skimage.transform.warp(
                    image=dem_mask, inverse_map=np.moveaxis(new_indices, 2, 0), output_shape=dem_arr.shape, cval=False
                )
                > 0
            )

        if dilate_mask:
            new_mask = scipy.ndimage.binary_dilation(new_mask, iterations=order[resampling]).astype(new_mask.dtype)

        warped[new_mask] = np.nan

    # Apply the Z-correction if apply_z_correction is True and if the coordinates are 3D (N, 3)
    if not no_vertical and apply_z_correction:
        grid_offsets = scipy.interpolate.griddata(
            points=destination_coords_scaled[:, :2],
            values=source_coords_scaled[:, 2] - destination_coords_scaled[:, 2],
            xi=(grid_x, grid_y),
            method=resampling,
            fill_value=np.nan,
        )
        if not trim_border:
            grid_offsets[np.isnan(grid_offsets)] = np.nanmean(grid_offsets)

        warped += grid_offsets

    assert not np.all(np.isnan(warped)), "All-NaN output."

    return warped.reshape(dem.shape)
