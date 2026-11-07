# Copyright (c) 2026 xDEM developers
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

"""Module for coregistration pipelines."""

from __future__ import annotations

import copy
import inspect
import logging
import warnings
from contextlib import ExitStack
from typing import (
    Any,
    Generator,
    Literal,
    overload,
)

import geopandas as gpd
import geoutils as gu
import numpy as np
import rasterio as rio
import rasterio.warp
from geoutils._dispatch import (
    _get_raster_interface,
    is_dask_array,
)
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.pointcloud import PointCloudType
from geoutils.raster import Raster, RasterType

from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregType,
    _preprocess_coreg_apply,
    _preprocess_coreg_fit,
)


class CoregPipeline(Coreg):
    """Apply coregistration steps in sequence, passing each aligned result to the next step."""

    ###############################
    # Construction and inspection
    ###############################

    def __init__(self, pipeline: list[Coreg]) -> None:
        """Instantiate a processing pipeline and flatten any nested pipelines.

        :param pipeline: Processing steps to run in the sequence they are given.
        """

        def put_coreg_in_series(pipeline: list[Coreg]) -> list[Coreg]:
            """Flatten nested pipelines while keeping the processing order."""

            list_coreg = []
            for step in pipeline:
                if not isinstance(step, CoregPipeline):
                    list_coreg.append(step)
                else:
                    list_coreg += put_coreg_in_series(step.pipeline)
            return list_coreg

        self.pipeline = put_coreg_in_series(pipeline)

        # Only the first step can use an initial shift because later steps receive an already aligned input
        for i, step in enumerate(self.pipeline):
            if i > 0 and "affine" in step.meta["inputs"] and "initial_shift" in step.meta["inputs"]["affine"]:
                warnings.warn(
                    message="No initial shift can be defined in a coregistration pipeline other than for the first "
                    f"step. Overriding to initial_shift=None for step number {i}. Remove initial shift parameters"
                    " outside of the first step to silence this warning.",
                    category=UserWarning,
                )
                del step.meta["inputs"]["affine"]["initial_shift"]

        super().__init__()

    def __repr__(self) -> str:
        """Return the ordered coregistration steps in the pipeline."""

        return f"Pipeline: {self.pipeline}"

    @overload
    def info(self, as_str: Literal[False] = ...) -> None: ...

    @overload
    def info(self, as_str: Literal[True]) -> str: ...

    def info(self, as_str: bool = False) -> None | str:
        """Summarize information about this coregistration."""

        # Get the pipeline information for each step as a string
        final_str = []
        for i, step in enumerate(self.pipeline):
            final_str.append(f"Pipeline step {i}:\n" f"################\n")
            step_str = step.info(as_str=True)
            final_str.append(step_str)

        # Return as string or print (default)
        if as_str:
            return "".join(final_str)
        else:
            print("".join(final_str))
            return None

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        new_coreg.__dict__ = {key: copy.deepcopy(value) for key, value in self.__dict__.items() if key != "pipeline"}
        new_coreg.pipeline = [step.copy() for step in self.pipeline]

        return new_coreg

    def _parse_bias_vars(self, step: int, bias_vars: dict[str, NDArrayf] | None) -> dict[str, NDArrayf]:
        """Parse bias variables for a pipeline step requiring them."""

        # Get number of non-affine coregistration requiring bias variables to be passed
        nb_needs_vars = sum(c._needs_vars for c in self.pipeline)

        # Get step object
        coreg = self.pipeline[step]

        # Check that all variable names of this were passed
        var_names = coreg._meta["inputs"]["fitorbin"]["bias_var_names"]

        # Raise error if bias_vars is None
        if bias_vars is None:
            msg = f"No `bias_vars` passed to .fit() for bias correction step {coreg.__class__} of the pipeline."
            if nb_needs_vars > 1:
                msg += (
                    " As you are using several bias correction steps requiring `bias_vars`, don't forget to "
                    "explicitly define their `bias_var_names` during "
                    "instantiation, e.g. {}(bias_var_names=['slope']).".format(coreg.__class__.__name__)
                )
            raise ValueError(msg)

        # Raise error if no variable were explicitly assigned and there is more than 1 step with bias_vars
        if var_names is None and nb_needs_vars > 1:
            raise ValueError(
                "When using several bias correction steps requiring `bias_vars` in a pipeline,"
                "the `bias_var_names` need to be explicitly defined at each step's "
                "instantiation, e.g. {}(bias_var_names=['slope']).".format(coreg.__class__.__name__)
            )

        # Raise error if the variables explicitly assigned don't match the ones passed in bias_vars
        if not all(n in bias_vars.keys() for n in var_names):
            raise ValueError(
                "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
                "instantiation of the bias correction step {}: {}.".format(coreg.__class__, var_names)
            )

        # Add subset dict for this pipeline step to args of fit and apply
        return {n: bias_vars[n] for n in var_names}

    # Need to override base Coreg method to work on pipeline steps
    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame | PointCloudType,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame | PointCloudType,
        inlier_mask: NDArrayb | Raster | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str | None = None,
        random_state: int | np.random.Generator | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> CoregType:
        """Fit each step and pass intermediate aligned elevations to the following step."""

        # Check if subsample arguments are different from their default value for any of the coreg steps:
        # get default value in argument spec and "subsample" stored in meta, and compare both are consistent
        argspec = [inspect.getfullargspec(c.__class__) for c in self.pipeline]
        sub_meta = [c.meta["inputs"]["random"]["subsample"] for c in self.pipeline]
        sub_is_default = [
            argspec[i].defaults[argspec[i].args.index("subsample") - 1] == sub_meta[i]  # type: ignore
            for i in range(len(argspec))
        ]
        if subsample is not None and not all(sub_is_default):
            warnings.warn(
                "Subsample argument passed to fit() will override non-default subsample values defined for"
                " individual steps of the pipeline. To silence this warning: only define 'subsample' in "
                "either fit(subsample=...) or instantiation e.g., VerticalShift(subsample=...)."
            )
            # Filter warnings of individual pipelines now that the one above was raised
            warnings.filterwarnings("ignore", message="Subsample argument passed to*", category=UserWarning)

        # Prepare inputs
        prepared = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            mp_config=mp_config,
            bias_vars=bias_vars,
            fit_options=kwargs,
        )
        reference_elev = prepared["ref_elev"]
        aligned = prepared["tba_elev"]
        if _get_raster_interface(aligned) is not None:
            tba_elev_mod = aligned.copy(deep=False)
        else:
            tba_elev_mod = aligned.copy()

        # Keep temporarily files during the pipeline run
        with ExitStack() as temporary_files:
            for i, coreg in enumerate(self.pipeline):
                logging.debug("Running pipeline step: %d / %d", i + 1, len(self.pipeline))

                main_args_fit = {
                    "reference_elev": reference_elev,
                    "to_be_aligned_elev": tba_elev_mod,
                    "inlier_mask": inlier_mask,
                    "weights": weights,
                    "subsample": subsample,
                    "random_state": random_state,
                    "mp_config": mp_config,
                }

                main_args_apply = {"elev": tba_elev_mod}
                # If non-affine method that expects a bias_vars argument
                if coreg._needs_vars:
                    step_bias_vars = self._parse_bias_vars(step=i, bias_vars=bias_vars)

                    main_args_fit.update({"bias_vars": step_bias_vars})
                    main_args_apply.update({"bias_vars": step_bias_vars})

                # Perform the step fit
                coreg.fit(**main_args_fit)

                # We only run this step if it's not the last, otherwise it is unused!
                if i != (len(self.pipeline) - 1):
                    if mp_config is not None:
                        main_args_apply["mp_config"] = temporary_files.enter_context(mp_config.temporary())
                    tba_elev_mod = coreg.apply(**main_args_apply)

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: NDArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: RasterType | gpd.GeoDataFrame | PointCloudType,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame | gu.PointCloud: ...

    # Need to override base Coreg method to work on pipeline steps
    def apply(
        self,
        elev: MArrayf | NDArrayf | RasterType | gpd.GeoDataFrame | PointCloudType,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> (
        RasterType
        | gpd.GeoDataFrame
        | gu.PointCloud
        | tuple[NDArrayf, rio.transform.Affine]
        | tuple[MArrayf, rio.transform.Affine]
    ):
        """Apply every fitted coregistration step in sequence to an elevation object or array."""

        # First step and preprocessing
        if not self._fit_called and self._meta["outputs"]["affine"].get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        # Normalize spatial metadata once for every step
        result, _, _, _ = _preprocess_coreg_apply(
            elev, transform, crs, z_name, bias_vars=bias_vars, mp_config=kwargs.get("mp_config")
        )

        # Keep temporary files while running pipeline
        with ExitStack() as temporary_files:
            # Apply each step of the coregistration
            for index, coreg in enumerate(self.pipeline):
                # If non-affine method that expects a bias_vars argument
                variables = self._parse_bias_vars(index, bias_vars) if coreg._needs_vars else None
                step_kwargs = kwargs.copy()
                config = kwargs.get("mp_config")
                if config is not None and index < len(self.pipeline) - 1:
                    step_kwargs["mp_config"] = temporary_files.enter_context(config.temporary())
                result = coreg.apply(
                    result, bias_vars=variables, resample=resample, resampling=resampling, **step_kwargs
                )

            # If inputs were array, get array outputs while temporary files are available
            if isinstance(elev, np.ndarray) or is_dask_array(elev):
                raster = _get_raster_interface(result)
                if is_dask_array(elev):
                    values = raster.data.reshape(elev.shape)
                else:
                    values = raster.to_nanarray().reshape(elev.shape)
                    if isinstance(elev, np.ma.MaskedArray):
                        values = np.ma.masked_invalid(values)
                return values, raster.transform

            return result

    def __iter__(self) -> Generator[Coreg]:
        """Iterate over the pipeline steps."""
        yield from self.pipeline

    def __add__(self, other: list[Coreg] | Coreg | CoregPipeline) -> CoregPipeline:
        """Append a processing step or a pipeline to the pipeline."""
        if not isinstance(other, Coreg):
            other = list(other)
        else:
            other = [other]

        pipelines = self.pipeline + other
        return CoregPipeline(pipelines)

    def to_matrix(self) -> NDArrayf:
        """Convert the transform to a 4x4 transformation matrix."""
        return self._to_matrix_func()

    def _to_matrix_func(self) -> NDArrayf:
        """Try to join the coregistration steps to a single transformation matrix."""

        total_transform = np.eye(4)
        for coreg in self.pipeline:
            new_matrix = coreg.to_matrix()
            total_transform = new_matrix @ total_transform

        return total_transform
