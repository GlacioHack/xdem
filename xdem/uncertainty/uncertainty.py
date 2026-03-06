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

"""Module for uncertainty quantification routines related to geospatial data and Rasters."""

from __future__ import annotations

import warnings
import logging
from typing import Callable, Any, Literal, TYPE_CHECKING

import pandas as pd
import numpy as np
import geopandas as gpd

import xdem
from geoutils.stats import nmad
from xdem._typing import NDArrayf, NDArrayb
from geoutils import Raster, Vector, PointCloud

from xdem import coreg
from xdem.cosampling import cosample
from xdem.uncertainty._spatial import _variogram, params_to_gstools_model
from xdem import terrain
from xdem._misc import import_optional

from xdem.spatialstats import _estimate_model_heteroscedasticity, fit_sum_model_variogram, correlation_from_variogram

if TYPE_CHECKING:
    from xdem.dem import DEM
    from xdem.epc import EPC
    import gstools as gs

def _postproc_coreg_metadata(c: coreg.Coreg) -> pd.DataFrame:

    # Get metadata: translations, rotations, centroid, last iteration and last translation/rotation
    output_matrix = c.to_matrix()
    trans_rot_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    trans_rot = coreg.translations_rotations_from_matrix(output_matrix)
    if "centroid" in c.meta["outputs"]["affine"]:
        output_centroid = c.meta["outputs"]["affine"]["centroid"]
    else:
        output_centroid = (np.nan, np.nan, np.nan)
    last_it = c.meta["outputs"]["iterative"]["last_iteration"]
    df_stats = c.meta["outputs"]["iterative"]["iteration_stats"]
    last_statistic_t = df_stats.loc[df_stats["iteration"] == last_it, "translation"].values[0]
    if "rotation" in df_stats.columns:
        last_statistic_r = df_stats.loc[df_stats["iteration"] == last_it, "rotation"].values[0]
    else:
        last_statistic_r = np.nan

    # Store in dataframe
    output_dict = {trans_rot_names[i]: trans_rot[i] for i in range(len(trans_rot_names))}
    output_dict.update({"ocx": output_centroid[0], "ocy": output_centroid[1], "ocz": output_centroid[2]})
    df = pd.DataFrame(data=[output_dict])
    df["last_iteration"] = last_it
    df["last_translation"] = last_statistic_t
    df["last_rotation"] = last_statistic_r

    return df


def _propag_uncertainty_coreg(
    reference_elev: DEM | gpd.GeoDataFrame | xdem.EPC,
    to_be_aligned_elev: DEM | gpd.GeoDataFrame | xdem.EPC,
    coreg_method: coreg.Coreg,
    nsim: int = 30,
    error_applied_to: Literal["ref", "tba"] = "tba",
    inlier_mask: Raster | NDArrayb = None,
    random_state: int | np.random.Generator | None = None,
    kwargs_coreg_fit: dict[str, Any] | None = None,
    kwargs_infer_uncertainty: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[coreg.Coreg]]:
    """
    Propagate uncertainties to any coregistration by Monte Carlo simulations of errors.

    Returns a summary dataframe (mean/STD per translation and rotation), the full dataframe of all simulations core
    metadata, and the full list of coreg objects for each run.

    :param reference_elev: Reference elevation.
    :param to_be_aligned_elev: To-be-aligned elevation.
    :param coreg_method: Coregistration method.
    :param nsim: Number of simulations to perform.
    :param inlier_mask: Inlier mask (valid = True).
    :param random_state: Random state.
    :param kwargs_coreg_fit: Keyword arguments passed to `Coreg.fit`.
    :param kwargs_infer_uncertainty: Keyword arguments passed to `DEM/EPC.infer_uncertainty`.
    """

    # Normalize input dicts if empty
    if kwargs_coreg_fit is None:
        kwargs_coreg_fit = {}
    if kwargs_infer_uncertainty is None:
        kwargs_infer_uncertainty = {}

    # Define random state
    rng = np.random.default_rng(random_state)

    # First, infer uncertainty
    if error_applied_to == "ref":
        source_elev, other_elev = reference_elev, to_be_aligned_elev
    else:
        source_elev, other_elev = to_be_aligned_elev, reference_elev

    logging.info(f"Inferring uncertainty from {error_applied_to}...")
    # Get correlation, heteroscedasticity, and build model for GSTools random fields
    hetesc_out, corr_out = _infer_uncertainty(source_elev=source_elev, other_elev=other_elev,
                                             stable_terrain=inlier_mask, random_state=rng,
                                             **kwargs_infer_uncertainty)
    sig_elev = hetesc_out[0]
    params = corr_out[1]
    corr_func = params_to_gstools_model(params)
    logging.info(f"Found spatial correlation parameters:\n{params}")

    # Then, run simulations
    list_df = []
    list_coreg = []
    for i in range(nsim):
        logging.info(f"Running simulation {i+1} out of {nsim}\n"
                     f"######################################")

        # Derive error field for this simulation
        logging.info(f"  Simulating error field...")
        error_field = _simu_random_error_field(elev=source_elev, sig_elev=sig_elev, corr_func=corr_func,
                                               random_state=rng)
        # Apply error to proper input
        if error_applied_to == "ref":
            ref_elev = reference_elev + error_field
            tba_elev = to_be_aligned_elev
        else:
            ref_elev = reference_elev
            tba_elev = to_be_aligned_elev + error_field

        # Run coreg fit
        logging.info(f"  Running coregistration fit...")
        c = coreg_method.copy()  # Avoid carrying over the state over multiple simulations
        c.fit(reference_elev=ref_elev, to_be_aligned_elev=tba_elev, inlier_mask=inlier_mask, random_state=rng,
              **kwargs_coreg_fit)
        df_it = _postproc_coreg_metadata(c)
        df_it["nsim"] = i + 1
        list_df.append(df_it)
        list_coreg.append(c)

    # Finally, estimate errors for all the translations/rotations in the simulations
    df = pd.concat(list_df, ignore_index=True)
    t_r_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    summary = pd.DataFrame(
        {"mean": df[t_r_names].mean(), "std": df[t_r_names].std(ddof=1)}
    )

    return summary, df, list_coreg

def _simu_random_error_field(
    elev: Raster | PointCloud,
    sig_elev: Raster | PointCloud | float,
    corr_func: Callable[[NDArrayf], NDArrayf] | gs.CovModel | None = None,
    random_state: None | np.random.Generator | int = None,
) -> Raster | PointCloud:
    """
    Simulate random error fields based on per-pixel error input and correlation range.

    :param elev: DEM or point cloud.
    :param sig_elev: Random error magnitude (1-sigma), either per-point (raster or point cloud) or constant value (same
        error for all pixels).
    :param corr_func: Spatial correlation function of error.
    :param random_state: Random state or seed number to use for calculations.
    """

    gs = import_optional("gstools")

    rng = np.random.default_rng(random_state)

    if ((isinstance(elev, (gpd.GeoDataFrame, PointCloud)) and not isinstance(sig_elev, (gpd.GeoDataFrame,
                                                                                       PointCloud)))
            or (isinstance(elev, Raster) and not isinstance(sig_elev, Raster))):
        raise ValueError("Input 'elev' and 'sig_elev' must both be of the same type (raster or point cloud).")

    # 1/ Standardized random field

    # Coordinates for random field
    if isinstance(elev, (gpd.GeoDataFrame, PointCloud)):
        if isinstance(elev, PointCloud):
            gdf = elev.ds
        else:
            gdf = elev
        geom = gdf.geometry
        x = geom.x.to_numpy()
        y = geom.y.to_numpy()
        coords = (x, y)
        mesh_type = "unstructured"
        out_shape = (len(x),)
    else:
        dem = elev
        x = np.arange(dem.shape[1]) * dem.res[0]
        y = np.arange(dem.shape[0]) * dem.res[1]
        coords = (x, y)
        mesh_type = "structured"
        out_shape = dem.shape

    # Generate random field
    if corr_func is None:
        error_field = rng.normal(loc=0.0, scale=1.0, size=out_shape)
    else:
        if isinstance(corr_func, gs.CovModel):
            model = corr_func
        else:
            user_corr = corr_func

            class _CallableCorrModel(gs.CovModel):
                def __init__(self):
                    super().__init__(dim=2, var=1.0)

                def correlation(self, r: NDArrayf) -> NDArrayf:
                    return user_corr(np.asarray(r, dtype=float))
            model = _CallableCorrModel()

        # Reproducible seed derived from rng (works for both int seed and generator input)
        seed = int(rng.integers(0, 2**32 - 1))
        srf = gs.SRF(model, seed=seed, mode_no=100)

        error_field = srf(coords, mesh_type=mesh_type)

        # GSTools sometimes returns (nx, ny), ensure (nrows, ncols)
        if mesh_type == "structured" and error_field.shape != out_shape and error_field.T.shape == out_shape:
            error_field = error_field.T

        # Enforce unit variance
        error_field = error_field / np.nanstd(error_field)

    # 2/ Scale with heteroscedasticity
    if np.isscalar(sig_elev):
        sigma = float(sig_elev)
    else:
        sigma = sig_elev.data  # Works for both raster and point cloud
    error_field = sigma * error_field

    # 3/ Return same type as input
    if isinstance(elev, PointCloud):
        out = elev.copy(new_array=error_field)
        return out
    else:
        out = Raster.from_array(error_field, transform=elev.transform, crs=elev.crs, nodata=elev.nodata)
        return out

def _infer_uncertainty(
    source_elev: Raster | gpd.GeoDataFrame,
    other_elev: Raster | gpd.GeoDataFrame,
    stable_terrain: Raster | NDArrayb = None,
    approach: Literal["H2022", "R2009", "Basic"] = "H2022",
    precision_of_other: Literal["finer"] | Literal["same"] = "finer",
    spread_estimator: Callable[[NDArrayf], np.floating[Any]] = nmad,
    variogram_estimator: Literal["matheron", "cressie", "genton", "dowd"] = "dowd",
    hetesc_vars: (tuple[Raster | pd.Series | str, ...] |
                  dict[str, Raster | pd.Series | str]) = ("slope", "max_curvature"),
    vario_model: str | tuple[str, ...] = ("gaussian", "spherical"),
    subsample_hetesc: int | float = 1_000_000,
    subsample_pairs_vario: int | float = 1_000_000,
    z_name: str | None = None,
    random_state: int | np.random.Generator | None = None,
) -> Any:
    """
    Infer the uncertainty (random error model) of an elevation data using the difference to another elevation data.

    This function infers variable errors (based on slope and curvature by default) and spatial correlation of error
    (between 0 and 1, varies with distance between observations).

    It uses elevation differences to another elevation data sources on static surfaces (or "stable terrain") as an
    error proxy. It assumes this other data sources has either much higher precision (all errors come from the
    source elevation) or similar precision (errors are divided by sqrt(2) to represent initial errors).

    See Hugonnet et al. (2022) for methodological details.

    :param source_elev: Source elevation dataset (Raster or PointCloud) to use.
    :param other_elev: Other elevation dataset (Raster or PointCloud) to use for inference, considered either of finer or similar
        precision (see `precision_of_other`).
    :param stable_terrain: Raster of stable terrain to use as error proxy.
    :param approach: Whether to use Hugonnet et al., 2022 (variable errors, multiple ranges of error correlation),
        or Rolstad et al., 2009 (constant error, multiple ranges of error correlation), or a basic approach
        (constant error, single range of error correlation). Note that all approaches use robust estimators of
        variance (NMAD) and variograms (Dowd) by default. These estimators can be set separately.
    :param precision_of_other: Whether other elevation dataset is of finer precision (3 times more precise = 95% of
        estimated error will come from the source) or similar precision (elevation difference is divided by sqrt(2) to
        avoid double-counting errors).
    :param spread_estimator: Estimator for statistical dispersion (e.g., standard deviation), defaults to the
        normalized median absolute deviation (NMAD) for robustness.
    :param variogram_estimator: Estimator for empirical variogram, defaults to Dowd for robustness and consistency
        with the NMAD estimator for the spread.
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
    :param random_state: Random state or seed number to use for subsampling and optimizer.
    :param hetesc_vars: Variables to use to predict error variability (= heteroscedasticity). Either rasters,
        pandas series if a point cloud was passed, or names of a terrain attributes. Defaults to slope and
        maximum curvature for the Raster.
    :param vario_model: Variogram forms to model the spatial correlation of error. A list translates into
        a sum of models. Uses three by default for methods allowing nested correlation ranges, otherwise one.
    :param random_state: State or seed to use for randomization.

    :return: Tuple of (Raster of spread of random errors (1-sigma), Binning dataframe, Empirical error function) and
             Tuple of (Empirical variogram dataframe, Model parameters dataframe, Spatial error correlation function).
    """

    # Summarize approach steps
    approach_dict = {
        "H2022": {"heterosc": True, "multi_range": True},
        "R2009": {"heterosc": False, "multi_range": True},
        "Basic": {"heterosc": False, "multi_range": False},
    }

    # # Difference the two datasets
    # dh = _difference(source_elev, other_elev)

    # # If the precision of the other Raster is the same, divide the dh values by sqrt(2)
    # # See Equation 7 and 8 of Hugonnet et al. (2022)
    # if precision_of_other == "same":
    #     dh = dh / np.sqrt(2)

    logging.info(f"Starting heteroscedasticity inference.")
    # Heteroscedasticity
    sig_dh, df_bin, fun_bin = _infer_heteroscedasticity(
        source_elev=source_elev,
        other_elev=other_elev,
        stable_terrain=stable_terrain,
        heterosc=approach_dict[approach]["heterosc"],
        hetesc_vars=hetesc_vars,
        z_name=z_name,
        subsample_hetesc=subsample_hetesc,
        spread_statistic=spread_estimator,
    )

    logging.info(f"Starting spatial correlation inference.")
    # Spatial error correlation
    df_vario, df_params, fun_corr = _infer_spatial_correlation(
        source_elev=source_elev,
        other_elev=other_elev,
        inlier_mask=stable_terrain,
        errors=sig_dh,
        estimator=variogram_estimator,
        random_state=random_state,
        list_models=vario_model,
        subsample=subsample_pairs_vario,
    )

    return (sig_dh, df_bin, fun_bin), (df_vario, df_params, fun_corr)


def _infer_heteroscedasticity(
    *,
    # Main elevation inputs to difference (always used to fit on stable terrain)
    source_elev: Raster | gpd.GeoDataFrame | PointCloud,
    other_elev: Raster | gpd.GeoDataFrame | PointCloud,
    # Masking
    stable_terrain: Raster | np.ndarray | Vector | gpd.GeoDataFrame | None = None,
    vector_mask_mode: Literal["inside", "outside"] = "inside",
    # Whether to infer a variable error (default) or constant
    heterosc: bool = True,
    # Heteroscedastic predictors
    hetesc_vars: (
        tuple[Raster | np.ndarray | str, ...]
        | dict[str, Raster | np.ndarray | str]
    ) = ("slope", "max_curvature"),
    # Point-cloud value column
    z_name: str | None = None,
    # Subsampling (internal only: output is always on the source domain)
    subsample_hetesc: int | float = int(1e8),
    random_state: int | np.random.Generator | None = None,
    # Estimation options (same as infer_heteroscedasticity)
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = nmad,
    list_var_bins: int | tuple[int, ...] | tuple[NDArrayf] | None = None,
    min_count: int | None = 100,
    fac_spread_outliers: float | None = 7,
) -> tuple[
    Raster | PointCloud,
    pd.DataFrame,
    Callable[[tuple[NDArrayf, ...]], NDArrayf],
]:
    """
    Infer heteroscedasticity from elevation differences on static terrain and explanatory variables.
    """

    # 0) Normalize point cloud inputs (PointCloud -> GeoDataFrame)
    if isinstance(source_elev, PointCloud):
        source_pc = source_elev.ds
        z_name_source = source_elev.data_column
    else:
        source_pc = source_elev if isinstance(source_elev, gpd.GeoDataFrame) else None
        z_name_source = z_name

    if isinstance(other_elev, PointCloud):
        other_pc = other_elev.ds
        z_name_other = other_elev.data_column
    else:
        other_pc = other_elev if isinstance(other_elev, gpd.GeoDataFrame) else None
        z_name_other = z_name

    # 1) Normalize heteroscedastic predictors and their names
    if isinstance(hetesc_vars, dict):
        het_names = list(hetesc_vars.keys())
        het_specs = list(hetesc_vars.values())
    else:
        het_specs = list(hetesc_vars)
        # Keep existing behavior: if strings are passed, names match the attribute name.
        het_names = [v if isinstance(v, str) else f"var{i+1}" for i, v in enumerate(het_specs)]

    # Derive terrain attributes of Raster if string is passed in the list of variables
    list_vars: list[Raster | np.ndarray] = []
    for var in het_specs:
        if isinstance(var, str):
            # Prefer deriving from the source Raster so we can always evaluate on the source domain later.
            dem_for_attr = source_elev if isinstance(source_elev, Raster) else (other_elev if isinstance(other_elev, Raster) else None)
            if dem_for_attr is None:
                raise TypeError(f"Cannot derive terrain attribute '{var}' without at least one Raster input.")
            list_vars.append(getattr(terrain, var)(dem_for_attr))
        else:
            # In this helper, predictors must be raster/grid-like or derivable as such.
            # (Point predictors are handled upstream via cosample as 1D arrays/Series, but for heterosc-vars
            #  we keep this minimal and consistent with the fitting model.)
            if not isinstance(var, (Raster, np.ndarray)):
                raise TypeError(
                    "hetesc_vars entries must be Raster/ndarray or str terrain attribute names."
                )
            list_vars.append(var)

    # 2) Fit heteroscedasticity model on stable terrain (co-sampled)
    logging.info(f"  Step 1: Sampling all datasets at colocated valid coordinates with subsample size"
                 f" {subsample_hetesc}...")
    rp1_fit, rp2_fit, aux_fit, _ = cosample(
        rst_pc1=source_elev if not isinstance(source_elev, PointCloud) else source_pc,
        rst_pc2=other_elev if not isinstance(other_elev, PointCloud) else other_pc,
        aux_vars=dict(zip(het_names, list_vars)),
        inlier_mask=stable_terrain,
        vector_mask_mode=vector_mask_mode,
        z_name=z_name_source if source_pc is not None else z_name_other,
        subsample=subsample_hetesc,
        random_state=random_state,
        return_coords=False,
    )

    # Elevation difference of the subsample
    dvalues_fit = rp1_fit - rp2_fit

    # 3) Perform binning and function fit on array inputs

    # 3A) If heteroscedastic, perform binning and fit
    if heterosc:
        logging.info(f"  Step 2: Estimating variable error with binned variables {list(aux_fit.keys())} and spread "
                     f"estimator {spread_statistic.__name__}...")
        df, fun = _estimate_model_heteroscedasticity(
            dvalues=dvalues_fit,
            list_var=list(aux_fit.values()),
            list_var_names=list(aux_fit.keys()),
            spread_statistic=spread_statistic,
            list_var_bins=list_var_bins,
            min_count=min_count,
            fac_spread_outliers=fac_spread_outliers,
        )

        # Evaluate the fitted model back on the *source* domain
        # If Source domain is raster
        if isinstance(source_elev, Raster):

            aux_src: list[NDArrayf] = []
            for name, var in zip(het_names, list_vars):

                if isinstance(var, Raster):
                    if not source_elev.georeferenced_grid_equal(var):
                        var = var.reproject(source_elev, silent=True)
                    v_arr = var.data.filled(np.nan) if np.ma.isMaskedArray(var.data) else var.data
                    aux_src.append(v_arr)

                else:
                    # ndarray must already match the source grid
                    if var.ndim != 2 or var.shape != source_elev.shape:
                        raise ValueError(
                            f"Heteroscedastic variable '{name}' was "
                            f"provided as an array but does not match the source grid."
                        )
                    aux_src.append(var)

            error_arr = fun(tuple(aux_src))
            sig_dh = source_elev.copy(new_array=error_arr)

        else:
            assert source_pc is not None

            # Evaluate predictors at *all* source points (interpolate rasters at those points)
            # No subsampling here, output is always on the full source domain
            _, _, aux_src_pts, _ = cosample(
                rst_pc1=source_pc,
                rst_pc2=other_elev,
                aux_vars=dict(zip(het_names, list_vars)),
                inlier_mask=None,
                vector_mask_mode=vector_mask_mode,
                z_name=z_name_source,
                subsample=1,
                random_state=random_state,
                return_coords=True,
                preserve_shape=True,
            )

            # Apply model on 1D predictor arrays at source points
            aux_src_list = [aux_src_pts[n] for n in het_names]
            sig_dh_arr = fun(tuple(aux_src_list))
            sig_dh = PointCloud.from_xyz(x=source_elev.geometry.x.values, y=source_elev.geometry.y.values, z=sig_dh_arr,
                                         data_column="sig_dh", crs=source_elev.crs)

    # 3B) Otherwise, return a constant error evaluated on the source input
    else:
        logging.info(f"  Step 2: Computing constant spread with spread estimator {spread_statistic.__name__}...")
        sig_dh_magn = float(spread_statistic(dvalues_fit))
        df = pd.DataFrame()
        def _const_fun(_: tuple[NDArrayf, ...]) -> NDArrayf:
            return np.asarray(sig_dh_magn, dtype=np.float32)
        fun = _const_fun

        if isinstance(source_elev, Raster):
            sig_dh = source_elev.copy(new_array=sig_dh_magn * np.ones(source_elev.shape))
        else:
            sig_dh = PointCloud.from_xyz(
                x=source_elev.geometry.x.values, y=source_elev.geometry.y.values, z=sig_dh_magn * np.ones(len(
                    source_elev.geometry)),
                data_column="sig_dh", crs=source_elev.crs)

    return sig_dh, df, fun

def _difference(source_elev: Raster | gpd.GeoDataFrame, other_elev: Raster | gpd.GeoDataFrame) -> Raster | PointCloud:
    """Difference two elevation datasets, with at least one that must be a Raster."""

    # Elevation change with the other Raster
    if isinstance(other_elev, Raster) and isinstance(source_elev, Raster):
        dh = other_elev.reproject(source_elev, silent=True) - source_elev

    # Other is point cloud, source is Raster: we interpolate at other coordinates
    elif isinstance(other_elev, (gpd.GeoDataFrame, PointCloud)) and isinstance(source_elev, Raster):
        if isinstance(other_elev, PointCloud):
            gdf = other_elev.ds
            z_name = other_elev.data_column
        else:
            gdf = other_elev

        gdf = gdf.to_crs(source_elev.crs)
        interp_h = source_elev.interp_points(PointCloud(gdf, data_column=z_name), as_array=True)

        dh_arr = gdf[z_name].values - interp_h
        dh = PointCloud.from_xyz(x=gdf.geometry.x, y=gdf.geometry.y, z=dh_arr, data_column=z_name,
                                 crs=gdf.crs)

    # Source is point cloud, other is Raster: we interpolate at source coordinates
    elif isinstance(source_elev, (gpd.GeoDataFrame, PointCloud)) and isinstance(other_elev, Raster):
        # Point cloud vs Raster (source is point cloud, other is Raster): dh at source point coords
        if isinstance(source_elev, PointCloud):
            gdf = source_elev.ds
            z_name = source_elev.data_column
        else:
            gdf = source_elev

        gdf = gdf.to_crs(other_elev.crs)
        interp_h = other_elev.interp_points(PointCloud(gdf, data_column=z_name), as_array=True)
        dh_arr = interp_h - gdf[z_name].values
        dh = PointCloud.from_xyz(x=gdf.geometry.x, y=gdf.geometry.y, z=dh_arr, data_column=z_name, crs=gdf.crs)
    else:
        raise TypeError("Elevation datasets should be Raster or elevation point cloud object.")

    return dh

def _infer_spatial_correlation(
    source_elev: Raster | gpd.GeoDataFrame | PointCloud,
    other_elev: Raster | gpd.GeoDataFrame | PointCloud,
    inlier_mask: NDArrayb | Raster | Vector | gpd.GeoDataFrame | None = None,
    vector_mask_mode: Literal["inside", "outside"] = "inside",
    errors: NDArrayf | Raster | None = None,
    estimator: Literal["matheron", "cressie", "genton", "dowd"] = "dowd",
    sampling: Literal["loglag", "random_xy"] = "loglag",
    subsample: int | float = 1,
    random_state: int | np.random.Generator | None = None,
    z_name: str | None = None,
    # Variogram model fitting (sum of models)
    list_models: list[str | Callable[[NDArrayf, float, float], NDArrayf]] = None,
    bounds: list[tuple[float, float]] | None = None,
    p0: list[float] | None = None,
    # Pass-through for _variogram sampling / binning arguments
    **sampling_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, Callable[[NDArrayf], NDArrayf]]:
    """
    Infer spatial correlation of errors from elevation differences on inlier terrain.

    - Accepts point or raster inputs (but at least one input must be raster/Raster).
    - Accepts inlier_mask as Vector/Raster/2D ndarray.
    - Keeps raster dh as 2D (masked to NaN outside inliers, Dask-safe).
    - For point dh, evaluates the raster-grid inlier mask on points via cosample().
    """

    if list_models is None:
        raise ValueError("'list_models' must be provided (list of variogram model names/functions).")

    # 1) Difference datasets (dh on the *source* domain in your implementation)
    dh = _difference(source_elev, other_elev)

    # 2) Choose a reference raster grid to interpret 2D masks (and for point masking via interpolation)
    ref: Raster
    if isinstance(source_elev, Raster):
        ref = source_elev
    elif isinstance(other_elev, Raster):
        ref = other_elev
    else:
        raise TypeError("At least one of source_elev or other_elev must be a Raster to infer a reference grid.")

    # 3) Standardize + empirical variogram

    # Use errors only if passed (otherwise constant value)
    if errors is None:
        aux_vars = None
    else:
        aux_vars = {"err": errors.data}

    # We need to preserve shape for lazy raster pairwise sampling, and need to get coords for points
    if isinstance(source_elev, Raster) and isinstance(other_elev, Raster):
        preserve_shape = True
        return_coords = False
        transform = source_elev.transform
    else:
        preserve_shape = False
        return_coords = True
        transform = None

    logging.info(f"  Step 1: Sampling at colocated values of the inlier mask...")
    # Apply mask to inputs and auxiliary "error" input (no subsampling)
    rp1, rp2, aux_e, coords = cosample(
        rst_pc1=source_elev,
        rst_pc2=other_elev,
        aux_vars=aux_vars,
        inlier_mask=inlier_mask,
        vector_mask_mode="inside",  # already encoded in inlier_grid
        z_name=z_name,
        subsample=1,
        random_state=random_state,
        return_coords=return_coords,
        preserve_shape=preserve_shape,
        )

    # Difference and standardize
    logging.info(f"  Step 2: Standardizing elevation differences...")
    dh_vals = rp1 - rp2
    if errors is not None:
        dh_vals = dh_vals / aux_e["err"]

    # Empirical variogram directly on 2D array (RegularLogLagMetricSpace path)
    logging.info(f"  Step 3: Sampling variogram with pairwise subsample size of {subsample} and estimator "
                 f"{estimator}...")
    df_emp = _variogram(
        values=dh_vals,
        coords=coords,  # This is None if the dh is a raster
        transform=transform,  # This is None if the dh is a point cloud
        sampling=sampling,
        estimator=estimator,
        random_state=random_state,
        samples=subsample,
        **sampling_kwargs,
    )

    # 5) Fit model(s) and return correlation function
    # Fit a sum of variogram models
    logging.info(f"  Step 4: Fitting variogram models {list_models}...")
    params_variogram_model = fit_sum_model_variogram(
        list_models=list_models,
        empirical_variogram=df_emp,
        bounds=bounds,
        p0=p0,
    )[1]

    spatial_correlation_func = correlation_from_variogram(params_variogram_model=params_variogram_model)

    return df_emp, params_variogram_model, spatial_correlation_func