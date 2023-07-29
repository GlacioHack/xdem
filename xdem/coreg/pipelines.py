"""Coregistration pipelines pre-defined with convenient user inputs and parameters."""

from __future__ import annotations

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from geoutils._typing import AnyNumber
from geoutils.raster import RasterType

from xdem._typing import NDArrayf
from xdem.coreg.affine import NuthKaab, VerticalShift
from xdem.coreg.base import Coreg
from xdem.dem import DEM
from xdem.spatialstats import nmad
from xdem.terrain import slope


def create_inlier_mask(
    src_dem: RasterType,
    ref_dem: RasterType,
    shp_list: list[str | gu.Vector | None] | tuple[str | gu.Vector] | tuple[()] = (),
    inout: list[int] | tuple[int] | tuple[()] = (),
    filtering: bool = True,
    dh_max: AnyNumber = None,
    nmad_factor: AnyNumber = 5,
    slope_lim: list[AnyNumber] | tuple[AnyNumber, AnyNumber] = (0.1, 40),
) -> NDArrayf:
    """
    Create a mask of inliers pixels to be used for coregistration. The following pixels can be excluded:
    - pixels within polygons of file(s) in shp_list (with corresponding inout element set to 1) - useful for \
    masking unstable terrain like glaciers.
    - pixels outside polygons of file(s) in shp_list (with corresponding inout element set to -1) - useful to \
delineate a known stable area.
    - pixels with absolute dh (=src-ref) are larger than a given threshold
    - pixels where absolute dh differ from the mean dh by more than a set threshold (with \
filtering=True and nmad_factor)
    - pixels with low/high slope (with filtering=True and set slope_lim values)

    :param src_dem: the source DEM to be coregistered, as a Raster or DEM instance.
    :param ref_dem: the reference DEM, must have same grid as src_dem. To be used for filtering only.
    :param shp_list: a list of one or several paths to shapefiles to use for masking. Default is none.
    :param inout: a list of same size as shp_list. For each shapefile, set to 1 (resp. -1) to specify whether \
to mask inside (resp. outside) of the polygons. Defaults to masking inside polygons for all shapefiles.
    :param filtering: if set to True, pixels will be removed based on dh values or slope (see next arguments).
    :param dh_max: remove pixels where abs(src - ref) is more than this value.
    :param nmad_factor: remove pixels where abs(src - ref) differ by nmad_factor * NMAD from the median.
    :param slope_lim: a list/tuple of min and max slope values, in degrees. Pixels outside this slope range will \
be excluded.

    :returns: A boolean array of same shape as src_dem set to True for inlier pixels
    """
    # - Sanity check on inputs - #
    # Check correct input type of shp_list
    if not isinstance(shp_list, (list, tuple)):
        raise ValueError("`shp_list` must be a list/tuple")
    for el in shp_list:
        if not isinstance(el, (str, gu.Vector)):
            raise ValueError("`shp_list` must be a list/tuple of strings or geoutils.Vector instance")

    # Check correct input type of inout
    if not isinstance(inout, (list, tuple)):
        raise ValueError("`inout` must be a list/tuple")

    if len(shp_list) > 0:
        if len(inout) == 0:
            # Fill inout with 1
            inout = [1] * len(shp_list)
        elif len(inout) == len(shp_list):
            # Check that inout contains only 1 and -1
            not_valid = [el for el in np.unique(inout) if ((el != 1) & (el != -1))]
            if len(not_valid) > 0:
                raise ValueError("`inout` must contain only 1 and -1")
        else:
            raise ValueError("`inout` must be of same length as shp")

    # Check slope_lim type
    if not isinstance(slope_lim, (list, tuple)):
        raise ValueError("`slope_lim` must be a list/tuple")
    if len(slope_lim) != 2:
        raise ValueError("`slope_lim` must contain 2 elements")
    for el in slope_lim:
        if (not isinstance(el, (int, float, np.integer, np.floating))) or (el < 0) or (el > 90):
            raise ValueError("`slope_lim` must be a tuple/list of 2 elements in the range [0-90]")

    # Initialize inlier_mask with no masked pixel
    inlier_mask = np.ones(src_dem.data.shape, dtype="bool")

    # - Create mask based on shapefiles - #
    if len(shp_list) > 0:
        for k, shp in enumerate(shp_list):
            if isinstance(shp, str):
                outlines = gu.Vector(shp)
            else:
                outlines = shp
            mask_temp = outlines.create_mask(src_dem, as_array=True).reshape(np.shape(inlier_mask))
            # Append mask for given shapefile to final mask
            if inout[k] == 1:
                inlier_mask[mask_temp] = False
            elif inout[k] == -1:
                inlier_mask[~mask_temp] = False

    # - Filter possible outliers - #
    if filtering:
        # Calculate dDEM
        ddem = src_dem - ref_dem

        # Remove gross blunders with absolute threshold
        if dh_max is not None:
            inlier_mask[np.abs(ddem.data) > dh_max] = False

        # Remove blunders where dh differ by nmad_factor * NMAD from the median
        nmad_val = nmad(ddem.data[inlier_mask])
        med = np.ma.median(ddem.data[inlier_mask])
        inlier_mask = inlier_mask & (np.abs(ddem.data - med) < nmad_factor * nmad_val).filled(False)

        # Exclude steep slopes for coreg
        slp = slope(ref_dem)
        inlier_mask[slp.data < slope_lim[0]] = False
        inlier_mask[slp.data > slope_lim[1]] = False

    return inlier_mask


def dem_coregistration(
    src_dem_path: str | RasterType,
    ref_dem_path: str | RasterType,
    out_dem_path: str | None = None,
    coreg_method: Coreg | None = NuthKaab() + VerticalShift(),
    grid: str = "ref",
    resample: bool = False,
    resampling: rio.warp.Resampling | None = rio.warp.Resampling.bilinear,
    shp_list: list[str | gu.Vector] | tuple[str | gu.Vector] | tuple[()] = (),
    inout: list[int] | tuple[int] | tuple[()] = (),
    filtering: bool = True,
    dh_max: AnyNumber = None,
    nmad_factor: AnyNumber = 5,
    slope_lim: list[AnyNumber] | tuple[AnyNumber, AnyNumber] = (0.1, 40),
    plot: bool = False,
    out_fig: str = None,
    verbose: bool = False,
) -> tuple[DEM, Coreg, pd.DataFrame, NDArrayf]:
    """
    A one-line function to coregister a selected DEM to a reference DEM.

    Reads both DEMs, reprojects them on the same grid, mask pixels based on shapefile(s), filter steep slopes and \
outliers, run the coregistration, returns the coregistered DEM and some statistics.
    Optionally, save the coregistered DEM to file and make a figure.
    For details on masking options, see `create_inlier_mask` function.

    :param src_dem_path: Path to the input DEM to be coregistered
    :param ref_dem_path: Path to the reference DEM
    :param out_dem_path: Path where to save the coregistered DEM. If set to None (default), will not save to file.
    :param coreg_method: The xdem coregistration method, or pipeline.
    :param grid: The grid to be used during coregistration, set either to "ref" or "src".
    :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, only \
the array/transform will be updated (if possible) and no resampling is done. Useful to avoid spreading data gaps.
    :param resampling: The resampling algorithm to be used if `resample` is True. Default is bilinear.
    :param shp_list: A list of one or several paths to shapefiles to use for masking.
    :param inout: A list of same size as shp_list. For each shapefile, set to 1 (resp. -1) to specify whether \
to mask inside (resp. outside) of the polygons. Defaults to masking inside polygons for all shapefiles.
    :param filtering: If set to True, filtering will be applied prior to coregistration.
    :param dh_max: Remove pixels where abs(src - ref) is more than this value.
    :param nmad_factor: Remove pixels where abs(src - ref) differ by nmad_factor * NMAD from the median.
    :param slope_lim: A list/tuple of min and max slope values, in degrees. Pixels outside this slope range will \
be excluded.
    :param plot: Set to True to plot a figure of elevation diff before/after coregistration.
    :param out_fig: Path to the output figure. If None will display to screen.
    :param verbose: Set to True to print details on screen during coregistration.

    :returns: A tuple containing 1) coregistered DEM as an xdem.DEM instance 2) the coregistration method \
3) DataFrame of coregistration statistics (count of obs, median and NMAD over stable terrain) before and after \
coregistration and 4) the inlier_mask used.
    """
    # Check inputs
    if not isinstance(coreg_method, Coreg):
        raise ValueError("`coreg_method` must be an xdem.coreg instance (e.g. xdem.coreg.NuthKaab())")

    if isinstance(ref_dem_path, str):
        if not isinstance(src_dem_path, str):
            raise ValueError(
                f"`ref_dem_path` is string but `src_dem_path` has type {type(src_dem_path)}."
                "Both must have same type."
            )
    elif isinstance(ref_dem_path, gu.Raster):
        if not isinstance(src_dem_path, gu.Raster):
            raise ValueError(
                f"`ref_dem_path` is of Raster type but `src_dem_path` has type {type(src_dem_path)}."
                "Both must have same type."
            )
    else:
        raise ValueError("`ref_dem_path` must be either a string or a Raster")

    if grid not in ["ref", "src"]:
        raise ValueError(f"`grid` must be either 'ref' or 'src' - currently set to {grid}")

    # Load both DEMs
    if verbose:
        print("Loading and reprojecting input data")

    if isinstance(ref_dem_path, str):
        if grid == "ref":
            ref_dem, src_dem = gu.raster.load_multiple_rasters([ref_dem_path, src_dem_path], ref_grid=0)
        elif grid == "src":
            ref_dem, src_dem = gu.raster.load_multiple_rasters([ref_dem_path, src_dem_path], ref_grid=1)
    else:
        ref_dem = ref_dem_path
        src_dem = src_dem_path
        if grid == "ref":
            src_dem = src_dem.reproject(ref_dem, silent=True)
        elif grid == "src":
            ref_dem = ref_dem.reproject(src_dem, silent=True)

    # Convert to DEM instance with Float32 dtype
    # TODO: Could only convert types int into float, but any other float dtype should yield very similar results
    ref_dem = DEM(ref_dem.astype(np.float32))
    src_dem = DEM(src_dem.astype(np.float32))

    # Create raster mask
    if verbose:
        print("Creating mask of inlier pixels")

    inlier_mask = create_inlier_mask(
        src_dem,
        ref_dem,
        shp_list=shp_list,
        inout=inout,
        filtering=filtering,
        dh_max=dh_max,
        nmad_factor=nmad_factor,
        slope_lim=slope_lim,
    )

    # Calculate dDEM
    ddem = src_dem - ref_dem

    # Calculate dDEM statistics on pixels used for coreg
    inlier_data = ddem.data[inlier_mask].compressed()
    nstable_orig, mean_orig = len(inlier_data), np.mean(inlier_data)
    med_orig, nmad_orig = np.median(inlier_data), nmad(inlier_data)

    # Coregister to reference - Note: this will spread NaN
    coreg_method.fit(ref_dem, src_dem, inlier_mask, verbose=verbose)
    dem_coreg = coreg_method.apply(src_dem, resample=resample, resampling=resampling)

    # Calculate coregistered ddem (might need resampling if resample set to False), needed for stats and plot only
    ddem_coreg = dem_coreg.reproject(ref_dem, silent=True) - ref_dem

    # Calculate new stats
    inlier_data = ddem_coreg.data[inlier_mask].compressed()
    nstable_coreg, mean_coreg = len(inlier_data), np.mean(inlier_data)
    med_coreg, nmad_coreg = np.median(inlier_data), nmad(inlier_data)

    # Plot results
    if plot:
        # Max colorbar value - 98th percentile rounded to nearest 5
        vmax = np.percentile(np.abs(ddem.data.compressed()), 98) // 5 * 5

        plt.figure(figsize=(11, 5))

        ax1 = plt.subplot(121)
        plt.imshow(ddem.data.squeeze(), cmap="coolwarm_r", vmin=-vmax, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label("Elevation change (m)")
        ax1.set_title(f"Before coreg\n\nmean = {mean_orig:.1f} m - med = {med_orig:.1f} m - NMAD = {nmad_orig:.1f} m")

        ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
        plt.imshow(ddem_coreg.data.squeeze(), cmap="coolwarm_r", vmin=-vmax, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label("Elevation change (m)")
        ax2.set_title(
            f"After coreg\n\n\nmean = {mean_coreg:.1f} m - med = {med_coreg:.1f} m - NMAD = {nmad_coreg:.1f} m"
        )

        plt.tight_layout()
        if out_fig is None:
            plt.show()
        else:
            plt.savefig(out_fig, dpi=200)
            plt.close()

    # Save coregistered DEM
    if out_dem_path is not None:
        dem_coreg.save(out_dem_path, tiled=True)

    # Save stats to DataFrame
    out_stats = pd.DataFrame(
        ((nstable_orig, med_orig, nmad_orig, nstable_coreg, med_coreg, nmad_coreg),),
        columns=("nstable_orig", "med_orig", "nmad_orig", "nstable_coreg", "med_coreg", "nmad_coreg"),
    )

    return dem_coreg, coreg_method, out_stats, inlier_mask
