# Copyright (c) 2024 xDEM developers
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

"""Terrain attribute calculations, such as slope, aspect, hillshade, curvature and ruggedness indexes."""
from __future__ import annotations

import warnings
from typing import Any, Literal, Sized, overload

import geoutils as gu
import numba
import numpy as np
from geoutils.raster import Raster, RasterType
from geoutils.raster.distributed_computing import (
    MultiprocConfig,
    map_overlap_multiproc_save,
)
from scipy.ndimage import generic_filter

from xdem._typing import DTypeLike, MArrayf, NDArrayf

# List available attributes
available_attributes = [
    "slope",
    "aspect",
    "hillshade",
    "curvature",
    "planform_curvature",
    "profile_curvature",
    "maximum_curvature",
    "topographic_position_index",
    "terrain_ruggedness_index",
    "roughness",
    "rugosity",
    "fractal_roughness",
]

##############################################################################
# SURFACE FIT ATTRIBUTES: DEPENDENT ON FIT COEFFICIENTS IN A FIXED WINDOW SIZE
##############################################################################

# Store coefficient of fixed window-size attributes outside functions
# to allow reuse with several engines (Numba, SciPy, Cuda)

# Zevenberg and Thorne (1987) coefficients, Equations 3 to 11
#############################################################

# A, B, C and I are effectively unused for terrain attributes, only useful to get quadric fit
zt_a = np.array([[1 / 4, -1 / 2, 1 / 4], [-1 / 2, 1, -1 / 2], [1 / 4, -1 / 2, 1 / 4]])
zt_b = np.array([[-1 / 4, 0, 1 / 4], [1 / 2, 0, -1 / 2], [-1 / 4, 0, 1 / 4]])
zt_c = np.array([[1 / 4, -1 / 2, 1 / 4], [0, 0, 0], [-1 / 4, 1 / 2, -1 / 4]])
zt_i = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# All below useful for curvature
zt_d = np.array([[0, 1 / 2, 0], [0, -1, 0], [0, 1 / 2, 0]])
zt_e = np.array([[0, 0, 0], [1 / 2, -1, 1 / 2], [0, 0, 0]])
zt_f = np.array([[-1 / 4, 0, 1 / 4], [0, 0, 0], [1 / 4, 0, -1 / 4]])

# The G and H coefficients are the only ones needed for slope/aspect/hillshade
zt_g = np.array([[0, 1 / 2, 0], [0, 0, 0], [0, -1 / 2, 0]])
zt_h = np.array([[0, 0, 0], [-1 / 2, 0, 1 / 2], [0, 0, 0]])
zv_coefs = {
    "zt_a": zt_a,
    "zt_b": zt_b,
    "zt_c": zt_c,
    "zt_d": zt_d,
    "zt_e": zt_e,
    "zt_f": zt_f,
    "zt_g": zt_g,
    "zt_h": zt_h,
    "zt_i": zt_i,
}

# Horn (1981) coefficients, page 18 bottom left equations
#########################################################

h1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
h2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
horn_coefs = {"h1": h1, "h2": h2}

# Florinsky (2009) coefficients: ASK TOM?
#########################################

all_coefs = zv_coefs.copy()
all_coefs.update(horn_coefs)

# Dividers associated with coefficients
#######################################


def _divider_method_coef(res: float, coef: str) -> float:
    """Divider for a given coefficient based on resolution."""

    mapping_div_coef = {
        "zt_a": res**4,
        "zt_b": res**3,
        "zt_c": res**3,
        "zt_d": res**2,
        "zt_e": res**2,
        "zt_f": res**2,
        "zt_g": res,
        "zt_h": res,
        "zt_i": 1,
        "h1": 8 * res,
        "h2": 8 * res,
    }

    return mapping_div_coef[coef]


def _preprocess_surface_fit(
    surface_attributes: list[str], resolution: float, slope_method: Literal["Horn", "ZevenbergThorne"]
) -> tuple[list[NDArrayf], list[int], list[int], list[bool], int]:
    """
    Pre-processing for surface fit attributes.

    Determine the list of surface coefficients that need to be derived given input attributes, and map ordered indexes
    to be used to derive them through SciPy or Numba loop efficiently. (to minimize memory and CPU usage)

    Returns list of arrays with coefficients, list of indexes to map coefs and attributes, list of boolean to make a
    given attribute, and the size of the output attribute array.
    """

    # Step 1: Get coefficients needed for the list of attributes

    # For slope, aspect and hillshade, only 2 coefs depending on method
    if any(att in surface_attributes for att in ["slope", "aspect", "hillshade"]):
        if slope_method == "Horn":
            c_sah = ["h1", "h2"]
        elif slope_method == "ZevenbergThorne":
            c_sah = ["zt_g", "zt_h"]
    else:
        c_sah = []

    # For simple curvature, only 2 coefs needed
    if "curvature" in surface_attributes:
        c_curv = ["zt_d", "zt_e"]
    else:
        c_curv = []

    # For other curvature, 5 coefs needed
    if any(att in surface_attributes for att in ["planform_curvature", "profile_curvature", "maximum_curvature"]):
        c_pcurv = ["zt_d", "zt_e", "zt_f", "zt_g", "zt_h"]
    else:
        c_pcurv = []

    coef_names = list(set(c_sah + c_curv + c_pcurv))

    # Coefficient arrays take almost no memory, so we want the finest precision
    coef_arrs = [all_coefs[cn].astype(np.float64) for cn in coef_names]

    # Divide coefficients by associated resolution factors
    for i in range(len(coef_names)):
        coef_arrs[i] /= _divider_method_coef(res=resolution, coef=coef_names[i])

    # Step 2: Derive ordered indexes for attributes/coefs outside of SciPy/Numba processing for speed

    # Define booleans for generating attributes
    make_slope = "slope" in surface_attributes or "hillshade" in surface_attributes
    make_aspect = "aspect" in surface_attributes or "hillshade" in surface_attributes
    make_hillshade = "hillshade" in surface_attributes
    make_curvature = "curvature" in surface_attributes
    make_planform_curvature = "planform_curvature" in surface_attributes or "maximum_curvature" in surface_attributes
    make_profile_curvature = "profile_curvature" in surface_attributes or "maximum_curvature" in surface_attributes
    make_maximum_curvature = "maximum_curvature" in surface_attributes

    make_attrs = [
        make_slope,
        make_aspect,
        make_hillshade,
        make_curvature,
        make_planform_curvature,
        make_profile_curvature,
        make_maximum_curvature,
    ]

    # Map index of attributes and coefficients to defined order
    order_attrs = [
        "slope",
        "aspect",
        "hillshade",
        "curvature",
        "planform_curvature",
        "profile_curvature",
        "maximum_curvature",
    ]
    order_coefs = ["zt_a", "zt_b", "zt_c", "zt_d", "zt_e", "zt_f", "zt_g", "zt_h", "zt_i", "h1", "h2"]

    idx_attrs = [surface_attributes.index(oa) if oa in surface_attributes else 99 for oa in order_attrs]
    idx_coefs = [coef_names.index(oc) if oc in coef_names else 99 for oc in order_coefs]

    # Because of the above indexes, we don't store the length of the output attributes anymore
    attrs_size = len(surface_attributes)

    return coef_arrs, idx_coefs, idx_attrs, make_attrs, attrs_size


def _make_attribute_from_coefs(
    coef_arrs: NDArrayf,
    h1_idx: int,
    h2_idx: int,
    zt_d_idx: int,
    zt_e_idx: int,
    zt_f_idx: int,
    zt_g_idx: int,
    zt_h_idx: int,
    slope_idx: int,
    aspect_idx: int,
    hs_idx: int,
    curv_idx: int,
    plancurv_idx: int,
    profcurv_idx: int,
    maxcurv_idx: int,
    make_attrs: list[bool],
    out_size: tuple[int, ...],
    slope_method_id: int,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    out_dtype: DTypeLike = np.float32,
) -> NDArrayf:
    """
    Compute surface attributes given coefficients, either on N-D arrays (for output of SciPy convolve) or along
    1D array (to use within Numba function).

    The coefficient names, surfaces attributes and slope method were pre-computed into integers or indexes to enable
    fast computation within Numba loops.
    """

    # Indexes of attributes and coefficients are already mapped to the same indexes to avoid solving outside Numba loop

    # For surface attributes
    # slope: 0,
    # aspect: 1,
    # hillshade: 2,
    # curvature: 3,
    # planform_curvature: 4,
    # profile_curvature: 5,
    # maximum_curvature: 6

    # For methods
    # horn: 0
    # zevenbergthorne: 1

    # For coefficients names
    # zt_a: 0
    # zt_b: 1
    # zt_c: 2
    # zt_d: 3
    # zt_e: 4
    # zt_f: 5
    # zt_g: 6
    # zt_h: 7
    # zt_i: 8
    # h1: 9
    # h2: 10

    C = coef_arrs

    attrs = np.full(out_size, fill_value=np.nan, dtype=out_dtype)

    # Extract conditions for making the various attributes (see mapping to integers above)
    (
        make_slope,
        make_aspect,
        make_hillshade,
        make_curvature,
        make_planform_curvature,
        make_profile_curvature,
        make_maximum_curvature,
    ) = make_attrs

    if make_slope:

        if slope_method_id == 0:

            # This calculation is based on page 18 (bottom left) and 20-21 of Horn (1981),
            # http://dx.doi.org/10.1109/PROC.1981.11918.
            slope = np.arctan((C[h1_idx] ** 2 + C[h2_idx] ** 2) ** 0.5)

        elif slope_method_id == 1:

            # This calculation is based on Equation 13 of Zevenbergen and Thorne (1987),
            # http://dx.doi.org/10.1002/esp.3290120107.
            # SLOPE = ARCTAN((G²+H²)**(1/2))
            slope = np.arctan((C[zt_g_idx] ** 2 + C[zt_h_idx] ** 2) ** 0.5)

        # In case slope is only derived for hillshade
        if slope_idx != 99:
            attrs[slope_idx] = slope

    if make_aspect:

        # ASPECT = ARCTAN(-H/-G)  # This did not work
        # ASPECT = (ARCTAN2(-G, H) + 0.5PI) % 2PI  did work.
        if slope_method_id == 0:

            # This uses the estimates from Horn (1981).
            aspect = (-np.arctan2(-C[h1_idx], C[h2_idx]) - np.pi) % (2 * np.pi)

        elif slope_method_id == 1:

            # This uses the estimate from Zevenbergen and Thorne (1987).
            aspect = (np.arctan2(-C[zt_g_idx], C[zt_h_idx]) + np.pi / 2) % (2 * np.pi)

        # In case aspect is only derived for hillshade
        if aspect_idx != 99:
            attrs[aspect_idx] = aspect

    if make_hillshade:

        # If a different z-factor was given, slopemap with exaggerated gradients.
        if hillshade_z_factor != 1.0:
            slopemap = np.arctan(np.tan(slope) * hillshade_z_factor)
        else:
            slopemap = slope

        azimuth_rad = np.deg2rad(360 - hillshade_azimuth)
        altitude_rad = np.deg2rad(hillshade_altitude)

        # The operation below yielded the closest hillshade to GDAL (multiplying by 255 did not work)
        # As 0 is generally no data for this uint8, we add 1 and then 0.5 for the rounding to occur between 1 and 255
        attrs[hs_idx] = 1.5 + 254 * (
            np.sin(altitude_rad) * np.cos(slopemap)
            + np.cos(altitude_rad) * np.sin(slopemap) * np.sin(azimuth_rad - aspect)
        )

    if make_curvature:

        # Curvature is the second derivative of the surface fit equation.
        # (URL in get_quadric_coefficients() docstring)
        # Curvature = -2(D + E) * 100, see Moore et al. (1991) Equation 16 based on Zevenberg and Thorne (1987)
        attrs[curv_idx] = -2.0 * (C[zt_d_idx] + C[zt_e_idx]) * 100

    if make_planform_curvature:

        # PLANC = 2(DH² + EG² -FGH)/(G²+H²)
        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output
        plancurv = np.where(
            C[zt_g_idx] ** 2 + C[zt_h_idx] ** 2 == 0.0,
            np.array([0.0]),
            -2
            * (
                C[zt_d_idx] * C[zt_h_idx] ** 2
                + C[zt_e_idx] * C[zt_g_idx] ** 2
                - C[zt_f_idx] * C[zt_g_idx] * C[zt_h_idx]
            )
            / (C[zt_g_idx] ** 2 + C[zt_h_idx] ** 2)
            * 100,
        )

        # In case plan curv is only derived for max curv
        if plancurv_idx != 99:
            attrs[plancurv_idx] = plancurv

    if make_profile_curvature:

        # PROFC = -2(DG² + EH² + FGH)/(G²+H²)
        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output
        profcurv = np.where(
            C[zt_g_idx] ** 2 + C[zt_h_idx] ** 2 == 0.0,
            np.array([0.0]),
            2
            * (
                C[zt_d_idx] * C[zt_g_idx] ** 2
                + C[zt_e_idx] * C[zt_h_idx] ** 2
                + C[zt_f_idx] * C[zt_g_idx] * C[zt_h_idx]
            )
            / (C[zt_g_idx] ** 2 + C[zt_h_idx] ** 2)
            * 100,
        )

        # In case profile curv is only derived for max curv
        if profcurv_idx != 99:
            attrs[profcurv_idx] = profcurv

    if make_maximum_curvature:

        minc = np.minimum(plancurv, profcurv)
        maxc = np.maximum(plancurv, profcurv)
        attrs[maxcurv_idx] = np.where(np.abs(minc) > maxc, minc, maxc)

    return attrs


@numba.njit(inline="always", cache=True)  # type: ignore
def _convolution_numba(
    dem: NDArrayf, filters: NDArrayf, row: int, col: int, out_dtype: DTypeLike = np.float32
) -> NDArrayf:
    """Convolution in Numba for a given row/col pixel."""

    n_M, M1, M2 = filters.shape

    # Compute coefficients from convolution
    coefs = np.zeros((n_M,), dtype=out_dtype)
    for m1 in range(M1):
        for m2 in range(M2):
            for ff in range(n_M):
                imgval = dem[row + m1, col + m2]
                filterval = filters[ff, -(m1 + 1), -(m2 + 1)]
                coefs[ff] += imgval * filterval

    return coefs


# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_make_attribute_from_coefs_numba = numba.njit(inline="always", cache=True)(_make_attribute_from_coefs)


@numba.njit(parallel=True, cache=True)  # type: ignore
def _get_surface_attributes_numba(
    dem: NDArrayf,
    filters: NDArrayf,
    make_attrs: list[bool],
    idx_coefs: list[int],
    idx_attrs: list[int],
    attrs_size: int,
    out_dtype: DTypeLike,
    slope_method_id: int = 0,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
) -> NDArrayf:
    """
    Run the pixel-wise analysis in parallel for a 3x3 window using the resolution.

    See the xdem.terrain.get_quadric_coefficients() docstring for more info.
    """

    # Get input shapes
    N1, N2 = dem.shape
    n_M, M1, M2 = filters.shape

    # This ugly unpacking outside the loop is required for a Numba speed-up by a factor of 10
    zt_d_idx = idx_coefs[3]
    zt_e_idx = idx_coefs[4]
    zt_f_idx = idx_coefs[5]
    zt_g_idx = idx_coefs[6]
    zt_h_idx = idx_coefs[7]
    h1_idx = idx_coefs[9]
    h2_idx = idx_coefs[10]
    slope_idx, aspect_idx, hs_idx, curv_idx, plancurv_idx, profcurv_idx, maxcurv_idx = idx_attrs

    # Define ranges to loop through given padding
    row_range = N1 - M1 + 1
    col_range = N2 - M2 + 1

    # Allocate output array
    outputs = np.full((attrs_size, row_range, col_range), fill_value=np.nan, dtype=out_dtype)

    # Loop over every pixel concurrently by using prange
    for row in numba.prange(row_range):
        for col in numba.prange(col_range):

            # Compute coefficients from convolution
            coefs = _convolution_numba(dem, filters, row, col, out_dtype=np.float64)

            # Synthesize coefficients into attributes
            attrs = _make_attribute_from_coefs_numba(
                coef_arrs=coefs,
                make_attrs=make_attrs,
                h1_idx=h1_idx,
                h2_idx=h2_idx,
                zt_d_idx=zt_d_idx,
                zt_e_idx=zt_e_idx,
                zt_f_idx=zt_f_idx,
                zt_g_idx=zt_g_idx,
                zt_h_idx=zt_h_idx,
                slope_idx=slope_idx,
                aspect_idx=aspect_idx,
                hs_idx=hs_idx,
                curv_idx=curv_idx,
                plancurv_idx=plancurv_idx,
                profcurv_idx=profcurv_idx,
                maxcurv_idx=maxcurv_idx,
                out_size=(attrs_size, 1),  # 2-d required for np.where inside func
                slope_method_id=slope_method_id,
                out_dtype=np.float64,
                hillshade_azimuth=hillshade_azimuth,
                hillshade_altitude=hillshade_altitude,
                hillshade_z_factor=hillshade_z_factor,
            )

            # Save output for this pixel
            outputs[:, row, col] = attrs[:, 0]  # Squeeze extra dimension of last axis

    return outputs


def _get_surface_attributes_scipy(
    dem: NDArrayf,
    filters: NDArrayf,
    make_attrs: list[bool],
    idx_coefs: list[int],
    idx_attrs: list[int],
    slope_method_id: int,
    attrs_size: int,
    out_dtype: DTypeLike = np.float32,
    **kwargs: Any,
) -> NDArrayf:

    # Perform convolution and squeeze output into 3D array
    from xdem.spatialstats import convolution

    coefs = convolution(imgs=dem.reshape((1, dem.shape[0], dem.shape[1])), filters=filters, method="scipy").squeeze()

    # Convert coefficients to attributes
    out_size = (attrs_size, dem.shape[0], dem.shape[1])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in remainder")

        # This ugly unpacking outside the function is required for a Numba speed-up by a factor of 10
        # (and SciPy uses the same function, so requires the arguments as well)
        zt_d_idx = idx_coefs[3]
        zt_e_idx = idx_coefs[4]
        zt_f_idx = idx_coefs[5]
        zt_g_idx = idx_coefs[6]
        zt_h_idx = idx_coefs[7]
        h1_idx = idx_coefs[9]
        h2_idx = idx_coefs[10]
        slope_idx, aspect_idx, hs_idx, curv_idx, plancurv_idx, profcurv_idx, maxcurv_idx = idx_attrs

        attrs = _make_attribute_from_coefs(
            coef_arrs=coefs,
            make_attrs=make_attrs,
            h1_idx=h1_idx,
            h2_idx=h2_idx,
            zt_d_idx=zt_d_idx,
            zt_e_idx=zt_e_idx,
            zt_f_idx=zt_f_idx,
            zt_g_idx=zt_g_idx,
            zt_h_idx=zt_h_idx,
            slope_idx=slope_idx,
            aspect_idx=aspect_idx,
            hs_idx=hs_idx,
            curv_idx=curv_idx,
            plancurv_idx=plancurv_idx,
            profcurv_idx=profcurv_idx,
            maxcurv_idx=maxcurv_idx,
            out_size=out_size,
            slope_method_id=slope_method_id,
            out_dtype=out_dtype,
            **kwargs,
        )

    return attrs


def _get_surface_attributes(
    dem: NDArrayf,
    resolution: float,
    surface_attributes: list[str],
    out_dtype: DTypeLike = np.float32,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    engine: Literal["scipy", "numba"] = "scipy",
    **kwargs: Any,
) -> NDArrayf:
    """
    Get surface attributes based on fit coefficients (quadric, quintic, etc) using SciPy or Numba convolution and
    reducer functions.

    - Slope, aspect and hillshade from Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918, page 18 bottom left
      equations computed on a 3x3 window.
    - Slope, aspect, hillshade and curvatures from Zevenbergen and Thorne (1987),
      http://dx.doi.org/10.1002/esp.3290120107 also computed on a 3x3 window.

    :param dem: Input DEM as 2D array.
    :param resolution: Resolution of the DEM (X and Y length are equal).
    :param surface_attributes: Names of surface attributes to compute.
    :param out_dtype: Output dtype of the terrain attributes, can only be a floating type. Defaults to that of the
        input DEM if floating type or to float32 if integer type.
    :param slope_method: Method for the slope, aspect and hillshade ("Horn" or "ZevenbergThorne").
    :param engine: Engine to compute the surface attributes ("scipy" or "numba").
    """

    # Get list of necessary coefficients depending on method and resolution
    coef_arrs, idx_coefs, idx_attrs, make_attrs, attrs_size = _preprocess_surface_fit(
        surface_attributes=surface_attributes, resolution=resolution, slope_method=slope_method
    )

    # Stack coefficients into a 3D convolution kernel along the first axis
    kern3d = np.stack(coef_arrs, axis=0)

    # Map slope method to integer ID to improve efficiency in Numba loop
    slope_method_id = 0 if slope_method.lower() == "horn" else 1

    # Run convolution to compute all coefficients, then reduce those to attributes through either SciPy or Numba
    # (For Numba: Reduction is done within loop to reduce memory usage of computing dozens of full-array coefficients)
    if engine == "scipy":
        output = _get_surface_attributes_scipy(
            dem=dem,
            filters=kern3d,
            idx_coefs=idx_coefs,
            idx_attrs=idx_attrs,
            make_attrs=make_attrs,
            slope_method_id=slope_method_id,
            attrs_size=attrs_size,
            out_dtype=out_dtype,
            **kwargs,
        )
    elif engine == "numba":
        _, M1, M2 = kern3d.shape
        half_M1 = int((M1 - 1) / 2)
        half_M2 = int((M2 - 1) / 2)
        dem = np.pad(dem, pad_width=((half_M1, half_M1), (half_M2, half_M2)), constant_values=np.nan)
        # Now required to declare list typing in latest Numba before deprecation
        typed_make_attrs, typed_idx_attrs, typed_idx_coefs = numba.typed.List(), numba.typed.List(), numba.typed.List()
        [typed_make_attrs.append(x) for x in make_attrs]
        [typed_idx_attrs.append(x) for x in idx_attrs]
        [typed_idx_coefs.append(x) for x in idx_coefs]
        output = _get_surface_attributes_numba(
            dem=dem,
            filters=kern3d,
            make_attrs=typed_make_attrs,
            idx_coefs=typed_idx_coefs,
            idx_attrs=typed_idx_attrs,
            attrs_size=attrs_size,
            out_dtype=out_dtype,
            slope_method_id=slope_method_id,
            **kwargs,
        )

    return output


############################################################################################
# WINDOWED INDEXES: ATTRIBUTES INDEPENDENT OF EACH OTHER WITH VARYING WINDOW SIZE (=FILTERS)
############################################################################################


def _tri_riley_func(arr: NDArrayf) -> float:
    """
    Terrain Ruggedness Index from Riley et al. (1999): squareroot of squared sum of differences between center and
    neighbouring pixels.
    """
    mid_ind = int(arr.shape[0] / 2)
    diff = np.abs(arr - arr[mid_ind])
    return np.sqrt(np.sum(diff**2))


def _tri_wilson_func(arr: NDArrayf, window_size: int) -> float:
    """Terrain Ruggedness Index from Wilson et al. (2007): mean difference between center and neighbouring pixels."""
    mid_ind = int(arr.shape[0] / 2)
    diff = np.abs(arr - arr[mid_ind])
    return np.sum(diff) / (window_size**2 - 1)


def _tpi_func(arr: NDArrayf, window_size: int) -> float:
    """Topographic Position Index from Weiss (2001): difference between center and mean of neighbouring pixels."""
    mid_ind = int(arr.shape[0] / 2)
    return arr[mid_ind] - (np.sum(arr) - arr[mid_ind]) / (window_size**2 - 1)


def _roughness_func(arr: NDArrayf) -> float:
    """Roughness from Dartnell (2000): difference between maximum and minimum of the window."""
    if np.count_nonzero(np.isnan(arr)) > 0:
        return float("nan")  # This is somehow necessary for Numba to not ignore NaNs
    else:
        return float(np.max(arr) - np.min(arr))


def _fractal_roughness_func(arr: NDArrayf, window_size: int, out_dtype: DTypeLike = np.float32) -> float:
    """Fractal roughness according to the box-counting method of Taud and Parrot (2005)."""

    # First, we compute the number of voxels for each pixel of Equation 4
    mid_ind = int(np.floor(arr.shape[0] / 2))
    hw = int(np.floor(window_size / 2))
    mid_val = arr[mid_ind]

    count = 0
    V = np.empty((window_size, window_size), dtype=out_dtype)
    for j in range(window_size):
        for k in range(window_size):
            T = arr[window_size * j + k] - mid_val
            # The following is the equivalent of np.clip, written like this for numba
            if T < 0:
                V[j, k] = 0
            elif T > window_size:
                V[j, k] = window_size
            else:
                V[j, k] = T
            count += 1

    # Then, we compute the maximum number of voxels for varying box splitting of the cube of side the window
    # size, following Equation 5

    # Get all the divisors of the half window size
    list_box_sizes = np.zeros((hw,), dtype=np.uint8)
    for j in range(1, hw + 1):
        if hw % j == 0:
            list_box_sizes[j - 1] = j

    valids = list_box_sizes != 0
    sub_list_box_sizes = list_box_sizes[valids]

    Ns = np.empty((len(sub_list_box_sizes),), dtype=out_dtype)
    for l0 in range(0, len(sub_list_box_sizes)):
        # We loop over boxes of size q x q in the cube
        q = sub_list_box_sizes[l0]
        sumNs = 0
        for j in range(0, int((window_size - 1) / q)):
            for k in range(0, int((window_size - 1) / q)):
                sumNs += np.max(V[slice(j * q, (j + 1) * q), slice(k * q, (k + 1) * q)].flatten())
        Ns[l0] = sumNs / q

    # Finally, we calculate the slope of the logarithm of Ns with q
    # We do the linear regression manually, as np.polyfit is not supported by numba
    x = np.log(sub_list_box_sizes)
    y = np.log(Ns)
    # The number of observations
    n = len(x)
    # Mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    # Cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    # Calculating slope
    b_1 = SS_xy / SS_xx

    # The fractal dimension D is the opposite of the slope
    D = -b_1

    return D


def _rugosity_func(arr: NDArrayf, resolution: float, out_dtype: DTypeLike = np.float32) -> float:
    """
    Rugosity from Jenness (2004): difference between real surface area and planimetric surface area.

    The below computation only works for a 3x3 array, would need more effort to generalize it.
    """

    # Works only on a 3x3 block
    Z = arr
    L = resolution

    # Rugosity is computed on a 3x3 window like the quadratic coefficients, see Jenness (2004) for details

    # For this, we need elevation differences and horizontal length of 16 segments
    dzs = np.zeros((16,), dtype=out_dtype)
    dls = np.zeros((16,), dtype=out_dtype)

    count_without_center = 0
    count_all = 0
    # First, the 8 connected segments from the center cells, the center cell is index 4
    for j in range(-1, 2):
        for k in range(-1, 2):

            # Skip if this is the center pixel
            if j == 0 and k == 0:
                count_all += 1
                continue
            # The first eight elevation differences from the cell center
            dzs[count_without_center] = Z[4] - Z[count_all]
            # The first eight planimetric length that can be diagonal or straight from the center
            dls[count_without_center] = np.sqrt(j**2 + k**2) * L
            count_all += 1
            count_without_center += 1

    # Manually for the remaining eight segments between surrounding pixels:
    # First, four elevation differences along the x axis
    dzs[8] = Z[0] - Z[1]
    dzs[9] = Z[1] - Z[2]
    dzs[10] = Z[6] - Z[7]
    dzs[11] = Z[7] - Z[8]
    # Second, along the y axis
    dzs[12] = Z[0] - Z[3]
    dzs[13] = Z[3] - Z[6]
    dzs[14] = Z[2] - Z[5]
    dzs[15] = Z[5] - Z[8]
    # For the planimetric lengths, all are equal to one
    dls[8:] = L

    # Finally, the half-surface length of each segment
    hsl = np.sqrt(dzs**2 + dls**2) / 2

    # Starting from up direction anticlockwise, every triangle has 2 segments between center and surrounding
    # pixels and 1 segment between surrounding pixels; pixel 4 is the center
    # above 4 the index of center-surrounding segment decrease by 1, as the center pixel was skipped
    # Triangle 1: pixels 3 and 0
    T1 = (hsl[3], hsl[0], hsl[12])
    # Triangle 2: pixels 0 and 1
    T2 = (hsl[0], hsl[1], hsl[8])
    # Triangle 3: pixels 1 and 2
    T3 = (hsl[1], hsl[2], hsl[9])
    # Triangle 4: pixels 2 and 5
    T4 = (hsl[2], hsl[4], hsl[14])
    # Triangle 5: pixels 5 and 8
    T5 = (hsl[4], hsl[7], hsl[15])
    # Triangle 6: pixels 8 and 7
    T6 = (hsl[7], hsl[6], hsl[11])
    # Triangle 7: pixels 7 and 6
    T7 = (hsl[6], hsl[5], hsl[10])
    # Triangle 8: pixels 6 and 3
    T8 = (hsl[5], hsl[3], hsl[13])

    list_T = [T1, T2, T3, T4, T5, T6, T7, T8]

    # Finally, we compute the 3D surface areas of the 8 triangles
    A = np.empty((8,), dtype=out_dtype)
    count = 0
    for T in list_T:
        # Half sum of lengths
        hs = sum(T) / 2
        # Surface area of triangle
        A[count] = np.sqrt(hs * (hs - T[0]) * (hs - T[1]) * (hs - T[2]))
        count += 1

    rug = sum(A) / L**2

    return rug


# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_tpi_func_numba = numba.njit(inline="always", cache=True)(_tpi_func)
_tri_riley_func_numba = numba.njit(inline="always", cache=True)(_tri_riley_func)
_tri_wilson_func_numba = numba.njit(inline="always", cache=True)(_tri_wilson_func)
_roughness_func_numba = numba.njit(inline="always", cache=True)(_roughness_func)
_rugosity_func_numba = numba.njit(inline="always", cache=True)(_rugosity_func)
_fractal_roughness_func_numba = numba.njit(inline="always", cache=True)(_fractal_roughness_func)


def _preprocess_windowed_indexes(windowed_indexes: list[str]) -> tuple[list[int], list[bool], int]:
    """
    Pre-processing for windowed indexes.

    Map ordered indexes to be used to derive them through SciPy or Numba loop efficiently. (to minimize memory and CPU
    usage)

    Returns list of indexes to map attributes, list of booleans to make attributes, and the size of the output
    attribute array.
    """

    # Step 2: Derive ordered indexes for attributes/coefs outside of SciPy/Numba processing for speed

    # Define booleans for generating attributes
    make_tpi = "topographic_position_index" in windowed_indexes
    make_tri = "terrain_ruggedness_index" in windowed_indexes
    make_roughness = "roughness" in windowed_indexes
    make_rugosity = "rugosity" in windowed_indexes
    make_fractal_roughness = "fractal_roughness" in windowed_indexes

    make_attrs = [make_tpi, make_tri, make_roughness, make_rugosity, make_fractal_roughness]

    # Map index of attributes and coefficients to defined order
    order_attrs = [
        "topographic_position_index",
        "terrain_ruggedness_index",
        "roughness",
        "rugosity",
        "fractal_roughness",
    ]
    idx_attrs = [windowed_indexes.index(oa) if oa in windowed_indexes else 99 for oa in order_attrs]

    # Because of the above indexes, we don't store the length of the output attributes anymore
    attrs_size = len(windowed_indexes)

    return idx_attrs, make_attrs, attrs_size


@numba.njit(inline="always", cache=True)  # type: ignore
def _make_windowed_indexes(
    dem_window: NDArrayf,
    window_size: int,
    resolution: float,
    make_attrs: list[bool],
    tpi_idx: int,
    tri_idx: int,
    roughness_idx: int,
    rugosity_idx: int,
    frac_roughness_idx: int,
    tri_method_id: int,
    out_size: tuple[int, ...],
    out_dtype: DTypeLike,
) -> NDArrayf:

    attrs = np.full(out_size, fill_value=np.nan, dtype=out_dtype)

    make_tpi, make_tri, make_roughness, make_rugosity, make_fractal_roughness = make_attrs

    # Topographic position index
    if make_tpi:

        attrs[tpi_idx] = _tpi_func_numba(dem_window, window_size=window_size)

    if make_tri:

        if tri_method_id == 0:
            attrs[tri_idx] = _tri_riley_func_numba(dem_window)

        elif tri_method_id == 1:
            attrs[tri_idx] = _tri_wilson_func_numba(dem_window, window_size=window_size)

    if make_roughness:

        attrs[roughness_idx] = _roughness_func_numba(dem_window)

    if make_rugosity:

        attrs[rugosity_idx] = _rugosity_func_numba(dem_window, resolution=resolution, out_dtype=out_dtype)

    if make_fractal_roughness:

        attrs[frac_roughness_idx] = _fractal_roughness_func_numba(
            dem_window, window_size=window_size, out_dtype=out_dtype
        )

    return attrs


@numba.njit(parallel=True, cache=True)  # type: ignore
def _get_windowed_indexes_numba(
    dem: NDArrayf,
    window_size: int,
    resolution: float,
    out_dtype: DTypeLike,
    attrs_size: int,
    make_attrs: list[bool],
    idx_attrs: list[int],
    tri_method_id: int,
) -> NDArrayf:
    """
    Run the pixel-wise analysis in parallel for any window size without using the resolution.

    See the xdem.terrain.get_windowed_indexes() docstring for more info.
    """

    # Get input shapes
    N1, N2 = dem.shape

    # Define ranges to loop through given padding
    row_range = N1 - window_size + 1
    col_range = N2 - window_size + 1

    # Ugly unpacking as integers outside loop required for Numba to speed-up
    tpi_idx, tri_idx, roughness_idx, rugosity_idx, frac_roughness_idx = idx_attrs

    # Allocate output array
    outputs = np.full((attrs_size, row_range, col_range), fill_value=np.nan, dtype=out_dtype)

    # Loop over every pixel concurrently by using prange
    for row in numba.prange(row_range):
        for col in numba.prange(col_range):

            dem_window = dem[row : row + window_size, col : col + window_size].flatten()
            out_size = (attrs_size,)
            attrs = _make_windowed_indexes(
                dem_window,
                window_size=window_size,
                resolution=resolution,
                make_attrs=make_attrs,
                tpi_idx=tpi_idx,
                tri_idx=tri_idx,
                roughness_idx=roughness_idx,
                rugosity_idx=rugosity_idx,
                frac_roughness_idx=frac_roughness_idx,
                tri_method_id=tri_method_id,
                out_size=out_size,
                out_dtype=out_dtype,
            )

            outputs[:, row, col] = attrs

    return outputs


def _get_windowed_indexes_scipy(
    dem: NDArrayf,
    window_size: int,
    resolution: float,
    make_attrs: list[bool],
    idx_attrs: list[int],
    tri_method_id: int,
    attrs_size: int,
    out_dtype: DTypeLike = np.float32,
) -> NDArrayf:

    outputs = np.full((attrs_size, dem.shape[0], dem.shape[1]), fill_value=np.nan, dtype=out_dtype)

    make_tpi, make_tri, make_roughness, make_rugosity, make_fractal_roughness = make_attrs

    # Topographic position index
    if make_tpi:
        tpi_idx = idx_attrs[0]
        outputs[tpi_idx] = generic_filter(
            dem, _tpi_func, mode="constant", size=window_size, cval=np.nan, extra_arguments=(window_size,)
        )

    if make_tri:

        tri_idx = idx_attrs[1]
        if tri_method_id == 0:
            outputs[tri_idx] = generic_filter(dem, _tri_riley_func, mode="constant", size=window_size, cval=np.nan)

        elif tri_method_id == 1:
            outputs[tri_idx] = generic_filter(
                dem, _tri_wilson_func, mode="constant", size=window_size, cval=np.nan, extra_arguments=(window_size,)
            )

    if make_roughness:
        roughness_idx = idx_attrs[2]
        outputs[roughness_idx] = generic_filter(dem, _roughness_func, mode="constant", size=window_size, cval=np.nan)

    if make_rugosity:
        rugosity_idx = idx_attrs[3]
        outputs[rugosity_idx] = generic_filter(
            dem, _rugosity_func, mode="constant", size=window_size, cval=np.nan, extra_arguments=(resolution, out_dtype)
        )

    if make_fractal_roughness:
        frac_roughness_idx = idx_attrs[4]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice.")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
            outputs[frac_roughness_idx] = generic_filter(
                dem,
                _fractal_roughness_func,
                mode="constant",
                size=window_size,
                cval=np.nan,
                extra_arguments=(window_size, out_dtype),
            )

    return outputs


def _get_windowed_indexes(
    dem: NDArrayf,
    window_size: int,
    windowed_indexes: list[str],
    resolution: float,
    out_dtype: DTypeLike = np.float32,
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf:
    """
    Derive windowed terrain indexes using SciPy or Numba based on a windowed calculation of variable size.

    Includes:

    - Terrain Ruggedness Index from Riley et al. (1999),
        http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf, for topography and from Wilson
        et al. (2007), http://dx.doi.org/10.1080/01490410701295962, for bathymetry.
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Roughness from Dartnell (2000), thesis referenced in Wilson et al. (2007) above.
    - Fractal roughness from Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    Nearly all methods are also referenced in Wilson et al. (2007).

    :param dem: Input DEM as 2D array.
    :param window_size: Window size to compute the index.
    :param windowed_indexes: Names of windowed indexes to compute.
    :param out_dtype: Output dtype of the terrain attributes, can only be a floating type. Defaults to that of the
        input DEM if floating type or to float32 if integer type.
    :param tri_method: Method for the terrain ruggedness index ("Riley" or "Wilson").
    :param engine: Engine to compute the windowed indexes ("scipy" or "numba").
    """

    # Get list of necessary coefficients depending on method and resolution
    idx_attrs, make_attrs, attrs_size = _preprocess_windowed_indexes(windowed_indexes=windowed_indexes)

    # Map slope method to integer ID to improve efficiency in Numba loop
    tri_method_id = 0 if tri_method.lower() == "riley" else 1

    # Run convolution to compute all coefficients, then reduce those to attributes through either SciPy or Numba
    # (For Numba: Reduction is done within loop to reduce memory usage of computing dozens of full-array coefficients)
    if engine == "scipy":
        output = _get_windowed_indexes_scipy(
            dem=dem,
            window_size=window_size,
            resolution=resolution,
            idx_attrs=idx_attrs,
            make_attrs=make_attrs,
            tri_method_id=tri_method_id,
            attrs_size=attrs_size,
            out_dtype=out_dtype,
        )
    elif engine == "numba":
        hw = int((window_size - 1) / 2)
        dem = np.pad(dem, pad_width=((hw, hw), (hw, hw)), constant_values=np.nan)
        # Now required to declare list typing in latest Numba before deprecation
        typed_make_attrs, typed_idx_attrs = numba.typed.List(), numba.typed.List()
        [typed_make_attrs.append(x) for x in make_attrs]
        [typed_idx_attrs.append(x) for x in idx_attrs]
        output = _get_windowed_indexes_numba(
            dem=dem,
            window_size=window_size,
            resolution=resolution,
            make_attrs=typed_make_attrs,
            idx_attrs=typed_idx_attrs,
            attrs_size=attrs_size,
            out_dtype=out_dtype,
            tri_method_id=tri_method_id,
        )

    return output


@overload
def get_terrain_attribute(
    dem: NDArrayf | MArrayf,
    attribute: str,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def get_terrain_attribute(
    dem: NDArrayf | MArrayf,
    attribute: list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> list[NDArrayf]: ...


@overload
def get_terrain_attribute(
    dem: RasterType,
    attribute: list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> list[RasterType]: ...


@overload
def get_terrain_attribute(
    dem: RasterType,
    attribute: str,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def get_terrain_attribute(
    dem: NDArrayf | MArrayf | RasterType,
    attribute: str | list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | list[NDArrayf] | RasterType | list[RasterType]:
    """
    Derive one or multiple terrain attributes from a DEM.
    The attributes are based on:

    - Slope, aspect, hillshade (first method) from Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918,
    - Slope, aspect, hillshade (second method), and terrain curvatures from Zevenbergen and Thorne (1987),
        http://dx.doi.org/10.1002/esp.3290120107, with curvature expanded in Moore et al. (1991),
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Terrain Ruggedness Index (topography) from Riley et al. (1999),
        http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf.
    - Terrain Ruggedness Index (bathymetry) from Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962.
    - Roughness from Dartnell (2000), thesis referenced in Wilson et al. (2007) above.
    - Rugosity from Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.
    - Fractal roughness from Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    Aspect and hillshade are derived using the slope, and thus depend on the same method.
    More details on the equations in the functions get_quadric_coefficients() and get_windowed_indexes().

    This function can be run out-of-memory in multiprocessing by passing a Multiproc config argument.

    Attributes:

    * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical). Default method: "Horn".
    * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W).
    * 'hillshade': The shaded slope in relation to its aspect.
    * 'curvature': The second derivative of elevation (the rate of slope change per pixel), multiplied by 100.
    * 'planform_curvature': The curvature perpendicular to the direction of the slope, multiplied by 100.
    * 'profile_curvature': The curvature parallel to the direction of the slope, multiplied by 100.
    * 'maximum_curvature': The maximum curvature.
    * 'surface_fit': A quadric surface fit for each individual pixel.
    * 'topographic_position_index': The topographic position index defined by a difference to the average of
        neighbouring pixels.
    * 'terrain_ruggedness_index': The terrain ruggedness index. For topography, defined by the squareroot of squared
        differences to neighbouring pixels. For bathymetry, defined by the mean absolute difference to neighbouring
        pixels. Default method: "Riley" (topography).
    * 'roughness': The roughness, i.e. maximum difference between neighbouring pixels.
    * 'rugosity': The rugosity, i.e. difference between real and planimetric surface area.
    * 'fractal_roughness': The roughness based on a volume box-counting estimate of the fractal dimension.

    :param dem: Input DEM.
    :param attribute: Terrain attribute(s) to calculate.
    :param resolution: Resolution of the DEM.
    :param degrees: Whether to convert radians to degrees.
    :param hillshade_altitude: Shading altitude in degrees (0-90°). 90° is straight from above.
    :param hillshade_azimuth: Shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param hillshade_z_factor: Vertical exaggeration factor.
    :param slope_method: Method to calculate the slope, aspect and hillshade: "Horn" or "ZevenbergThorne".
    :param tri_method: Method to calculate the Terrain Ruggedness Index: "Riley" (topography) or "Wilson" (bathymetry).
    :param window_size: Window size for windowed ruggedness and roughness indexes.
    :param engine: Engine to use for computing the attributes with convolution or other windowed calculations, currently
        supports "scipy" or "numba".
    :param out_dtype: Output dtype of the terrain attributes, can only be a floating type. Defaults to that of the
        input DEM if floating type or to float32 if integer type.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted or are invalid.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3)[::-1].reshape(3, 3)
        >>> dem
        array([[2, 2, 2],
               [1, 1, 1],
               [0, 0, 0]])
        >>> slope, aspect = get_terrain_attribute(dem, ["slope", "aspect"], resolution=1)
        >>> slope[1, 1]
        np.float32(45.0)
        >>> aspect[1, 1]
        np.float32(180.0)

    :returns: One or multiple arrays of the requested attribute(s)
    """
    if mp_config is not None:
        if not isinstance(dem, Raster):
            raise TypeError("The DEM must be a Raster")
        if isinstance(attribute, str):
            attribute = [attribute]

        list_raster = []
        for attr in attribute:
            mp_config_copy = mp_config.copy()
            if mp_config.outfile is not None and len(attribute) > 1:
                mp_config_copy.outfile = mp_config_copy.outfile.split(".")[0] + "_" + attr + ".tif"
            list_raster.append(
                map_overlap_multiproc_save(
                    _get_terrain_attribute,
                    dem,
                    mp_config_copy,
                    attr,
                    resolution,
                    degrees,
                    hillshade_altitude,
                    hillshade_azimuth,
                    hillshade_z_factor,
                    slope_method,
                    tri_method,
                    window_size,
                    engine,
                    out_dtype,
                    depth=1,
                )
            )
        if len(list_raster) == 1:
            return list_raster[0]
        return list_raster
    else:
        return _get_terrain_attribute(  # type: ignore
            dem,
            attribute,  # type: ignore
            resolution,
            degrees,
            hillshade_altitude,
            hillshade_azimuth,
            hillshade_z_factor,
            slope_method,
            tri_method,
            window_size,
            engine,
            out_dtype,
        )


@overload
def _get_terrain_attribute(
    dem: NDArrayf | MArrayf,
    attribute: str,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
) -> NDArrayf: ...


@overload
def _get_terrain_attribute(
    dem: NDArrayf | MArrayf,
    attribute: list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
) -> list[NDArrayf]: ...


@overload
def _get_terrain_attribute(
    dem: RasterType,
    attribute: list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
) -> list[RasterType]: ...


@overload
def _get_terrain_attribute(
    dem: RasterType,
    attribute: str,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
) -> RasterType: ...


def _get_terrain_attribute(
    dem: NDArrayf | MArrayf | RasterType,
    attribute: str | list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    engine: Literal["scipy", "numba"] = "numba",
    out_dtype: DTypeLike | None = None,
) -> NDArrayf | list[NDArrayf] | RasterType | list[RasterType]:
    """
    See description of get_terrain_attribute().
    """
    if isinstance(dem, gu.Raster):
        if resolution is None:
            resolution = dem.res

    # Validate and format the inputs
    if isinstance(attribute, str):
        attribute = [attribute]

    # If output dtype is None, used that of input DEM
    if out_dtype is None:
        if np.issubdtype(dem.dtype, np.integer):
            out_dtype = np.float32
        else:
            out_dtype = np.dtype(dem.dtype)

    # These require the get_quadric_coefficients() function, which require the same X/Y resolution.
    list_requiring_surface_fit = [
        "slope",
        "aspect",
        "hillshade",
        "curvature",
        "planform_curvature",
        "profile_curvature",
        "maximum_curvature",
    ]
    attributes_requiring_surface_fit = [attr for attr in attribute if attr in list_requiring_surface_fit]

    list_requiring_windowed_index = [
        "terrain_ruggedness_index",
        "topographic_position_index",
        "roughness",
        "rugosity",
        "fractal_roughness",
    ]
    attributes_requiring_windowed_index = [attr for attr in attribute if attr in list_requiring_windowed_index]

    attributes_requiring_resolution = attributes_requiring_surface_fit + (
        ["rugosity"] if "rugosity" in attribute else []
    )
    if len(attributes_requiring_resolution) > 0:
        if resolution is None:
            raise ValueError(
                f"'resolution' must be provided as an argument for attributes: {attributes_requiring_resolution}"
            )

        if not isinstance(resolution, Sized):
            resolution = (float(resolution), float(resolution))  # type: ignore
        if resolution[0] != resolution[1]:
            raise ValueError(
                f"Surface fit and rugosity require the same X and Y resolution ({resolution} was given). "
                f"This was required by: {attributes_requiring_resolution}."
            )
    # If resolution is still None and there was no error, use a placeholder
    if resolution is None:
        resolution = 1
    # If sized, used the first
    elif isinstance(resolution, Sized):
        resolution = resolution[0]

    choices = list_requiring_surface_fit + list_requiring_windowed_index
    for attr in attribute:
        if attr not in choices:
            raise ValueError(f"Attribute '{attr}' is not supported. Choices: {choices}")

    list_slope_methods = ["Horn", "ZevenbergThorne"]
    if slope_method.lower() not in [sm.lower() for sm in list_slope_methods]:
        raise ValueError(f"Slope method '{slope_method}' is not supported. Must be one of: {list_slope_methods}")
    list_tri_methods = ["Riley", "Wilson"]
    if tri_method.lower() not in [tm.lower() for tm in list_tri_methods]:
        raise ValueError(f"TRI method '{tri_method}' is not supported. Must be one of: {list_tri_methods}")
    if (hillshade_azimuth < 0.0) or (hillshade_azimuth > 360.0):
        raise ValueError(f"Azimuth must be a value between 0 and 360 degrees (given value: {hillshade_azimuth})")
    if (hillshade_altitude < 0.0) or (hillshade_altitude > 90):
        raise ValueError("Altitude must be a value between 0 and 90 degrees (given value: {altitude})")
    if (hillshade_z_factor < 0.0) or not np.isfinite(hillshade_z_factor):
        raise ValueError(f"z_factor must be a non-negative finite value (given value: {hillshade_z_factor})")

    # Raise warning if CRS is not projected and using a surface fit attribute
    if isinstance(dem, gu.Raster) and not dem.crs.is_projected and len(attributes_requiring_surface_fit) > 0:
        warnings.warn(
            category=UserWarning,
            message=f"DEM is not in a projected CRS, the following surface fit attributes might be "
            f"wrong: {list_requiring_surface_fit}."
            f"Use DEM.reproject(crs=DEM.get_metric_crs()) to reproject in a projected CRS.",
        )

    # Get array of DEM
    dem_arr = gu.raster.get_array_and_mask(dem)[0]
    # We need to be able to use NaNs to propagate invalid values in attributes
    if np.issubdtype(dem_arr.dtype, np.integer):
        dem_arr = dem_arr.astype(np.float32)

    # Process surface attributes
    if len(attributes_requiring_surface_fit) > 0:

        # Keyword arguments
        surface_kwargs = {
            "hillshade_azimuth": hillshade_azimuth,
            "hillshade_altitude": hillshade_altitude,
            "hillshade_z_factor": hillshade_z_factor,
        }

        # Get attributes
        surface_attributes = _get_surface_attributes(
            dem=dem_arr,
            resolution=resolution,
            surface_attributes=attributes_requiring_surface_fit,
            out_dtype=out_dtype,
            slope_method=slope_method,
            engine=engine,
            **surface_kwargs,
        )

        # Convert the unit if wanted
        if degrees:
            for attr in ["slope", "aspect"]:
                if attr not in attributes_requiring_surface_fit:
                    continue
                idx_attr = attributes_requiring_surface_fit.index(attr)
                surface_attributes[idx_attr] = np.rad2deg(surface_attributes[idx_attr])

        # Clip values for hillshade
        if "hillshade" in attributes_requiring_surface_fit:
            idx_hs = attributes_requiring_surface_fit.index("hillshade")
            surface_attributes[idx_hs] = np.clip(surface_attributes[idx_hs], 0, 255)

        # Convert to list in-place to save memory
        surface_attributes = [surface_attributes[i] for i in range(surface_attributes.shape[0])]  # type: ignore
    else:
        surface_attributes = []  # type: ignore

    # Process windowed attributes
    if len(attributes_requiring_windowed_index) > 0:

        windowed_indexes = _get_windowed_indexes(
            dem=dem_arr,
            windowed_indexes=attributes_requiring_windowed_index,
            window_size=window_size,
            resolution=resolution,
            out_dtype=out_dtype,
            tri_method=tri_method,
            engine=engine,
        )
        windowed_indexes = [windowed_indexes[i] for i in range(windowed_indexes.shape[0])]  # type: ignore
    else:
        windowed_indexes = []  # type: ignore

    # Convert 3D array output to list of 2D arrays
    output_attributes = surface_attributes + windowed_indexes
    order_indices = [attribute.index(a) for a in attributes_requiring_surface_fit + attributes_requiring_windowed_index]
    output_attributes[:] = [output_attributes[idx] for idx in order_indices]

    if isinstance(dem, gu.Raster):
        output_attributes = [
            gu.Raster.from_array(attr, transform=dem.transform, crs=dem.crs, nodata=-99999)
            for attr in output_attributes
        ]  # type: ignore

    return output_attributes if len(output_attributes) > 1 else output_attributes[0]


@overload
def slope(
    dem: NDArrayf | MArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    degrees: bool = True,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def slope(
    dem: RasterType,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    degrees: bool = True,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> Raster: ...


def slope(
    dem: NDArrayf | MArrayf | RasterType,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    degrees: bool = True,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | Raster:
    """
    Generate a slope map for a DEM, returned in degrees by default.

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918 and on Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to generate a slope map for.
    :param method: Method to calculate slope: "Horn" or "ZevenbergThorne".
    :param degrees: Whether to use degrees or radians (False means radians).
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> slope(dem, resolution=1, degrees=True)[1, 1] # Slope in degrees
        np.float32(45.0)
        >>> np.round(np.tan(slope(dem, resolution=2, degrees=True)[1, 1] * np.pi / 180.), 1) # Slope in percentage
        np.float32(0.5)

    :returns: A slope map of the same shape as 'dem' in degrees or radians.
    """
    return get_terrain_attribute(
        dem,
        attribute="slope",
        slope_method=method,
        resolution=resolution,
        degrees=degrees,
        mp_config=mp_config,
    )


@overload
def aspect(
    dem: NDArrayf | MArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    degrees: bool = True,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def aspect(
    dem: RasterType,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    degrees: bool = True,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def aspect(
    dem: NDArrayf | MArrayf | RasterType,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    degrees: bool = True,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | Raster:
    """
    Calculate the aspect of each cell in a DEM, returned in degrees by default. The aspect of flat slopes is 180° by
    default (as in GDAL).

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918 and on Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107.

    0=N, 90=E, 180=S, 270=W.

    Note that aspect, representing only the orientation of the slope, is independent of the grid resolution.

    :param dem: The DEM to calculate the aspect from.
    :param method: Method to calculate aspect: "Horn" or "ZevenbergThorne".
    :param degrees: Whether to use degrees or radians (False means radians).
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :examples:
        >>> dem = np.tile(np.arange(3), (3,1))
        >>> dem
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])
        >>> aspect(dem, degrees=True)[1, 1]
        np.float32(270.0)
        >>> dem2 = np.repeat(np.arange(3), 3)[::-1].reshape(3, 3)
        >>> dem2
        array([[2, 2, 2],
               [1, 1, 1],
               [0, 0, 0]])
        >>> aspect(dem2, degrees=True)[1, 1]
        np.float32(180.0)

    """
    return get_terrain_attribute(
        dem,
        attribute="aspect",
        slope_method=method,
        resolution=1.0,
        degrees=degrees,
        mp_config=mp_config,
    )


@overload
def hillshade(
    dem: NDArrayf | MArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def hillshade(
    dem: RasterType,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def hillshade(
    dem: NDArrayf | MArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = "Horn",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Generate a hillshade from the given DEM. The value 0 is used for nodata, and 1 to 255 for hillshading.

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918.

    :param dem: The input DEM to calculate the hillshade from.
    :param method: Method to calculate the slope and aspect used for hillshading.
    :param azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param z_factor: Vertical exaggeration factor.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.


    :raises AssertionError: If the given DEM is not a 2D array.
    :raises ValueError: If invalid argument types or ranges were given.

    :returns: A hillshade with the dtype "float32" with value ranges of 0-255.
    """
    return get_terrain_attribute(
        dem,
        attribute="hillshade",
        resolution=resolution,
        slope_method=method,
        hillshade_azimuth=azimuth,
        hillshade_altitude=altitude,
        hillshade_z_factor=z_factor,
        mp_config=mp_config,
    )


@overload
def curvature(
    dem: NDArrayf | MArrayf,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def curvature(
    dem: NDArrayf | MArrayf | RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculate the terrain curvature (second derivative of elevation) in m-1 multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    Information:
       * Curvature is positive on convex surfaces and negative on concave surfaces.
       * Per convention, it is multiplied by 100 to obtain more reasonable numbers. \
               For analytic purposes, dividing by 100 is needed.
       * The unit is the second derivative of elevation (times 100), so '100m²/m' or '100/m' (assuming the unit is m).
       * It is created from the second derivative of a quadric surface fit for each pixel. \
               See xdem.terrain.get_quadric_coefficients() for more information.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> curvature(dem, resolution=1.0)[1, 1] / 100.
        np.float32(4.0)

    :returns: The curvature array of the DEM.
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="curvature",
        resolution=resolution,
        mp_config=mp_config,
    )


@overload
def planform_curvature(
    dem: NDArrayf | MArrayf,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def planform_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def planform_curvature(
    dem: NDArrayf | MArrayf | RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculate the terrain curvature perpendicular to the direction of the slope in m-1 multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> planform_curvature(dem, resolution=1.0)[1, 1] / 100.
        np.float32(-0.0)
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> planform_curvature(dem, resolution=1.0)[1, 1] / 100.
        np.float32(-4.0)

    :returns: The planform curvature array of the DEM.
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="planform_curvature",
        resolution=resolution,
        mp_config=mp_config,
    )


@overload
def profile_curvature(
    dem: NDArrayf | MArrayf,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def profile_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def profile_curvature(
    dem: NDArrayf | MArrayf | RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculate the terrain curvature parallel to the direction of the slope in m-1 multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> profile_curvature(dem, resolution=1.0)[1, 1] / 100.
        np.float32(1.0)
        >>> dem = np.array([[1, 2, 3],
        ...                 [1, 2, 3],
        ...                 [1, 2, 3]], dtype="float32")
        >>> profile_curvature(dem, resolution=1.0)[1, 1] / 100.
        np.float32(0.0)

    :returns: The profile curvature array of the DEM.
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="profile_curvature",
        resolution=resolution,
        mp_config=mp_config,
    )


@overload
def maximum_curvature(
    dem: NDArrayf | MArrayf,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def maximum_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def maximum_curvature(
    dem: NDArrayf | MArrayf | RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculate the signed maximum profile or planform curvature parallel to the direction of the slope in m-1
    multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The profile curvature array of the DEM.
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="maximum_curvature",
        resolution=resolution,
        mp_config=mp_config,
    )


@overload
def topographic_position_index(
    dem: NDArrayf | MArrayf,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def topographic_position_index(
    dem: RasterType,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def topographic_position_index(
    dem: NDArrayf | MArrayf | RasterType,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculates the Topographic Position Index, the difference to the average of neighbouring pixels. Output is in the
    unit of the DEM (typically meters).

    Based on: Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.

    :param dem: The DEM to calculate the topographic position index from.
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> topographic_position_index(dem)[1, 1]
        np.float32(1.0)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> topographic_position_index(dem)[1, 1]
        np.float32(0.0)

    :returns: The topographic position index array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="topographic_position_index",
        window_size=window_size,
        mp_config=mp_config,
    )


@overload
def terrain_ruggedness_index(
    dem: NDArrayf | MArrayf,
    method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def terrain_ruggedness_index(
    dem: RasterType,
    method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def terrain_ruggedness_index(
    dem: NDArrayf | MArrayf | RasterType,
    method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculates the Terrain Ruggedness Index, the cumulated differences to neighbouring pixels. Output is in the
    unit of the DEM (typically meters).

    Based either on:

    * Riley et al. (1999), http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf that derives
        the squareroot of squared differences to neighbouring pixels, preferred for topography.
    * Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962 that derives the mean absolute difference to
        neighbouring pixels, preferred for bathymetry.

    :param dem: The DEM to calculate the terrain ruggedness index from.
    :param method: The algorithm used ("Riley" for topography or "Wilson" for bathymetry).
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> terrain_ruggedness_index(dem)[1, 1]
        np.float32(2.828427)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> terrain_ruggedness_index(dem)[1, 1]
        np.float32(0.0)

    :returns: The terrain ruggedness index array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="terrain_ruggedness_index",
        tri_method=method,
        window_size=window_size,
        mp_config=mp_config,
    )


@overload
def roughness(
    dem: NDArrayf | MArrayf,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def roughness(
    dem: RasterType,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def roughness(
    dem: NDArrayf | MArrayf | RasterType,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculates the roughness, the maximum difference between neighbouring pixels, for any window size. Output is in the
    unit of the DEM (typically meters).

    Based on: Dartnell (2000), https://environment.sfsu.edu/node/11292.

    :param dem: The DEM to calculate the roughness from.
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> roughness(dem)[1, 1]
        np.float32(1.0)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> roughness(dem)[1, 1]
        np.float32(0.0)

    :returns: The roughness array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="roughness",
        window_size=window_size,
        mp_config=mp_config,
    )


@overload
def rugosity(
    dem: NDArrayf | MArrayf,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def rugosity(
    dem: RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def rugosity(
    dem: NDArrayf | MArrayf | RasterType,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculates the rugosity, the ratio between real area and planimetric area. Only available for a 3x3 window. The
    output is unitless.

    Based on: Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.

    :param dem: The DEM to calculate the rugosity from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> rugosity(dem, resolution=1.)[1, 1]
        np.float32(1.4142135)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> np.round(rugosity(dem, resolution=1.)[1, 1], 5)
        np.float32(1.0)

    :returns: The rugosity array of the DEM (unitless).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="rugosity",
        resolution=resolution,
        mp_config=mp_config,
    )


@overload
def fractal_roughness(
    dem: NDArrayf | MArrayf,
    window_size: int = 13,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def fractal_roughness(
    dem: RasterType,
    window_size: int = 13,
    mp_config: MultiprocConfig | None = None,
) -> RasterType: ...


def fractal_roughness(
    dem: NDArrayf | MArrayf | RasterType,
    window_size: int = 13,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterType:
    """
    Calculates the fractal roughness, the local 3D fractal dimension. Can only be computed on window sizes larger or
    equal to 5x5, defaults to 13x13. Output unit is a fractal dimension between 1 and 3.

    Based on: Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    :param dem: The DEM to calculate the roughness from.
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[1, 1] = 6.5
        >>> np.round(fractal_roughness(dem)[6, 6], 5) # The fractal dimension of a line is 1
        np.float32(1.0)
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[:, 1] = 13
        >>> np.round(fractal_roughness(dem)[6, 6]) # The fractal dimension of plane is 2
        np.float32(2.0)
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[:, :6] = 13
        >>> np.round(fractal_roughness(dem)[6, 6]) # The fractal dimension of cube is 3
        np.float32(3.0)

    :returns: The fractal roughness array of the DEM in fractal dimension (between 1 and 3).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="fractal_roughness",
        window_size=window_size,
        mp_config=mp_config,
    )
