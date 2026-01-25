# Copyright (c) 2025 xDEM developers
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

"""Terrain submodule on surface fit attributes: deriving and combining co-dependent fit coefficients."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import numpy as np
from scipy.ndimage import binary_dilation

from xdem._misc import import_optional
from xdem._typing import DTypeLike, NDArrayf

# Manage numba as an optional dependency
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake jit decorator if numba is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


###########################################################################
# SURFACE FIT ATTRIBUTES: DEPENDENT FIT COEFFICIENTS IN A GIVEN WINDOW SIZE
###########################################################################

# Store coefficient of fixed window-size attributes outside functions
# to allow reuse with several engines (Numba, SciPy, Cuda)

# Remove black formatting to facilitate reading the arrays
# fmt: off

# Zevenberg and Thorne (1987) coefficients, Equations 3 to 11
#############################################################

# A, B, C and I are effectively unused for terrain attributes, only useful to get quadric fit
zt_a = np.array(
    [
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ]
)
zt_b = np.array(
    [
        [-1, 0, 1],
        [2, 0, -2],
        [-1, 0, 1]
    ]
)
zt_c = np.array(
    [
        [1, -2, 1],
        [0, 0, 0],
        [-1, 2, -1]])
zt_i = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
)

# All below useful for curvature
zt_d = np.array(
    [
        [0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]
    ]
)
zt_e = np.array(
    [
        [0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]
    ]
)
zt_f = np.array(
    [
        [-1, 0, 1],
        [0, 0, 0],
        [1, 0, -1]
    ]
)

# The G and H coefficients are the only ones needed for slope/aspect/hillshade
zt_g = np.array(
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, -1, 0]
    ]
)
zt_h = np.array(
    [
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]
    ]
)
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

h1 = np.array(
    [
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ]
)
h2 = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]
)

horn_coefs = {"h1": h1, "h2": h2}

# Florinsky (2009) coefficients, equations 12-20
#########################################

fl_a = np.array(
    [
        [-1, 2, 0, -2, 1],
        [-1, 2, 0, -2, 1],
        [-1, 2, 0, -2, 1],
        [-1, 2, 0, -2, 1],
        [-1, 2, 0, -2, 1],
    ]
)

fl_d = np.array(
    [
        [1, 1, 1, 1, 1],
        [-2, -2, -2, -2, -2],
        [0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2],
        [-1, -1, -1, -1, -1],
    ]
)

fl_b = np.array(
    [
        [4, -2, -4, -2, 4],
        [2, -1, -2, -1, 2],
        [0, 0, 0, 0, 0],
        [-2, 1, 2, 1, -2],
        [-4, 2, 4, 2, -4],
    ]
)

fl_c = np.array(
    [
        [-4, -2, 0, 2, 4],
        [2, 1, 0, -1, -2],
        [4, 2, 0, -2, -4],
        [2, 1, 0, -1, -2],
        [-4, -2, 0, 2, 4],
    ]
)

fl_r = np.array(
    [
        [2, -1, -2, -1, 2],
        [2, -1, -2, -1, 2],
        [2, -1, -2, -1, 2],
        [2, -1, -2, -1, 2],
        [2, -1, -2, -1, 2],
    ]
)

fl_t = np.array(
    [
        [2, 2, 2, 2, 2],
        [-1, -1, -1, -1, -1],
        [-2, -2, -2, -2, -2],
        [-1, -1, -1, -1, -1],
        [2, 2, 2, 2, 2],
    ]
)

fl_s = np.array(
    [
        [-4, -2, 0, 2, 4],
        [-2, -1, 0, 1, 2],
        [0, 0, 0, 0, 0],
        [2, 1, 0, -1, -2],
        [4, 2, 0, -2, -4],
    ]
)

fl_p = np.array(
    [
        [31, -44, 0, 44, -31],
        [-5, -62, 0, 62, 5],
        [-17, -68, 0, 68, 17],
        [-5, -62, 0, 62, 5],
        [31, -44, 0, 44, -31],
    ]
)

fl_q = np.array(
    [
        [-31, 5, 17, 5, -31],
        [44, 62, 68, 62, 44],
        [0, 0, 0, 0, 0],
        [-44, -62, -68, -62, -44],
        [31, -5, -17, -5, 31],
    ]
)

# Reacting black formatting
# fmt: on

fl_coefs = {
    "fl_a": fl_a,
    "fl_d": fl_d,
    "fl_b": fl_b,
    "fl_c": fl_c,
    "fl_r": fl_r,
    "fl_t": fl_t,
    "fl_s": fl_s,
    "fl_p": fl_p,
    "fl_q": fl_q,
}


all_coefs = zv_coefs.copy()
all_coefs.update(horn_coefs)
all_coefs.update(fl_coefs)

# Dividers associated with coefficients
#######################################


def _divider_method_coef(res: float, coef: str) -> float:
    """Divider for a given coefficient based on resolution."""

    mapping_div_coef = {
        "zt_a": 4 * res**4,
        "zt_b": 4 * res**3,
        "zt_c": 4 * res**3,
        "zt_d": res**2,  # Divided by 2 compared to Zevenberg to match z_xx definition
        "zt_e": res**2,  # Divided by 2 compared to Zevenberg to match z_xx definition
        "zt_f": 4 * res**2,  # Times 2 what is reported in ZevenbergThorne because later formula multiplies by 2
        "zt_g": 2 * res,
        "zt_h": 2 * res,
        "zt_i": 1,
        "h1": 8 * res,
        "h2": 8 * res,
        "fl_a": 10 * res**3,
        "fl_d": 10 * res**3,
        "fl_b": 70 * res**3,
        "fl_c": 70 * res**3,
        "fl_r": 35 * res**2,
        "fl_t": 35 * res**2,
        "fl_s": 100 * res**2,
        "fl_p": 420 * res,
        "fl_q": 420 * res,
    }

    return mapping_div_coef[coef]


def _preprocess_surface_fit(
    surface_attributes: list[str],
    resolution: float,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"],
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
        if surface_fit == "Horn":
            c_sah = ["h1", "h2"]
        elif surface_fit == "ZevenbergThorne":
            c_sah = ["zt_g", "zt_h"]
        elif surface_fit == "Florinsky":
            c_sah = ["fl_p", "fl_q"]
    else:
        c_sah = []

    # For simple curvature, only 2 coefs needed
    if "curvature" in surface_attributes:
        if surface_fit == "ZevenbergThorne":
            c_curv = ["zt_d", "zt_e"]
        elif surface_fit == "Florinsky":
            c_curv = ["fl_r", "fl_t"]
        # For other methods not supporting curvatures (e.g. Horn)
        else:
            c_curv = []
    else:
        c_curv = []

    # For other curvature, 5 coefs needed
    # if any(att in surface_attributes for att in ["planform_curvature", "profile_curvature", "maximum_curvature"]):
    if any(
        att in surface_attributes
        for att in [
            "profile_curvature",
            "tangential_curvature",
            "planform_curvature",
            "flowline_curvature",
            "max_curvature",
            "min_curvature",
        ]
    ):
        if surface_fit == "ZevenbergThorne":
            c_pcurv = ["zt_d", "zt_e", "zt_f", "zt_g", "zt_h"]
        elif surface_fit == "Florinsky":
            c_pcurv = ["fl_r", "fl_t", "fl_s", "fl_p", "fl_q"]
        # For other methods not supporting curvatures (e.g. Horn)
        else:
            c_pcurv = []
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
    make_profile_curvature = "profile_curvature" in surface_attributes
    make_tangential_curvature = "tangential_curvature" in surface_attributes
    make_planform_curvature = "planform_curvature" in surface_attributes
    make_flowline_curvature = "flowline_curvature" in surface_attributes
    make_max_curvature = "max_curvature" in surface_attributes
    make_min_curvature = "min_curvature" in surface_attributes

    make_attrs = [
        make_slope,
        make_aspect,
        make_hillshade,
        make_curvature,
        make_profile_curvature,
        make_tangential_curvature,
        make_planform_curvature,
        make_flowline_curvature,
        make_max_curvature,
        make_min_curvature,
    ]

    # Map index of attributes and coefficients to defined order
    order_attrs = [
        "slope",
        "aspect",
        "hillshade",
        "curvature",
        "profile_curvature",
        "tangential_curvature",
        "planform_curvature",
        "flowline_curvature",
        "max_curvature",
        "min_curvature",
    ]
    order_coefs = [
        "zt_a",
        "zt_b",
        "zt_c",
        "zt_d",
        "zt_e",
        "zt_f",
        "zt_g",
        "zt_h",
        "zt_i",
        "h1",
        "h2",
        "fl_a",
        "fl_d",
        "fl_b",
        "fl_c",
        "fl_r",
        "fl_t",
        "fl_s",
        "fl_p",
        "fl_q",
    ]

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
    fl_a_idx: int,
    fl_d_idx: int,
    fl_b_idx: int,
    fl_c_idx: int,
    fl_r_idx: int,
    fl_t_idx: int,
    fl_s_idx: int,
    fl_p_idx: int,
    fl_q_idx: int,
    slope_idx: int,
    aspect_idx: int,
    hs_idx: int,
    curv_idx: int,
    profcurv_idx: int,
    tancurv_idx: int,
    plancurv_idx: int,
    flowcurv_idx: int,
    maxcurv_idx: int,
    mincurv_idx: int,
    make_attrs: list[bool],
    out_size: tuple[int, ...],
    surface_fit_id: int,
    curv_method_id: int,
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
    # profile_curvature: 4
    # tangential_curvature: 5
    # planform_curvature: 6
    # flowline_curvature: 7
    # max_curvature: 8
    # min_curvature: 9

    # For surface fits (surface_fit_id)
    # horn: 0
    # zevenbergthorne: 1
    # florinsky: 2

    # For curvature approach (curv_method_id)
    # geometric: 0
    # directional: 1

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
        make_profile_curvature,
        make_tangential_curvature,
        make_planform_curvature,
        make_flowline_curvature,
        make_max_curvature,
        make_min_curvature,
    ) = make_attrs

    if surface_fit_id == 0:

        # Extract surface derivatives based on Horn (1981).
        # http://dx.doi.org/10.1109/PROC.1981.11918.

        z_x_idx = h2_idx
        z_y_idx = h1_idx

    elif surface_fit_id == 1:

        # Extract surface derivatives based on Zevenbergen and Thorne (1987).
        # http://dx.doi.org/10.1002/esp.3290120107.

        # Current output provisions do not currently require b, c

        z_x_idx = zt_h_idx
        z_y_idx = zt_g_idx
        z_xx_idx = zt_e_idx
        z_yy_idx = zt_d_idx
        z_xy_idx = zt_f_idx
        # z_xxy = zt_b_idx
        # z_xyy = zt_c_idx

    elif surface_fit_id == 2:

        # Extract surface derivatives based on Florinsky (2017).
        # https://doi.org/10.1177/0309133317733667.

        # Current output provisions do not currently require a, b, d, c.

        z_x_idx = fl_p_idx
        z_y_idx = fl_q_idx
        z_xx_idx = fl_r_idx
        z_yy_idx = fl_t_idx
        z_xy_idx = fl_s_idx
        # z_xxx_idx = fl_a_idx
        # z_yyy_idx = fl_d_idx
        # z_xxy_idx = fl_b_idx
        # z_xyy_idx = fl_c_idx

    if make_slope:

        slope = np.arctan((C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 0.5)

        # In case slope is only derived for hillshade
        if slope_idx != 99:
            attrs[slope_idx] = slope

    if make_aspect:

        aspect = (-np.arctan2(-C[z_x_idx], C[z_y_idx])) % (2 * np.pi)

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

    # From here on out: no action is taken if surface_fit_id == 0. Cannot raise
    # ValueError here because numba cannot parallelise assertions. However,
    # checks should have occurred prior to this private function being called.

    if make_curvature and surface_fit_id in [1, 2]:

        # THIS FUNCTION IS NOT FOLLOWING THE MINÁR ET AL (2020).
        # RETAINED FOR BACKWARD COMPATIBILITY BUT WITH A WARNING IN `curvature()`

        # Curvature is the second derivative of the surface fit equation.
        # (URL in get_quadric_coefficients() docstring)
        # Curvature = -2(D + E) * 100, see Moore et al. (1991) Equation 16 based on Zevenberg and Thorne (1987)
        attrs[curv_idx] = -2.0 * (C[z_xx_idx] + C[z_yy_idx]) * 100

    if make_profile_curvature and surface_fit_id in [1, 2]:

        # # Completely flat surfaces need to be set to zero to avoid division by zero
        # # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # # so we use a 1-d array and write in a 2-d array output

        if curv_method_id == 0:

            # Geometric profile curvature (normal slope line curvature) following Evans, 1979.
            # profcurv = - (z_xx * z_x**2 + 2 * z_xy * z_x * z_y + z_yy * z_y**2) /
            # ((z_x**2 + z_y**2) * sqrt((1 + z_x**2 + z_y**2)**3))

            profcurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                -(
                    C[z_xx_idx] * C[z_x_idx] ** 2
                    + 2 * C[z_xy_idx] * C[z_x_idx] * C[z_y_idx]
                    + C[z_yy_idx] * C[z_y_idx] ** 2
                )
                / ((C[z_x_idx] ** 2 + C[z_y_idx] ** 2) * np.sqrt((1 + C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 3)),
            )

            # convert from m-1 to 100 m-1
            profcurv *= 100

        elif curv_method_id == 1:

            # Directional derivative slope curvature (2nd slope line derivative) following Krcho, 1973.
            # -(z_xx * z_x**2 + 2 * z_xy * z_x * z_y + z_yy * z_y**2) / (z_x**2 + z_y**2)

            profcurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                -(
                    C[z_xx_idx] * C[z_x_idx] ** 2
                    + 2 * C[z_xy_idx] * C[z_x_idx] * C[z_y_idx]
                    + C[z_yy_idx] * C[z_y_idx] ** 2
                )
                / (C[z_x_idx] ** 2 + C[z_y_idx] ** 2),
            )

            # convert from m-1 to 100 m-1
            profcurv *= 100

        # In case profile curv is only derived for max curv
        if profcurv_idx != 99:
            attrs[profcurv_idx] = profcurv

    if make_tangential_curvature and surface_fit_id in [1, 2]:

        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output

        if curv_method_id == 0:

            # Geometric tangential curvature (normal contour curvature) following Krcho, 1983.
            # tancurv = - (z_xx * z_y**2 - 2 * z_xy * z_x * z_y + z_yy * z_x**2) /
            # ((z_x**2 + z_y**2) * sqrt(1 + z_x**2 + z_y**2))

            tancurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                -(
                    C[z_xx_idx] * C[z_y_idx] ** 2
                    - 2 * C[z_xy_idx] * C[z_x_idx] * C[z_y_idx]
                    + C[z_yy_idx] * C[z_x_idx] ** 2
                )
                / ((C[z_x_idx] ** 2 + C[z_y_idx] ** 2) * np.sqrt(1 + C[z_x_idx] ** 2 + C[z_y_idx] ** 2)),
            )

            # convert from m-1 to 100 m-1
            tancurv *= 100

        if curv_method_id == 1:

            # Directional derivative tangential curvature: 2nd contour derivative "plan curvature",
            # following Zevenberg and Thorne, 1979.
            # tancurv = -(z_xx * z_y**2 - 2 * z_xy * z_x * z_y + z_yy * z_x**2) / (z_x**2 + z_y**2)

            tancurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                -(
                    C[z_xx_idx] * C[z_y_idx] ** 2
                    - 2 * C[z_xy_idx] * C[z_x_idx] * C[z_y_idx]
                    + C[z_yy_idx] * C[z_x_idx] ** 2
                )
                / (C[z_x_idx] ** 2 + C[z_y_idx] ** 2),
            )

            # convert from m-1 to 100 m-1
            tancurv *= 100

        if tancurv_idx != 99:
            attrs[tancurv_idx] = tancurv

    if make_planform_curvature and surface_fit_id in [1, 2]:

        # # Completely flat surfaces need to be set to zero to avoid division by zero
        # # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # # so we use a 1-d array and write in a 2-d array output

        if curv_method_id in [0, 1]:

            # Geometric planform curvature following Sobolevsky, 1932
            # Planform derivation is the same in a geometric and directional derivative context
            # (see Minár et al. 2020 following Jenčo, 1992)
            # plancurv = - (z_xx * z_y**2 - 2 * z_xy * z_x * z_y + z_yy * z_x**2) / sqrt((z_x**2 + z_y**2)**3)

            plancurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 < 10e-15,
                np.array([0.0]),
                -(
                    C[z_xx_idx] * C[z_y_idx] ** 2
                    - 2 * C[z_xy_idx] * C[z_x_idx] * C[z_y_idx]
                    + C[z_yy_idx] * C[z_x_idx] ** 2
                )
                / np.sqrt((C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 3),
            )

            # convert from m-1 to 100 m-1
            plancurv *= 100

        # In case plan curv is only derived for max curv
        if plancurv_idx != 99:
            attrs[plancurv_idx] = plancurv

    if make_flowline_curvature and surface_fit_id in [1, 2]:

        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output

        if curv_method_id == 0:

            # Geometric flowline curvature is geodesic slope line curvature following Minár et al. 2020
            # flowcurv = (z_x * z_y * (z_xx - z_yy) - z_xy * (z_x**2 - z_y**2)) /
            # (((z_x**2 + z_y**2)**3)**0.5 * (1 + z_x**2 + z_y**2)**0.5)

            flowcurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 < 10e-15,
                np.array([0.0]),
                (
                    C[z_x_idx] * C[z_y_idx] * (C[z_xx_idx] - C[z_yy_idx])
                    - C[z_xy_idx] * (C[z_x_idx] ** 2 - C[z_y_idx] ** 2)
                )
                / (((C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 3) ** 0.5 * (1 + C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 0.5),
            )

            # convert from m-1 to 100 m-1
            flowcurv *= 100

        elif curv_method_id == 1:

            # Directional derivative flowline curvature (projected slope line curvature) following Shary et al. 1992
            # flowcurv = (z_x * z_y * (z_xx - z_yy) - z_xy * (z_x**2 - z_y**2)) / ((z_x**2 + z_y**2)**3)**0.5

            flowcurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                (
                    C[z_x_idx] * C[z_y_idx] * (C[z_xx_idx] - C[z_yy_idx])
                    - C[z_xy_idx] * (C[z_x_idx] ** 2 - C[z_y_idx] ** 2)
                )
                / ((C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 3) ** 0.5,
            )

            # convert from m-1 to 100 m-1
            flowcurv *= 100

        if flowcurv_idx != 99:
            attrs[flowcurv_idx] = flowcurv

    if (make_max_curvature or make_min_curvature) and surface_fit_id in [1, 2] and curv_method_id == 0:

        # Mean curvature and unsphericity curvature required for maximal and minimal
        # curvature (could choose to make this explicit and exposed in future)

        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output

        # Mean geometric curvature (Gauss, 1928)
        # mean = -((1 + z_y**2) * z_xx - 2 * z_y * z_x * z_xy + (1 + z_x**2) * z_yy) /
        # (2 * ((1 + z_x**2 + z_y**2)**3)**0.5)
        mean = np.where(
            C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
            np.array([0.0]),
            -(
                (1 + C[z_y_idx] ** 2) * C[z_xx_idx]
                - 2 * C[z_xy_idx] * C[z_x_idx] * C[z_y_idx]
                + (1 + C[z_x_idx] ** 2) * C[z_yy_idx]
            )
            / (2 * ((1 + C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 3) ** 0.5),
        )

        # Not converted from m-1 to 100 m-1 until post-calculation of maximal/minimal curvatures

        # NB - the equivalent directional derivative mean curvature in Minár et al (2020) paper is defined as:
        # mean = (z_xx + z_yy) / 2 ,
        # following Wilson et al. (2007), but this seems to produce a mean curvature of the same
        # magnitude but opposite sign to the geometric approach. I am confident in the geometric
        # approach, however, as when I calculate the directional mean differently as (maximum_curv + minumum_curv)/2,
        # the result aligns with the geometric method.
        # If we were to expose 'mean' and 'unsphericity' as requestable outputs, we would need to diagnose what is
        # going on with the mean approach...

        # Unsphericity curvature (Shary, 1995)
        # unsphericity = (((1 + z_y**2) * z_xx - 2 * z_y * z_x * z_xy + (1 + z_x**2) * z_yy)
        # / (2 * ((1 + z_x**2 + z_y**2)**3)**0.5))**2 - (z_xx * z_yy - z_xy**2) / ((1 + z_x**2 + z_y**2)**2)**0.5

        unsphericity = np.where(
            C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
            np.array([0.0]),
            (
                (
                    (
                        (1 + C[z_y_idx] ** 2) * C[z_xx_idx]
                        - 2 * C[z_y_idx] * C[z_x_idx] * C[z_xy_idx]
                        + (1 + C[z_x_idx] ** 2) * C[z_yy_idx]
                    )
                    / (2 * ((1 + C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 3) ** 0.5)
                )
                ** 2
                - (C[z_xx_idx] * C[z_yy_idx] - C[z_xy_idx] ** 2) / ((1 + C[z_x_idx] ** 2 + C[z_y_idx] ** 2) ** 2)
            )
            ** 0.5,
        )

        # Not converted from m-1 to 100 m-1 until post-calculation of maximal/minimal curvatures

    if make_max_curvature and surface_fit_id in [1, 2]:

        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output

        if curv_method_id == 0:

            # maximual curvature (Shary, 1995) = minimal curvature (Euler, 1760)
            # maxcurv = mean + unsphericity

            maxcurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                mean + unsphericity,
            )

            # convert from m-1 to 100 m-1
            maxcurv *= 100

        elif curv_method_id == 1:

            # maximum curvature (Wood, 1996) is the minimum second derivative
            # maxcurv = -((z_xx + z_yy) / 2 - (((z_xx - z_yy) / 2)**2 + z_xy**2)**0.5)

            maxcurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                -((C[z_xx_idx] + C[z_yy_idx]) / 2 - (((C[z_xx_idx] - C[z_yy_idx]) / 2) ** 2 + C[z_xy_idx] ** 2) ** 0.5),
            )

            # convert from m-1 to 100 m-1
            maxcurv *= 100

        if maxcurv_idx != 99:
            attrs[maxcurv_idx] = maxcurv

    if make_min_curvature and surface_fit_id in [1, 2]:

        # Completely flat surfaces need to be set to zero to avoid division by zero
        # Unfortunately np.where doesn't support scalar input or 0d-array for the Numba parallel case,
        # so we use a 1-d array and write in a 2-d array output

        if curv_method_id == 0:

            # minimal curvature (Shary, 1995) = maximal curvature (Euler, 1760)
            # maxcurv = mean - unsphericity

            mincurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                mean - unsphericity,
            )

            # convert from m-1 to 100 m-1
            mincurv *= 100

        elif curv_method_id == 1:

            # minimum curvature (Wood, 1996) is the maximum second derivative
            # mincurv = -((z_xx + z_yy) / 2 + (((z_xx - z_yy) / 2)**2 + z_xy**2)**0.5)

            mincurv = np.where(
                C[z_x_idx] ** 2 + C[z_y_idx] ** 2 == 0.0,
                np.array([0.0]),
                -((C[z_xx_idx] + C[z_yy_idx]) / 2 + (((C[z_xx_idx] - C[z_yy_idx]) / 2) ** 2 + C[z_xy_idx] ** 2) ** 0.5),
            )

            # convert from m-1 to 100 m-1
            mincurv *= 100

        if mincurv_idx != 99:
            attrs[mincurv_idx] = mincurv

    return attrs


@njit(inline="always", cache=True)  # type: ignore
def _convolution_numba(
    dem: NDArrayf,
    filters: NDArrayf,
    row: int,
    col: int,
    out_dtype: DTypeLike = np.float32,
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
_make_attribute_from_coefs_numba = njit(inline="always", cache=True)(_make_attribute_from_coefs)


@njit(parallel=True, cache=True)  # type: ignore
def _get_surface_attributes_numba(
    dem: NDArrayf,
    filters: NDArrayf,
    make_attrs: list[bool],
    idx_coefs: list[int],
    idx_attrs: list[int],
    attrs_size: int,
    out_dtype: DTypeLike,
    surface_fit_id: int,
    curv_method_id: int,
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
    fl_a_idx = idx_coefs[11]
    fl_d_idx = idx_coefs[12]
    fl_b_idx = idx_coefs[13]
    fl_c_idx = idx_coefs[14]
    fl_r_idx = idx_coefs[15]
    fl_t_idx = idx_coefs[16]
    fl_s_idx = idx_coefs[17]
    fl_p_idx = idx_coefs[18]
    fl_q_idx = idx_coefs[19]
    (
        slope_idx,
        aspect_idx,
        hs_idx,
        curv_idx,
        profcurv_idx,
        tancurv_idx,
        plancurv_idx,
        flowcurv_idx,
        maxcurv_idx,
        mincurv_idx,
    ) = idx_attrs

    # Define ranges to loop through given padding
    row_range = N1 - M1 + 1
    col_range = N2 - M2 + 1

    # Allocate output array
    outputs = np.full((attrs_size, row_range, col_range), fill_value=np.nan, dtype=out_dtype)

    # Loop over every pixel concurrently by using prange
    for row in prange(row_range):
        for col in prange(col_range):

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
                fl_a_idx=fl_a_idx,
                fl_d_idx=fl_d_idx,
                fl_b_idx=fl_b_idx,
                fl_c_idx=fl_c_idx,
                fl_r_idx=fl_r_idx,
                fl_t_idx=fl_t_idx,
                fl_s_idx=fl_s_idx,
                fl_p_idx=fl_p_idx,
                fl_q_idx=fl_q_idx,
                slope_idx=slope_idx,
                aspect_idx=aspect_idx,
                hs_idx=hs_idx,
                curv_idx=curv_idx,
                profcurv_idx=profcurv_idx,
                tancurv_idx=tancurv_idx,
                plancurv_idx=plancurv_idx,
                flowcurv_idx=flowcurv_idx,
                maxcurv_idx=maxcurv_idx,
                mincurv_idx=mincurv_idx,
                out_size=(attrs_size, 1),  # 2-d required for np.where inside func
                surface_fit_id=surface_fit_id,
                curv_method_id=curv_method_id,
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
    surface_fit_id: int,
    curv_method_id: int,
    attrs_size: int,
    out_dtype: DTypeLike = np.float32,
    **kwargs: Any,
) -> NDArrayf:

    # Perform convolution and squeeze output into 3D array
    from xdem.spatialstats import convolution

    coefs = convolution(
        imgs=dem.reshape((1, dem.shape[0], dem.shape[1])),
        filters=filters,
        method="scipy",
    ).squeeze()

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
        fl_a_idx = idx_coefs[11]
        fl_d_idx = idx_coefs[12]
        fl_b_idx = idx_coefs[13]
        fl_c_idx = idx_coefs[14]
        fl_r_idx = idx_coefs[15]
        fl_t_idx = idx_coefs[16]
        fl_s_idx = idx_coefs[17]
        fl_p_idx = idx_coefs[18]
        fl_q_idx = idx_coefs[19]

        (
            slope_idx,
            aspect_idx,
            hs_idx,
            curv_idx,
            profcurv_idx,
            tancurv_idx,
            plancurv_idx,
            flowcurv_idx,
            maxcurv_idx,
            mincurv_idx,
        ) = idx_attrs

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
            fl_a_idx=fl_a_idx,
            fl_d_idx=fl_d_idx,
            fl_b_idx=fl_b_idx,
            fl_c_idx=fl_c_idx,
            fl_r_idx=fl_r_idx,
            fl_t_idx=fl_t_idx,
            fl_s_idx=fl_s_idx,
            fl_p_idx=fl_p_idx,
            fl_q_idx=fl_q_idx,
            slope_idx=slope_idx,
            aspect_idx=aspect_idx,
            hs_idx=hs_idx,
            curv_idx=curv_idx,
            profcurv_idx=profcurv_idx,
            tancurv_idx=tancurv_idx,
            plancurv_idx=plancurv_idx,
            flowcurv_idx=flowcurv_idx,
            maxcurv_idx=maxcurv_idx,
            mincurv_idx=mincurv_idx,
            out_size=out_size,
            surface_fit_id=surface_fit_id,
            curv_method_id=curv_method_id,
            out_dtype=out_dtype,
            **kwargs,
        )
    # Force NaN at the coordinates surrounding (some kernels don't use the center value and thus don't propagate NaNs)
    mask_invalid = ~np.isfinite(dem)
    if surface_fit_id == 2:
        struct = np.ones((5, 5), dtype=bool)
    else:
        struct = np.ones((3, 3), dtype=bool)
    eroded_mask_invalid = binary_dilation(mask_invalid.astype(int), structure=struct, iterations=1)
    attrs[:, eroded_mask_invalid] = np.nan

    return attrs


def _get_surface_attributes(
    dem: NDArrayf,
    resolution: float,
    surface_attributes: list[str],
    out_dtype: DTypeLike = np.float32,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    engine: Literal["scipy", "numba"] = "scipy",
    **kwargs: Any,
) -> NDArrayf:
    """
    Get surface attributes based on fit coefficients (quadric, quintic, etc) using SciPy or Numba convolution and
    reducer functions.

    - Slope, aspect and hillshade from Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918, page 18 bottom left
      equations computed on a 3x3 window.
    - Slope, aspect, hillshade and curvatures from Zevenbergen and Thorne (1987),
      http://dx.doi.org/10.1002/esp.3290120107, computed on a 3x3 window.
    - Slope, aspect, hillshade and curvatures from Florinsky (2008),
      https://doi.org/10.1080/13658810802527499, computed on a 5x5 window.

    :param dem: Input DEM as 2D array.
    :param resolution: Resolution of the DEM (X and Y length are equal).
    :param surface_attributes: Names of surface attributes to compute.
    :param out_dtype: Output dtype of the terrain attributes, can only be a floating type. Defaults to that of the
        input DEM if floating type or to float32 if integer type.
    :param surface_fit: Method for the slope, aspect and hillshade ("Horn", "ZevenbergThorne", or "Florinsky").
    :param curv_method: Method for the curvatures ("geometric" or "directional").
    :param engine: Engine to compute the surface attributes ("scipy" or "numba").
    """

    # Get list of necessary coefficients depending on method and resolution
    coef_arrs, idx_coefs, idx_attrs, make_attrs, attrs_size = _preprocess_surface_fit(
        surface_attributes=surface_attributes,
        resolution=resolution,
        surface_fit=surface_fit,
    )

    # Stack coefficients into a 3D convolution kernel along the first axis
    kern3d = np.stack(coef_arrs, axis=0)

    # Map slope method to integer ID to improve efficiency in Numba loop
    # surface_fit_id = 0 if surface_fit.lower() == "horn" else 1
    surface_fit_mapping = {"horn": 0, "zevenbergthorne": 1, "florinsky": 2}
    surface_fit_id = surface_fit_mapping.get(surface_fit.lower(), -1)

    # Same, but for curvautre method
    curv_method_mapping = {"geometric": 0, "directional": 1}
    curv_method_id = curv_method_mapping.get(curv_method.lower(), -1)

    # Run convolution to compute all coefficients, then reduce those to attributes through either SciPy or Numba
    # (For Numba: Reduction is done within loop to reduce memory usage of computing dozens of full-array coefficients)
    if engine == "scipy":

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice.")
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )
            output = _get_surface_attributes_scipy(
                dem=dem,
                filters=kern3d,
                idx_coefs=idx_coefs,
                idx_attrs=idx_attrs,
                make_attrs=make_attrs,
                surface_fit_id=surface_fit_id,
                curv_method_id=curv_method_id,
                attrs_size=attrs_size,
                out_dtype=out_dtype,
                **kwargs,
            )
    elif engine == "numba":

        # Fail and raise error if optional dependency is not installed
        numba = import_optional("numba")

        _, M1, M2 = kern3d.shape
        half_M1 = int((M1 - 1) / 2)
        half_M2 = int((M2 - 1) / 2)
        dem = np.pad(
            dem,
            pad_width=((half_M1, half_M1), (half_M2, half_M2)),
            constant_values=np.nan,
        )
        # Now required to declare list typing in latest Numba before deprecation
        typed_make_attrs, typed_idx_attrs, typed_idx_coefs = (
            numba.typed.List(),
            numba.typed.List(),
            numba.typed.List(),
        )
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
            surface_fit_id=surface_fit_id,
            curv_method_id=curv_method_id,
            **kwargs,
        )

    return output
