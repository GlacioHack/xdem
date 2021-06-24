"""Terrain attribute calculations, such as the slope, aspect etc."""
from __future__ import annotations

import warnings
from typing import Sized, overload

import numba
import numpy as np

import xdem.spatial_tools


@numba.njit(parallel=True)
def _get_quadric_coefficients(
    dem: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """
    Run the pixel-wise analysis in parallel.

    See the xdem.terrain.get_quadric_coefficients() docstring for more info.
    """
    # Rename the resolution to be consistent with the ArcGIS reference.
    L = resolution

    # Allocate the output.
    output = np.empty((9,) + dem.shape, dtype=dem.dtype) + np.nan

    # Loop over every pixel concurrently.
    for i in numba.prange(dem.size):
        # Derive its associated row and column index.
        col = i % dem.shape[1]
        row = int(i / dem.shape[1])

        # Extract the pixel and its 8 immediate neighbours.
        # If the border is reached, just duplicate the closest neighbour to obtain 9 values.
        Z = np.empty((9,), dtype=dem.dtype)
        count = 0
        for j in range(-1, 2):
            for k in range(-1, 2):
                row_indexer = min(max(row + k, 0), dem.shape[0] - 1)
                col_indexer = min(max(col + j, 0), dem.shape[1] - 1)
                Z[count] = dem[row_indexer, col_indexer]
                count += 1

        # Get a mask of all invalid (nan or inf) values.
        invalids = ~np.isfinite(Z)

        # Skip the pixel if it and all of its neighbours are invalid
        if np.all(invalids):
            continue

        # Fill all non-finite values with the most common value.
        Z[invalids] = np.nanmedian(Z)

        # Assign the A, B, C, D etc., factors to the output. This ugly syntax is needed to make parallel numba happy.
        output[0, row, col] = ((Z[0] + Z[2] + Z[6] + Z[8]) / 4 - (Z[1] + Z[3] + Z[5] + Z[7]) / 2 + Z[4]) / (L ** 4)  # A
        output[1, row, col] = ((Z[0] + Z[2] - Z[6] - Z[8]) / 4 - (Z[1] - Z[7]) / 2) / (L ** 3)  # B
        output[2, row, col] = ((-Z[0] + Z[2] - Z[6] + Z[8]) / 4 + (Z[3] - Z[5]) / 2) / (L ** 3)  # C
        output[3, row, col] = ((Z[3] + Z[5]) / 2 - Z[4]) / (L ** 2)  # D
        output[4, row, col] = ((Z[1] + Z[7]) / 2 - Z[4]) / (L ** 2)  # E
        output[5, row, col] = (-Z[0] + Z[2] + Z[6] - Z[8]) / (4 * L ** 2)  # F
        output[6, row, col] = (-Z[3] + Z[5]) / (2 * L)  # G
        output[7, row, col] = (Z[1] - Z[7]) / (2 * L)  # H
        output[8, row, col] = Z[4]  # I

    return output


def get_quadric_coefficients(dem: np.ndarray, resolution: float) -> np.ndarray:
    """
    Return the 9 coefficients of a quadric surface fit to every pixel in the raster.

    Mostly inspired by: https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm

    The function that is solved is:
    Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I

    Where Z is the elevation, x is the distance from left-right and y is the distance from top-bottom.
    Each pixel's fit can be accessed by coefficients[:, row, col], returning an array of shape 9.
    The 9 coefficients correspond to those in the equation above.

    Quirks:
        * Edges are naively treated by filling the closest value, so that a 3x3 matrix is always calculated.\
                It may therefore be slightly off in the edges.
        * NaNs and infs are filled with the median of the finites in the matrix, possibly affecting the fit.
        * The X and Y resolution needs to be the same. It does not work if they differ.

    :param dem: The 2D DEM to be analyzed (3D DEMs of shape (1, row, col) are not supported)
    :param resolution: The X/Y resolution of the DEM.

    :raises ValueError: If the inputs are poorly formatted.
    :raises RuntimeError: If unexpected backend errors occurred.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> coeffs = get_quadric_coefficients(dem, resolution=1.0)
        >>> coeffs.shape
        (9, 3, 3)
        >>> coeffs[:, 1, 1]
        array([ 1.,  0.,  0., -1., -1.,  0.,  0.,  0.,  2.])

    :returns: An array of coefficients for each pixel of shape (9, row, col).
    """
    # This function only formats and validates the inputs. For the true functionality, see _get_quadric_coefficients()
    dem_arr = xdem.spatial_tools.get_array_and_mask(dem)[0]

    if len(dem_arr.shape) != 2:
        raise ValueError(
            f"Invalid input array shape: {dem.shape}, parsed into {dem_arr.shape}. "
            "Expected 2D array or 3D array of shape (1, row, col)"
        )

    if any(dim < 3 for dim in dem_arr.shape):
        raise ValueError(f"DEM (shape: {dem.shape}) is too small. Smallest supported shape is (3, 3)")

    # Resolution is in other tools accepted as a tuple. Here, it must be just one number, so it's best to sanity check.
    if isinstance(resolution, Sized):
        raise ValueError("Resolution must be the same for X and Y directions")

    # Try to run the numba JIT code. It should never fail at this point, so if it does, it should be reported!
    try:
        coeffs = _get_quadric_coefficients(dem_arr, resolution)
    except Exception as exception:
        raise RuntimeError("Unhandled numba exception. Please raise an issue of what happened.") from exception

    return _get_quadric_coefficients(dem_arr, resolution)


@overload
def get_terrain_attribute(
    dem: np.ndarray | np.ma.masked_array,
    attribute: str,
    resolution: tuple[float, float] | float,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
) -> np.ndarray:
    ...


@overload
def get_terrain_attribute(
    dem: np.ndarray | np.ma.masked_array,
    attribute: list[str],
    resolution: tuple[float, float] | float,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
) -> list[np.ndarray]:
    ...


def get_terrain_attribute(
    dem: np.ndarray | np.ma.masked_array,
    attribute: str | list[str],
    resolution: tuple[float, float] | float,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
) -> np.ndarray | list[np.ndarray]:
    """
    Derive one or multiple terrain attributes from a DEM.

    Attributes:
        * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical).
        * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W)
        * 'hillshade': The shaded slope in relation to its aspect.
        * 'curvature': The second derivative of elevation (the rate of slope change per pixel), multiplied by 100.
        * 'planform_curvature': The curvature perpendicular to the direction of the slope.
        * 'profile_curvature': The curvature parallel to the direction of the slope.
        * 'surface_fit': A quadric surface fit for each individual pixel. For more info, see get_quadric_coefficients()

    :param dem: The DEM to analyze.
    :param attribute: The terrain attribute(s) to calculate.
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param degrees: Convert radians to degrees?
    :param hillshade_altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param hillshade_azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param hillshade_z_factor: Vertical exaggeration factor.

    :raises ValueError: If the inputs are poorly formatted or are invalid.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> slope, aspect = get_terrain_attribute(dem, ["slope", "aspect"], resolution=1)
        >>> slope
        array([[45., 45., 45.],
               [45., 45., 45.],
               [45., 45., 45.]], dtype=float32)
        >>> aspect
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]], dtype=float32)

    :returns: One or multiple arrays of the requested attribute(s)
    """
    # Validate and format the inputs
    if isinstance(attribute, str):
        attribute = [attribute]

    choices = ["slope", "aspect", "hillshade", "curvature", "planform_curvature", "profile_curvature", "surface_fit"]
    for attr in attribute:
        if attr not in choices:
            raise ValueError(f"Attribute '{attr}' is not supported. Choices: {choices}")

    if not isinstance(resolution, Sized):
        resolution = (float(resolution), float(resolution))

    if (hillshade_azimuth < 0.0) or (hillshade_azimuth > 360.0):
        raise ValueError(f"Azimuth must be a value between 0 and 360 degrees (given value: {hillshade_azimuth})")
    if (hillshade_altitude < 0.0) or (hillshade_altitude > 90):
        raise ValueError("Altitude must be a value between 0 and 90 degress (given value: {altitude})")
    if (hillshade_z_factor < 0.0) or not np.isfinite(hillshade_z_factor):
        raise ValueError(f"z_factor must be a non-negative finite value (given value: {hillshade_z_factor})")

    dem_arr = xdem.spatial_tools.get_array_and_mask(dem)[0]

    # Initialize the terrain_attributes dictionary, which will be filled with the requested values.
    terrain_attributes: dict[str, np.ndarray] = {}

    attributes_requiring_surface_fit = [
        attr
        for attr in attribute
        if attr in ["curvature", "planform_curvature", "profile_curvature", "slope", "hillshade", "aspect"]
    ]

    # Check which products should be made
    make_aspect = any(attr in attribute for attr in ["aspect", "hillshade"])
    make_slope = any(
        attr in attribute for attr in ["slope", "hillshade", "planform_curvature", "aspect", "profile_curvature"]
    )
    make_hillshade = "hillshade" in attribute
    make_surface_fit = len(attributes_requiring_surface_fit) > 0
    make_curvature = "curvature" in attribute
    make_planform_curvature = "planform_curvature" in attribute
    make_profile_curvature = "profile_curvature" in attribute

    if make_surface_fit:
        if resolution[0] != resolution[1]:
            raise ValueError(
                f"Quadric surface fit requires the same X and Y resolution ({resolution} was given). "
                f"This was required by: {attributes_requiring_surface_fit}"
            )
        terrain_attributes["surface_fit"] = get_quadric_coefficients(dem_arr, resolution[0])

    if make_slope:
        # This calculation is based on (p18, left side): https://ieeexplore.ieee.org/document/1456186
        # SLOPE = -(G²+H²)**(1/2)
        terrain_attributes["slope"] = np.arctan(
            (terrain_attributes["surface_fit"][6, :, :] ** 2 + terrain_attributes["surface_fit"][7, :, :] ** 2) ** 0.5
        )

    if make_aspect:
        # ASPECT = ARCTAN(-H/-G)  # This did not work
        # ASPECT = ARCTAN2(-G, H)  did work.
        terrain_attributes["aspect"] = np.arctan2(
            -terrain_attributes["surface_fit"][6, :, :], terrain_attributes["surface_fit"][7, :, :]
        )

    if make_hillshade:
        # If a different z-factor was given, slopemap with exaggerated gradients.
        if hillshade_z_factor != 1.0:
            slopemap = np.arctan(np.tan(terrain_attributes["slope"]) * hillshade_z_factor)
        else:
            slopemap = terrain_attributes["slope"]

        azimuth_rad = np.deg2rad(360 - hillshade_azimuth)
        altitude_rad = np.deg2rad(hillshade_altitude)
        terrain_attributes["hillshade"] = np.clip(
            255
            * (
                np.sin(altitude_rad) * np.cos(slopemap)
                + np.cos(altitude_rad) * np.sin(slopemap) * np.cos(azimuth_rad - terrain_attributes["aspect"] - np.pi)
            ),
            0,
            255,
        ).astype("float32")

    if make_curvature:
        # Curvature is the second derivative of the surface fit equation. See the ArcGIS documentation.
        # (URL in get_quadric_coefficients() docstring)
        # Curvature = -2(D + E) * 100
        terrain_attributes["curvature"] = (
            -2 * (terrain_attributes["surface_fit"][3, :, :] + terrain_attributes["surface_fit"][4, :, :]) * 100
        )

    if make_planform_curvature:
        # PLANC = 2(DH² + EG² -FGH)/(G²+H²)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
            terrain_attributes["planform_curvature"] = (
                2
                * (
                    terrain_attributes["surface_fit"][3, :, :] * terrain_attributes["surface_fit"][7, :, :] ** 2
                    + terrain_attributes["surface_fit"][4, :, :] * terrain_attributes["surface_fit"][6, :, :] ** 2
                    - terrain_attributes["surface_fit"][5, :, :]
                    * terrain_attributes["surface_fit"][6, :, :]
                    * terrain_attributes["surface_fit"][7, :, :]
                )
                / (terrain_attributes["surface_fit"][6, :, :] ** 2 + terrain_attributes["surface_fit"][7, :, :] ** 2)
            )

        # Completely flat surfaces trigger the warning above. These need to be set to zero
        terrain_attributes["planform_curvature"][terrain_attributes["slope"] == 0.0] = 0.0

    if make_profile_curvature:
        # PROFC = -2(DH² + EG² + FGH)/(G²+H²)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
            terrain_attributes["profile_curvature"] = (
                -2
                * (
                    terrain_attributes["surface_fit"][3, :, :] * terrain_attributes["surface_fit"][7, :, :] ** 2
                    + terrain_attributes["surface_fit"][4, :, :] * terrain_attributes["surface_fit"][6, :, :] ** 2
                    + terrain_attributes["surface_fit"][5, :, :]
                    * terrain_attributes["surface_fit"][6, :, :]
                    * terrain_attributes["surface_fit"][7, :, :]
                )
                / (terrain_attributes["surface_fit"][6, :, :] ** 2 + terrain_attributes["surface_fit"][7, :, :] ** 2)
            )

        # Completely flat surfaces trigger the warning above. These need to be set to zero
        terrain_attributes["profile_curvature"][terrain_attributes["slope"] == 0.0] = 0.0

    # Convert the unit if wanted.
    if degrees:
        for attr in ["slope", "aspect"]:
            if attr not in terrain_attributes:
                continue
            terrain_attributes[attr] = np.rad2deg(terrain_attributes[attr])

    output_attributes = [terrain_attributes[key].reshape(dem.shape) for key in attribute]

    return output_attributes if len(output_attributes) > 1 else output_attributes[0]


def slope(
    dem: np.ndarray | np.ma.masked_array, resolution: float | tuple[float, float], degrees: bool = True
) -> np.ndarray:
    """
    Generate a slope map for a DEM.

    :param dem: The DEM to generate a slope map for.
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param degrees: Return a slope map in degrees (False means radians)

    :returns: A slope map of the same shape as 'dem' in degrees or radians.
    """
    return get_terrain_attribute(dem, attribute="slope", resolution=resolution, degrees=degrees)


def aspect(dem: np.ndarray | np.ma.masked_array, degrees: bool = True) -> np.ndarray:
    """
    Calculate the aspect of each cell in a DEM.

    0=N, 90=E, 180=S, 270=W

    :param dem: The DEM to calculate the aspect from.
    :param degrees: Return an aspect map in degrees (if False, returns radians)

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> aspect(dem, degrees=True)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]], dtype=float32)
        >>> dem.T
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])
        >>> aspect(dem.T, degrees=True)
        array([[270., 270., 270.],
               [270., 270., 270.],
               [270., 270., 270.]], dtype=float32)

    """
    return get_terrain_attribute(dem, attribute="aspect", resolution=1.0, degrees=degrees)


def hillshade(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float],
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """
    Generate a hillshade from the given DEM.

    :param dem: The input DEM to calculate the hillshade from.
    :param resolution: One or two values specifying the resolution of the DEM.
    :param azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param z_factor: Vertical exaggeration factor.

    :raises AssertionError: If the given DEM is not a 2D array.
    :raises ValueError: If invalid argument types or ranges were given.

    :returns: A hillshade with the dtype "float32" with value ranges of 0-255.
    """
    return get_terrain_attribute(
        dem,
        attribute="hillshade",
        resolution=resolution,
        hillshade_azimuth=azimuth,
        hillshade_altitude=altitude,
        hillshade_z_factor=z_factor,
    )


def curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float,
) -> np.ndarray:
    """
    Get the terrain curvature (second derivative of elevation).

    Information:
       * Curvature is positive on convex surfaces and negative on concave surfaces.
       * Per convention, it is multiplied by 100 to obtain more reasonable numbers. \
               For analytic purposes, dividing by 100 is needed.
       * The unit is the second derivative of elevation (times 100), so '100m²/m' or '100/m' (assuming the unit is m).
       * It is created from the second derivative of a quadric surface fit for each pixel. \
               See xdem.terrain.get_quadric_coefficients() for more information.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> curvature(dem, resolution=1.0)
        array([[  -0., -100.,   -0.],
               [-100.,  400., -100.],
               [  -0., -100.,   -0.]])

    :returns: The curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="curvature", resolution=resolution)
