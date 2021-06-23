"""Terrain attribute calculations, such as the slope, aspect etc."""
from __future__ import annotations

from typing import Sized, overload

import numpy as np

import xdem.spatial_tools


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

    :param attribute: The terrain attribute(s) to calculate:
        * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical).
        * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W)
        * 'hillshade': The shaded slope in relation to its aspect.
    """
    if isinstance(attribute, str):
        attribute = [attribute]

    choices = ["slope", "aspect", "hillshade"]
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
    x_gradient, y_gradient = np.gradient(dem_arr)

    # Normalize by the radius of the resolution to make it resolution variant.
    x_gradient /= resolution[0]
    y_gradient /= resolution[1]

    terrain_attributes = {"hillshade": np.array([np.nan]), "aspect": np.array([np.nan]), "slope": np.array([np.nan])}
    make_aspect = any(attr in attribute for attr in ["aspect", "hillshade"])
    make_slope = any(attr in attribute for attr in ["slope", "hillshade"])
    make_hillshade = "hillshade" in attribute

    if make_aspect:
        terrain_attributes["aspect"] = np.arctan2(-x_gradient, y_gradient)

    if make_slope:
        # Calculate slope
        terrain_attributes["slope"] = np.pi / 2.0 - np.arctan(np.sqrt(x_gradient ** 2 + y_gradient ** 2))

    if make_hillshade:
        if hillshade_z_factor != 1.0:
            slopemap = np.pi / 2.0 - np.arctan(np.sqrt((x_gradient * hillshade_z_factor) ** 2 + (y_gradient * hillshade_z_factor) ** 2))
        else:
            slopemap = terrain_attributes["slope"]

        azimuth_rad = np.deg2rad(360 - hillshade_azimuth)
        altitude_rad = np.deg2rad(hillshade_altitude)
        terrain_attributes["hillshade"] = np.clip(
            255
            * (
                np.sin(altitude_rad) * np.sin(slopemap)
                + np.cos(altitude_rad)
                * np.cos(terrain_attributes["slope"])
                * np.cos((azimuth_rad - np.pi / 2.0) - terrain_attributes["aspect"])
            ),
            0,
            255,
        ).astype("float32")

    # Convert the unit if wanted.
    if degrees:
        terrain_attributes["slope"] = 90 - np.rad2deg(terrain_attributes["slope"])
        with np.errstate(invalid="ignore"):  # It may warn for nans (which is okay)
            terrain_attributes["aspect"] = (270 - np.rad2deg(terrain_attributes["aspect"])) % 360

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
    :param azimuth: The azimuth in degrees (0-360°) going clockwise, starting from north.
    :param altitude: The altitude in degrees (0-90°). 90° is straight from above.
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
