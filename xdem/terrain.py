"""Terrain attribute calculations, such as the slope, aspect etc."""
from __future__ import annotations

import warnings
from typing import Sized, overload

import numba
import numpy as np

import xdem.spatial_tools
import geoutils as gu
from geoutils.georaster import RasterType, Raster


@numba.njit(parallel=True)
def _get_quadric_coefficients(
    dem: np.ndarray, resolution: float, fill_method: str = "median", edge_method: str = "nearest"
) -> np.ndarray:
    """
    Run the pixel-wise analysis in parallel.

    See the xdem.terrain.get_quadric_coefficients() docstring for more info.
    """
    # Rename the resolution
    L = resolution

    # Allocate the output.
    output = np.empty((9,) + dem.shape, dtype=dem.dtype) + np.nan

    # Convert the string to a number (fewer bytes to compare each iteration)
    if fill_method == "median":
        fill_method_n = numba.uint8(0)
    elif fill_method == "mean":
        fill_method_n = numba.uint8(1)
    elif fill_method == "none":
        fill_method_n = numba.uint8(2)

    if edge_method == "nearest":
        edge_method_n = numba.uint8(0)
    elif edge_method == "wrap":
        edge_method_n = numba.uint8(1)
    elif edge_method == "none":
        edge_method_n = numba.uint8(2)

    # Loop over every pixel concurrently.
    for i in numba.prange(dem.size):
        # Derive its associated row and column index.
        col = i % dem.shape[1]
        row = int(i / dem.shape[1])

        # Extract the pixel and its 8 immediate neighbours.
        # If the border is reached, just duplicate the closest neighbour to obtain 9 values.
        Z = np.empty((9,), dtype=dem.dtype)
        count = 0

        # If edge_method == "none", validate that it's not near an edge. If so, leave the nans without filling.
        if edge_method_n == 2:
            if (row < 1) or (row > (dem.shape[0] - 2)) or (col < 1) or (col > (dem.shape[1] - 2)):
                continue

        for j in range(-1, 2):
            for k in range(-1, 2):
                # Here the "nearest" edge_method is performed.
                if edge_method_n == 0:
                    row_indexer = min(max(row + k, 0), dem.shape[0] - 1)
                    col_indexer = min(max(col + j, 0), dem.shape[1] - 1)
                elif edge_method_n == 1:
                    row_indexer = (row + k) % dem.shape[0]
                    col_indexer = (col + j) % dem.shape[1]
                Z[count] = dem[row_indexer, col_indexer]
                count += 1

        # Get a mask of all invalid (nan or inf) values.
        invalids = ~np.isfinite(Z)
        n_invalid = np.count_nonzero(invalids)

        # Skip the pixel if it and all of its neighbours are invalid
        if np.all(invalids):
            continue

        if np.count_nonzero(invalids) > 0:
            if fill_method_n == 0:
                # Fill all non-finite values with the most common value.
                Z[invalids] = np.nanmedian(Z)
            elif fill_method_n == 1:
                # Fill all non-finite values with the mean.
                Z[invalids] = np.nanmean(Z)
            elif fill_method_n == 2:
                # Skip the pixel if any of its neighbours are nan.
                continue
            else:
                # This should not occur.
                pass

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


def get_quadric_coefficients(
    dem: np.ndarray, resolution: float, fill_method: str = "median", edge_method: str = "nearest"
) -> np.ndarray:
    """
    Return the 9 coefficients of a quadric surface fit to every pixel in the raster.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107, also described in the documentation:
    https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm

    The function that is solved is:
    Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I

    Where Z is the elevation, x is the distance from left-right and y is the distance from top-bottom.
    Each pixel's fit can be accessed by coefficients[:, row, col], returning an array of shape 9.
    The 9 coefficients correspond to those in the equation above.

    Fill methods
        If the 3x3 matrix to fit the quadric function on has NaNs, these need to be handled:
        * 'median': NaNs are filled with the median value of the matrix.
        * 'mean': NaNs are filled with the mean value of the matrix.
        * 'none': If NaNs are encountered, skip the entire cell (default for GDAL and SAGA).

    Edge methods
        Each iteration requires a 3x3 matrix, so special edge cases have to be made.
        * 'nearest': Pixels outside the range are filled using the closest pixel value.
        * 'wrap': The array is wrapped so pixels near the right edge will be sampled from the left, etc.
        * 'none': Edges will not be analyzed, leaving a 1 pixel edge of NaNs.

    Quirks:
        * Edges are naively treated by filling the closest value, so that a 3x3 matrix is always calculated.\
                It may therefore be slightly off in the edges.
        * NaNs and infs are filled with the median of the finites in the matrix, possibly affecting the fit.
        * The X and Y resolution needs to be the same. It does not work if they differ.

    :param dem: The 2D DEM to be analyzed (3D DEMs of shape (1, row, col) are not supported)
    :param resolution: The X/Y resolution of the DEM.
    :param fill_method: Fill method to use for NaNs in the 3x3 matrix.
    :param edge_method: The method to use near the array edge.

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

    allowed_fill_methods = ["median", "mean", "none"]
    allowed_edge_methods = ["nearest", "wrap", "none"]
    for value, name, allowed in zip(
        [fill_method, edge_method], ["fill", "edge"], (allowed_fill_methods, allowed_edge_methods)
    ):
        if value.lower() not in allowed:
            raise ValueError(f"Invalid {name} method: '{value}'. Choices: {allowed}")

    # Try to run the numba JIT code. It should never fail at this point, so if it does, it should be reported!
    try:
        coeffs = _get_quadric_coefficients(
            dem_arr, resolution, fill_method=fill_method.lower(), edge_method=edge_method.lower()
        )
    except Exception as exception:
        raise RuntimeError("Unhandled numba exception. Please raise an issue of what happened.") from exception

    return coeffs

@numba.njit(parallel=True)
def _get_windowed_indexes(
    dem: np.ndarray, fill_method: str = "median", edge_method: str = "nearest", window_size: int = 3)\
        -> np.ndarray:
    """
    Run the pixel-wise analysis in parallel.

    See the xdem.terrain.get_windowed_indexes() docstring for more info.
    """

    # Allocate the outputs.
    output = np.empty((5,) + dem.shape, dtype=dem.dtype) + np.nan

    # Half window size
    hw = int(np.floor(window_size / 2))

    # Convert the string to a number (fewer bytes to compare each iteration)
    if fill_method == "median":
        fill_method_n = numba.uint8(0)
    elif fill_method == "mean":
        fill_method_n = numba.uint8(1)
    elif fill_method == "none":
        fill_method_n = numba.uint8(2)

    if edge_method == "nearest":
        edge_method_n = numba.uint8(0)
    elif edge_method == "wrap":
        edge_method_n = numba.uint8(1)
    elif edge_method == "none":
        edge_method_n = numba.uint8(2)

    # Loop over every pixel concurrently.
    for i in numba.prange(dem.size):
        # Derive its associated row and column index.
        col = i % dem.shape[1]
        row = int(i / dem.shape[1])

        # Extract the pixel and its 8 immediate neighbours.
        # If the border is reached, just duplicate the closest neighbour to obtain 9 values.
        Z = np.empty((window_size**2,), dtype=dem.dtype)
        count = 0

        # If edge_method == "none", validate that it's not near an edge. If so, leave the nans without filling.
        if edge_method_n == 2:
            if (row < window_size - 2) or (row > (dem.shape[0] - window_size + 1)) or (col < window_size - 2) or \
                    (col > (dem.shape[1] - window_size + 1)):
                continue

        for j in range(-hw, -hw+window_size):
            for k in range(-hw, -hw+window_size):
                # Here the "nearest" edge_method is performed.
                if edge_method_n == 0:
                    row_indexer = min(max(row + k, 0), dem.shape[0] - 1)
                    col_indexer = min(max(col + j, 0), dem.shape[1] - 1)
                elif edge_method_n == 1:
                    row_indexer = (row + k) % dem.shape[0]
                    col_indexer = (col + j) % dem.shape[1]
                Z[count] = dem[row_indexer, col_indexer]
                count += 1

        # Get a mask of all invalid (nan or inf) values.
        invalids = ~np.isfinite(Z)
        n_invalid = np.count_nonzero(invalids)

        # Skip the pixel if it and all of its neighbours are invalid
        if np.all(invalids):
            continue

        if np.count_nonzero(invalids) > 0:
            if fill_method_n == 0:
                # Fill all non-finite values with the most common value.
                Z[invalids] = np.nanmedian(Z)
            elif fill_method_n == 1:
                # Fill all non-finite values with the mean.
                Z[invalids] = np.nanmean(Z)
            elif fill_method_n == 2:
                # Skip the pixel if any of its neighbours are nan.
                continue
            else:
                # This should not occur.
                pass

        # Difference pixels between specific cells: only useful for Terrain Ruggedness Index
        count = 0
        index_middle_pixel = int((window_size**2 - 1)/2)
        S = np.empty((window_size**2,))
        for j in range(-hw, -hw + window_size):
            for k in range(-hw, -hw + window_size):
                S[count] = np.abs(Z[count] - Z[index_middle_pixel])
                count += 1


        # Elevation differences and horizontal length
        dzs = np.empty((16,))
        dls = np.empty((16,))

        count = 0
        # First, the 8 connected segments from the center cells, the center cell is index 4
        for j in range(-hw, -hw + window_size):
            for k in range(-hw, -hw + window_size):
                # Skip if this is the center pixel
                if j == 0 and k == 0:
                    continue
                # The first eight elevation differences from the cell center
                dzs[count] = Z[4] - Z[count]
                # The first eight planimetric length that can be diagonal or straight from the center
                dls[count] = np.sqrt(j**2 + k**2)
                count += 1

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
        dls[8:] = 1

        # Finally, the half-surface length of each segment
        L = np.sqrt(dzs**2 + dls**2)/2

        # Starting from up direction anticlockwise, every triangle has 2 segments between center and surrounding pixels
        # and 1 segment between surrounding pixels; pixel 4 is the center
        # above 4 the index of center-surrounding segment decrease by 1, as the center pixel was skipped
        # Triangle 1: pixels 3 and 0
        T1 = [L[3], L[0], L[12]]
        # Triangle 2: pixels 0 and 1
        T2 = [L[0], L[1], L[8]]
        # Triangle 3: pixels 1 and 2
        T3 = [L[1], L[2], L[9]]
        # Triangle 4: pixels 2 and 5
        T4 = [L[2], L[4], L[14]]
        # Triangle 5: pixels 5 and 8
        T5 = [L[4], L[7], L[15]]
        # Triangle 6: pixels 8 and 7
        T6 = [L[7], L[6], L[11]]
        # Triangle 7: pixels 7 and 6
        T7 = [L[6], L[5], L[10]]
        # Triangle 8: pixels 6 and 3
        T8 = [L[5], L[3], L[13]]

        list_T = [T1, T2, T3, T4, T5, T6, T7, T8]

        # Finally, we compute the 3D surface areas of the 8 triangles
        A = np.empty((8,))
        count = 0
        for T in list_T:
            # Half sum of lengths
            hs = sum(T)/2
            # Surface area of triangle
            A[count] = np.sqrt(hs*(hs-T[0])*(hs-T[1])*(hs-T[2]))
            count += 1


        # First output is the Terrain Ruggedness Index from Riley et al. (1999): squareroot of squared sum of
        # differences between center and neighbouring pixels
        output[0, row, col] = np.sqrt(np.sum(S**2))
        # Second output is the Terrain Ruggedness Index from Wilson et al. (2007): mean difference between center
        # and neighbouring pixels
        output[1, row, col] = np.sum(S) / (window_size**2 - 1)
        # Third output is the Topographic Position Index from Weiss (2001): difference between center and mean of
        # neighbouring pixels
        output[2, row, col] =  Z[index_middle_pixel] - (np.sum(Z) - Z[index_middle_pixel]) / (window_size**2 - 1)
        # Fourth output is the Roughness: difference between maximum and minimum of the window
        output[3, row, col] = np.max(Z) - np.min(Z)
        # Fifth output is the Rugosity: difference between real surface area and planimetric surface area
        output[4, row, col] = sum(A)

    return output


def get_windowed_indexes(
    dem: np.ndarray, fill_method: str = "median", edge_method: str = "nearest", window_size: int = 3,
) -> np.ndarray:
    """
    Return terrain indexes based on a windowed calculation of variable size.

    Includes:
    - Terrain Ruggedness Index from Riley et al. (1999) for topgraphy and from Wilson et al. (2007) for bathymetry.
    - Topographic Position Index from Weiss (2001).
    - Roughness from Dartnell (2000).
    Also all referenced in Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962.

    Where Z is the elevation, x is the distance from left-right and y is the distance from top-bottom.
    Each pixel's index can be accessed at [:, row, col], returning an array of shape 4.

    Fill methods
        If the 3x3 matrix to fit the quadric function on has NaNs, these need to be handled:
        * 'median': NaNs are filled with the median value of the matrix.
        * 'mean': NaNs are filled with the mean value of the matrix.
        * 'none': If NaNs are encountered, skip the entire cell (default for GDAL and SAGA).

    Edge methods
        Each iteration requires a 3x3 matrix, so special edge cases have to be made.
        * 'nearest': Pixels outside the range are filled using the closest pixel value.
        * 'wrap': The array is wrapped so pixels near the right edge will be sampled from the left, etc.
        * 'none': Edges will not be analyzed, leaving a 1 pixel edge of NaNs.

    Quirks:
        * Edges are naively treated by filling the closest value, so that a 3x3 matrix is always calculated.\
                It may therefore be slightly off in the edges.
        * NaNs and infs are filled with the median of the finites in the matrix, possibly affecting the fit.
        * The X and Y resolution needs to be the same. It does not work if they differ.

    :param dem: The 2D DEM to be analyzed (3D DEMs of shape (1, row, col) are not supported)
    :param fill_method: Fill method to use for NaNs in the 3x3 matrix.
    :param edge_method: The method to use near the array edge.
    :param window_size: The size of the window

    :raises ValueError: If the inputs are poorly formatted.
    :raises RuntimeError: If unexpected backend errors occurred.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> indexes = get_windowed_indexes(dem)
        >>> indexes.shape
        (5, 3, 3)
        >>> indexes[:, 1, 1]
        array([2.82842712, 1., 1., 1., 1.27716652])

    :returns: An array of coefficients for each pixel of shape (5, row, col).
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

    if not isinstance(window_size, int) or window_size % 2 != 1:
        raise ValueError("Window size must be an odd integer.")

    allowed_fill_methods = ["median", "mean", "none"]
    allowed_edge_methods = ["nearest", "wrap", "none"]
    for value, name, allowed in zip(
        [fill_method, edge_method], ["fill", "edge"], (allowed_fill_methods, allowed_edge_methods)
    ):
        if value.lower() not in allowed:
            raise ValueError(f"Invalid {name} method: '{value}'. Choices: {allowed}")

    # Try to run the numba JIT code. It should never fail at this point, so if it does, it should be reported!
    try:
        indexes = _get_windowed_indexes(
            dem_arr, fill_method=fill_method.lower(), edge_method=edge_method.lower(),
            window_size=window_size
        )
    except Exception as exception:
        raise RuntimeError("Unhandled numba exception. Please raise an issue of what happened.") from exception

    return indexes


@overload
def get_terrain_attribute(
    dem: np.ndarray | np.ma.masked_array,
    attribute: str,
    resolution: tuple[float, float] | float | None,
    degrees: bool,
    hillshade_altitude: float,
    hillshade_azimuth: float,
    hillshade_z_factor: float,
    fill_method: str,
    edge_method: str,
    window_size: int
) -> np.ndarray:
    ...


@overload
def get_terrain_attribute(
    dem: np.ndarray | np.ma.masked_array,
    attribute: list[str],
    resolution: tuple[float, float] | float | None,
    degrees: bool,
    hillshade_altitude: float,
    hillshade_azimuth: float,
    hillshade_z_factor: float,
    fill_method: str,
    edge_method: str,
    window_size: int
) -> list[np.ndarray]:
    ...

@overload
def get_terrain_attribute(
    dem: RasterType,
    attribute: str,
    resolution: tuple[float, float] | float | None,
    degrees: bool,
    hillshade_altitude: float,
    hillshade_azimuth: float,
    hillshade_z_factor: float,
    fill_method: str,
    edge_method: str,
    window_size: int
) -> Raster:
    ...

@overload
def get_terrain_attribute(
    dem: RasterType,
    attribute: list[str],
    resolution: tuple[float, float] | float | None,
    degrees: bool,
    hillshade_altitude: float,
    hillshade_azimuth: float,
    hillshade_z_factor: float,
    fill_method: str,
    edge_method: str,
    window_size: int
) -> list[Raster]:
    ...


def get_terrain_attribute(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    attribute: str | list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    fill_method: str = "median",
    edge_method: str = "nearest",
    window_size: int = 3
) -> np.ndarray | list[np.ndarray] | Raster | list[Raster]:
    """
    Derive one or multiple terrain attributes from a DEM.
    The attributes are derived following Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962 which is
    directy based on the following references:
    - Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918,
    - Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.
    - Riley et al. (1999), http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf.
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Roughness from Dartnell (2000), http://dx.doi.org/10.14358/PERS.70.9.1081.
    More details on the equations in the functions get_quadric_coefficients() and get_windowed_indexes().

    Attributes:
        * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical).
        * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W)
        * 'hillshade': The shaded slope in relation to its aspect.
        * 'curvature': The second derivative of elevation (the rate of slope change per pixel), multiplied by 100.
        * 'planform_curvature': The curvature perpendicular to the direction of the slope.
        * 'profile_curvature': The curvature parallel to the direction of the slope.
        * 'maximum_curvature': The maximum curvature.
        * 'surface_fit': A quadric surface fit for each individual pixel.
        * 'terrain_ruggedness_index_topo': The terrain ruggedness index generally used for topography, defined by the
        squareroot of squared differences to neighbouring pixels.
        * 'terrain_ruggedness_index_bathy': The terrain ruggedness index generally used for bathymetry, defined by the
        mean absolute difference to neighbouring pixels.
        * 'topographic_position_index': The topographic position index defined by a difference to the average of
        neighbouring pixels.
        * 'roughness': The roughness, i.e. maximum difference to neighbouring pixels.

    :param dem: The DEM to analyze.
    :param attribute: The terrain attribute(s) to calculate.
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param degrees: Convert radians to degrees?
    :param hillshade_altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param hillshade_azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param hillshade_z_factor: Vertical exaggeration factor.
    :param fill_method: See the 'get_quadric_coefficients()' docstring for information.
    :param edge_method: See the 'get_quadric_coefficients()' docstring for information.
    :param window_size: The window size for windowed ruggedness and roughness indexes.

    :raises ValueError: If the inputs are poorly formatted or are invalid.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> slope, aspect = get_terrain_attribute(dem, ["slope", "aspect"], resolution=1)
        >>> slope  # Note the flattening edge effect; see 'get_quadric_coefficients()' for more.
        array([[26.56505118, 26.56505118, 26.56505118],
               [45.        , 45.        , 45.        ],
               [26.56505118, 26.56505118, 26.56505118]])
        >>> aspect
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

    :returns: One or multiple arrays of the requested attribute(s)
    """
    if isinstance(dem, gu.Raster):
        if resolution is None:
            resolution = dem.res

    # Validate and format the inputs
    if isinstance(attribute, str):
        attribute = [attribute]

    # These require the get_quadric_coefficients() function, which require the same X/Y resolution.
    list_requiring_surface_fit = ["curvature", "planform_curvature", "profile_curvature", "maximum_curvature",
                                  "slope", "hillshade", "aspect", "surface_fit"]
    attributes_requiring_surface_fit = [attr for attr in attribute if attr in list_requiring_surface_fit]

    list_requiring_windowed_index = ["terrain_ruggedness_index_topo", "terrain_ruggedness_index_bathy",
                                     "topographic_position_index", "roughness"]
    attributes_requiring_windowed_index = [attr for attr in attribute if attr in list_requiring_windowed_index]

    if resolution is None and len(attributes_requiring_surface_fit)>1:
        raise ValueError(f"'resolution' must be provided as an argument for attributes: {list_requiring_surface_fit}")

    choices = list_requiring_surface_fit + list_requiring_windowed_index
    for attr in attribute:
        if attr not in choices:
            raise ValueError(f"Attribute '{attr}' is not supported. Choices: {choices}")

    if (hillshade_azimuth < 0.0) or (hillshade_azimuth > 360.0):
        raise ValueError(f"Azimuth must be a value between 0 and 360 degrees (given value: {hillshade_azimuth})")
    if (hillshade_altitude < 0.0) or (hillshade_altitude > 90):
        raise ValueError("Altitude must be a value between 0 and 90 degress (given value: {altitude})")
    if (hillshade_z_factor < 0.0) or not np.isfinite(hillshade_z_factor):
        raise ValueError(f"z_factor must be a non-negative finite value (given value: {hillshade_z_factor})")

    # Initialize the terrain_attributes dictionary, which will be filled with the requested values.
    terrain_attributes: dict[str, np.ndarray] = {}

    # Check which products should be made to optimize the processing
    make_aspect = any(attr in attribute for attr in ["aspect", "hillshade"])
    make_slope = any(
        attr in attribute for attr in ["slope", "hillshade", "planform_curvature", "aspect", "profile_curvature",
                                       "maximum_curvature"]
    )
    make_hillshade = "hillshade" in attribute
    make_surface_fit = len(attributes_requiring_surface_fit) > 0
    make_curvature = "curvature" in attribute
    make_planform_curvature = "planform_curvature" in attribute or "maximum_curvature" in attribute
    make_profile_curvature = "profile_curvature" in attribute or  "maximum_curvature" in attribute
    make_maximum_curvature = "maximum_curvature" in attribute
    make_windowed_index = len(attributes_requiring_windowed_index) > 0
    make_terrain_ruggedness_topo = "terrain_ruggedness_index_topo" in attribute
    make_terrain_ruggedness_bathy = "terrain_ruggedness_index_bathy" in attribute
    make_topographic_position = "topographic_position_index" in attribute
    make_roughness = "roughness" in attribute

    # Get array of DEM
    dem_arr = xdem.spatial_tools.get_array_and_mask(dem)[0]

    if make_surface_fit:
        if not isinstance(resolution, Sized):
            resolution = (float(resolution), float(resolution))
        if resolution[0] != resolution[1]:
            raise ValueError(
                f"Quadric surface fit requires the same X and Y resolution ({resolution} was given). "
                f"This was required by: {attributes_requiring_surface_fit}"
            )
        terrain_attributes["surface_fit"] = get_quadric_coefficients(
            dem=dem_arr, resolution=resolution[0], fill_method=fill_method, edge_method=edge_method
        )

    if make_slope:
        # This calculation is based on (p18, left side): https://ieeexplore.ieee.org/document/1456186
        # SLOPE = -(G²+H²)**(1/2)
        terrain_attributes["slope"] = np.arctan(
            (terrain_attributes["surface_fit"][6, :, :] ** 2 + terrain_attributes["surface_fit"][7, :, :] ** 2) ** 0.5
        )

    if make_aspect:
        # ASPECT = ARCTAN(-H/-G)  # This did not work
        # ASPECT = (ARCTAN2(-G, H) + 0.5PI) % 2PI  did work.
        terrain_attributes["aspect"] = (
            np.arctan2(-terrain_attributes["surface_fit"][6, :, :], terrain_attributes["surface_fit"][7, :, :])
            + np.pi / 2
        ) % (2 * np.pi)

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
                + np.cos(altitude_rad) * np.sin(slopemap) * np.sin(azimuth_rad - terrain_attributes["aspect"])
            ),
            0,
            255,
        ).astype("float32")

    if make_curvature:
        # Curvature is the second derivative of the surface fit equation.
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
                * 100
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
                * 100
            )

        # Completely flat surfaces trigger the warning above. These need to be set to zero
        terrain_attributes["profile_curvature"][terrain_attributes["slope"] == 0.0] = 0.0

    if make_maximum_curvature:
        minc = np.minimum(terrain_attributes["profile_curvature"], terrain_attributes["planform_curvature"])
        maxc = np.maximum(terrain_attributes["profile_curvature"], terrain_attributes["planform_curvature"])
        terrain_attributes["maximum_curvature"] = np.where(np.abs(minc)>maxc, minc, maxc)

    if make_windowed_index:
        terrain_attributes["windowed_indexes"] = \
            get_windowed_indexes(dem=dem_arr, fill_method=fill_method, edge_method=edge_method, window_size=window_size)

    if make_terrain_ruggedness_topo:
        terrain_attributes["terrain_ruggedness_index_topo"] = terrain_attributes["windowed_indexes"][0, :, :]

    if make_terrain_ruggedness_bathy:
        terrain_attributes["terrain_ruggedness_index_bathy"] = terrain_attributes["windowed_indexes"][1, :, :]

    if make_topographic_position:
        terrain_attributes["topographic_position_index"] = terrain_attributes["windowed_indexes"][2, :, :]

    if make_roughness:
        terrain_attributes["roughness"] = terrain_attributes["windowed_indexes"][3, :, :]

    # Convert the unit if wanted.
    if degrees:
        for attr in ["slope", "aspect"]:
            if attr not in terrain_attributes:
                continue
            terrain_attributes[attr] = np.rad2deg(terrain_attributes[attr])

    output_attributes = [terrain_attributes[key].reshape(dem.shape) for key in attribute]

    if isinstance(dem, gu.Raster):
        output_attributes = [gu.Raster.from_array(attr, transform=dem.transform, crs=dem.crs, nodata=None) for attr in output_attributes]

    return output_attributes if len(output_attributes) > 1 else output_attributes[0]

@overload
def slope(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
    degrees: bool
) -> Raster: ...

@overload
def slope(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
    degrees: bool
) -> np.ndarray: ...

def slope(
    dem: np.ndarray | np.ma.masked_array | RasterType, resolution: float | tuple[float, float] | None = None, degrees: bool = True
) -> np.ndarray | Raster:
    """
    Generate a slope map for a DEM.
    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918.

    :param dem: The DEM to generate a slope map for.
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param degrees: Return a slope map in degrees (False means radians)

    :returns: A slope map of the same shape as 'dem' in degrees or radians.
    """
    return get_terrain_attribute(dem, attribute="slope", resolution=resolution, degrees=degrees)

@overload
def aspect(
    dem: np.ndarray | np.ma.masked_array,
    degrees: bool
) -> np.ndarray: ...

@overload
def aspect(
    dem: RasterType,
    degrees: bool
) -> Raster: ...

def aspect(dem: np.ndarray | np.ma.masked_array | RasterType, degrees: bool = True) -> np.ndarray | Raster:
    """
    Calculate the aspect of each cell in a DEM.
    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918.

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
               [0., 0., 0.]])
        >>> dem.T
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])
        >>> aspect(dem.T, degrees=True)
        array([[270., 270., 270.],
               [270., 270., 270.],
               [270., 270., 270.]])

    """
    return get_terrain_attribute(dem, attribute="aspect", resolution=1.0, degrees=degrees)

@overload
def hillshade(
    dem: RasterType,
    resolution: float | tuple[float, float],
    azimuth: float,
    altitude: float,
    z_factor: float,
) -> Raster: ...

@overload
def hillshade(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float],
    azimuth: float,
    altitude: float,
    z_factor: float,
) -> np.ndarray: ...

def hillshade(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None = None,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray | Raster:
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

@overload
def curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
) -> Raster: ...

@overload
def curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
) -> np.ndarray: ...

def curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
) -> np.ndarray | Raster:
    """
    Calculate the terrain curvature (second derivative of elevation).
    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

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


@overload
def planform_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
) -> Raster: ...

@overload
def planform_curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
) -> np.ndarray: ...

def planform_curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
) -> np.ndarray | Raster:
    """
    Calculate the terrain curvature perpendicular to the direction of the slope.
    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The planform curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="planform_curvature", resolution=resolution)


@overload
def profile_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
) -> Raster: ...

@overload
def profile_curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
) -> np.ndarray: ...

def profile_curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
) -> np.ndarray | Raster:
    """
    Calculate the terrain curvature parallel to the direction of the slope.
    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The profile curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="profile_curvature", resolution=resolution)


@overload
def maximum_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
) -> Raster: ...

@overload
def maximum_curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
) -> np.ndarray: ...

def maximum_curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
) -> np.ndarray | Raster:
    """
    Calculate the maximum profile or planform curvature parallel to the direction of the slope.
    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The profile curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="maximum_curvature", resolution=resolution)

@overload
def terrain_ruggedness_index(
    dem: RasterType,
    method: str,
    window_size: int
) -> Raster: ...

@overload
def terrain_ruggedness_index(
    dem: np.ndarray | np.ma.masked_array,
    method: str,
    window_size: int
) -> np.ndarray: ...

def terrain_ruggedness_index(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    method: str = "Riley",
    window_size: int = 3
) -> np.ndarray | Raster:
    """
    Calculates the Terrain Ruggedness Index.
    Based either on:
     - Riley et al. (1999),  http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf, preferred
     for topography.
     - Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962, preferred for bathymetry.

    :param dem: The DEM to calculate the terrain ruggedness index from.
    :param method: The algorithm used ("Riley" for topography or "Wilson" for bathymetry)
    :param window_size: The size of the window for deriving the terrain index

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The terrain ruggedness index array of the DEM.
    """
    if method.lower() == 'riley':
        return get_terrain_attribute(dem=dem, attribute="terrain_ruggedness_index_topo", window_size=window_size)
    elif method.lower() == 'wilson':
        return get_terrain_attribute(dem=dem, attribute="terrain_ruggedness_index_bathy", window_size=window_size)
    else:
        raise ValueError('Method for Terrain Ruggedness Index must be "Riley" or "Wilson".')


@overload
def topographic_position_index(
    dem: RasterType,
    window_size: int,
) -> Raster: ...


@overload
def topographic_position_index(
    dem: np.ndarray | np.ma.masked_array,
    window_size: int,
) -> np.ndarray: ...


def topographic_position_index(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    window_size: int = 3
) -> np.ndarray | Raster:
    """
    Calculates the Topographic Position Index.
    Based on: Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.

    :param dem: The DEM to calculate the topographic position index from.
    :param window_size: The size of the window for deriving the terrain index

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The topographic position index array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="topographic_position_index", window_size=window_size)


@overload
def roughness(
    dem: RasterType,
    window_size: int
) -> Raster: ...


@overload
def roughness(
    dem: np.ndarray | np.ma.masked_array,
    window_size: int
) -> np.ndarray: ...


def roughness(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    window_size: int = 3
) -> np.ndarray | Raster:
    """
    Calculates the roughness.
    Based on: Dartnell (2000), http://dx.doi.org/10.14358/PERS.70.9.1081.

    :param dem: The DEM to calculate the roughness from.
    :param window_size: The size of the window for deriving the terrain index

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The roughness array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="roughness", window_size=window_size)
