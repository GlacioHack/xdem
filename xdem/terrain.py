"""Terrain attribute calculations, such as slope, aspect, hillshade, curvature and ruggedness indexes."""
from __future__ import annotations

import warnings
from typing import Sized, overload

import numba
import numpy as np
import rasterio as rio

import geoutils as gu
from geoutils.georaster import RasterType, Raster

try:
    import richdem as rd
    _has_rd = True
except ImportError:
    _has_rd = False


def _rio_to_rda(ds: rio.DatasetReader) -> rd.rdarray:
    """
    Get georeferenced richDEM array from rasterio dataset
    :param ds: DEM
    :return: DEM
    """
    arr = ds.read(1)
    rda = rd.rdarray(arr, no_data=ds.get_nodatavals()[0])
    rda.geotransform = ds.get_transform()
    rda.projection = ds.get_gcps()

    return rda


def _get_terrainattr_richdem(ds: rio.DatasetReader, attribute='slope_radians') -> np.ndarray:
    """
    Derive terrain attribute for DEM opened with rasterio. One of "slope_degrees", "slope_percentage", "aspect",
    "profile_curvature", "planform_curvature", "curvature" and others (see RichDEM documentation).
    :param ds: DEM
    :param attribute: RichDEM terrain attribute
    :return:
    """
    rda = _rio_to_rda(ds)
    terrattr = rd.TerrainAttribute(rda, attrib=attribute)

    return np.array(terrattr)


@numba.njit(parallel=True)
def _get_quadric_coefficients(
    dem: np.ndarray, resolution: float, fill_method: str = "none", edge_method: str = "none",
        make_rugosity: bool = False) -> np.ndarray:
    """
    Run the pixel-wise analysis in parallel for a 3x3 window using the resolution.

    See the xdem.terrain.get_quadric_coefficients() docstring for more info.
    """
    # Rename the resolution
    L = resolution

    # Allocate the output.
    output = np.empty((12,) + dem.shape, dtype=dem.dtype) + np.nan

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
                else:
                    row_indexer = row + k
                    col_indexer = col + j
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

        if make_rugosity:

            # Rugosity is computed on a 3x3 window like the quadratic coefficients, see Jenness (2004) for details

            # For this, we need elevation differences and horizontal length of 16 segments
            dzs = np.zeros((16,))
            dls = np.zeros((16,))

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
                    dls[count_without_center] = np.sqrt(j ** 2 + k ** 2)*L
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
            hsl = np.sqrt(dzs ** 2 + dls ** 2) / 2

            # Starting from up direction anticlockwise, every triangle has 2 segments between center and surrounding pixels
            # and 1 segment between surrounding pixels; pixel 4 is the center
            # above 4 the index of center-surrounding segment decrease by 1, as the center pixel was skipped
            # Triangle 1: pixels 3 and 0
            T1 = [hsl[3], hsl[0], hsl[12]]
            # Triangle 2: pixels 0 and 1
            T2 = [hsl[0], hsl[1], hsl[8]]
            # Triangle 3: pixels 1 and 2
            T3 = [hsl[1], hsl[2], hsl[9]]
            # Triangle 4: pixels 2 and 5
            T4 = [hsl[2], hsl[4], hsl[14]]
            # Triangle 5: pixels 5 and 8
            T5 = [hsl[4], hsl[7], hsl[15]]
            # Triangle 6: pixels 8 and 7
            T6 = [hsl[7], hsl[6], hsl[11]]
            # Triangle 7: pixels 7 and 6
            T7 = [hsl[6], hsl[5], hsl[10]]
            # Triangle 8: pixels 6 and 3
            T8 = [hsl[5], hsl[3], hsl[13]]

            list_T = [T1, T2, T3, T4, T5, T6, T7, T8]

            # Finally, we compute the 3D surface areas of the 8 triangles
            A = np.empty((8,))
            count = 0
            for T in list_T:
                # Half sum of lengths
                hs = sum(T) / 2
                # Surface area of triangle
                A[count] = np.sqrt(hs * (hs - T[0]) * (hs - T[1]) * (hs - T[2]))
                count += 1

        # Assign the A, B, C, D etc., factors to the output. This ugly syntax is needed to make parallel numba happy.

        # Coefficients of Zevenberg and Thorne (1987), Equations 3 to 11
        output[0, row, col] = ((Z[0] + Z[2] + Z[6] + Z[8]) / 4 - (Z[1] + Z[3] + Z[5] + Z[7]) / 2 + Z[4]) / (L ** 4)  # A
        output[1, row, col] = ((Z[0] + Z[2] - Z[6] - Z[8]) / 4 - (Z[1] - Z[7]) / 2) / (L ** 3)  # B
        output[2, row, col] = ((-Z[0] + Z[2] - Z[6] + Z[8]) / 4 + (Z[3] - Z[5]) / 2) / (L ** 3)  # C
        output[3, row, col] = ((Z[3] + Z[5]) / 2 - Z[4]) / (L ** 2)  # D
        output[4, row, col] = ((Z[1] + Z[7]) / 2 - Z[4]) / (L ** 2)  # E
        output[5, row, col] = (-Z[0] + Z[2] + Z[6] - Z[8]) / (4 * L ** 2)  # F
        output[6, row, col] = (-Z[3] + Z[5]) / (2 * L)  # G
        output[7, row, col] = (Z[1] - Z[7]) / (2 * L)  # H
        output[8, row, col] = Z[4]  # I

        # Refined coefficients for slope of Horn (1981), page 18 bottom left equations.
        output[9, row, col] = ((Z[6] + 2 * Z[7] + Z[8]) - (Z[0] + 2 * Z[1] + Z[2])) / (8 * L)
        output[10, row, col] = ((Z[6] + 2 * Z[3] + Z[0]) - (Z[8] + 2 * Z[5] + Z[2])) / (8 * L)

        # Rugosity from Jenness (2004): difference between real surface area and planimetric surface area
        if make_rugosity:
            output[11, row, col] = sum(A) / L**2

    return output


def get_quadric_coefficients(
    dem: np.ndarray, resolution: float, fill_method: str = "none", edge_method: str = "none",
        make_rugosity : bool = False) -> np.ndarray:
    """
    Computes quadric and other coefficients on a fixed 3x3 pixel window, and that depends on the resolution.
    Returns the 9 coefficients of a quadric surface fit to every pixel in the raster, the 2 coefficients of optimized
    slope gradient, and the rugosity.

    For the quadric surface, based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107, also
    described in the documentation:
    https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
    The function that is solved is:
    Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I
    Where Z is the elevation, x is the distance from left-right and y is the distance from top-bottom.

    For the 2 coefficients of optimized slope, based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918, page 18
    bottom left equations.

    For the rugosity, based on Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.

    Each pixel's fit can be accessed by coefficients[:, row, col], returning an array of shape 12.

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
    :param make_rugosity: Whether to compute coefficients for rugosity.

    :raises ValueError: If the inputs are poorly formatted.
    :raises RuntimeError: If unexpected backend errors occurred.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> coeffs = get_quadric_coefficients(dem, resolution=1.0, make_rugosity=True)
        >>> coeffs.shape
        (12, 3, 3)
        >>> coeffs[:9, 1, 1]
        array([ 1.,  0.,  0., -1., -1.,  0.,  0.,  0.,  2.])
        >>> coeffs[9:11, 1, 1]
        array([0., 0.])
        >>> coeffs[11, 1, 1]
        1.4142135623730954

    :returns: An array of coefficients for each pixel of shape (9, row, col).
    """
    # This function only formats and validates the inputs. For the true functionality, see _get_quadric_coefficients()
    dem_arr = gu.spatial_tools.get_array_and_mask(dem)[0]

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
            dem_arr, resolution, fill_method=fill_method.lower(), edge_method=edge_method.lower(),
            make_rugosity=make_rugosity)
    except Exception as exception:
        raise RuntimeError("Unhandled numba exception. Please raise an issue of what happened.") from exception

    return coeffs

@numba.njit(parallel=True)
def _get_windowed_indexes(
    dem: np.ndarray, fill_method: str = "median", edge_method: str = "nearest", window_size: int = 3,
    make_fractal_roughness: bool = False) -> np.ndarray:
    """
    Run the pixel-wise analysis in parallel for any window size without using the resolution.

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
            if (row < hw) or (row >= (dem.shape[0] - hw)) or (col < hw) or (col >= (dem.shape[1] - hw)):
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
                else:
                    row_indexer = row + k
                    col_indexer = col + j
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

        if make_fractal_roughness:
            # Fractal roughness computation according to the box-counting method of Taud and Parrot (2005)
            # First, we compute the number of voxels for each pixel of Equation 4
            count = 0
            V = np.empty((window_size, window_size))
            for j in range(-hw, -hw + window_size):
                for k in range(-hw, -hw + window_size):
                    T = (Z[count] - Z[index_middle_pixel])
                    # The following is the equivalent of np.clip, written like this for numba
                    if T < 0:
                        V[hw + j, hw + k] = 0
                    elif T > window_size:
                        V[hw + j, hw + k] = window_size
                    else:
                        V[hw + j, hw + k] = T
                    count += 1

            # Then, we compute the maximum number of voxels for varying box splitting of the cube of side the window size,
            # following Equation 5

            # Get all the divisors of the half window size
            list_box_sizes = []
            for j in range(1, hw + 1):
                if hw % j == 0:
                    list_box_sizes.append(j)

            Ns = np.empty((len(list_box_sizes),))
            for l in range(0, len(list_box_sizes)):
                # We loop over boxes of size q x q in the cube
                q = list_box_sizes[l]
                sumNs = 0
                for j in range(0, int((window_size-1)/q)):
                    for k in range(0, int((window_size-1)/q)):
                        sumNs += np.max(V[slice(j*q, (j+1)*q), slice(k*q, (k+1)*q)].flatten())
                Ns[l] = sumNs / q

            # Finally, we calculate the slope of the logarithm of Ns with q
            # We do the linear regression manually, as np.polyfit is not supported by numba
            x = np.log(np.array(list_box_sizes))
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

        # First output is the Terrain Ruggedness Index from Riley et al. (1999): squareroot of squared sum of
        # differences between center and neighbouring pixels
        output[0, row, col] = np.sqrt(np.sum(S**2))
        # Second output is the Terrain Ruggedness Index from Wilson et al. (2007): mean difference between center
        # and neighbouring pixels
        output[1, row, col] = np.sum(S) / (window_size**2 - 1)
        # Third output is the Topographic Position Index from Weiss (2001): difference between center and mean of
        # neighbouring pixels
        output[2, row, col] =  Z[index_middle_pixel] - (np.sum(Z) - Z[index_middle_pixel]) / (window_size**2 - 1)
        # Fourth output is the Roughness from Dartnell (2000): difference between maximum and minimum of the window
        output[3, row, col] = np.max(Z) - np.min(Z)

        if make_fractal_roughness:
            # Fifth output is the Fractal Roughness from Taud et Parrot (2005): estimate of the fractal dimension
            # based on volume box-counting
            output[4, row, col] = D

    return output


def get_windowed_indexes(
    dem: np.ndarray, fill_method: str = "none", edge_method: str = "none", window_size: int = 3,
    make_fractal_roughness: bool = False) -> np.ndarray:
    """
    Return terrain indexes based on a windowed calculation of variable size, independent of the resolution.

    Includes:

    - Terrain Ruggedness Index from Riley et al. (1999),  http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf, for topography and from Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962, for bathymetry.
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Roughness from Dartnell (2000), http://dx.doi.org/10.14358/PERS.70.9.1081.
    - Fractal roughness from Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.
    Nearly all are also referenced in Wilson et al. (2007).

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

    :param dem: The 2D DEM to be analyzed (3D DEMs of shape (1, row, col) are not supported).
    :param fill_method: Fill method to use for NaNs in the 3x3 matrix.
    :param edge_method: The method to use near the array edge.
    :param window_size: The size of the window.
    :param make_fractal_roughness: Whether to compute fractal roughness.

    :raises ValueError: If the inputs are poorly formatted.
    :raises RuntimeError: If unexpected backend errors occurred.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> indexes = get_windowed_indexes(dem)
        >>> indexes.shape
        (5, 3, 3)
        >>> indexes[:4, 1, 1]
        array([2.82842712, 1.        , 1.        , 1.        ])

    :returns: An array of coefficients for each pixel of shape (5, row, col).
    """
    # This function only formats and validates the inputs. For the true functionality, see _get_quadric_coefficients()
    dem_arr = gu.spatial_tools.get_array_and_mask(dem)[0]

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
            window_size=window_size, make_fractal_roughness=make_fractal_roughness
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
    slope_method: str,
    tri_method: str,
    fill_method: str,
    edge_method: str,
    use_richdem: bool,
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
    slope_method: str,
    tri_method: str,
    fill_method: str,
    edge_method: str,
    use_richdem: bool,
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
    slope_method: str,
    tri_method: str,
    fill_method: str,
    edge_method: str,
    use_richdem: bool,
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
    slope_method: str,
    tri_method: str,
    fill_method: str,
    edge_method: str,
    use_richdem: bool,
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
    slope_method: str = "Horn",
    tri_method: str = "Riley",
    fill_method: str = "none",
    edge_method: str = "none",
    use_richdem: bool = False,
    window_size: int = 3
) -> np.ndarray | list[np.ndarray] | Raster | list[Raster]:
    """
    Derive one or multiple terrain attributes from a DEM.
    The attributes are based on:

    - Slope, aspect, hillshade (first method) from Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918,
    - Slope, aspect, hillshade (second method), and terrain curvatures from Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Terrain Ruggedness Index (topography) from Riley et al. (1999), http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf.
    - Terrain Ruggedness Index (bathymetry) from Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962.
    - Roughness from Dartnell (2000), http://dx.doi.org/10.14358/PERS.70.9.1081.
    - Rugosity from Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.
    - Fractal roughness from Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    Aspect and hillshade are derived using the slope, and thus depend on the same method.
    More details on the equations in the functions get_quadric_coefficients() and get_windowed_indexes().
    The slope, aspect ("Horn" method), and all curvatures ("ZevenbergThorne" method) can also be derived using RichDEM.

    Attributes:

    * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical). Default method: "Horn".
    * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W).
    * 'hillshade': The shaded slope in relation to its aspect.
    * 'curvature': The second derivative of elevation (the rate of slope change per pixel), multiplied by 100.
    * 'planform_curvature': The curvature perpendicular to the direction of the slope, multiplied by 100.
    * 'profile_curvature': The curvature parallel to the direction of the slope, multiplied by 100.
    * 'maximum_curvature': The maximum curvature.
    * 'surface_fit': A quadric surface fit for each individual pixel.
    * 'topographic_position_index': The topographic position index defined by a difference to the average of neighbouring pixels.
    * 'terrain_ruggedness_index': The terrain ruggedness index. For topography, defined by the squareroot of squared differences to neighbouring pixels. For bathymetry, defined by the mean absolute difference to neighbouring pixels. Default method: "Riley" (topography).
    * 'roughness': The roughness, i.e. maximum difference between neighbouring pixels.
    * 'rugosity': The rugosity, i.e. difference between real and planimetric surface area.
    * 'fractal_roughness': The roughness based on a volume box-counting estimate of the fractal dimension.

    :param dem: The DEM to analyze.
    :param attribute: The terrain attribute(s) to calculate.
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param degrees: Convert radians to degrees?
    :param hillshade_altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param hillshade_azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param hillshade_z_factor: Vertical exaggeration factor.
    :param slope_method: Method to calculate the slope, aspect and hillshade: "Horn" or "ZevenbergThorne".
    :param tri_method: Method to calculate the Terrain Ruggedness Index: "Riley" (topography) or "Wilson" (bathymetry).
    :param fill_method: See the 'get_quadric_coefficients()' docstring for information.
    :param edge_method: See the 'get_quadric_coefficients()' docstring for information.
    :param use_richdem: Whether to use richDEM for slope, aspect and curvature calculations.
    :param window_size: The window size for windowed ruggedness and roughness indexes.

    :raises ValueError: If the inputs are poorly formatted or are invalid.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> slope, aspect = get_terrain_attribute(dem, ["slope", "aspect"], resolution=1, edge_method='nearest')
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
                                  "slope", "hillshade", "aspect", "surface_fit", "rugosity"]
    attributes_requiring_surface_fit = [attr for attr in attribute if attr in list_requiring_surface_fit]

    if use_richdem:

        if not _has_rd:
            raise ValueError("Optional dependency needed. Install 'richdem'")

        if ("slope" in attribute or "aspect" in attribute) and slope_method == 'ZevenbergThorne':
            raise ValueError("RichDEM can only compute the slope and aspect using the default method of Horn (1981)")

        list_requiring_richdem = ["slope", "aspect", "hillshade", "curvature", "planform_curvature",
                                  "profile curvature", "maximum_curvature"]
        attributes_using_richdem = [attr for attr in attribute if attr in list_requiring_richdem]
        for attr in attributes_using_richdem:
            attributes_requiring_surface_fit.remove(attr)

        if isinstance(dem, gu.Raster):
            # Prepare rasterio.Dataset to pass to RichDEM
            ds = dem.ds
        else:
            # Here, maybe we could pass the geotransform based on the resolution, and add a "default" projection as
            # this is mandated but likely not used by the rdarray format of RichDEM...
            # For now, not supported
            raise ValueError("To derive RichDEM attributes, the DEM passed must be a Raster object")

    list_requiring_windowed_index = ["terrain_ruggedness_index", "topographic_position_index", "roughness",
                                     "fractal_roughness"]
    attributes_requiring_windowed_index = [attr for attr in attribute if attr in list_requiring_windowed_index]

    if resolution is None and len(attributes_requiring_surface_fit)>1:
        raise ValueError(f"'resolution' must be provided as an argument for attributes: {list_requiring_surface_fit}")

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
    make_topographic_position = "topographic_position_index" in attribute
    make_terrain_ruggedness = "terrain_ruggedness_index" in attribute
    make_roughness = "roughness" in attribute
    make_rugosity = "rugosity" in attribute
    make_fractal_roughness = "fractal_roughness" in attribute

    # Get array of DEM
    dem_arr = gu.spatial_tools.get_array_and_mask(dem)[0]

    if make_surface_fit:
        if not isinstance(resolution, Sized):
            resolution = (float(resolution), float(resolution))
        if resolution[0] != resolution[1]:
            raise ValueError(
                f"Quadric surface fit requires the same X and Y resolution ({resolution} was given). "
                f"This was required by: {attributes_requiring_surface_fit}"
            )
        terrain_attributes["surface_fit"] = get_quadric_coefficients(
            dem=dem_arr, resolution=resolution[0], fill_method=fill_method, edge_method=edge_method,
            make_rugosity=make_rugosity)

    if make_slope:

        if use_richdem:
            terrain_attributes["slope"] = _get_terrainattr_richdem(ds, attribute="slope_radians")

        else:
            if slope_method == "Horn":
                # This calculation is based on page 18 (bottom left) and 20-21 of Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918.
                terrain_attributes["slope"] = np.arctan(
                    (terrain_attributes["surface_fit"][9, :, :] ** 2 + terrain_attributes["surface_fit"][10, :, :] ** 2) ** 0.5
                )

            elif slope_method == "ZevenbergThorne":
                # This calculation is based on Equation 13 of Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.
                # SLOPE = ARCTAN((G²+H²)**(1/2))
                terrain_attributes["slope"] = np.arctan(
                    (terrain_attributes["surface_fit"][6, :, :] ** 2 + terrain_attributes["surface_fit"][7, :, :] ** 2) ** 0.5
                )


    if make_aspect:

        if use_richdem:
            # The aspect of RichDEM is returned in degrees, we convert to radians to match the others
            terrain_attributes["aspect"] = np.deg2rad(_get_terrainattr_richdem(ds, attribute="aspect"))
            # For flat slopes, RichDEM returns a 90° aspect by default, while GDAL return a 180° aspect
            # We stay consistent with GDAL
            slope_tmp = _get_terrainattr_richdem(ds, attribute="slope_radians")
            terrain_attributes["aspect"][slope_tmp == 0] = np.pi

        else:
            # ASPECT = ARCTAN(-H/-G)  # This did not work
            # ASPECT = (ARCTAN2(-G, H) + 0.5PI) % 2PI  did work.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in remainder")
                if slope_method == "Horn":
                    # This uses the estimates from Horn (1981).
                    terrain_attributes["aspect"] = (-
                                                           np.arctan2(-terrain_attributes["surface_fit"][9, :, :],
                                                                      terrain_attributes["surface_fit"][10, :, :])
                                                           -  np.pi
                                                   ) % (2 * np.pi)

                elif slope_method == "ZevenbergThorne":
                    # This uses the slope estimate from Zevenbergen and Thorne (1987).
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

        # The operation below yielded the closest hillshade to GDAL (multiplying by 255 did not work)
        # As 0 is generally no data for this uint8, we add 1 and then 0.5 for the rounding to occur between 1 and 255
        terrain_attributes["hillshade"] = np.clip(
            1.5 + 254
            * (
                np.sin(altitude_rad) * np.cos(slopemap)
                + np.cos(altitude_rad) * np.sin(slopemap) * np.sin(azimuth_rad - terrain_attributes["aspect"])
            ),
            0,
            255,
        ).astype("float32")

    if make_curvature:

        if use_richdem:
            terrain_attributes["curvature"] = _get_terrainattr_richdem(ds, attribute="curvature")

        else:
            # Curvature is the second derivative of the surface fit equation.
            # (URL in get_quadric_coefficients() docstring)
            # Curvature = -2(D + E) * 100
            terrain_attributes["curvature"] = (
                -2 * (terrain_attributes["surface_fit"][3, :, :] + terrain_attributes["surface_fit"][4, :, :]) * 100
            )

    if make_planform_curvature:

        if use_richdem:
            terrain_attributes["planform_curvature"] = _get_terrainattr_richdem(ds, attribute="planform_curvature")

        else:
            # PLANC = 2(DH² + EG² -FGH)/(G²+H²)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
                terrain_attributes["planform_curvature"] = (
                    - 2
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
            terrain_attributes["planform_curvature"][terrain_attributes["surface_fit"][6, :, :] ** 2 +
                                                     terrain_attributes["surface_fit"][7, :, :] ** 2 == 0.0] = 0.0

    if make_profile_curvature:

        if use_richdem:
            terrain_attributes["profile_curvature"] = _get_terrainattr_richdem(ds, attribute="profile_curvature")

        else:
            # PROFC = -2(DG² + EH² + FGH)/(G²+H²)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
                terrain_attributes["profile_curvature"] = (
                    2
                    * (
                        terrain_attributes["surface_fit"][3, :, :] * terrain_attributes["surface_fit"][6, :, :] ** 2
                        + terrain_attributes["surface_fit"][4, :, :] * terrain_attributes["surface_fit"][7, :, :] ** 2
                        + terrain_attributes["surface_fit"][5, :, :]
                        * terrain_attributes["surface_fit"][6, :, :]
                        * terrain_attributes["surface_fit"][7, :, :]
                    )
                    / (terrain_attributes["surface_fit"][6, :, :] ** 2 + terrain_attributes["surface_fit"][7, :, :] ** 2)
                    * 100
                )

            # Completely flat surfaces trigger the warning above. These need to be set to zero
            terrain_attributes["profile_curvature"][terrain_attributes["surface_fit"][6, :, :] ** 2 +
                                                     terrain_attributes["surface_fit"][7, :, :] ** 2 == 0.0] = 0.0

    if make_maximum_curvature:
        minc = np.minimum(terrain_attributes["profile_curvature"], terrain_attributes["planform_curvature"])
        maxc = np.maximum(terrain_attributes["profile_curvature"], terrain_attributes["planform_curvature"])
        terrain_attributes["maximum_curvature"] = np.where(np.abs(minc)>maxc, minc, maxc)

    if make_windowed_index:
        terrain_attributes["windowed_indexes"] = \
            get_windowed_indexes(dem=dem_arr, fill_method=fill_method, edge_method=edge_method,
                                 window_size=window_size, make_fractal_roughness=make_fractal_roughness)

    if make_topographic_position:
        terrain_attributes["topographic_position_index"] = terrain_attributes["windowed_indexes"][2, :, :]

    if make_terrain_ruggedness:

        if tri_method == "Riley":
            terrain_attributes["terrain_ruggedness_index"] = terrain_attributes["windowed_indexes"][0, :, :]

        elif tri_method == "Wilson":
            terrain_attributes["terrain_ruggedness_index"] = terrain_attributes["windowed_indexes"][1, :, :]

    if make_roughness:
        terrain_attributes["roughness"] = terrain_attributes["windowed_indexes"][3, :, :]

    if make_rugosity:
        terrain_attributes["rugosity"] = terrain_attributes["surface_fit"][11, :, :]

    if make_fractal_roughness:
        terrain_attributes["fractal_roughness"] = terrain_attributes["windowed_indexes"][4, :, :]

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
    method: str,
    degrees: bool,
    use_richdem: bool,
) -> Raster: ...

@overload
def slope(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
    method: str,
    degrees: bool,
    use_richdem: bool,
) -> np.ndarray: ...

def slope(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
    method: str = "Horn",
    degrees: bool = True,
    use_richdem: bool = False
) -> np.ndarray | Raster:
    """
    Generate a slope map for a DEM, returned in degrees by default.

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918 and on Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to generate a slope map for.
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param method: Method to calculate slope: "Horn" or "ZevenbergThorne".
    :param degrees: Return a slope map in degrees (False means radians).
    :param use_richdem: Whether to use RichDEM to compute the attribute.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> slope(dem, resolution=1, degrees=True)[1, 1] # Slope in degrees
        45.0
        >>> np.tan(slope(dem, resolution=2, degrees=True)[1, 1] * np.pi / 180.) # Slope in percentage
        0.5

    :returns: A slope map of the same shape as 'dem' in degrees or radians.
    """
    return get_terrain_attribute(dem, attribute="slope", slope_method=method, resolution=resolution, degrees=degrees,
                                 use_richdem=use_richdem)

@overload
def aspect(
    dem: np.ndarray | np.ma.masked_array,
    method: str,
    degrees: bool,
    use_richdem: bool,
) -> np.ndarray: ...

@overload
def aspect(
    dem: RasterType,
    method: str,
    degrees: bool,
    use_richdem: bool,
) -> Raster: ...

def aspect(dem: np.ndarray | np.ma.masked_array | RasterType,
           method: str = "Horn",
           degrees: bool = True,
           use_richdem: bool = False,
           ) -> np.ndarray | Raster:
    """
    Calculate the aspect of each cell in a DEM, returned in degrees by default. The aspect of flat slopes is 180° by
    default (as in GDAL).

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918 and on Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107.

    0=N, 90=E, 180=S, 270=W.

    :param dem: The DEM to calculate the aspect from.
    :param method: Method to calculate aspect: "Horn" or "ZevenbergThorne".
    :param degrees: Return an aspect map in degrees (if False, returns radians)
    :param use_richdem: Whether to use RichDEM to compute the attribute.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> aspect(dem, degrees=True)[1, 1]
        0.0
        >>> dem.T
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])
        >>> aspect(dem.T, degrees=True)[1, 1]
        270.0

    """
    return get_terrain_attribute(dem, attribute="aspect", slope_method=method, resolution=1.0, degrees=degrees,
                                 use_richdem=use_richdem)

@overload
def hillshade(
    dem: RasterType,
    resolution: float | tuple[float, float],
    method: str,
    azimuth: float,
    altitude: float,
    z_factor: float,
    use_richdem: bool,
) -> Raster: ...

@overload
def hillshade(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float],
    method: str,
    azimuth: float,
    altitude: float,
    z_factor: float,
    use_richdem: bool,
) -> np.ndarray: ...

def hillshade(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None = None,
    method: str = "Horn",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    use_richdem: bool = False,
) -> np.ndarray | Raster:
    """
    Generate a hillshade from the given DEM. The value 0 is used for nodata, and 1 to 255 for hillshading.

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918.

    :param dem: The input DEM to calculate the hillshade from.
    :param resolution: One or two values specifying the resolution of the DEM.
    :param method: Method to calculate the slope and aspect used for hillshading.
    :param azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param z_factor: Vertical exaggeration factor.
    :param use_richdem: Whether to use RichDEM to compute the slope and aspect used for the hillshade.


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
        use_richdem=use_richdem
    )

@overload
def curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> Raster: ...

@overload
def curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> np.ndarray: ...

def curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
    use_richdem: bool = False,
) -> np.ndarray | Raster:
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
    :param resolution: The X/Y resolution of the DEM.
    :param use_richdem: Whether to use RichDEM to compute the attribute.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> curvature(dem, resolution=1.0)[1, 1] / 100.
        4.0

    :returns: The curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="curvature", resolution=resolution, use_richdem=use_richdem)


@overload
def planform_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> Raster: ...

@overload
def planform_curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> np.ndarray: ...

def planform_curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
    use_richdem: bool = False,
) -> np.ndarray | Raster:
    """
    Calculate the terrain curvature perpendicular to the direction of the slope in m-1 multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.
    :param use_richdem: Whether to use RichDEM to compute the attribute.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> planform_curvature(dem, resolution=1.0)[1, 1] / 100.
        -0.0
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> planform_curvature(dem, resolution=1.0)[1, 1] / 100.
        -4.0

    :returns: The planform curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="planform_curvature", resolution=resolution, use_richdem=use_richdem)


@overload
def profile_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> Raster: ...

@overload
def profile_curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> np.ndarray: ...

def profile_curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
    use_richdem: bool = False,
) -> np.ndarray | Raster:
    """
    Calculate the terrain curvature parallel to the direction of the slope in m-1 multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.
    :param use_richdem: Whether to use RichDEM to compute the attribute.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> profile_curvature(dem, resolution=1.0)[1, 1] / 100.
        1.0
        >>> dem = np.array([[1, 2, 3],
        ...                 [1, 2, 3],
        ...                 [1, 2, 3]], dtype="float32")
        >>> profile_curvature(dem, resolution=1.0)[1, 1] / 100.
        0.0

    :returns: The profile curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="profile_curvature", resolution=resolution, use_richdem=use_richdem)


@overload
def maximum_curvature(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> Raster: ...

@overload
def maximum_curvature(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
    use_richdem: bool,
) -> np.ndarray: ...

def maximum_curvature(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None,
    use_richdem: bool = False,
) -> np.ndarray | Raster:
    """
    Calculate the signed maximum profile or planform curvature parallel to the direction of the slope in m-1
    multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM.
    :param use_richdem: Whether to use RichDEM to compute the attribute.

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The profile curvature array of the DEM.
    """
    return get_terrain_attribute(dem=dem, attribute="maximum_curvature", resolution=resolution, use_richdem=use_richdem)

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
    Calculates the Topographic Position Index, the difference to the average of neighbouring pixels. Output is in the
    unit of the DEM (typically meters).

    Based on: Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.

    :param dem: The DEM to calculate the topographic position index from.
    :param window_size: The size of the window for deriving the metric.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> topographic_position_index(dem)[1, 1]
        1.0
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> topographic_position_index(dem)[1, 1]
        0.0

    :returns: The topographic position index array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(dem=dem, attribute="topographic_position_index", window_size=window_size)


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
    Calculates the Terrain Ruggedness Index, the cumulated differences to neighbouring pixels. Output is in the
    unit of the DEM (typically meters).

    Based either on:

    * Riley et al. (1999), http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf that derives the squareroot of squared differences to neighbouring pixels, preferred for topography.
    * Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962 that derives the mean absolute difference to neighbouring pixels, preferred for bathymetry.

    :param dem: The DEM to calculate the terrain ruggedness index from.
    :param method: The algorithm used ("Riley" for topography or "Wilson" for bathymetry).
    :param window_size: The size of the window for deriving the metric.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> terrain_ruggedness_index(dem)[1, 1]
        2.8284271247461903
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> terrain_ruggedness_index(dem)[1, 1]
        0.0

    :returns: The terrain ruggedness index array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(dem=dem, attribute="terrain_ruggedness_index", tri_method=method, window_size=window_size)


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
    Calculates the roughness, the maximum difference between neighbouring pixels, for any window size. Output is in the
    unit of the DEM (typically meters).

    Based on: Dartnell (2000), http://dx.doi.org/10.14358/PERS.70.9.1081.

    :param dem: The DEM to calculate the roughness from.
    :param window_size: The size of the window for deriving the metric.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> roughness(dem)[1, 1]
        1.0
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> roughness(dem)[1, 1]
        0.0

    :returns: The roughness array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(dem=dem, attribute="roughness", window_size=window_size)


@overload
def rugosity(
    dem: RasterType,
    resolution: float | tuple[float, float] | None,
) -> Raster: ...


@overload
def rugosity(
    dem: np.ndarray | np.ma.masked_array,
    resolution: float | tuple[float, float] | None,
) -> np.ndarray: ...


def rugosity(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    resolution: float | tuple[float, float] | None = None
) -> np.ndarray | Raster:
    """
    Calculates the rugosity, the ratio between real area and planimetric area. Only available for a 3x3 window. The
    output is unitless.

    Based on: Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.

    :param dem: The DEM to calculate the rugosity from.
    :param resolution: The X/Y resolution of the DEM.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> rugosity(dem, resolution=1.)[1, 1]
        1.4142135623730954
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> np.round(rugosity(dem, resolution=1.)[1, 1], 5)
        1.0

    :returns: The rugosity array of the DEM (unitless).
    """
    return get_terrain_attribute(dem=dem, attribute="rugosity", resolution=resolution)


@overload
def fractal_roughness(
    dem: RasterType,
    window_size: int
) -> Raster: ...


@overload
def fractal_roughness(
    dem: np.ndarray | np.ma.masked_array,
    window_size: int
) -> np.ndarray: ...


def fractal_roughness(
    dem: np.ndarray | np.ma.masked_array | RasterType,
    window_size: int = 13
) -> np.ndarray | Raster:
    """
    Calculates the fractal roughness, the local 3D fractal dimension. Can only be computed on window sizes larger or
    equal to 5x5, defaults to 13x13. Output unit is a fractal dimension between 1 and 3.

    Based on: Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    :param dem: The DEM to calculate the roughness from.
    :param window_size: The size of the window for deriving the metric.

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[1, 1] = 6.5
        >>> np.round(fractal_roughness(dem)[6, 6], 5) # The fractal dimension of a line is 1
        1.0
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[:, 1] = 13
        >>> np.round(fractal_roughness(dem)[6, 6]) # The fractal dimension of plane is 2
        2.0
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[:, :6] = 13 # The fractal dimension of a cube is 3
        >>> np.round(fractal_roughness(dem)[6, 6]) # The fractal dimension of cube is 3
        3.0

    :returns: The fractal roughness array of the DEM in fractal dimension (between 1 and 3).
    """
    return get_terrain_attribute(dem=dem, attribute="fractal_roughness", window_size=window_size)
