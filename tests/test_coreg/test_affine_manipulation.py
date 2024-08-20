from typing import Optional

import geopandas as gpd
import geoutils as gu
import numpy as np
import pytest
import pytransform3d.rotations
import rasterio as rio
from scipy.ndimage import binary_dilation

from xdem import coreg, misc, spatialstats
from xdem._typing import NDArrayf
from xdem.coreg.base import apply_matrix

# Identity transformation
matrix_identity = np.diag(np.ones(4, float))

# Vertical shift
matrix_vertical = matrix_identity.copy()
matrix_vertical[2, 3] = 1

# Vertical and horizontal shifts
matrix_translations = matrix_identity.copy()
matrix_translations[:3, 3] = [0.5, 1, 1.5]

# Single rotation
rotation = np.deg2rad(5)
matrix_rotations = matrix_identity.copy()
matrix_rotations[1, 1] = np.cos(rotation)
matrix_rotations[2, 2] = np.cos(rotation)
matrix_rotations[2, 1] = -np.sin(rotation)
matrix_rotations[1, 2] = np.sin(rotation)

# Mix of translations and rotations in all axes (X, Y, Z) simultaneously
rotation_x = 5
rotation_y = 10
rotation_z = 3
e = np.deg2rad(np.array([rotation_x, rotation_y, rotation_z]))
# This is a 3x3 rotation matrix
rot_matrix = pytransform3d.rotations.matrix_from_euler(e=e, i=0, j=1, k=2, extrinsic=True)
matrix_all = matrix_rotations.copy()
matrix_all[0:3, 0:3] = rot_matrix
matrix_all[:3, 3] = [0.5, 1, 1.5]

list_matrices = [matrix_identity, matrix_vertical, matrix_translations, matrix_rotations, matrix_all]


@pytest.mark.parametrize("matrix", list_matrices)  # type: ignore
def test_apply_matrix__points_opencv(matrix: NDArrayf) -> None:
    """
    Test that apply matrix's exact transformation for points (implemented with NumPy matrix multiplication)
    is exactly the same as the one of OpenCV (optional dependency).
    """

    # Create random points
    points = np.random.default_rng(42).normal(size=(10, 3))

    # Convert to a geodataframe and use apply_matrix for the point cloud
    epc = gpd.GeoDataFrame(data={"z": points[:, 2]}, geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]))
    trans_epc = apply_matrix(epc, matrix=matrix)

    # Run the same operation with openCV
    import cv2

    trans_cv2_arr = cv2.perspectiveTransform(points[:, :].reshape(1, -1, 3), matrix)[0, :, :]

    # Transform point cloud back to array
    trans_numpy = np.array([trans_epc.geometry.x.values, trans_epc.geometry.y.values, trans_epc["z"].values]).T
    assert np.allclose(trans_numpy, trans_cv2_arr)


@pytest.mark.parametrize("regrid_method", [None, "iterative", "griddata"])  # type: ignore
@pytest.mark.parametrize("matrix", list_matrices)  # type: ignore
def test_apply_matrix__raster(regrid_method: Optional[str], matrix: NDArrayf) -> None:
    """Test that apply matrix gives consistent results between points and rasters (thus validating raster
    implementation, as point implementation is validated above), for all possible regridding methods."""

    # Create a synthetic raster and convert to point cloud
    # dem = gu.Raster(.ref)
    dem_arr = np.linspace(0, 2, 25).reshape(5, 5)
    transform = rio.transform.from_origin(0, 5, 1, 1)
    dem = gu.Raster.from_array(dem_arr, transform=transform, crs=4326, nodata=100)
    epc = dem.to_pointcloud(data_column_name="z").ds

    # If a centroid was not given, default to the center of the DEM (at Z=0).
    centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)

    # Apply affine transformation to both datasets
    trans_dem = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method=regrid_method)
    trans_epc = apply_matrix(epc, matrix=matrix, centroid=centroid)

    # Interpolate transformed DEM at coordinates of the transformed point cloud
    # Because the raster created as a constant slope (plan-like), the interpolated values should be very close
    z_points = trans_dem.interp_points(points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values))
    valids = np.isfinite(z_points)
    assert np.count_nonzero(valids) > 0
    assert np.allclose(z_points[valids], trans_epc.z.values[valids], rtol=10e-5)


def test_apply_matrix__raster_nodata() -> None:
    """Test the nodatas created by apply_matrix are consistent between methods"""

    # Use matrix with all transformations
    matrix = matrix_all

    # Create a synthetic raster, add NaNs, and convert to point cloud
    dem_arr = np.linspace(0, 2, 400).reshape(20, 20)
    dem_arr[10:14, 10:14] = np.nan
    dem_arr[5, 5] = np.nan
    dem_arr[:2, :] = np.nan
    transform = rio.transform.from_origin(0, 5, 1, 1)
    dem = gu.Raster.from_array(dem_arr, transform=transform, crs=4326, nodata=100)
    epc = dem.to_pointcloud(data_column_name="z").ds

    centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)

    trans_dem_it = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="iterative")
    trans_dem_gd = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="griddata")

    # Get nodata mask
    mask_nodata_it = trans_dem_it.data.mask
    mask_nodata_gd = trans_dem_gd.data.mask

    # The iterative mask should be larger and contain the other (as griddata interpolates up to 1 pixel away)
    assert np.array_equal(np.logical_or(mask_nodata_gd, mask_nodata_it), mask_nodata_it)

    # Verify nodata masks are located within two pixels of each other (1 pixel can be added by griddata,
    # and 1 removed by regular-grid interpolation by the iterative method)
    smallest_mask = ~binary_dilation(
        ~mask_nodata_it, iterations=2
    )  # Invert before dilate to avoid spreading at the edges
    # All smallest mask value should exist in the mask of griddata
    assert np.array_equal(np.logical_or(smallest_mask, mask_nodata_gd), mask_nodata_gd)


def test_apply_matrix__raster_realdata(load_examples) -> None:
    """Testing real data no complex matrix only to avoid all loops"""

    ref, _, _, _, _ = load_examples
    # Use real data
    dem = ref
    dem.crop((dem.bounds.left, dem.bounds.bottom, dem.bounds.left + 2000, dem.bounds.bottom + 2000))
    epc = dem.to_pointcloud(data_column_name="z").ds

    # Only testing complex matrices for speed
    matrix = matrix_all

    # If a centroid was not given, default to the center of the DEM (at Z=0).
    centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)

    # Apply affine transformation to both datasets
    trans_dem_it = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="iterative")
    trans_dem_gd = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="griddata")
    trans_epc = apply_matrix(epc, matrix=matrix, centroid=centroid)

    # Interpolate transformed DEM at coordinates of the transformed point cloud, and check values are very close
    z_points_it = trans_dem_it.interp_points(points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values))
    z_points_gd = trans_dem_gd.interp_points(points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values))

    valids = np.logical_and(np.isfinite(z_points_it), np.isfinite(z_points_gd))
    assert np.count_nonzero(valids) > 0

    diff_it = z_points_it[valids] - trans_epc.z.values[valids]
    diff_gd = z_points_gd[valids] - trans_epc.z.values[valids]

    # Because of outliers, noise and slope near 90°, several solutions can exist
    # Additionally, contrary to the check in the __raster test which uses a constant slope DEM, the slopes vary
    # here so the interpolation check is less accurate so all values can vary a bit
    assert np.percentile(np.abs(diff_it), 90) < 1
    assert np.percentile(np.abs(diff_it), 50) < 0.2
    assert np.percentile(np.abs(diff_gd), 90) < 1
    assert np.percentile(np.abs(diff_gd), 50) < 0.2

    # But, between themselves, the two re-gridding methods should yield much closer results
    # (no errors due to 2D interpolation for checking)
    diff_it_gd = z_points_gd[valids] - z_points_it[valids]
    assert np.percentile(np.abs(diff_it_gd), 99) < 1  # 99% of values are within a meter (instead of 90%)
    assert np.percentile(np.abs(diff_it_gd), 50) < 0.02  # 10 times more precise than above


def test_warp_dem() -> None:
    """Test that the warp_dem function works expectedly."""

    small_dem = np.zeros((5, 10), dtype="float32")
    small_transform = rio.transform.from_origin(0, 5, 1, 1)

    source_coords = np.array([[0, 0, 0], [0, 5, 0], [10, 0, 0], [10, 5, 0]]).astype(small_dem.dtype)

    dest_coords = source_coords.copy()
    dest_coords[0, 0] = -1e-5

    warped_dem = coreg.base.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
        trim_border=False,
    )
    assert np.nansum(np.abs(warped_dem - small_dem)) < 1e-6

    elev_shift = 5.0
    dest_coords[1, 2] = elev_shift
    warped_dem = coreg.base.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
    )

    # The warped DEM should have the value 'elev_shift' in the upper left corner.
    assert warped_dem[0, 0] == elev_shift
    # The corner should be zero, so the corner pixel (represents the corner minus resolution / 2) should be close.
    # We select the pixel before the corner (-2 in X-axis) to avoid the NaN propagation on the bottom row.
    assert warped_dem[-2, -1] < 1

    # Synthesise some X/Y/Z coordinates on the DEM.
    source_coords = np.array(
        [
            [0, 0, 200],
            [480, 20, 200],
            [460, 480, 200],
            [10, 460, 200],
            [250, 250, 200],
        ]
    )

    # Copy the source coordinates and apply some shifts
    dest_coords = source_coords.copy()
    # Apply in the X direction
    dest_coords[0, 0] += 20
    dest_coords[1, 0] += 7
    dest_coords[2, 0] += 10
    dest_coords[3, 0] += 5

    # Apply in the Y direction
    dest_coords[4, 1] += 5

    # Apply in the Z direction
    dest_coords[3, 2] += 5
    test_shift = 6  # This shift will be validated below
    dest_coords[4, 2] += test_shift

    # Generate a semi-random DEM
    transform = rio.transform.from_origin(0, 500, 1, 1)
    shape = (500, 550)
    dem = misc.generate_random_field(shape, 100) * 200 + misc.generate_random_field(shape, 10) * 50

    # Warp the DEM using the source-destination coordinates.
    transformed_dem = coreg.base.warp_dem(
        dem=dem, transform=transform, source_coords=source_coords, destination_coords=dest_coords, resampling="linear"
    )

    # Try to undo the warp by reversing the source-destination coordinates.
    untransformed_dem = coreg.base.warp_dem(
        dem=transformed_dem,
        transform=transform,
        source_coords=dest_coords,
        destination_coords=source_coords,
        resampling="linear",
    )
    # Validate that the DEM is now more or less the same as the original.
    # Due to the randomness, the threshold is quite high, but would be something like 10+ if it was incorrect.
    assert spatialstats.nmad(dem - untransformed_dem) < 0.5

    if False:
        import matplotlib.pyplot as plt

        plt.figure(dpi=200)
        plt.subplot(141)

        plt.imshow(dem, vmin=0, vmax=300)
        plt.subplot(142)
        plt.imshow(transformed_dem, vmin=0, vmax=300)
        plt.subplot(143)
        plt.imshow(untransformed_dem, vmin=0, vmax=300)

        plt.subplot(144)
        plt.imshow(dem - untransformed_dem, cmap="coolwarm_r", vmin=-10, vmax=10)
        plt.show()
