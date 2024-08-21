"""Functions to test the affine coregistrations."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import rasterio as rio

import xdem
from xdem import coreg
from xdem.coreg.affine import AffineCoreg, CoregDict


def test_from_classmethods(three_d_coordinates) -> None:

    # Check that the from_matrix function works as expected.
    vshift = 5
    matrix = np.diag(np.ones(4, dtype=float))
    matrix[2, 3] = vshift
    coreg_obj = AffineCoreg.from_matrix(matrix)
    transformed_points = coreg_obj.apply(three_d_coordinates)
    assert all(transformed_points["z"].values == vshift)

    # Check that the from_translation function works as expected.
    x_offset = 5
    coreg_obj2 = AffineCoreg.from_translation(x_off=x_offset)
    transformed_points2 = coreg_obj2.apply(three_d_coordinates)
    assert np.array_equal(three_d_coordinates.geometry.x.values + x_offset, transformed_points2.geometry.x.values)

    # Try to make a Coreg object from a nan translation (should fail).
    try:
        AffineCoreg.from_translation(np.nan)
    except ValueError as exception:
        if "non-finite values" not in str(exception):
            raise exception


def test_vertical_shift(load_examples, three_d_coordinates) -> None:

    ref, tba, _, inlier_mask, fit_params = load_examples

    # Create a vertical shift correction instance
    vshiftcorr = coreg.VerticalShift()
    # Fit the vertical shift model to the data
    vshiftcorr.fit(**fit_params)

    res = ref.res[0]

    # Check that a vertical shift was found.
    assert vshiftcorr.meta.get("shift_z") is not None
    assert vshiftcorr.meta["shift_z"] != 0.0

    # Copy the vertical shift to see if it changes in the test (it shouldn't)
    vshift = copy.copy(vshiftcorr.meta["shift_z"])

    # Check that the to_matrix function works as it should
    matrix = vshiftcorr.to_matrix()
    assert matrix[2, 3] == vshift, matrix

    # Check that the first z coordinate is now the vertical shift
    assert all(vshiftcorr.apply(three_d_coordinates)["z"].values == vshiftcorr.meta["shift_z"])

    # Apply the model to correct the DEM
    tba_unshifted, _ = vshiftcorr.apply(tba.data, transform=ref.transform, crs=ref.crs)

    # Create a new vertical shift correction model
    vshiftcorr2 = coreg.VerticalShift()
    # Check that this is indeed a new object
    assert vshiftcorr is not vshiftcorr2
    # Fit the corrected DEM to see if the vertical shift will be close to or at zero
    vshiftcorr2.fit(
        reference_elev=ref.data,
        to_be_aligned_elev=tba_unshifted,
        transform=ref.transform,
        crs=ref.crs,
        inlier_mask=inlier_mask,
    )
    # Test the vertical shift
    newmeta: CoregDict = vshiftcorr2.meta
    new_vshift = newmeta["shift_z"]
    assert np.abs(new_vshift) * res < 0.01

    # Check that the original model's vertical shift has not changed
    # (that the _.meta dicts are two different objects)
    assert vshiftcorr.meta["shift_z"] == vshift


def test_all_nans() -> None:
    """Check that the coregistration approaches fail gracefully when given only nans."""
    dem1 = np.ones((50, 50), dtype=float)
    dem2 = dem1.copy() + np.nan
    affine = rio.transform.from_origin(0, 0, 1, 1)
    crs = rio.crs.CRS.from_epsg(4326)

    vshiftcorr = coreg.VerticalShift()
    icp = coreg.ICP()

    pytest.raises(ValueError, vshiftcorr.fit, dem1, dem2, transform=affine)
    pytest.raises(ValueError, icp.fit, dem1, dem2, transform=affine)

    dem2[[3, 20, 40], [2, 21, 41]] = 1.2

    vshiftcorr.fit(dem1, dem2, transform=affine, crs=crs)

    pytest.raises(ValueError, icp.fit, dem1, dem2, transform=affine)


def test_coreg_example(load_examples, verbose: bool = False) -> None:
    """
    Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
    test_examples.py, but helps identify from where differences arise.
    """

    ref, tba, _, inlier_mask, _ = load_examples

    # Run co-registration
    nuth_kaab = xdem.coreg.NuthKaab()
    nuth_kaab.fit(ref, tba, inlier_mask=inlier_mask, verbose=verbose, random_state=42)

    # Check the output .metadata is always the same
    shifts = (nuth_kaab.meta["shift_x"], nuth_kaab.meta["shift_y"], nuth_kaab.meta["shift_z"])
    res = ref.res[0]
    assert shifts == pytest.approx((-0.463 * res, -0.1339999 * res, -1.9922009))


def test_gradientdescending(
    load_examples, subsample: int = 10000, inlier_mask: bool = True, verbose: bool = False
) -> None:
    """
    Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
    test_examples.py, but helps identify from where differences arise.

    It also implicitly tests the z_name kwarg and whether a geometry column can be provided instead of E/N cols.
    """

    ref, tba, _, inlier_mask_data, _ = load_examples

    if inlier_mask:
        inlier_mask = inlier_mask_data

    # Run co-registration
    gds = xdem.coreg.GradientDescending(subsample=subsample)
    gds.fit(
        ref.to_pointcloud(data_column_name="z").ds,
        tba,
        inlier_mask=inlier_mask,
        verbose=verbose,
        random_state=42,
    )

    res = ref.res[0]
    shifts = (gds.meta["shift_x"], gds.meta["shift_y"], gds.meta["shift_z"])
    assert shifts == pytest.approx((0.03525 * res, -0.59775 * res, -2.39144), abs=10e-5)


@pytest.mark.parametrize("shift_px", [(1, 1), (2, 2)])  # type: ignore
@pytest.mark.parametrize("coreg_class", [coreg.NuthKaab, coreg.GradientDescending, coreg.ICP])  # type: ignore
@pytest.mark.parametrize("points_or_raster", ["raster", "points"])
def test_coreg_example_shift(load_examples, shift_px, coreg_class, points_or_raster, verbose=False, subsample=5000):
    """
    For comparison of coreg algorithms:
    Shift a ref_dem on purpose, e.g. shift_px = (1,1), and then applying coreg to shift it back.
    """

    ref, tba, _, inlier_mask_data, _ = load_examples

    res = ref.res[0]

    # shift DEM by shift_px
    shifted_ref = ref.copy()
    shifted_ref.translate(shift_px[0] * res, shift_px[1] * res, inplace=True)

    shifted_ref_points = shifted_ref.to_pointcloud(subsample=subsample, force_pixel_offset="center", random_state=42).ds
    shifted_ref_points["E"] = shifted_ref_points.geometry.x
    shifted_ref_points["N"] = shifted_ref_points.geometry.y
    shifted_ref_points.rename(columns={"b1": "z"}, inplace=True)

    kwargs = {} if coreg_class.__name__ != "GradientDescending" else {"subsample": subsample}

    coreg_obj = coreg_class(**kwargs)

    best_east_diff = 1e5
    best_north_diff = 1e5
    if points_or_raster == "raster":
        coreg_obj.fit(shifted_ref, ref, verbose=verbose, random_state=42)
    elif points_or_raster == "points":
        coreg_obj.fit(shifted_ref_points, ref, verbose=verbose, random_state=42)

    if coreg_class.__name__ == "ICP":
        matrix = coreg_obj.to_matrix()
        # The ICP fit only creates a matrix and doesn't normally show the alignment in pixels
        # Since the test is formed to validate pixel shifts, these calls extract the approximate pixel shift
        # from the matrix (it's not perfect since rotation/scale can change it).
        coreg_obj.meta["shift_x"] = -matrix[0][3]
        coreg_obj.meta["shift_y"] = -matrix[1][3]

    # ICP can never be expected to be much better than 1px on structured data, as its implementation often finds a
    # minimum between two grid points. This is clearly warned for in the documentation.
    precision = 1e-2 if coreg_class.__name__ != "ICP" else 1

    if coreg_obj.meta["shift_x"] == pytest.approx(-shift_px[0] * res, rel=precision) and coreg_obj.meta[
        "shift_y"
    ] == pytest.approx(-shift_px[0] * res, rel=precision):
        return
    best_east_diff = coreg_obj.meta["shift_x"] - shift_px[0]
    best_north_diff = coreg_obj.meta["shift_y"] - shift_px[1]

    raise AssertionError(f"Diffs are too big. east: {best_east_diff:.2f} px, north: {best_north_diff:.2f} px")


def test_nuth_kaab(load_examples, three_d_coordinates) -> None:

    ref, tba, _, inlier_mask_data, fit_params = load_examples

    nuth_kaab = coreg.NuthKaab(max_iterations=10)

    # Synthesize a shifted and vertically offset DEM
    pixel_shift = 2
    vshift = 5
    shifted_dem = ref.data.squeeze().copy()
    shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
    shifted_dem[:, :pixel_shift] = np.nan
    shifted_dem += vshift

    # Fit the synthesized shifted DEM to the original
    nuth_kaab.fit(
        ref.data.squeeze(),
        shifted_dem,
        transform=ref.transform,
        crs=ref.crs,
        verbose=fit_params["verbose"],
    )

    # Make sure that the estimated offsets are similar to what was synthesized.
    res = ref.res[0]
    assert nuth_kaab.meta["shift_x"] == pytest.approx(pixel_shift * res, abs=0.03)
    assert nuth_kaab.meta["shift_y"] == pytest.approx(0, abs=0.03)
    assert nuth_kaab.meta["shift_z"] == pytest.approx(-vshift, 0.03)

    # Apply the estimated shift to "revert the DEM" to its original state.
    unshifted_dem, _ = nuth_kaab.apply(shifted_dem, transform=ref.transform, crs=ref.crs)
    # Measure the difference (should be more or less zero)
    diff = ref.data.squeeze() - unshifted_dem
    diff = diff.compressed()  # turn into a 1D array with only unmasked values

    # Check that the median is very close to zero
    assert np.abs(np.median(diff)) < 0.01
    # Check that the RMSE is low
    assert np.sqrt(np.mean(np.square(diff))) < 1

    # Transform some arbitrary points.
    transformed_points = nuth_kaab.apply(three_d_coordinates)

    # Check that the x shift is close to the pixel_shift * image resolution
    assert all(
        abs((transformed_points.geometry.x.values - three_d_coordinates.geometry.x.values) - pixel_shift * ref.res[0])
        < 0.1
    )
    # Check that the z shift is close to the original vertical shift.
    assert all(abs((transformed_points["z"].values - three_d_coordinates["z"].values) + vshift) < 0.1)


def test_tilt(load_examples) -> None:

    ref, tba, _, inlier_mask, fit_params = load_examples

    # Try a 1st degree deramping.
    tilt = coreg.Tilt()

    # Fit the data
    tilt.fit(**fit_params, random_state=42)

    # Apply the deramping to a DEM
    tilted_dem = tilt.apply(tba)

    # Get the periglacial offset after deramping
    periglacial_offset = (ref - tilted_dem)[inlier_mask]
    # Get the periglacial offset before deramping
    pre_offset = (ref - tba)[inlier_mask]

    # Check that the error improved
    assert np.abs(np.mean(periglacial_offset)) < np.abs(np.mean(pre_offset))

    # Check that the mean periglacial offset is low
    assert np.abs(np.mean(periglacial_offset)) < 0.02


def test_icp_opencv(load_examples) -> None:

    ref, tba, _, inlier_mask, fit_params = load_examples

    # Do a fast and dirty 3 iteration ICP just to make sure it doesn't error out.
    icp = coreg.ICP(max_iterations=3)
    icp.fit(**fit_params)

    aligned_dem, _ = icp.apply(tba.data, transform=ref.transform, crs=ref.crs)

    assert aligned_dem.shape == ref.data.squeeze().shape
