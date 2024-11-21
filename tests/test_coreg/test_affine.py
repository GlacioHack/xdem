"""Functions to test the affine coregistrations."""

from __future__ import annotations

import warnings

import geopandas as gpd
import geoutils
import numpy as np
import pytest
import pytransform3d
import rasterio as rio
from geoutils import Raster, Vector
from geoutils._typing import NDArrayNum
from geoutils.raster import RasterType
from geoutils.raster.geotransformations import _translate
from scipy.ndimage import binary_dilation

from xdem import coreg, examples
from xdem.coreg.affine import AffineCoreg, _reproject_horizontal_shift_samecrs


def load_examples(crop: bool = True) -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    if crop:
        # Crop to smaller extents for test speed
        res = reference_dem.res
        crop_geom = (
            reference_dem.bounds.left,
            reference_dem.bounds.bottom,
            reference_dem.bounds.left + res[0] * 300,
            reference_dem.bounds.bottom + res[1] * 300,
        )
        reference_dem = reference_dem.crop(crop_geom)
        to_be_aligned_dem = to_be_aligned_dem.crop(crop_geom)

    return reference_dem, to_be_aligned_dem, glacier_mask


def gdal_reproject_horizontal_shift_samecrs(filepath_example: str, xoff: float, yoff: float) -> NDArrayNum:
    """
    Reproject horizontal shift in same CRS with GDAL for testing purposes.

    :param filepath_example: Path to raster file.
    :param xoff: X shift in georeferenced unit.
    :param yoff: Y shift in georeferenced unit.

    :return: Reprojected shift array in the same CRS.
    """

    from osgeo import gdal, gdalconst

    # Open source raster from file
    src = gdal.Open(filepath_example, gdalconst.GA_ReadOnly)

    # Create output raster in memory
    driver = "MEM"
    method = gdal.GRA_Bilinear
    drv = gdal.GetDriverByName(driver)
    dest = drv.Create("", src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Float32)
    proj = src.GetProjection()
    ndv = src.GetRasterBand(1).GetNoDataValue()
    dest.SetProjection(proj)

    # Shift the horizontally shifted geotransform
    gt = src.GetGeoTransform()
    gtl = list(gt)
    gtl[0] += xoff
    gtl[3] += yoff
    dest.SetGeoTransform(tuple(gtl))

    # Copy the raster metadata of the source to dest
    dest.SetMetadata(src.GetMetadata())
    dest.GetRasterBand(1).SetNoDataValue(ndv)
    dest.GetRasterBand(1).Fill(ndv)

    # Reproject with resampling
    gdal.ReprojectImage(src, dest, proj, proj, method)

    # Extract reprojected array
    array = dest.GetRasterBand(1).ReadAsArray().astype("float32")
    array[array == ndv] = np.nan

    return array


class TestAffineCoreg:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    # Check all point-raster possibilities supported
    # Use the reference DEM for both, it will be artificially misaligned during tests
    # Raster-Raster
    fit_args_rst_rst = dict(reference_elev=ref, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    # Convert DEMs to points with a bit of subsampling for speed-up
    ref_pts = ref.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
    tba_pts = ref.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds

    # Raster-Point
    fit_args_rst_pts = dict(reference_elev=ref, to_be_aligned_elev=tba_pts, inlier_mask=inlier_mask)

    # Point-Raster
    fit_args_pts_rst = dict(reference_elev=ref_pts, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    all_fit_args = [fit_args_rst_rst, fit_args_rst_pts, fit_args_pts_rst]

    # Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )

    @pytest.mark.parametrize(
        "xoff_yoff",
        [(ref.res[0], ref.res[1]), (10 * ref.res[0], 10 * ref.res[1]), (-1.2 * ref.res[0], -1.2 * ref.res[1])],
    )  # type: ignore
    def test_reproject_horizontal_shift_samecrs__gdal(self, xoff_yoff: tuple[float, float]) -> None:
        """Check that the same-CRS reprojection based on SciPy (replacing Rasterio due to subpixel errors)
        is accurate by comparing to GDAL."""

        ref = load_examples(crop=False)[0]

        # Reproject with SciPy
        xoff, yoff = xoff_yoff
        dst_transform = _translate(transform=ref.transform, xoff=xoff, yoff=yoff, distance_unit="georeferenced")
        output = _reproject_horizontal_shift_samecrs(
            raster_arr=ref.data, src_transform=ref.transform, dst_transform=dst_transform
        )

        # Reproject with GDAL
        output2 = gdal_reproject_horizontal_shift_samecrs(filepath_example=ref.filename, xoff=xoff, yoff=yoff)

        # Reproject and NaN propagation is exactly the same for shifts that are a multiple of pixel resolution
        if xoff % ref.res[0] == 0 and yoff % ref.res[1] == 0:
            assert np.array_equal(output, output2, equal_nan=True)

        # For sub-pixel shifts, NaN propagation differs slightly (within 1 pixel) but the resampled values are the same
        else:
            # Verify all close values
            valids = np.logical_and(np.isfinite(output), np.isfinite(output2))
            # Max relative tolerance that is reached just for a small % of points
            assert np.allclose(output[valids], output2[valids], rtol=10e-2)
            # Median precision is much higher
            # (here absolute, equivalent to around 10e-7 relative as raster values are in the 1000s)
            assert np.nanmedian(np.abs(output[valids] - output2[valids])) < 0.0001

            # NaNs differ by 1 pixel max, i.e. the mask dilated by one includes the other
            mask_nans = ~np.isfinite(output)
            mask_dilated_plus_one = binary_dilation(mask_nans, iterations=1).astype(bool)
            assert np.array_equal(np.logical_or(mask_dilated_plus_one, ~np.isfinite(output2)), mask_dilated_plus_one)

    def test_from_classmethods(self) -> None:

        # Check that the from_matrix function works as expected.
        vshift = 5
        matrix = np.diag(np.ones(4, dtype=float))
        matrix[2, 3] = vshift
        coreg_obj = AffineCoreg.from_matrix(matrix)
        transformed_points = coreg_obj.apply(self.points)
        assert all(transformed_points["z"].values == vshift)

        # Check that the from_translation function works as expected.
        x_offset = 5
        coreg_obj2 = AffineCoreg.from_translations(x_off=x_offset)
        transformed_points2 = coreg_obj2.apply(self.points)
        assert np.array_equal(self.points.geometry.x.values + x_offset, transformed_points2.geometry.x.values)

        # Try to make a Coreg object from a nan translation (should fail).
        try:
            AffineCoreg.from_translations(np.nan)
        except ValueError as exception:
            if "non-finite values" not in str(exception):
                raise exception

    def test_raise_all_nans(self) -> None:
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

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("shifts", [(20, 5, 2), (-50, 100, 2)])  # type: ignore
    @pytest.mark.parametrize("coreg_method", [coreg.NuthKaab, coreg.DhMinimize, coreg.ICP])  # type: ignore
    def test_coreg_translations__synthetic(self, fit_args, shifts, coreg_method) -> None:
        """
        Test the horizontal/vertical shift coregistrations with synthetic shifted data. These tests include NuthKaab,
        ICP and DhMinimize.

        We test all combinaison of inputs: raster-raster, point-raster and raster-point.

        We verify that the shifts found by the coregistration are within 1% of the synthetic shifts with opposite sign
        of the ones introduced, and that applying the coregistration to the shifted elevations corrects more than
        99% of the variance from the initial elevation differences (hence, that the direction of coregistration has
        to be the right one; and that there is no other errors introduced in the process).
        """

        warnings.filterwarnings("ignore", message="Covariance of the parameters*")

        horizontal_coreg = coreg_method()

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        # Create synthetic translation from the reference DEM
        ref = self.ref
        ref_shifted = ref.translate(shifts[0], shifts[1]) + shifts[2]
        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_shifted = ref_shifted.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
        elev_fit_args["to_be_aligned_elev"] = ref_shifted

        # Run coregistration
        coreg_elev = horizontal_coreg.fit_and_apply(**elev_fit_args, subsample=50000, random_state=42)

        # Check all fit parameters are the opposite of those used above, within a relative 1% (10% for ICP)
        fit_shifts = [-horizontal_coreg.meta["outputs"]["affine"][k] for k in ["shift_x", "shift_y", "shift_z"]]

        # ICP can be less precise than other methods
        rtol = 10e-2 if coreg_method == coreg.NuthKaab else 10e-1
        assert np.allclose(shifts, fit_shifts, rtol=rtol)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_shifted.geometry.x.values, ref_shifted.geometry.y.values)) - ref_shifted["z"]
            )
            dh = ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values)) - coreg_elev["z"]
        else:
            init_dh = ref - ref_shifted.reproject(ref)
            dh = ref - coreg_elev.reproject(ref)

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            init_dh.plot()
            plt.show()
            dh.plot()
            plt.show()

        # Check applying the coregistration removes 99% of the variance (95% for ICP)
        # Need to standardize by the elevation difference spread to avoid huge/small values close to infinity
        tol = 0.01 if coreg_method == coreg.NuthKaab else 0.05
        assert np.nanvar(dh / np.nanstd(init_dh)) < tol

    @pytest.mark.parametrize(
        "coreg_method__shift",
        [
            (coreg.NuthKaab, (9.202739, 2.735573, -1.97733)),
            (coreg.DhMinimize, (10.0850892, 2.898172, -1.943001)),
            (coreg.ICP, (8.73833, 1.584255, -1.943957)),
        ],
    )  # type: ignore
    def test_coreg_translations__example(
        self, coreg_method__shift: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the translation co-registration outputs are always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba = load_examples(crop=False)[0:2]
        inlier_mask = ~self.outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_shifts = coreg_method__shift

        c = coreg_method(subsample=50000)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations match the exact values
        shifts = [c.meta["outputs"]["affine"][k] for k in ["shift_x", "shift_y", "shift_z"]]  # type: ignore
        assert shifts == pytest.approx(expected_shifts)

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("vshift", [0.2, 10.0, 1000.0])  # type: ignore
    def test_coreg_vertical_translation__synthetic(self, fit_args, vshift) -> None:
        """
        Test the vertical shift coregistration with synthetic shifted data. These tests include VerticalShift.

        We test all combinaison of inputs: raster-raster, point-raster and raster-point.
        """

        # Create a vertical shift correction instance
        vshiftcorr = coreg.VerticalShift()

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        # Create synthetic vertical shift from the reference DEM
        ref = self.ref
        ref_vshifted = ref + vshift

        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_vshifted = ref_vshifted.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
        elev_fit_args["to_be_aligned_elev"] = ref_vshifted

        # Fit the vertical shift model to the data
        coreg_elev = vshiftcorr.fit_and_apply(**elev_fit_args, subsample=50000, random_state=42)

        # Check that the right vertical shift was found
        assert vshiftcorr.meta["outputs"]["affine"]["shift_z"] == pytest.approx(-vshift, rel=10e-2)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_vshifted.geometry.x.values, ref_vshifted.geometry.y.values)) - ref_vshifted["z"]
            )
            dh = ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values)) - coreg_elev["z"]
        else:
            init_dh = ref - ref_vshifted
            dh = ref - coreg_elev

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            init_dh.plot()
            plt.show()
            dh.plot()
            plt.show()

        # Check that the median difference is zero, and that no additional variance
        # was introduced, so that the variance is also close to zero (no variance for a constant vertical shift)
        assert np.nanmedian(dh) == pytest.approx(0, abs=10e-6)
        assert np.nanvar(dh) == pytest.approx(0, abs=10e-6)

    @pytest.mark.parametrize("coreg_method__vshift", [(coreg.VerticalShift, -2.305015)])  # type: ignore
    def test_coreg_vertical_translation__example(
        self, coreg_method__vshift: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the vertical translation co-registration output is always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba = load_examples(crop=False)[0:2]
        inlier_mask = ~self.outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_vshift = coreg_method__vshift

        # Run co-registration
        c = coreg_method(subsample=50000)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations match the exact values
        vshift = c.meta["outputs"]["affine"]["shift_z"]
        assert vshift == pytest.approx(expected_vshift)

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("shifts_rotations", [(20, 5, 0, 0.02, 0.05, 0.1), (-50, 100, 0, 10, 5, 4)])  # type: ignore
    @pytest.mark.parametrize("coreg_method", [coreg.ICP])  # type: ignore
    def test_coreg_rigid__synthetic(self, fit_args, shifts_rotations, coreg_method) -> None:
        """
        Test the rigid coregistrations with synthetic misaligned (shifted and rotatedà data. These tests include ICP.

        We test all combinaison of inputs: raster-raster, point-raster and raster-point.

        We verify that the matrix found by the coregistration is within 1% of the synthetic matrix, and inverted from
        the one introduced, and that applying the coregistration to the misaligned elevations corrects more than
        95% of the variance from the initial elevation differences (hence, that the direction of coregistration has
        to be the right one; and that there is no other errors introduced in the process).
        """

        # Initiate coregistration
        horizontal_coreg = coreg_method()

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        ref = self.ref

        # Create synthetic rigid transformation (translation and rotation) from the reference DEM
        sr = np.array(shifts_rotations)
        shifts = sr[:3]
        rotations = sr[3:6]
        e = np.deg2rad(rotations)
        # Derive is a 3x3 rotation matrix
        rot_matrix = pytransform3d.rotations.matrix_from_euler(e=e, i=0, j=1, k=2, extrinsic=True)
        matrix = np.diag(np.ones(4, float))
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = shifts

        # Pass a centroid
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_shifted_rotated = ref_shifted_rotated.to_pointcloud(
                data_column_name="z", subsample=50000, random_state=42
            ).ds
        elev_fit_args["to_be_aligned_elev"] = ref_shifted_rotated

        # Run coregistration
        coreg_elev = horizontal_coreg.fit_and_apply(**elev_fit_args, subsample=50000, random_state=42)

        # Check that fit matrix is the invert of those used above, within a relative 10% for rotations, and within
        # a large 100% margin for shifts, as ICP has relative difficulty resolving shifts with large rotations
        fit_matrix = horizontal_coreg.meta["outputs"]["affine"]["matrix"]
        invert_fit_matrix = coreg.invert_matrix(fit_matrix)
        invert_fit_shifts = invert_fit_matrix[:3, 3]
        invert_fit_rotations = pytransform3d.rotations.euler_from_matrix(
            invert_fit_matrix[0:3, 0:3], i=0, j=1, k=2, extrinsic=True
        )
        invert_fit_rotations = np.rad2deg(invert_fit_rotations)
        assert np.allclose(shifts, invert_fit_shifts, rtol=1)
        assert np.allclose(rotations, invert_fit_rotations, rtol=10e-1)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_shifted_rotated.geometry.x.values, ref_shifted_rotated.geometry.y.values))
                - ref_shifted_rotated["z"]
            )
            dh = ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values)) - coreg_elev["z"]
        else:
            init_dh = ref - ref_shifted_rotated
            dh = ref - coreg_elev

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            init_dh.plot()
            plt.show()
            dh.plot()
            plt.show()

        # Need to standardize by the elevation difference spread to avoid huge/small values close to infinity
        # Only 95% of variance as ICP cannot always resolve the last shifts
        assert np.nanvar(dh / np.nanstd(init_dh)) < 0.05

    @pytest.mark.parametrize(
        "coreg_method__shifts_rotations",
        [(coreg.ICP, (8.738332, 1.584255, -1.943957, 0.0069004, -0.00703, -0.0119733))],
    )  # type: ignore
    def test_coreg_rigid__example(
        self, coreg_method__shifts_rotations: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the rigid co-registration outputs is always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba = load_examples(crop=False)[0:2]
        inlier_mask = ~self.outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_shifts_rots = coreg_method__shifts_rotations

        # Run co-registration
        c = coreg_method(subsample=50000)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations and rotations match the exact values
        fit_matrix = c.meta["outputs"]["affine"]["matrix"]
        fit_shifts = fit_matrix[:3, 3]
        fit_rotations = pytransform3d.rotations.euler_from_matrix(fit_matrix[0:3, 0:3], i=0, j=1, k=2, extrinsic=True)
        fit_rotations = np.rad2deg(fit_rotations)
        fit_shifts_rotations = tuple(np.concatenate((fit_shifts, fit_rotations)))

        assert fit_shifts_rotations == pytest.approx(expected_shifts_rots, abs=10e-6)
