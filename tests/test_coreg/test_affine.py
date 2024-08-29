"""Functions to test the affine coregistrations."""
from __future__ import annotations

import copy

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from geoutils import Raster, Vector
from geoutils._typing import NDArrayNum
from geoutils.raster import RasterType
from geoutils.raster.raster import _shift_transform
from scipy.ndimage import binary_dilation

import xdem
from xdem import coreg, examples
from xdem.coreg.affine import (
    AffineCoreg,
    CoregDict,
    _reproject_horizontal_shift_samecrs,
)


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_raster = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


def gdal_reproject_horizontal_samecrs(filepath_example: str, xoff: float, yoff: float) -> NDArrayNum:
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

    fit_params = dict(
        reference_elev=ref.data,
        to_be_aligned_elev=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        crs=ref.crs,
        verbose=True,
    )
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

        # Reproject with SciPy
        xoff, yoff = xoff_yoff
        dst_transform = _shift_transform(
            transform=self.ref.transform, xoff=xoff, yoff=yoff, distance_unit="georeferenced"
        )
        output = _reproject_horizontal_shift_samecrs(
            raster_arr=self.ref.data, src_transform=self.ref.transform, dst_transform=dst_transform
        )

        # Reproject with GDAL
        output2 = gdal_reproject_horizontal_samecrs(filepath_example=self.ref.filename, xoff=xoff, yoff=yoff)

        # Reproject and NaN propagation is exactly the same for shifts that are a multiple of pixel resolution
        if xoff % self.ref.res[0] == 0 and yoff % self.ref.res[1] == 0:
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
        coreg_obj2 = AffineCoreg.from_translation(x_off=x_offset)
        transformed_points2 = coreg_obj2.apply(self.points)
        assert np.array_equal(self.points.geometry.x.values + x_offset, transformed_points2.geometry.x.values)

        # Try to make a Coreg object from a nan translation (should fail).
        try:
            AffineCoreg.from_translation(np.nan)
        except ValueError as exception:
            if "non-finite values" not in str(exception):
                raise exception

    def test_vertical_shift(self) -> None:

        # Create a vertical shift correction instance
        vshiftcorr = coreg.VerticalShift()
        # Fit the vertical shift model to the data
        vshiftcorr.fit(**self.fit_params)

        res = self.ref.res[0]

        # Check that a vertical shift was found.
        assert vshiftcorr.meta["outputs"]["affine"].get("shift_z") is not None
        assert vshiftcorr.meta["outputs"]["affine"]["shift_z"] != 0.0

        # Copy the vertical shift to see if it changes in the test (it shouldn't)
        vshift = copy.copy(vshiftcorr.meta["outputs"]["affine"]["shift_z"])

        # Check that the to_matrix function works as it should
        matrix = vshiftcorr.to_matrix()
        assert matrix[2, 3] == vshift, matrix

        # Check that the first z coordinate is now the vertical shift
        assert all(vshiftcorr.apply(self.points)["z"].values == vshiftcorr.meta["outputs"]["affine"]["shift_z"])

        # Apply the model to correct the DEM
        tba_unshifted, _ = vshiftcorr.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        # Create a new vertical shift correction model
        vshiftcorr2 = coreg.VerticalShift()
        # Check that this is indeed a new object
        assert vshiftcorr is not vshiftcorr2
        # Fit the corrected DEM to see if the vertical shift will be close to or at zero
        vshiftcorr2.fit(
            reference_elev=self.ref.data,
            to_be_aligned_elev=tba_unshifted,
            transform=self.ref.transform,
            crs=self.ref.crs,
            inlier_mask=self.inlier_mask,
        )
        # Test the vertical shift
        newmeta: CoregDict = vshiftcorr2.meta
        new_vshift = newmeta["outputs"]["affine"]["shift_z"]
        assert np.abs(new_vshift) * res < 0.01

        # Check that the original model's vertical shift has not changed
        # (that the _.meta dicts are two different objects)
        assert vshiftcorr.meta["outputs"]["affine"]["shift_z"] == vshift

    def test_all_nans(self) -> None:
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

    def test_coreg_example(self, verbose: bool = False) -> None:
        """
        Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
        test_examples.py, but helps identify from where differences arise.
        """

        # Run co-registration
        nuth_kaab = xdem.coreg.NuthKaab()
        nuth_kaab.fit(self.ref, self.tba, inlier_mask=self.inlier_mask, verbose=verbose, random_state=42)

        # Check the output .metadata is always the same
        shifts = (nuth_kaab.meta["outputs"]["affine"]["shift_x"], nuth_kaab.meta["outputs"]["affine"]["shift_y"],
                  nuth_kaab.meta["outputs"]["affine"]["shift_z"])
        assert shifts == pytest.approx((-9.200801, -2.785496, -1.9818556))

    def test_gradientdescending(self, subsample: int = 10000, inlier_mask: bool = True, verbose: bool = False) -> None:
        """
        Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
        test_examples.py, but helps identify from where differences arise.

        It also implicitly tests the z_name kwarg and whether a geometry column can be provided instead of E/N cols.
        """
        if inlier_mask:
            inlier_mask = self.inlier_mask

        # Run co-registration
        gds = xdem.coreg.GradientDescending(subsample=subsample)
        gds.fit(
            self.ref.to_pointcloud(data_column_name="z").ds,
            self.tba,
            inlier_mask=inlier_mask,
            verbose=verbose,
            random_state=42,
        )

        shifts = (gds.meta["outputs"]["affine"]["shift_x"], gds.meta["outputs"]["affine"]["shift_y"],
                  gds.meta["outputs"]["affine"]["shift_z"])
        assert shifts == pytest.approx((-10.625, -2.65625, 1.940031), abs=10e-5)

    @pytest.mark.parametrize("shift_px", [(1, 1), (2, 2)])  # type: ignore
    @pytest.mark.parametrize("coreg_class", [coreg.NuthKaab, coreg.GradientDescending, coreg.ICP])  # type: ignore
    @pytest.mark.parametrize("points_or_raster", ["raster", "points"])
    def test_coreg_example_shift(self, shift_px, coreg_class, points_or_raster, verbose=False, subsample=5000):
        """
        For comparison of coreg algorithms:
        Shift a ref_dem on purpose, e.g. shift_px = (1,1), and then applying coreg to shift it back.
        """
        res = self.ref.res[0]

        # shift DEM by shift_px
        shifted_ref = self.ref.copy()
        shifted_ref.translate(shift_px[0] * res, shift_px[1] * res, inplace=True)

        shifted_ref_points = shifted_ref.to_pointcloud(subsample=subsample, random_state=42).ds
        shifted_ref_points.rename(columns={"b1": "z"}, inplace=True)

        kwargs = {} if coreg_class.__name__ != "GradientDescending" else {"subsample": subsample}

        coreg_obj = coreg_class(**kwargs)

        if points_or_raster == "raster":
            coreg_obj.fit(shifted_ref, self.ref, verbose=verbose, random_state=42)
        elif points_or_raster == "points":
            coreg_obj.fit(shifted_ref_points, self.ref, verbose=verbose, random_state=42)

        if coreg_class.__name__ == "ICP":
            matrix = coreg_obj.to_matrix()
            # The ICP fit only creates a matrix and doesn't normally show the alignment in pixels
            # Since the test is formed to validate pixel shifts, these calls extract the approximate pixel shift
            # from the matrix (it's not perfect since rotation/scale can change it).
            coreg_obj.meta["outputs"]["affine"]["shift_x"] = -matrix[0][3]
            coreg_obj.meta["outputs"]["affine"]["shift_y"] = -matrix[1][3]

        # ICP can never be expected to be much better than 1px on structured data, as its implementation often finds a
        # minimum between two grid points. This is clearly warned for in the documentation.
        precision = 1e-2 if coreg_class.__name__ != "ICP" else 1

        assert coreg_obj.meta["outputs"]["affine"]["shift_x"] == pytest.approx(-shift_px[0] * res, rel=precision)
        assert coreg_obj.meta["outputs"]["affine"]["shift_y"] == pytest.approx(-shift_px[0] * res, rel=precision)

    def test_nuth_kaab(self) -> None:

        nuth_kaab = coreg.NuthKaab(max_iterations=50)

        # Synthesize a shifted and vertically offset DEM
        pixel_shift = 2
        vshift = 5
        shifted_dem = self.ref.data.squeeze().copy()
        shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
        shifted_dem[:, :pixel_shift] = np.nan
        shifted_dem += vshift

        # Fit the synthesized shifted DEM to the original
        nuth_kaab.fit(
            self.ref.data.squeeze(),
            shifted_dem,
            transform=self.ref.transform,
            crs=self.ref.crs,
            verbose=self.fit_params["verbose"],
        )

        # Make sure that the estimated offsets are similar to what was synthesized.
        res = self.ref.res[0]
        assert nuth_kaab.meta["outputs"]["affine"]["shift_x"] == pytest.approx(pixel_shift * res, abs=0.03)
        assert nuth_kaab.meta["outputs"]["affine"]["shift_y"] == pytest.approx(0, abs=0.03)
        assert nuth_kaab.meta["outputs"]["affine"]["shift_z"] == pytest.approx(-vshift, 0.03)

        # Apply the estimated shift to "revert the DEM" to its original state.
        unshifted_dem, _ = nuth_kaab.apply(shifted_dem, transform=self.ref.transform, crs=self.ref.crs)
        # Measure the difference (should be more or less zero)
        diff = self.ref.data.squeeze() - unshifted_dem
        diff = diff.compressed()  # turn into a 1D array with only unmasked values

        # Check that the median is very close to zero
        assert np.abs(np.median(diff)) < 0.01
        # Check that the RMSE is low
        assert np.sqrt(np.mean(np.square(diff))) < 1

        # Transform some arbitrary points.
        transformed_points = nuth_kaab.apply(self.points)

        # Check that the x shift is close to the pixel_shift * image resolution
        assert all(
            abs((transformed_points.geometry.x.values - self.points.geometry.x.values) + pixel_shift * self.ref.res[0])
            < 0.1
        )
        # Check that the z shift is close to the original vertical shift.
        assert all(abs((transformed_points["z"].values - self.points["z"].values) + vshift) < 0.1)

    def test_icp_opencv(self) -> None:

        # Do a fast and dirty 3 iteration ICP just to make sure it doesn't error out.
        icp = coreg.ICP(max_iterations=3)
        icp.fit(**self.fit_params)

        aligned_dem, _ = icp.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        assert aligned_dem.shape == self.ref.data.squeeze().shape
