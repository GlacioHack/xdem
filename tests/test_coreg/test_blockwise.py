"""Functions to test the coregistration blockwise classes."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from geoutils import Raster, Vector
from geoutils.raster import RasterType
from geoutils.stats import nmad

import xdem
from xdem import coreg, examples, misc
from xdem.coreg import BlockwiseCoreg
from xdem.coreg.base import Coreg


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

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


class TestBlockwiseCoreg:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_elev=ref.data,
        to_be_aligned_elev=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        crs=ref.crs,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )

    @pytest.mark.parametrize(
        "pipeline", [coreg.VerticalShift(), coreg.VerticalShift() + coreg.NuthKaab()]
    )  # type: ignore
    @pytest.mark.parametrize("subdivision", [4, 10])  # type: ignore
    def test_blockwise_coreg(self, pipeline: Coreg, subdivision: int) -> None:

        blockwise = coreg.BlockwiseCoreg(step=pipeline, subdivision=subdivision)

        # Results can not yet be extracted (since fit has not been called) and should raise an error
        with pytest.raises(AssertionError, match="No coreg results exist.*"):
            blockwise.to_points()

        blockwise.fit(**self.fit_params)
        points = blockwise.to_points()

        # Validate that the number of points is equal to the amount of subdivisions.
        assert points.shape[0] == subdivision

        # Validate that the points do not represent only the same location.
        assert np.sum(np.linalg.norm(points[:, :, 0] - points[:, :, 1], axis=1)) != 0.0

        z_diff = points[:, 2, 1] - points[:, 2, 0]

        # Validate that all values are different
        assert np.unique(z_diff).size == z_diff.size, "Each coreg cell should have different results."

        # Validate that the BlockwiseCoreg doesn't accept uninstantiated Coreg classes
        with pytest.raises(ValueError, match="instantiated Coreg subclass"):
            coreg.BlockwiseCoreg(step=coreg.VerticalShift, subdivision=1)  # type: ignore

        # Metadata copying has been an issue. Validate that all chunks have unique ids
        chunk_numbers = [m["i"] for m in blockwise.meta["step_meta"]]
        assert np.unique(chunk_numbers).shape[0] == len(chunk_numbers)

        transformed_dem = blockwise.apply(self.tba)

        ddem_pre = (self.ref - self.tba)[~self.inlier_mask]
        ddem_post = (self.ref - transformed_dem)[~self.inlier_mask]

        # Check that the periglacial difference is lower after coregistration.
        assert abs(np.ma.median(ddem_post)) < abs(np.ma.median(ddem_pre))

        stats = blockwise.stats()

        # Check that nans don't exist (if they do, something has gone very wrong)
        assert np.all(np.isfinite(stats["nmad"]))
        # Check that offsets were actually calculated.
        assert np.sum(np.abs(np.linalg.norm(stats[["x_off", "y_off", "z_off"]], axis=0))) > 0

    def test_blockwise_coreg_large_gaps(self) -> None:
        """Test BlockwiseCoreg when large gaps are encountered, e.g. around the frame of a rotated DEM."""
        reference_dem = self.ref.reproject(crs="EPSG:3413", res=self.ref.res, resampling="bilinear")
        dem_to_be_aligned = self.tba.reproject(ref=reference_dem, resampling="bilinear")

        blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), 64, warn_failures=False)

        # This should not fail or trigger warnings as warn_failures is False
        blockwise.fit(reference_dem, dem_to_be_aligned)

        stats = blockwise.stats()

        # We expect holes in the blockwise coregistration, but not in stats due to nan padding for failing chunks
        assert stats.shape[0] == 64

        # Copy the TBA DEM and set a square portion to nodata
        tba = self.tba.copy()
        mask = np.zeros(np.shape(tba.data), dtype=bool)
        mask[450:500, 450:500] = True
        tba.set_mask(mask=mask)

        blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), 8, warn_failures=False)

        # Align the DEM and apply blockwise to a zero-array (to get the z_shift)
        aligned = blockwise.fit(self.ref, tba).apply(tba)
        zshift, _ = blockwise.apply(np.zeros_like(tba.data), transform=tba.transform, crs=tba.crs)

        # Validate that the zshift is not something crazy high and that no negative values exist in the data.
        assert np.nanmax(np.abs(zshift)) < 50
        assert np.count_nonzero(aligned.data.compressed() < -50) == 0

        # Check that coregistration improved the alignment
        ddem_post = (aligned - self.ref).data.compressed()
        ddem_pre = (tba - self.ref).data.compressed()
        assert abs(np.nanmedian(ddem_pre)) > abs(np.nanmedian(ddem_post))
        # assert np.nanstd(ddem_pre) > np.nanstd(ddem_post)

    def test_failed_chunks_return_nan(self) -> None:
        blockwise = BlockwiseCoreg(xdem.coreg.NuthKaab(), subdivision=4)
        blockwise.fit(**self.fit_params)
        # Missing chunk 1 to simulate failure
        blockwise._meta["step_meta"] = [meta for meta in blockwise._meta["step_meta"] if meta.get("i") != 1]

        result_df = blockwise.stats()

        # Check that chunk 1 (index 1) has NaN values for the statistics
        assert np.isnan(result_df.loc[1, "inlier_count"])
        assert np.isnan(result_df.loc[1, "nmad"])
        assert np.isnan(result_df.loc[1, "median"])
        assert isinstance(result_df.loc[1, "center_x"], float)
        assert isinstance(result_df.loc[1, "center_y"], float)
        assert np.isnan(result_df.loc[1, "center_z"])
        assert np.isnan(result_df.loc[1, "x_off"])
        assert np.isnan(result_df.loc[1, "y_off"])
        assert np.isnan(result_df.loc[1, "z_off"])

    def test_successful_chunks_return_values(self) -> None:
        blockwise = BlockwiseCoreg(xdem.coreg.NuthKaab(), subdivision=2)
        blockwise.fit(**self.fit_params)
        result_df = blockwise.stats()

        # Check that the correct statistics are returned for successful chunks
        assert result_df.loc[0, "inlier_count"] == blockwise._meta["step_meta"][0]["inlier_count"]
        assert result_df.loc[0, "nmad"] == blockwise._meta["step_meta"][0]["nmad"]
        assert result_df.loc[0, "median"] == blockwise._meta["step_meta"][0]["median"]

        assert result_df.loc[1, "inlier_count"] == blockwise._meta["step_meta"][1]["inlier_count"]
        assert result_df.loc[1, "nmad"] == blockwise._meta["step_meta"][1]["nmad"]
        assert result_df.loc[1, "median"] == blockwise._meta["step_meta"][1]["median"]


def test_warp_dem() -> None:
    """Test that the warp_dem function works expectedly."""

    small_dem = np.zeros((5, 10), dtype="float32")
    small_transform = rio.transform.from_origin(0, 5, 1, 1)

    source_coords = np.array([[0, 0, 0], [0, 5, 0], [10, 0, 0], [10, 5, 0]]).astype(small_dem.dtype)

    dest_coords = source_coords.copy()
    dest_coords[0, 0] = -1e-5

    warped_dem = coreg.blockwise.warp_dem(
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
    warped_dem = coreg.blockwise.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
    )

    # The warped DEM should have the value 'elev_shift' in the upper left corner.
    assert warped_dem[0, 0] == -elev_shift
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
    transformed_dem = coreg.blockwise.warp_dem(
        dem=dem, transform=transform, source_coords=source_coords, destination_coords=dest_coords, resampling="linear"
    )

    # Try to undo the warp by reversing the source-destination coordinates.
    untransformed_dem = coreg.blockwise.warp_dem(
        dem=transformed_dem,
        transform=transform,
        source_coords=dest_coords,
        destination_coords=source_coords,
        resampling="linear",
    )
    # Validate that the DEM is now more or less the same as the original.
    # Due to the randomness, the threshold is quite high, but would be something like 10+ if it was incorrect.
    assert nmad(dem - untransformed_dem) < 0.5

    # Test with Z-correction disabled
    transformed_dem_no_z = coreg.blockwise.warp_dem(
        dem=dem,
        transform=transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
        apply_z_correction=False,
    )

    # Try to undo the warp by reversing the source-destination coordinates with Z-correction disabled
    untransformed_dem_no_z = coreg.blockwise.warp_dem(
        dem=transformed_dem_no_z,
        transform=transform,
        source_coords=dest_coords,
        destination_coords=source_coords,
        resampling="linear",
        apply_z_correction=False,
    )

    # Validate that the DEM is now more or less the same as the original, with Z-correction disabled.
    # The result should be similar to the original, but with no Z-shift applied.
    assert nmad(dem - untransformed_dem_no_z) < 0.5

    # The difference between the two DEMs should be the vertical shift.
    # We expect the difference to be approximately equal to the average vertical shift.
    expected_vshift = np.mean(dest_coords[:, 2] - source_coords[:, 2])

    # Check that the mean difference between the DEMs matches the expected vertical shift.
    assert np.nanmean(transformed_dem_no_z - transformed_dem) == pytest.approx(expected_vshift, rel=0.3)

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
