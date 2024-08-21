import numpy as np
import pytest

import xdem
from xdem import coreg
from xdem.coreg.base import Coreg


@pytest.mark.parametrize("pipeline", [coreg.VerticalShift(), coreg.VerticalShift() + coreg.NuthKaab()])  # type: ignore
@pytest.mark.parametrize("subdivision", [4, 10])  # type: ignore
def test_blockwise_coreg(pipeline: Coreg, subdivision: int, load_examples) -> None:

    ref, tba, _, inlier_mask, fit_params = load_examples

    blockwise = coreg.BlockwiseCoreg(step=pipeline, subdivision=subdivision)

    # Results can not yet be extracted (since fit has not been called) and should raise an error
    with pytest.raises(AssertionError, match="No coreg results exist.*"):
        blockwise.to_points()

    blockwise.fit(**fit_params)
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

    transformed_dem = blockwise.apply(tba)

    ddem_pre = (ref - tba)[~inlier_mask]
    ddem_post = (ref - transformed_dem)[~inlier_mask]

    # Check that the periglacial difference is lower after coregistration.
    assert abs(np.ma.median(ddem_post)) < abs(np.ma.median(ddem_pre))

    stats = blockwise.stats()

    # Check that nans don't exist (if they do, something has gone very wrong)
    assert np.all(np.isfinite(stats["nmad"]))
    # Check that offsets were actually calculated.
    assert np.sum(np.abs(np.linalg.norm(stats[["x_off", "y_off", "z_off"]], axis=0))) > 0


def test_blockwise_coreg_large_gaps(load_examples) -> None:
    """Test BlockwiseCoreg when large gaps are encountered, e.g. around the frame of a rotated DEM."""

    ref, tba, _, inlier_mask, fit_params = load_examples

    reference_dem = ref.reproject(crs="EPSG:3413", res=ref.res, resampling="bilinear")
    dem_to_be_aligned = tba.reproject(ref=reference_dem, resampling="bilinear")

    blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), 64, warn_failures=False)

    # This should not fail or trigger warnings as warn_failures is False
    blockwise.fit(reference_dem, dem_to_be_aligned)

    stats = blockwise.stats()

    # We expect holes in the blockwise coregistration, so there should not be 64 "successful" blocks.
    assert stats.shape[0] < 64

    # Statistics are only calculated on finite values, so all of these should be finite as well.
    assert np.all(np.isfinite(stats))

    # Copy the TBA DEM and set a square portion to nodata
    tba = tba.copy()
    mask = np.zeros(np.shape(tba.data), dtype=bool)
    mask[450:500, 450:500] = True
    tba.set_mask(mask=mask)

    blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), 8, warn_failures=False)

    # Align the DEM and apply the blockwise to a zero-array (to get the zshift)
    aligned = blockwise.fit(ref, tba).apply(tba)
    zshift, _ = blockwise.apply(np.zeros_like(tba.data), transform=tba.transform, crs=tba.crs)

    # Validate that the zshift is not something crazy high and that no negative values exist in the data.
    assert np.nanmax(np.abs(zshift)) < 50
    assert np.count_nonzero(aligned.data.compressed() < -50) == 0

    # Check that coregistration improved the alignment
    ddem_post = (aligned - ref).data.compressed()
    ddem_pre = (tba - ref).data.compressed()
    assert abs(np.nanmedian(ddem_pre)) > abs(np.nanmedian(ddem_post))
    assert np.nanstd(ddem_pre) > np.nanstd(ddem_post)
