"""Functions to test the coregistration blockwise classes."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geoutils import Raster, Vector
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.raster import RasterType
from geoutils.raster.distributed_computing import MultiprocConfig

import xdem
from xdem import examples


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_outlines = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    # Create a stable ground mask (not glacierized) to mark "inlier data"
    inlier_mask = ~glacier_outlines.create_mask(reference_dem)

    return reference_dem, to_be_aligned_dem, inlier_mask


def coreg_object(path: Path) -> xdem.coreg.BlockwiseCoreg:
    """
    Fixture to create a coregistration object
    """

    mp_config = MultiprocConfig(chunk_size=500, outfile=path / "test.tif")
    step = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_obj = xdem.coreg.BlockwiseCoreg(step=step, mp_config=mp_config)

    return coreg_obj


class TestBlockwiseCoreg:
    """
    Class for testing Blockwise coregistration
    """

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.

    def test_init_with_valid_parameters(self, tmp_path: Path) -> None:
        """
        Test initialisation of CoregBlockwise class
        """

        coreg_obj = coreg_object(tmp_path)

        assert coreg_obj.block_size == 500
        assert coreg_obj.apply_z_correction is False
        assert coreg_obj.output_path_reproject == tmp_path / "reprojected_dem.tif"
        assert coreg_obj.output_path_aligned == tmp_path / "aligned_dem.tif"
        assert coreg_obj.meta == {"inputs": {}, "outputs": {}}

    def test_ransac_with_large_data(self, tmp_path: Path) -> None:

        coreg_obj = coreg_object(tmp_path)

        np.random.seed(0)
        num_points = 1000
        x_coords = np.random.rand(num_points) * 100
        y_coords = np.random.rand(num_points) * 100
        # add noises
        shifts = 2 * x_coords + 3 * y_coords + 5 + np.random.randn(num_points) * 0.1
        a, b, c = coreg_obj.ransac(x_coords, y_coords, shifts)

        assert np.isclose(a, 2.0, atol=0.2)
        assert np.isclose(b, 3.0, atol=0.2)
        assert np.isclose(c, 5.0, atol=0.2)

    def test_ransac_with_insufficient_points(self, tmp_path: Path) -> None:
        """
        Test ransac function failure and user warning
        """

        coreg_obj = coreg_object(tmp_path)

        x_coords = np.array([1, 2, 3, 4, 5])
        y_coords = np.array([1, 2, 3, 4, 5])
        shifts = np.array([2, 4, 6, 8, 10])

        with pytest.raises(ValueError):
            coreg_obj.ransac(x_coords, y_coords, shifts)

    def test_wrapper_apply_epc(self, tmp_path: Path) -> None:
        """
        test wrapper_apply_epc function
        """

        coreg_obj = coreg_object(tmp_path)

        _, tba_dem_tile, _ = load_examples()

        # To pointcloud
        epc = tba_dem_tile.to_pointcloud(data_column_name="z").ds
        # Unpack coefficients
        a_x, b_x, d_x = [1, 1, 1]
        a_y, b_y, d_y = [1, 1, 1]
        a_z, b_z, d_z = [1, 1, 1]

        # Extract x, y, z from the point cloud
        x = epc.geometry.x.values
        y = epc.geometry.y.values
        z = epc["z"].values

        # Compute modeled shift fields
        shift_x = a_x * x + b_x * y + d_x
        shift_y = a_y * x + b_y * y + d_y
        shift_z = a_z * x + b_z * y + d_z

        # Apply shifts to the coordinates
        x_new = x + shift_x
        y_new = y + shift_y
        z_new = z + shift_z

        trans_epc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x_new, y_new, crs=epc.crs),
            data={"z": z_new if True else z},
        )

        with warnings.catch_warnings():
            # CRS mismatch between the CRS of left geometries and the CRS of right geometries.
            warnings.filterwarnings("ignore", category=UserWarning)
            # To raster
            new_dem = _grid_pointcloud(
                trans_epc,
                grid_coords=tba_dem_tile.coords(grid=False),
                data_column_name="z",
            )

        applied_dem_tile_vt = Raster.from_array(new_dem, tba_dem_tile.transform, tba_dem_tile.crs, tba_dem_tile.nodata)

        applied_dem_tile = coreg_obj.wrapper_apply_epc(tba_dem_tile, (1, 1, 1), (1, 1, 1), (1, 1, 1))

        assert applied_dem_tile == applied_dem_tile_vt

    def test_blockwise_coreg(self, tmp_path: Path) -> None:
        """
        test blockwise pipeline with Nuth and Kaab coregistration
        """
        mp_config = MultiprocConfig(chunk_size=500, outfile=tmp_path / "test.tif")
        blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(vertical_shift=False), mp_config=mp_config)

        reference_dem, to_be_aligned_dem, inlier_mask = load_examples()

        blockwise.fit(reference_dem, to_be_aligned_dem, inlier_mask)
        blockwise.apply()

        aligned_dem = xdem.DEM(tmp_path / "aligned_dem.tif")

        # Ground truth is global coregistration
        nuth_kaab = xdem.coreg.NuthKaab()
        aligned_dem_vt = nuth_kaab.fit_and_apply(reference_dem, to_be_aligned_dem, inlier_mask)

        mask = (aligned_dem_vt.data.data != aligned_dem.nodata) & (aligned_dem.data.data != aligned_dem.nodata)

        assert np.allclose(aligned_dem_vt.data.data[mask], aligned_dem.data.data[mask], atol=10)
