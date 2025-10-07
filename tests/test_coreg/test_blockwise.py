"""Tests for the BlockwiseCoreg class."""

# mypy: disable-error-code=no-untyped-def
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
from xdem.coreg import BlockwiseCoreg, Coreg


@pytest.fixture(scope="module")  # type: ignore
def example_data() -> tuple[RasterType, RasterType, Vector]:
    """Load example DEMs and glacier outlines with inlier mask."""
    ref_dem = Raster(xdem.examples.get_path("longyearbyen_ref_dem"))
    tba_dem = Raster(xdem.examples.get_path("longyearbyen_tba_dem"))
    outlines = Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    inlier_mask = ~outlines.create_mask(ref_dem)
    return ref_dem, tba_dem, inlier_mask


@pytest.fixture  # type: ignore
def step() -> Coreg:
    return xdem.coreg.NuthKaab(vertical_shift=False)


@pytest.fixture  # type: ignore
def mp_config(tmp_path: Path) -> MultiprocConfig:
    return MultiprocConfig(chunk_size=500, outfile=tmp_path / "test.tif")


@pytest.fixture  # type: ignore
def blockwise_coreg(step, mp_config) -> BlockwiseCoreg:
    return xdem.coreg.BlockwiseCoreg(step=step, mp_config=mp_config)


class TestBlockwiseCoreg:
    """Tests for the xdem.coreg.BlockwiseCoreg class."""

    def test_init_with_valid_parameters(self, mp_config, step, tmp_path) -> None:
        """Test initialization with valid multiprocessing config only."""
        coreg_obj = xdem.coreg.BlockwiseCoreg(step=step, mp_config=mp_config)
        assert coreg_obj.block_size_apply == 500
        assert coreg_obj.block_size_fit == 500
        assert coreg_obj.apply_z_correction is False
        assert coreg_obj.output_path_reproject == tmp_path / "reprojected_dem.tif"
        assert coreg_obj.output_path_aligned == tmp_path / "aligned_dem.tif"
        assert coreg_obj.meta == {"inputs": {}, "outputs": {}}

    def test_init_raises_if_both_mp_config_and_parent_path_are_provided(self, mp_config, step, tmp_path) -> None:
        """Test error is raised when both 'mp_config' and 'parent_path' are set."""
        with pytest.raises(
            ValueError, match="Only one of the parameters 'mp_config' or 'parent_path' may be specified."
        ):
            xdem.coreg.BlockwiseCoreg(step=step, mp_config=mp_config, parent_path=tmp_path)

    def test_init_raises_if_neither_mp_config_nor_parent_path_are_provided(self, step) -> None:
        """Test error is raised when neither 'mp_config' nor 'parent_path' are set."""
        with pytest.raises(
            ValueError, match="Exactly one of the parameters 'mp_config' or 'parent_path' must be provided."
        ):
            xdem.coreg.BlockwiseCoreg(step=step, mp_config=None, parent_path=None)

    def test_init_success_with_only_mp_config(self, step, mp_config, tmp_path) -> None:
        """Test successful initialization with only 'mp_config' set."""
        obj = xdem.coreg.BlockwiseCoreg(step=step, mp_config=mp_config, parent_path=None)
        assert isinstance(obj, xdem.coreg.BlockwiseCoreg)
        assert obj.mp_config == mp_config
        assert obj.parent_path == tmp_path

    def test_init_success_with_only_parent_path(self, step, tmp_path) -> None:
        """Test successful initialization with only 'parent_path' set."""
        obj = xdem.coreg.BlockwiseCoreg(step=step, mp_config=None, parent_path=tmp_path)
        assert isinstance(obj, xdem.coreg.BlockwiseCoreg)
        assert obj.parent_path == tmp_path

    def test_ransac_with_large_data(self, blockwise_coreg) -> None:
        """Test RANSAC estimates with synthetic data and known coefficients."""
        np.random.seed(0)
        x = np.random.rand(1000) * 100
        y = np.random.rand(1000) * 100
        z = 2 * x + 3 * y + 5 + np.random.randn(1000) * 0.1

        a, b, c = blockwise_coreg._ransac(x, y, z)

        assert np.isclose(a, 2.0, atol=0.2)
        assert np.isclose(b, 3.0, atol=0.2)
        assert np.isclose(c, 5.0, atol=0.2)

    def test_wrapper_apply_epc(self, blockwise_coreg, example_data) -> None:
        """Test point cloud coefficient application via _wrapper_apply_epc."""
        _, tba_dem, _ = example_data
        epc = tba_dem.to_pointcloud(data_column_name="z").ds
        x, y, z = epc.geometry.x.values, epc.geometry.y.values, epc["z"].values

        shift_x = x + y + 1
        shift_y = x + y + 1
        shift_z = x + y + 1

        x_new = x + shift_x
        y_new = y + shift_y
        z_new = z + shift_z

        trans_epc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x_new, y_new, crs=epc.crs),
            data={"z": z_new},
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            new_dem = _grid_pointcloud(
                trans_epc,
                grid_coords=tba_dem.coords(grid=False),
                data_column_name="z",
            )

        expected = Raster.from_array(new_dem, tba_dem.transform, tba_dem.crs, tba_dem.nodata)
        actual = blockwise_coreg._wrapper_apply_epc(tba_dem, (1, 1, 1), (1, 1, 1), (1, 1, 1))

        assert actual == expected

    @pytest.mark.parametrize("block_size", [500, 985, 1332], ids=["2d_shifts", "1d_shifts_x", "monotile"])
    def test_blockwise_coreg_pipeline(self, step, example_data, tmp_path, block_size):
        """Test end-to-end blockwise coregistration and validate output."""
        ref, tba, mask = example_data

        config_mc = MultiprocConfig(chunk_size=block_size, outfile=tmp_path / "test.tif")
        blockwise_coreg = xdem.coreg.BlockwiseCoreg(step=step, mp_config=config_mc, block_size_fit=block_size)
        blockwise_coreg.fit(ref, tba, mask)
        blockwise_coreg.apply()

        aligned = xdem.DEM(tmp_path / "aligned_dem.tif")

        # Ground truth comparison with full image coregistration
        nuth_kaab = xdem.coreg.NuthKaab()
        expected = nuth_kaab.fit_and_apply(ref, tba, mask)

        valid = (expected.data.data != expected.nodata) & (aligned.data.data != aligned.nodata)
        assert np.allclose(expected.data.data[valid], aligned.data.data[valid], atol=20)

    def test_ransac_on_horizontal_tiles(self, blockwise_coreg) -> None:
        """Test case where RANSAC works on horizontal tiles."""
        x = np.linspace(0, 100, 50)
        y = np.full_like(x, 50)
        shift = 0.2 * x + 3.0

        a, b, c = blockwise_coreg._ransac(x, y, shift)

        assert np.isclose(a, 0.2, atol=1e-2)
        assert np.isclose(b, 0.0, atol=1e-6)
        assert np.isclose(c, 3.0, atol=1e-2)

    def test_ransac_on_vertical_tiles(self, blockwise_coreg):
        """Test case where RANSAC works on vertical tiles."""
        y = np.linspace(0, 100, 50)
        x = np.full_like(y, 50)
        shift = -0.1 * y + 1.5

        a, b, c = blockwise_coreg._ransac(x, y, shift)

        assert np.isclose(a, 0.0, atol=1e-6)
        assert np.isclose(b, -0.1, atol=1e-2)
        assert np.isclose(c, 1.5, atol=1e-2)

    def test_ransac_on_2d_grid(self, blockwise_coreg) -> None:
        """Test case where RANSAC works on 2D grid."""
        x, y = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 5, 10))
        x = x.ravel()
        y = y.ravel()
        shift = 0.3 * x - 0.2 * y + 1.0

        a, b, c = blockwise_coreg._ransac(x, y, shift)

        assert np.isclose(a, 0.3, atol=1e-2)
        assert np.isclose(b, -0.2, atol=1e-2)
        assert np.isclose(c, 1.0, atol=1e-2)
