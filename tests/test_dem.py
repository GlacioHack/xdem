""" Functions to test the DEM tools."""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from geoutils.raster.raster import _default_rio_attrs
from pyproj import CRS
from pyproj.transformer import Transformer

import xdem
import xdem.vcrs
from xdem.dem import DEM

DO_PLOT = False


class TestDEM:
    def test_init(self) -> None:
        """Test that inputs work properly in DEM class init."""
        fn_img = xdem.examples.get_path("longyearbyen_ref_dem")

        # From filename
        dem = DEM(fn_img)
        assert isinstance(dem, DEM)

        # From DEM
        dem2 = DEM(dem)
        assert isinstance(dem2, DEM)

        # From Raster
        r = gu.Raster(fn_img)
        dem3 = DEM(r)
        assert isinstance(dem3, DEM)

        list_dem = [dem, dem2, dem3]

        # Check all attributes
        attrs = [at for at in _default_rio_attrs if at not in ["name", "dataset_mask", "driver"]]
        all_attrs = attrs + xdem.dem.dem_attrs
        for attr in all_attrs:
            attrs_per_dem = [idem.__getattribute__(attr) for idem in list_dem]
            assert all(at == attrs_per_dem[0] for at in attrs_per_dem)

        assert np.logical_and.reduce(
            (
                np.array_equal(dem.data, dem2.data, equal_nan=True),
                np.array_equal(dem2.data, dem3.data, equal_nan=True),
            )
        )

        assert np.logical_and.reduce(
            (
                np.all(dem.data.mask == dem2.data.mask),
                np.all(dem2.data.mask == dem3.data.mask),
            )
        )

        # Check that an error is raised when more than one band is provided
        with pytest.raises(ValueError, match="DEM rasters should be composed of one band only"):
            xdem.DEM.from_array(
                data=np.zeros(shape=(2, 5, 5)),
                transform=rio.transform.from_bounds(0, 0, 1, 1, 5, 5),
                crs=None,
                nodata=None,
            )

    def test_init__vcrs(self) -> None:
        """Test that vcrs is set properly during instantiation."""

        # Tests 1: instantiation with a file that has a 2D CRS

        # First, check a DEM that does not have any vertical CRS set
        fn_img = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_img)
        assert dem.vcrs is None

        # Setting a vertical CRS during instantiation should work here
        dem = DEM(fn_img, vcrs="EGM96")
        assert dem.vcrs_name == "EGM96 height"

        # Tests 2: instantiation with a file that has a 3D CRS
        # Create such a file
        dem = DEM(fn_img)
        dem_reproj = dem.reproject(crs=4979)

        # Save to temporary folder
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.tif")
        dem_reproj.save(temp_file)

        # Check opening a DEM with a 3D CRS sets the vcrs
        dem_3d = DEM(temp_file)
        assert dem_3d.vcrs == "Ellipsoid"

        # Check that a warning is raised when trying to override with user input
        with pytest.warns(
            UserWarning,
            match="The CRS in the raster metadata already has a vertical component, "
            "the user-input 'EGM08' will override it.",
        ):
            DEM(temp_file, vcrs="EGM08")

    def test_from_array(self) -> None:
        """Test that overridden from_array works as expected."""

        # Create a 5x5 DEM
        data = np.ones((5, 5))
        transform = rio.transform.from_bounds(0, 0, 1, 1, 5, 5)
        crs = CRS("EPSG:4326")
        nodata = -9999
        vcrs = "EGM08"
        dem = DEM.from_array(data=data, transform=transform, crs=crs, nodata=nodata, vcrs=vcrs)

        # Check output matches
        assert isinstance(dem, DEM)
        assert isinstance(dem.data, np.ma.masked_array)
        assert np.array_equal(dem.data.data, np.ones((5, 5)))
        assert dem.transform == transform
        assert dem.crs == crs
        assert dem.nodata == nodata
        assert dem.vcrs == xdem.vcrs._vcrs_from_user_input(vcrs_input=vcrs)

    def test_from_array__vcrs(self) -> None:
        """Test that overridden from_array rightly sets the vertical CRS."""

        # Create a 5x5 DEM with a 2D CRS
        transform = rio.transform.from_bounds(0, 0, 1, 1, 5, 5)
        dem = DEM.from_array(data=np.ones((5, 5)), transform=transform, crs=CRS("EPSG:4326"), nodata=None, vcrs=None)
        assert dem.vcrs is None

        # One with a 3D ellipsoid CRS
        dem = DEM.from_array(data=np.ones((5, 5)), transform=transform, crs=CRS("EPSG:4979"), nodata=None, vcrs=None)
        assert dem.vcrs == "Ellipsoid"

        # One with a 2D and the ellipsoid vertical CRS
        dem = DEM.from_array(
            data=np.ones((5, 5)), transform=transform, crs=CRS("EPSG:4326"), nodata=None, vcrs="Ellipsoid"
        )
        assert dem.vcrs == "Ellipsoid"

        # One with a compound CRS
        dem = DEM.from_array(
            data=np.ones((5, 5)), transform=transform, crs=CRS("EPSG:4326+5773"), nodata=None, vcrs=None
        )
        assert dem.vcrs == CRS("EPSG:5773")

        # One with a CRS and vertical CRS
        dem = DEM.from_array(
            data=np.ones((5, 5)), transform=transform, crs=CRS("EPSG:4326"), nodata=None, vcrs=CRS("EPSG:5773")
        )
        assert dem.vcrs == CRS("EPSG:5773")

    def test_from_array__cast_mask(self) -> None:
        """Test that DEMs are cast into mask for a logical operation."""

        transform = rio.transform.from_bounds(0, 0, 1, 1, 5, 5)
        dem = DEM.from_array(data=np.ones((5, 5)), transform=transform, crs=CRS("EPSG:4326"), nodata=None)

        mask_dem = dem > 1
        assert isinstance(mask_dem, gu.Mask)

    def test_copy(self) -> None:
        """
        Test that the copy method works as expected for DEM. In particular
        when copying r to r2:
        - if r.data is modified and r copied, the updated data is copied
        - if r is copied, r.data changed, r2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        r = xdem.dem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a DEM
        assert isinstance(r2, xdem.dem.DEM)

        # Check all immutable attributes are equal
        # raster_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'bands', 'nodata',
        #                    'res', 'shape', 'transform', 'width']
        # satimg_attrs = ['satellite', 'sensor', 'product', 'version', 'tile_name', 'datetime']
        # dem_attrs = ['vcrs', 'vcrs_grid', 'vcrs_name', 'ccrs']

        # using list directly available in Class
        attrs = [at for at in _default_rio_attrs if at not in ["name", "dataset_mask", "driver", "profile"]]
        all_attrs = attrs + xdem.dem.dem_attrs
        for attr in all_attrs:
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert np.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.all(r.data.mask == r2.data.mask)

        # Check that if r.data is modified, it does not affect r2.data
        r.data += 5
        assert not np.array_equal(r.data, r2.data, equal_nan=True)

        # Check that the new_array argument indeed modifies the raster
        r3 = r.copy(new_array=r2.data)

        assert np.array_equal(r3.data, r2.data)

    def test_set_vcrs(self) -> None:
        """Tests to set the vertical CRS."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_dem)

        # -- Test 1: we check with names --

        # Check setting ellipsoid
        dem.set_vcrs(new_vcrs="Ellipsoid")
        assert dem.vcrs_name is not None
        assert "Ellipsoid (No vertical CRS)." in dem.vcrs_name
        assert dem.vcrs_grid is None

        # Check setting EGM96
        dem.set_vcrs(new_vcrs="EGM96")
        assert dem.vcrs_name == "EGM96 height"
        assert dem.vcrs_grid == "us_nga_egm96_15.tif"

        # Check setting EGM08
        dem.set_vcrs(new_vcrs="EGM08")
        assert dem.vcrs_name == "EGM2008 height"
        assert dem.vcrs_grid == "us_nga_egm08_25.tif"

        # -- Test 2: we check with grids --
        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid not found in *")

        dem.set_vcrs(new_vcrs="us_nga_egm96_15.tif")
        assert dem.vcrs_name == "unknown using geoidgrids=us_nga_egm96_15.tif"
        assert dem.vcrs_grid == "us_nga_egm96_15.tif"

        dem.set_vcrs(new_vcrs="us_nga_egm08_25.tif")
        assert dem.vcrs_name == "unknown using geoidgrids=us_nga_egm08_25.tif"
        assert dem.vcrs_grid == "us_nga_egm08_25.tif"

        # Check that other existing grids are well detected in the pyproj.datadir
        dem.set_vcrs(new_vcrs="is_lmi_Icegeoid_ISN93.tif")

        # Check that non-existing grids raise errors
        with pytest.warns(UserWarning, match="Grid not found in*"):
            with pytest.raises(
                ValueError,
                match="The provided grid 'the best grid' does not exist at https://cdn.proj.org/. "
                "Provide an existing grid.",
            ):
                dem.set_vcrs(new_vcrs="the best grid")

    def test_to_vcrs(self) -> None:
        """Tests the conversion of vertical CRS."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_dem)

        # Reproject in WGS84 2D
        dem = dem.reproject(crs=4326)
        dem_before_trans = dem.copy()

        # Set ellipsoid as vertical reference
        dem.set_vcrs(new_vcrs="Ellipsoid")
        ccrs_init = dem.ccrs
        median_before = np.nanmean(dem)
        # Transform to EGM96 geoid not inplace (default)
        trans_dem = dem.to_vcrs(vcrs="EGM96")

        # The output should be a DEM, input shouldn't have changed
        assert isinstance(trans_dem, DEM)
        assert dem.raster_equal(dem_before_trans)

        # Compare to inplace
        should_be_none = dem.to_vcrs(vcrs="EGM96", inplace=True)
        assert should_be_none is None
        assert dem.raster_equal(trans_dem)

        # Save the median of after
        median_after = np.nanmean(trans_dem)

        # About 32 meters of difference in Svalbard between EGM96 geoid and ellipsoid
        assert median_after - median_before == pytest.approx(-32, rel=0.1)

        # Check that the results are consistent with the operation done independently
        ccrs_dest = xdem.vcrs._build_ccrs_from_crs_and_vcrs(dem.crs, xdem.vcrs._vcrs_from_user_input("EGM96"))
        transformer = Transformer.from_crs(crs_from=ccrs_init, crs_to=ccrs_dest, always_xy=True)

        xx, yy = dem.coords()
        x = xx[5, 5]
        y = yy[5, 5]
        z = dem_before_trans.data[5, 5]
        z_out = transformer.transform(xx=x, yy=y, zz=z)[2]

        assert z_out == pytest.approx(dem.data[5, 5])

    def test_to_vcrs__equal_warning(self) -> None:
        """Test that DEM.to_vcrs() does not transform if both 3D CRS are equal."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_dem)

        # With both inputs as names
        dem.set_vcrs("EGM96")
        with pytest.warns(
            UserWarning, match="Source and destination vertical CRS are the same, " "skipping vertical transformation."
        ):
            dem.to_vcrs("EGM96")

        # With one input as name, the other as CRS
        dem.set_vcrs("Ellipsoid")
        with pytest.warns(
            UserWarning, match="Source and destination vertical CRS are the same, " "skipping vertical transformation."
        ):
            dem.to_vcrs(CRS("EPSG:4979"))

    # Compare to manually-extracted shifts at specific coordinates for the geoid grids
    egm96_chile = {"grid": "us_nga_egm96_15.tif", "lon": -68, "lat": -20, "shift": 42}
    egm08_chile = {"grid": "us_nga_egm08_25.tif", "lon": -68, "lat": -20, "shift": 42}
    geoid96_alaska = {"grid": "us_noaa_geoid06_ak.tif", "lon": -145, "lat": 62, "shift": 15}
    isn93_iceland = {"grid": "is_lmi_Icegeoid_ISN93.tif", "lon": -18, "lat": 65, "shift": 68}

    @pytest.mark.parametrize("grid_shifts", [egm08_chile, egm08_chile, geoid96_alaska, isn93_iceland])  # type: ignore
    def test_to_vcrs__grids(self, grid_shifts: dict[str, Any]) -> None:
        """Tests grids to convert vertical CRS."""

        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid not found in *")

        # Using an arbitrary elevation of 100 m (no influence on the transformation)
        dem = DEM.from_array(
            data=np.array([[100, 100]]),
            transform=rio.transform.from_bounds(
                grid_shifts["lon"], grid_shifts["lat"], grid_shifts["lon"] + 0.01, grid_shifts["lat"] + 0.01, 0.01, 0.01
            ),
            crs=CRS.from_epsg(4326),
            nodata=None,
        )
        dem.set_vcrs("Ellipsoid")

        # Transform to the vertical CRS of the grid
        trans_dem = dem.to_vcrs(grid_shifts["grid"])

        # Compare the elevation difference
        z_diff = 100 - trans_dem.data[0, 0]

        # Check the shift is the one expect within 10%
        assert z_diff == pytest.approx(grid_shifts["shift"], rel=0.1)

    @pytest.mark.parametrize("terrain_attribute", xdem.terrain.available_attributes)  # type: ignore
    def test_terrain_attributes_wrappers(self, terrain_attribute: str) -> None:
        """Check the terrain attributes corresponds to the ones derived in the terrain module."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_dem)

        dem_class_attr = getattr(dem, terrain_attribute)()
        terrain_module_attr = getattr(xdem.terrain, terrain_attribute)(dem)

        assert dem_class_attr.raster_equal(terrain_module_attr)

    def test_coregister_3d_wrapper(self) -> None:

        fn_ref = xdem.examples.get_path("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)

        dem_class_aligned = dem_tba.coregister_3d(dem_ref, random_state=42)

        nk = xdem.coreg.NuthKaab()
        nk.fit(dem_ref, dem_tba, random_state=42)
        coreg_module_aligned = nk.apply(dem_tba)

        assert dem_class_aligned.raster_equal(coreg_module_aligned)

    def test_estimate_uncertainty(self) -> None:

        fn_ref = xdem.examples.get_path("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)

        sig_h, corr_sig = dem_tba.estimate_uncertainty(dem_ref, random_state=42)

        assert isinstance(sig_h, gu.Raster)
        assert callable(corr_sig)
