"""Test module for EPC class."""

import os
import tempfile
import warnings

import geopandas as gpd
import geoutils as gu
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from pyproj.transformer import Transformer
from shapely import Polygon

import xdem
from xdem import DEM, EPC


class TestEPC:

    # 1/ Elevation point cloud with 3D points
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(100, 3)) + rng.normal(0, 0.15, size=(100, 3))
    gdf1 = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1], z=arr_points[:, 2]), crs=4326
    )

    # 2/ Elevation point cloud with 2D points and data column
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(100, 3)) + rng.normal(0, 0.15, size=(100, 3))
    gdf2 = gpd.GeoDataFrame(
        data={"Z": arr_points[:, 2]}, geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1]), crs=4326
    )

    # 3/ LAS file
    fn_las = gu.examples.get_path_test("coromandel_lidar")

    # 4/ Non-point vector (for error raising)
    poly = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    gdf3 = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

    def test_init(self) -> None:
        """Test that inputs work properly in EPC class init."""

        # 1/ For a single column point cloud with 3D points
        epc = EPC(self.gdf1)

        # Assert that both the dataframe and data column name are equal
        assert epc.data_column is None
        assert_geodataframe_equal(epc.ds, self.gdf1)

        # 2/ For a single column point cloud with 2D points and a data column
        epc2 = EPC(self.gdf2, data_column="Z")

        # Assert that both the dataframe and data column name are equal
        assert epc2.data_column == "Z"
        assert_geodataframe_equal(epc2.ds, self.gdf2)

    def test_init__las(self) -> None:
        """Test that LAS files work properly in EPC class init."""

        # From filename
        epc = EPC(self.fn_las)
        assert isinstance(epc, EPC)

        # From EPC
        epc2 = EPC(epc)
        assert isinstance(epc2, EPC)

        # From PointCloud
        r = gu.PointCloud(self.fn_las)
        epc3 = EPC(r)
        assert isinstance(epc3, EPC)

        assert np.logical_and.reduce(
            (
                np.array_equal(epc.data, epc2.data, equal_nan=True),
                np.array_equal(epc2.data, epc3.data, equal_nan=True),
            )
        )

    def test_init__vcrs(self) -> None:
        """Test that vcrs is set properly during instantiation."""

        # Tests 1: instantiation with a file that has a 2D CRS

        # First, check a EPC that does not have any vertical CRS set
        epc = EPC(self.gdf1)
        assert epc.vcrs is None

        # Setting a vertical CRS during instantiation should work here
        epc = EPC(self.gdf1, vcrs="EGM96")
        assert epc.vcrs_name == "EGM96 height"

        # Tests 2: instantiation with a file that has a 3D CRS
        # Create such a file
        epc = EPC(self.gdf1)
        epc_reproj = epc.reproject(crs=4979)

        # Save to temporary folder
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.tif")
        epc_reproj.save(temp_file)

        # Check opening a EPC with a 3D CRS sets the vcrs
        epc_3d = EPC(temp_file)
        assert epc_3d.vcrs == "Ellipsoid"

        # Check that a warning is raised when trying to override with user input
        with pytest.warns(
            UserWarning,
            match="The CRS in the point cloud metadata already has a vertical component, "
            "the user-input 'EGM08' will override it.",
        ):
            EPC(temp_file, vcrs="EGM08")

    def test_copy(self) -> None:
        """
        Test that the copy method works as expected for EPC. In particular
        when copying pc to pc2:
        - if pc.data is modified and pc copied, the updated data is copied
        - if pc is copied, pc.data changed, pc2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        epc = xdem.EPC(self.gdf1)
        epc.data += 5
        epc2 = epc.copy()

        # Objects should be different (not pointing to the same memory)
        assert epc is not epc2

        # Check the object is a EPC
        assert isinstance(epc2, xdem.EPC)

        # Check data array
        assert np.array_equal(epc.data, epc2.data, equal_nan=True)

        # Check that if pc.data is modified, it does not affect pc2.data
        epc.data += 5
        assert not np.array_equal(epc.data, epc2.data, equal_nan=True)

        # Check that the new_array argument indeed modifies the point cloud
        epc3 = epc.copy(new_array=epc2.data)

        assert np.array_equal(epc3.data, epc2.data)

    def test_set_vcrs(self) -> None:
        """Tests to set the vertical CRS of an EPC."""

        epc = EPC(self.gdf1)

        # -- Test 1: we check with names --

        # Check setting ellipsoid
        epc.set_vcrs(new_vcrs="Ellipsoid")
        assert epc.vcrs_name is not None
        assert "Ellipsoid (No vertical CRS)." in epc.vcrs_name
        assert epc.vcrs_grid is None

        # Check setting EGM96
        epc.set_vcrs(new_vcrs="EGM96")
        assert epc.vcrs_name == "EGM96 height"
        assert epc.vcrs_grid == "us_nga_egm96_15.tif"

        # Check setting EGM08
        epc.set_vcrs(new_vcrs="EGM08")
        assert epc.vcrs_name == "EGM2008 height"
        assert epc.vcrs_grid == "us_nga_egm08_25.tif"

        # -- Test 2: we check with grids --
        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid not found in *")

        epc.set_vcrs(new_vcrs="us_nga_egm96_15.tif")
        assert epc.vcrs_name == "unknown using geoidgrids=us_nga_egm96_15.tif"
        assert epc.vcrs_grid == "us_nga_egm96_15.tif"

        epc.set_vcrs(new_vcrs="us_nga_egm08_25.tif")
        assert epc.vcrs_name == "unknown using geoidgrids=us_nga_egm08_25.tif"
        assert epc.vcrs_grid == "us_nga_egm08_25.tif"

        # Check that other existing grids are well detected in the pyproj.datadir
        epc.set_vcrs(new_vcrs="is_lmi_Icegeoid_ISN93.tif")

        # Check that non-existing grids raise errors
        with pytest.warns(UserWarning, match="Grid*"):
            with pytest.raises(
                ValueError,
                match="The provided grid 'thebestgrid.tif' does not exist at https://cdn.proj.org/. "
                "Provide an existing grid.",
            ):
                epc.set_vcrs(new_vcrs="thebestgrid.tif")

    def test_to_vcrs(self) -> None:
        """Tests the conversion of vertical CRS for an EPC."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = xdem.DEM(fn_dem)
        epc = EPC(dem.to_pointcloud(subsample=500))

        # Reproject in WGS84 2D
        epc = epc.reproject(crs=4326)
        epc_before_trans = epc.copy()

        # Set ellipsoid as vertical reference
        epc.set_vcrs(new_vcrs="Ellipsoid")
        ccrs_init = epc.ccrs
        median_before = np.nanmean(epc)
        # Transform to EGM96 geoid not inplace (default)
        trans_epc = epc.to_vcrs(vcrs="EGM96")

        # The output should be a EPC, input shouldn't have changed
        assert isinstance(trans_epc, EPC)
        assert epc.pointcloud_equal(epc_before_trans)

        # Compare to inplace
        should_be_none = epc.to_vcrs(vcrs="EGM96", inplace=True)
        assert should_be_none is None
        assert epc.pointcloud_equal(trans_epc)

        # Save the median of after
        median_after = np.nanmean(trans_epc)

        # About 32 meters of difference in Svalbard between EGM96 geoid and ellipsoid
        assert median_after - median_before == pytest.approx(-32, rel=0.1)

        # Check that the results are consistent with the operation done independently
        ccrs_dest = xdem.vcrs._build_ccrs_from_crs_and_vcrs(epc.crs, xdem.vcrs._vcrs_from_user_input("EGM96"))
        transformer = Transformer.from_crs(crs_from=ccrs_init, crs_to=ccrs_dest, always_xy=True)

        xx, yy = epc.geometry.x.values, epc.geometry.y.values
        x = xx[5]
        y = yy[5]
        z = epc_before_trans.data[5]
        z_out = transformer.transform(xx=x, yy=y, zz=z)[2]

        assert z_out == pytest.approx(epc.data[5])

    def test_to_vcrs__equal_warning(self) -> None:
        """Test that EPC.to_vcrs() does not transform if both 3D CRS are equal."""

        fn_epc = gu.examples.get_path_test("coromandel_lidar")
        epc = EPC(fn_epc)

        # With both inputs as names
        epc.set_vcrs("EGM96")
        with pytest.warns(
            UserWarning, match="Source and destination vertical CRS are the same, " "skipping vertical transformation."
        ):
            epc.to_vcrs("EGM96")

        # With one input as name, the other as CRS
        epc.set_vcrs("Ellipsoid")
        with pytest.warns(
            UserWarning, match="Source and destination vertical CRS are the same, " "skipping vertical transformation."
        ):
            epc.to_vcrs(CRS("EPSG:4979"))

    @staticmethod
    @pytest.mark.parametrize(  # type: ignore
        "coreg_method, expected_pipeline_types",
        [
            pytest.param(
                xdem.coreg.NuthKaab(initial_shift=(10, 5)) + xdem.coreg.VerticalShift(),
                [xdem.coreg.AffineCoreg, xdem.coreg.VerticalShift],
                id="Pipeline: NuthKaab + VerticalShift with initial shift",
            ),
            pytest.param(
                xdem.coreg.NuthKaab(initial_shift=(10, 5)),
                [xdem.coreg.AffineCoreg],
                id="Pipeline: NuthKaab with initial shift",
            ),
            pytest.param(
                xdem.coreg.NuthKaab() + xdem.coreg.VerticalShift(),
                [xdem.coreg.AffineCoreg, xdem.coreg.VerticalShift],
                id="Pipeline: NuthKaab + VerticalShift without initial shift",
            ),
            pytest.param(
                xdem.coreg.DhMinimize(),
                [xdem.coreg.AffineCoreg],
                id="Simple affine method: DhMinimize without initial shift",
            ),
        ],
    )
    def test_coregister_3d(coreg_method, expected_pipeline_types) -> None:  # type: ignore
        """
        Test coregister_3d works for an EPC.
        """
        fn_ref = xdem.examples.get_path("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        epc_tba = dem_tba.to_pointcloud(subsample=5000)

        # Run coregistration
        dem_aligned = epc_tba.coregister_3d(dem_ref, coreg_method=coreg_method, random_state=42)

        assert isinstance(dem_aligned, xdem.EPC)
        assert isinstance(coreg_method, xdem.coreg.Coreg)

        # Test pipeline
        pipeline = coreg_method.pipeline if hasattr(coreg_method, "pipeline") else [coreg_method]
        for i, expected_type in enumerate(expected_pipeline_types):
            assert isinstance(pipeline[i], expected_type)


    def test_coregister_3d__raises(self) -> None:  # type: ignore
        """
        Test coregister_3d functionality raises propers errors for an EPC.
        """
        fn_ref = xdem.examples.get_path("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        epc_tba = dem_tba.to_pointcloud(subsample=5000)

        coreg_method = xdem.coreg.Deramp()

        # Run coregistration
        with pytest.raises(ValueError, match=".* has no implemented _apply_pts."):
            dem_aligned = epc_tba.coregister_3d(dem_ref, coreg_method=coreg_method, random_state=42)
