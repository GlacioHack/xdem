import os
import warnings
from pathlib import Path

import geoutils as gu
import numpy as np
import pytest
from geoutils._typing import NDArrayNum
from geoutils.raster import Raster
from geoutils.raster.distributed_computing import AbstractCluster, ClusterGenerator

import xdem
from xdem import DEM
from xdem.multiproc import get_terrain_attribute_multiproc

PLOT = True


def _get_multiproc_attr(
    dem_path: str,
    attribute: str | list[str],
    tile_size: int,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: str = "Horn",
    tri_method: str = "Riley",
    fill_method: str = "none",
    edge_method: str = "none",
    window_size: int = 3,
    cluster: AbstractCluster | None = None,
) -> list[NDArrayNum]:
    get_terrain_attribute_multiproc(
        dem_path,
        attribute,
        tile_size,
        None,
        resolution,
        degrees,
        hillshade_altitude,
        hillshade_azimuth,
        hillshade_z_factor,
        slope_method,
        tri_method,
        fill_method,
        edge_method,
        window_size,
        cluster,
    )
    dem_path = Path(dem_path)  # type: ignore

    attr_list = []
    if isinstance(attribute, str):
        attribute = [attribute]

    for attr in attribute:
        filename = dem_path.stem + "_" + attr + ".tif"
        outfile = dem_path.parent.joinpath(filename)
        attr_list.append(Raster(outfile, load_data=True))
        os.remove(outfile)

    return attr_list


class TestMultiprocTerrainAttribute:
    filepath = xdem.examples.get_path("longyearbyen_ref_dem")
    dem = DEM(filepath)
    cluster = ClusterGenerator("multi", nb_workers=4)

    @pytest.mark.parametrize(
        "attribute",
        [
            "slope_Horn",
            "aspect_Horn",
            "hillshade_Horn",
            "slope_Zevenberg",
            "aspect_Zevenberg",
            "hillshade_Zevenberg",
            "tri_Riley",
            "tri_Wilson",
            "tpi",
            "roughness",
        ],
    )  # type: ignore
    @pytest.mark.parametrize("tile_size", [200])  # type: ignore
    def test_attribute_functions_against_gdaldem(self, attribute: str, tile_size: int, get_test_data_path) -> None:
        """
        Test that all attribute functions give the same results as those of GDALDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        functions = {
            "slope_Horn": lambda file: _get_multiproc_attr(
                file, "slope", tile_size, resolution=self.dem.res, degrees=True, cluster=self.cluster
            ),
            "aspect_Horn": lambda file: _get_multiproc_attr(
                file, "aspect", tile_size, degrees=True, cluster=self.cluster
            ),
            "hillshade_Horn": lambda file: _get_multiproc_attr(
                file, "hillshade", tile_size, resolution=self.dem.res, cluster=self.cluster
            ),
            "slope_Zevenberg": lambda file: _get_multiproc_attr(
                file,
                "slope",
                tile_size,
                resolution=self.dem.res,
                slope_method="ZevenbergThorne",
                degrees=True,
                cluster=self.cluster,
            ),
            "aspect_Zevenberg": lambda file: _get_multiproc_attr(
                file, "aspect", tile_size, slope_method="ZevenbergThorne", degrees=True, cluster=self.cluster
            ),
            "hillshade_Zevenberg": lambda file: _get_multiproc_attr(
                file,
                "hillshade",
                tile_size,
                resolution=self.dem.res,
                slope_method="ZevenbergThorne",
                cluster=self.cluster,
            ),
            "tri_Riley": lambda file: _get_multiproc_attr(
                file, "terrain_ruggedness_index", tile_size, tri_method="Riley", cluster=self.cluster
            ),
            "tri_Wilson": lambda file: _get_multiproc_attr(
                file, "terrain_ruggedness_index", tile_size, tri_method="Wilson", cluster=self.cluster
            ),
            "tpi": lambda file: _get_multiproc_attr(
                file, "topographic_position_index", tile_size, cluster=self.cluster
            ),
            "roughness": lambda file: _get_multiproc_attr(file, "roughness", tile_size, cluster=self.cluster),
        }

        # Derive the attribute using both GDAL and xdem
        attr_xdem = functions[attribute](self.filepath)[0].data.squeeze()
        attr_gdal = gu.Raster(get_test_data_path(os.path.join("gdal", f"{attribute}.tif"))).data

        # For hillshade, we round into an integer to match GDAL's output
        if attribute in ["hillshade_Horn", "hillshade_Zevenberg"]:
            with warnings.catch_warnings():
                # Normal that a warning would be raised here, so we catch it
                warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
                attr_xdem = attr_xdem.astype("int").astype("float32")

        # We compute the difference and keep only valid values
        diff = (attr_xdem - attr_gdal).filled(np.nan)
        diff_valid = diff[np.isfinite(diff)]

        try:
            # Difference between xdem and GDAL attribute
            # Mean of attribute values to get an order of magnitude of the attribute unit
            magn = np.nanmean(np.abs(attr_xdem))

            # Check that the attributes are similar within a tolerance of a thousandth of the magnitude
            # For instance, slopes have an average magnitude of around 30 deg, so the tolerance is 0.030 deg
            if attribute in ["hillshade_Horn", "hillshade_Zevenberg"]:
                # For hillshade, check 0 or 1 difference due to integer rounding
                assert np.all(np.logical_or(diff_valid == 0.0, np.abs(diff_valid) == 1.0))

            elif attribute in ["aspect_Horn", "aspect_Zevenberg"]:
                # For aspect, check the tolerance within a 360 degree modulo due to the circularity of the variable
                diff_valid = np.mod(np.abs(diff_valid), 360)
                assert np.all(np.minimum(diff_valid, np.abs(360 - diff_valid)) < 10 ** (-3) * magn)
            else:
                # All attributes other than hillshade and aspect are non-circular floats, so we check within a tolerance
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and GDAL attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(121)
                plt.imshow(attr_gdal.squeeze())
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(attr_xdem.squeeze())
                plt.colorbar()
                plt.show()

            raise exception

    @pytest.mark.parametrize(
        "attribute",
        ["slope_Horn", "aspect_Horn", "hillshade_Horn", "curvature", "profile_curvature", "planform_curvature"],
    )  # type: ignore
    @pytest.mark.parametrize("tile_size", [200])  # type: ignore
    def test_attribute_functions_against_richdem(self, attribute: str, tile_size: int, get_test_data_path) -> None:
        """
        Test that all attribute functions give the same results as those of RichDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        # Functions for xdem-implemented methods
        functions_xdem = {
            "slope_Horn": lambda file: _get_multiproc_attr(
                file, "slope", tile_size, resolution=self.dem.res, degrees=True, cluster=self.cluster
            ),
            "aspect_Horn": lambda file: _get_multiproc_attr(
                file, "aspect", tile_size, degrees=True, cluster=self.cluster
            ),
            "hillshade_Horn": lambda file: _get_multiproc_attr(
                file, "hillshade", tile_size, resolution=self.dem.res, cluster=self.cluster
            ),
            "curvature": lambda file: _get_multiproc_attr(
                file, "curvature", tile_size, resolution=self.dem.res, cluster=self.cluster
            ),
            "profile_curvature": lambda file: _get_multiproc_attr(
                file, "profile_curvature", tile_size, resolution=self.dem.res, cluster=self.cluster
            ),
            "planform_curvature": lambda file: _get_multiproc_attr(
                file, "planform_curvature", tile_size, resolution=self.dem.res, cluster=self.cluster
            ),
        }

        # Derive the attribute using both RichDEM and xdem
        attr_xdem = gu.raster.get_array_and_mask(functions_xdem[attribute](self.filepath)[0])[0].squeeze()
        attr_richdem_rst = gu.Raster(get_test_data_path(os.path.join("richdem", f"{attribute}.tif")), load_data=True)
        attr_richdem = gu.raster.get_array_and_mask(attr_richdem_rst)[0].squeeze()

        # We compute the difference and keep only valid values
        diff = attr_xdem - attr_richdem
        diff_valid = diff[np.isfinite(diff)]

        try:
            # Difference between xdem and RichDEM attribute
            # Mean of attribute values to get an order of magnitude of the attribute unit
            magn = np.nanmean(np.abs(attr_xdem))

            # Check that the attributes are similar within a tolerance of a thousandth of the magnitude
            # For instance, slopes have an average magnitude of around 30 deg, so the tolerance is 0.030 deg
            if attribute in ["aspect_Horn"]:
                # For aspect, check the tolerance within a 360 degree modulo due to the circularity of the variable
                diff_valid = np.mod(np.abs(diff_valid), 360)
                assert np.all(np.minimum(diff_valid, np.abs(360 - diff_valid)) < 10 ** (-3) * magn)

            else:
                # All attributes other than aspect are non-circular floats, so we check within a tolerance
                # Here hillshade is not rounded as integer by our calculation, so no need to differentiate as with GDAL
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and RichDEM attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(221)
                plt.imshow(attr_richdem)
                plt.colorbar()
                plt.subplot(222)
                plt.imshow(attr_xdem)
                plt.colorbar()
                plt.subplot(223)
                plt.imshow(diff)
                plt.colorbar()
                plt.show()

            raise exception

    @pytest.mark.parametrize("tile_size", [200])  # type: ignore
    def test_multiple_attributes(self, tile_size: int) -> None:
        # test with dem and not filepath this time
        slope, aspect = _get_multiproc_attr(self.dem, ["slope", "aspect"], tile_size, cluster=self.cluster)

        assert slope != aspect

        assert isinstance(slope, type(aspect))
        assert all(isinstance(r, gu.Raster) for r in (aspect, slope, self.dem))

        assert slope.transform == self.dem.transform == aspect.transform
        assert slope.crs == self.dem.crs == aspect.crs
