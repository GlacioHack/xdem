"""Utility functions to download and find example data."""
import asyncio
import os

import geopandas as gpd
import rasterio as rio

EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples/"))
# Absolute filepaths to the example files.
FILEPATHS = {
    "longyearbyen_ref_dem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen/data/DEM_2009_ref.tif"),
    "longyearbyen_tba_dem": os.path.join(EXAMPLES_DIRECTORY, "Longyearbyen/data/DEM_1990.tif"),
    "longyearbyen_glacier_outlines": os.path.join(
        EXAMPLES_DIRECTORY,
        "Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
    )
}

# The URLs for where to find the data.
URLS = {
    "longyearbyen_ref_dem": ("zip+https://publicdatasets.data.npolar.no/kartdata/S0_Terrengmodell/"
                             "Mosaikk/NP_S0_DTM20.zip!NP_S0_DTM20/S0_DTM20.tif"),
    "longyearbyen_tba_dem": ("zip+https://publicdatasets.data.npolar.no/kartdata/S0_Terrengmodell/"
                             "Historisk/NP_S0_DTM20_199095_33.zip!NP_S0_DTM20_199095_33/S0_DTM20_199095_33.tif"),
    "longyearbyen_glacier_outlines": "http://public.data.npolar.no/cryoclim/CryoClim_GAO_SJ_1990.zip"
}


async def _async_load_svalbard():
    """Load the datasets asynchronously."""
    # The bounding coordinates to crop the datasets.
    bounds = {
        "west": 502810,
        "east": 529450,
        "south": 8654330,
        "north": 8674030
    }

    async def crop_dem(input_path, output_path, bounds):
        """Read the input path and crop it to the given bounds."""
        dem = rio.open(input_path)

        upper, left = dem.index(bounds["west"], bounds["north"])
        lower, right = dem.index(bounds["east"], bounds["south"])
        window = rio.windows.Window.from_slices((upper, lower), (left, right))

        data = dem.read(1, window=window)
        meta = dem.meta.copy()
        meta.update({
            "transform": dem.window_transform(window),
            "height": window.height,
            "width": window.width
        })
        with rio.open(output_path, "w", **meta) as raster:
            raster.write(data, 1)
        print(f"Saved {output_path}")

    async def read_outlines(input_path, output_path):
        """Read outlines from a path and save them."""
        outlines = gpd.read_file(input_path)
        for col in outlines:
            if outlines[col].dtype == "object":
                outlines[col] = outlines[col].astype(str)
        outlines.to_file(output_path)
        print(f"Saved {output_path}")

    await asyncio.gather(
        crop_dem(URLS["longyearbyen_ref_dem"], FILEPATHS["longyearbyen_ref_dem"], bounds=bounds),
        crop_dem(URLS["longyearbyen_tba_dem"], FILEPATHS["longyearbyen_tba_dem"], bounds=bounds),
        read_outlines(URLS["longyearbyen_glacier_outlines"], FILEPATHS["longyearbyen_glacier_outlines"])
    )


def download_longyearbyen_examples(overwrite: bool = False):
    """
    Fetch the Longyearbyen example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(FILEPATHS.values()))):
        print("Datasets exist")
        return
    print("Downloading datasets from Longyearbyen")
    os.makedirs(os.path.dirname(FILEPATHS["longyearbyen_glacier_outlines"]), exist_ok=True)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_async_load_svalbard())
