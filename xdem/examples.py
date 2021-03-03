import asyncio
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio


async def async_load_svalbard():
    """Load the datasets asynchronously."""
    # The URLs for the files to load
    urls = {
        "ref": "zip+https://publicdatasets.data.npolar.no/kartdata/S0_Terrengmodell/Mosaikk/NP_S0_DTM20.zip!NP_S0_DTM20/S0_DTM20.tif",
        "tba": "zip+https://publicdatasets.data.npolar.no/kartdata/S0_Terrengmodell/Historisk/NP_S0_DTM20_199095_33.zip!NP_S0_DTM20_199095_33/S0_DTM20_199095_33.tif",
        "outlines": "http://public.data.npolar.no/cryoclim/CryoClim_GAO_SJ_1990.zip"
    }

    # The output filepaths to save the files at.
    output_paths = {
        "ref": "examples/Longyearbyen/data/DEM_2009_ref.tif",
        "tba": "examples/Longyearbyen/data/DEM_1995.tif",
        "outlines": "examples/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
    }
    # The bounding coordinates to crop the datasets.
    bounds = {
        "west": 502810,
        "east": 529450,
        "south": 8654330,
        "north": 8674030
    }

    async def crop_dem(input_path, output_path, bounds):
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

        outlines = gpd.read_file(input_path)
        for col in outlines:
            if outlines[col].dtype == "object":
                outlines[col] = outlines[col].astype(str)
        outlines.to_file(output_path)
        print(f"Saved {output_path}")

    await asyncio.gather(
        crop_dem(urls["ref"], output_paths["ref"], bounds=bounds),
        crop_dem(urls["tba"], output_paths["tba"], bounds=bounds),
        read_outlines(urls["outlines"], output_paths["outlines"])
    )


def load_longyearbyen_examples(exist_ok: bool = True):
    """
    Fetch the Longyearbyen example files.

    :param exist_ok: Do not download the files again if they already exist.
    """
    if exist_ok and os.path.isfile("examples/Longyearbyen/data/DEM_2009_ref.tif"):
        print("Datasets exist")
        return
    print("Downloading datasets from Longyearbyen")
    os.makedirs("examples/Longyearbyen/data/glacier_mask", exist_ok=True)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_load_svalbard())
