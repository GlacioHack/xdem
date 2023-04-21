---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
(vertical-ref)=

# Vertical referencing

xDEM supports the use of vertical coordinate reference systems (CRSs) and vertical transformations for DEMs
by conveniently wrapping PROJ pipelines through [Pyproj](https://pyproj4.github.io/pyproj/stable/) in the {class}`~xdem.DEM` class.

## What is a vertical CRS?

A vertical CRS is a 1D, often gravity-related, coordinate reference system used for geospatial elevation data, used to expand a [2D CRS](https://en.wikipedia.org/wiki/Spatial_reference_system) to a 3D CRS.

There are two types of 3D CRSs:
- Ellipsoidal heights, that are simply 2D CRS promoted to 3D by adding an elevation axis for transformation,
- Geoid heights, that are a compound of a 2D CRS and a vertical CRS (2D + 1D).

In xDEM, we merge these into a single vertical CRS attribute {class}`DEM.vcrs<xdem.DEM.vcrs>` that takes two type of values:
- the string `"Ellipsoid"` for an ellipsoidal CRS promoted to 3D (e.g., the WGS84 ellipsoid), or
- a {class}`~pyproj.crs.VerticalCRS` for a CRS based on geoid heights or grids (e.g., the EGM96 geoid).

```{caution}
As yet, [Rasterio](https://rasterio.readthedocs.io/en/stable/) does not support vertical transformations during CRS reprojection (even if the CRS provided contains a vertical axis).
We therefore advise to perform horizontal transformation and vertical transformation independently using {func}`DEM.reproject<xdem.DEM.reproject>` and {func}`DEM.to_vcrs<xdem.DEM.to_vcrs>`, respectively.
```

## User-input consistent and convenient

The parsing, setting and transformation of vertical CRSs revolves around three methods:
- The **instantiation** of {class}`~xdem.DEM` that can optionally be passed a `vcrs` argument (further described below in {ref}`vref-setting`),
- The **setting** method {class}`~xdem.DEM.set_vcrs`,
- The **transformation** method {class}`~xdem.DEM.to_vcrs`.

For these three functions, the `vcrs` argument accepts a **wide variety of user inputs**:
- Simple strings for the three most common references: `"Ellipsoid"`, `"EGM08"` or `"EGM96"`,
- Any grid name supported by PROJ available at [https://cdn.proj.org/](https://cdn.proj.org/) **with automatic download** (e.g., `"us_noaa_geoid06_ak.tif"`),
- Any EPSG code as {class}`int` (e.g., `5773`),
- Any {class}`~pyproj.CRS` that possesses a vertical axis (if more than 1D, only the vertical axis is kept but a warning is raised).

```{code-cell} ipython3
import xdem

# Open DEM
filename_dem = xdem.examples.get_path("longyearbyen_ref_dem")
dem = xdem.DEM(filename_dem)
```

(vref-setting)=
## Automated setting of the vertical CRS

During instantiation of a {class}`~xdem.DEM`, the vertical CRS {attr}`~xdem.DEM.vcrs` is tentatively set with the following priority order:
1. From the user-input `vcrs` argument of {class}`~xdem.DEM` if passed (e.g., `vcrs="EGM96"`),
2. From the {attr}`~xdem.DEM.crs` if 3-dimensional (e.g., [WGS84 with EPSG code 4979](https://epsg.io/4979) corresponding to the 3D version of [4326](https://epsg.io/4326)),
3. From the {attr}`~xdem.DEM.product` name of the DEM, if parsed from the filename by {class}`geoutils.SatelliteImage` (e.g., COPDEM uses EGM08, GDEM uses EGM96 and TanDEM-X DEM the ellipsoid).

If any of these conflict, a warning is raised to the user.

If, after instantiation, the {attr}`~xdem.DEM.vcrs` is not yet defined or wrongly so, it can be set manually using {func}`~xdem.DEM.set_vcrs`.

## Transforming

To transform a {class}`~xdem.DEM` to a different vertical CRS, {func}`~xdem.DEM.to_vcrs` is used.
