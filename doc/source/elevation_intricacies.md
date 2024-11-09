(elevation-intricacies)=
# Georeferencing intricacies

Georeferenced elevation data comes in different types and relies on different attributes than typical georeferenced
data, which **can make quantitative analysis more delicate**.

Below, we summarize these aspects to help grasp a general understanding of the intricacies of elevation data.

## Types of elevation data

There are a number of types of elevation data:

1. **Digital elevation models (DEMs)** are a type of raster, i.e. defined on a regular grid with each grid cell representing an elevation. Additionally, DEMs are rasters than are almost always chosen to be single-band, and floating-type to accurately represent elevation in meters,
2. **Elevation point clouds** are simple point clouds with XYZ coordinates, often with a list of attributes attached to each point,
3. **Contour or breaklines** are 2D or 3D vector lines representing elevation. **They are usually not used for analysis**, instead for visualisation,
4. **Elevation triangle irregular networks (TINs)** are a triangle mesh representing the continuous elevation surface. **They are usually not used for analysis**, instead in-memory for visualization or for conversion from an elevation point cloud to a DEM.

```{note}
xDEM supports the two elevation data types primarily used for quantitative analysis: the {class}`~xdem.DEM` and the elevation point cloud (currently as a {class}`~geopandas.GeoDataFrame` for some operations, more soon!).

See the **{ref}`elevation-objects` features pages** for more details.
```

```{eval-rst}
.. plot:: code/intricacies_datatypes.py
    :width: 90%
```


Additionally, there are a critical differences for elevation point clouds depending on point density:
- **Sparse elevation point clouds** (e.g., altimetry) are generally stored as small vector-type datasets (e.g., SHP). Due to their sparsity, for subsequent analysis, they are rarely gridded into a DEM, and instead compared with DEMs at the point cloud coordinates by interpolation of the DEM,
- **Dense elevation point clouds** (e.g., lidar) are large datasets generally stored in specific formats (LAS). Due to their high density, they are often gridded into DEMs by triangular interpolation of the point cloud.

```{note}
For point–DEM interfacing, xDEM inherit functionalities from [GeoUtils's point–raster interfacing](https://geoutils.readthedocs.io/en/stable/raster_vector_point.html#rasterpoint-operations).
See for instance {class}`xdem.DEM.interp_points`.
```

## A third georeferenced dimension

Elevation data is unique among georeferenced data, in the sense that it **adds a third vertical dimension that also requires georeferencing**.

For this purpose, elevation data is related to a vertical [coordinate reference system (CRS)](https://en.wikipedia.org/wiki/Spatial_reference_system). A vertical CRS is a **1D, often gravity-related, coordinate reference system of surface elevation** (or height), used to expand a 2D horizontal CRS to a 3D CRS.

There are two types of models of surface elevation:
- **Ellipsoids** model the surface of the Earth as a three-dimensional shape created from a two-dimensional ellipse, which are already used by 2D CRS,
- **Geoids** model the surface of the Earth based on its gravity field (approximately mean sea-level). Since Earth's mass is not uniform, and the direction of gravity slightly changes, the shape of a geoid is irregular,

which are directly associated with two types of 3D CRSs:
- **Ellipsoidal heights** CRSs, that are simply 2D CRS promoted to 3D by explicitly adding an elevation axis to the ellipsoid used by the 2D CRS,
- **Geoid heights** CRSs, that are a compound of a 2D CRS and a vertical CRS (2D + 1D), where the vertical CRS of the geoid is added relative to the ellipsoid.


Problematically, until the early 2020s, **most elevation data was distributed without a 3D CRS in its metadata**. The vertical reference was generally provided separately, in a user guide or website of the data provider.
Therefore, it is important to either define your vertical CRSs manually before any analysis, or double-check that all your datasets are on the same vertical reference.

```{note}
For this reason, xDEM includes {ref}`tools to easily set a vertical CRS<vref-setting>`. See for instance {class}`xdem.DEM.set_vcrs`.
```

## The interpretation of pixel value for DEMs

Among the elevation data types listed above, DEMs are the only gridded dataset. While gridded datasets have become
ubiquitous for quantitative anaysis, they also suffer from a problem of pixel interpretation.

Pixel interpretation describes how a grid cell value should be interpreted, and has two definitions:
- **“Area” (the most common)** where the value represents a sampling over the region of the pixel (and typically refers to the upper-left corner coordinate), or
- **“Point”** where the value relates to a point sample (and typically refers to the center of the pixel).

**This interpretation difference disproportionally affects DEMs** as they are the primary type of gridded data associated with the least-common "Point" interpretation, and often rely on auxiliary point data such as ground-control points (GCPs).

**In different software packages, gridded data are interpreted differently**, resulting in (undesirable) half-pixel shifts during analysis. Additionally, different storage formats have different standards for grid coordinate interpretation, also sometimes resulting in a half-pixel shift (e.g., GeoTIFF versus netCDF).

```{note}
To perform consistent pixel interpretation of DEMs, xDEM relies on [the raster pixel interpretation of GeoUtils, which mirrors GDAL's GCP behaviour](https://geoutils.readthedocs.io/en/stable/georeferencing.html#pixel-interpretation-only-for-rasters).

This means that, by default, pixel interpretation induces a half-pixel shift during DEM–point interfacing for a “Point” interpretation, but only raises a warning for DEM–DEM operations if interpretations differ.
This default behaviour can be modified at the package-level by using [GeoUtils’ configuration](https://geoutils.readthedocs.io/en/stable/config.html).

See {class}`xdem.DEM.set_area_or_point` to re-set the pixel interpretation of your DEM.
```

----------------

:::{admonition} References and more reading
:class: tip

For more information about **vertical referencing**, see [educational material from the National Geodetic Survey](https://geodesy.noaa.gov/datums/index.shtml) and [NOAA's VDatum tutorials](https://vdatum.noaa.gov/docs/datums.html).

For more information about **pixel interpretation**, see [GIS StackExchange discussions](https://gis.stackexchange.com/questions/122670/is-there-a-standard-for-the-coordinates-of-pixels-in-georeferenced-rasters) and [GeoTIFF standard from the Open Geospatial Consortium](https://docs.ogc.org/is/19-008r4/19-008r4.html#_requirements_class_gtrastertypegeokey).
:::
