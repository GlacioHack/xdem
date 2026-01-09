(release-notes)=
# Release notes

Below, the release notes for all minor versions and our roadmap to a first major version.

## 0.2.0

xDEM version 0.2 is the **second minor release**, notably marked by the merge in efforts with [demcompare](https://github.com/CNES/demcompare) (no longer maintained) to build a unified tool!
It is the result of months of work to consolidate several features, in particular support for **elevation point clouds** (EPCs). Parallel work on scalability with Dask and Multiprocessing
is ongoing, and should soon be released in a 0.3.

This new version adds:
- Elevation point cloud objects (for both vector and LAS files) with relevant methods (coregistration, vertical referencing) and that builds on the point cloud object of GeoUtils (to interface easily with rasters and vectors),
- Support for elevation point clouds in coregistration and bias-correction pipelines,
- New terrain attributes including new fits to estimate derivatives (used for slope, aspect, ...), more rigorous curvature definitions and texture shading for visualization,
- An experimental command-line interface for multi-step workflows such as DEM accuracy assessment,
- Statistics features common to rasters and point clouds based on GeoUtils (being expanded with binning and spatial statistics).

Computationally, we started introducing parallel out-of-memory functionalities (for now through Multiprocessing, soon available also through Dask) for:
- Blockwise coregistration,
- Terrain attributes (that also benefit from a general performance improvement, and the option of using SciPy as engine instead of Numba).

Internally, xDEM has now defined a governance, nearly doubled its number of tests, robustified its packaging and minimized its required dependencies.

A few changes might be required to adapt from previous versions (from deprecation warnings):

- The `curvature()` function is being deprecated in favor of other clearly defined curvatures,
- The `slope_method=` argument of `slope()` and `aspect()` is being deprecated in favor of a `surface_fit=` argument for all surface derivatives (including curvatures),
- The `spatialstats.nmad()` function is being deprecator in favor of `geoutils.stats.nmad()`,
- The `.save()` function is being deprecated in favor of `.to_file()`,
- Specify `DEM.interp_points(as_array=True)` to mirror the previous behaviour of returning a 1D array of interpolated values, otherwise now returns an elevation point cloud by default,
- The `Mask` class of GeoUtils is deprecated in favor of `Raster(is_mask=True)` to declare a boolean-type Raster, but should keep working until 0.3.

## 0.1.0

xDEM version 0.1 is the **first minor release** since the creation of the project in 2020. It is the result of years of work
to consolidate and re-structure features into a mature and stable API to minimize future breaking changes.

**All the core features drafted at the start of the project are now supported**, and there is a **clear roadmap
towards a first major release 1.0**. This minor release also adds many tests and improves significantly the documentation
from the early-development state of the package.

The re-structuring created some breaking changes, though minor.

See details below, including **a guide to help migrate code from early-development versions**.

### Features

xDEM now gathers the following core features:
- **Elevation data objects** core to quantatiative analysis, which are DEMs and elevation point clouds,
- **Vertical referencing** including automatic 3D CRS fetching,
- **Terrain analysis** for many attributes,
- **Coregistration** with the choice of several methods, including modular pipeline building,
- **Bias corrections** for any variable, also modular and supported by pipelines,
- **Uncertainty analysis** based on several robust methods.

Recent additions include in particular **point-raster support for coregistration**, and the **expansion of
`DEM` class methods** to cover all features of the package, with for instance `DEM.coregister_3d()` or `DEM.slope()`.

### Guides and other resources

xDEM integrates **background material on quantitative analysis of elevation data** to help users use the various methods
of the package. This material includes **several illustrated guide pages**, **a cheatsheet** on how to recognize and correct
typical elevation errors, and more.

### Future deprecations

We have added warnings throughout the documentation and API related to planned deprecations:
- **Gap-filling features specific to glacier-applications** will be moved to a separate package,
- **Uncertainty analysis tools related to variography** will change API to rely on SciKit-GStat variogram objects,
- The **dDEM** and **DEMCollection** classes will likely be refactored or removed.

Changes related to **gap-filling** and **uncertainty analysis** will have deprecation warnings, while the function
remain available during a few more releases.

(migrate-early)=
### Migrate from early versions

The following changes **might be required to solve breaking changes**, depending on your early-development version:
- Rename `.show()` to `.plot()` for all data objects,
- Rename `.dtypes` to `dtype` for `DEM` objects,
- Operations `.crop()`, `shift()` and `to_vcrs()` are not done in-place by default anymore, replace by `dem = dem.crop()` or `dem.crop(..., inplace=True)` to mirror the old default behaviour,
- Rename `.shift()` to `.translate()` for `DEM` objects,
- Several function arguments are renamed, in particular `dst_xxx` arguments of `.reproject()` are all renamed to `xxx` e.g. `dst_crs` to `crs`, as well as the arguments of `Coreg.fit()` renamed from `xxx_dem` to `xxx_elev` to be generic to any elevation data,
- All `BiasCorr1D`, `BiasCorr2D` and `BiasCorrND` classes are removed in favor of a single `BiasCorr` class that implicitly understands the number of dimensions from the length of input `bias_vars`,
- New user warnings are sometimes raised, in particular if some metadata is not properly defined such as `.nodata`. Those should give an indication as how to silence them.

Additionally, **some important yet non-breaking changes**:
- The sequential use of `Coreg.fit()` and `Coreg.apply()` to the same `tba_elev` is now discouraged and updated everywhere in the documentation, use `Coreg.fit_and_apply()` or `DEM.coregister_3d()` instead,
- The use of a separate module for terrain attributes such as `xdem.terrain.slope()` is now discouraged, use `DEM.slope()` instead.

## Roadmap to 1.0

Based on recent and ongoing progress, we envision the following roadmap.

**Releases of 0.2, 0.3, 0.4, etc**, for the following planned (ongoing) additions:
- The **addition of a command-line interface for features such as coregistration**, in the frame of the merging effort with [demcompare](https://github.com/CNES/demcompare),
- The **addition of an elevation point cloud `EPC` data object**, inherited from the ongoing `PointCloud` object of GeoUtils alongside many features at the interface of point and raster,
- The **addition of a Xarray accessor `dem`** mirroring the `DEM` object, to work natively with Xarray objects and add support on out-of-memory Dask operations for most of xDEM's features,
- The **addition of a GeoPandas accessor `epc`** mirroring the `EPC` object, to work natively with GeoPandas objects,
- The **re-structuration of uncertainty analysis features** to rely directly on SciKit-GStat's `Variogram` object.

**Release of 1.0** once all these additions are fully implemented, and after feedback from the community.
