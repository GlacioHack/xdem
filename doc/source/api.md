(api)=

# API reference


This page provides a summary of xDEM’s API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

```{eval-rst}
.. currentmodule:: xdem
```

```{eval-rst}
.. minigallery:: xdem.DEM
      :add-heading:
```

## DEM

```{important}
A {class}`~xdem.DEM` inherits all raster methods and attributes from the {class}`~geoutils.Raster` object of GeoUtils.
Below, we only repeat some core attributes and methods of GeoUtils, see
[the Raster API in GeoUtils](https://geoutils.readthedocs.io/en/latest/api.html#raster) for the full list.
```

### Opening or saving

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM
    DEM.save
```

### Plotting or summarize info

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.info
    DEM.plot
```

### Create from an array

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.from_array
```

(api-dem-attrs)=

### Unique attributes

#### Inherited from {class}`~geoutils.Raster`

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.data
    DEM.crs
    DEM.transform
    DEM.nodata
    DEM.area_or_point
```

#### Specific to {class}`~xdem.DEM`

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.vcrs
```

### Other attributes

#### Inherited from {class}`~geoutils.Raster`

See the full list in [the Raster API of GeoUtils](https://geoutils.readthedocs.io/en/latest/api.html#raster).

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.res
    DEM.bounds
    DEM.width
    DEM.height
    DEM.shape
```

### Georeferencing

#### Inherited from {class}`~geoutils.Raster`

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.set_nodata
    DEM.set_area_or_point
    DEM.info
    DEM.reproject
    DEM.crop
```

#### Vertical referencing for {class}`~xdem.DEM`

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.set_vcrs
    DEM.to_vcrs
```

### Raster-vector interface

```{note}
See the full list of vector methods in [GeoUtils' documentation](https://geoutils.readthedocs.io/en/latest/api.html#vector).
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.polygonize
    DEM.proximity
    DEM.to_pointcloud
    DEM.interp_points
```

### Terrain attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.slope
    DEM.aspect
    DEM.hillshade
    DEM.curvature
    DEM.profile_curvature
    DEM.planform_curvature
    DEM.maximum_curvature
    DEM.topographic_position_index
    DEM.terrain_ruggedness_index
    DEM.roughness
    DEM.rugosity
    DEM.fractal_roughness
```

Or to get multiple related terrain attributes at once (for performance):

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.get_terrain_attribute
```

### Coregistration and bias corrections

```{tip}
To build and pass your coregistration pipeline to {func}`~xdem.DEM.coregister_3d`, see the API of {ref}`api-geo-handle`.
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.coregister_3d
```

### Uncertainty analysis

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.estimate_uncertainty
```

(api-geo-handle)=

## Coreg

**Overview of co-registration class structure**:

```{eval-rst}
.. inheritance-diagram:: xdem.coreg.base.Coreg xdem.coreg.affine xdem.coreg.biascorr
        :top-classes: xdem.coreg.Coreg
```

### Coregistration, pipeline and blockwise

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.Coreg
    coreg.CoregPipeline
    coreg.BlockwiseCoreg
```

#### Fitting and applying transforms

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.Coreg.fit_and_apply
    coreg.Coreg.fit
    coreg.Coreg.apply
```

#### Extracting metadata

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.Coreg.info
    coreg.Coreg.meta
```

### Affine coregistration

#### Parent object (to define custom methods)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.AffineCoreg
```

#### Coregistration methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.VerticalShift
    coreg.NuthKaab
    coreg.DhMinimize
    coreg.LZD
    coreg.ICP
    coreg.CPD
```

#### Manipulating affine transforms

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.apply_matrix
    coreg.invert_matrix
    coreg.matrix_from_translations_rotations
    coreg.translations_rotations_from_matrix
```

### Bias-correction

#### Parent object (to define custom methods)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.BiasCorr
```

#### Bias-correction methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    coreg.Deramp
    coreg.DirectionalBias
    coreg.TerrainBias
```

## Uncertainty analysis

```{important}
Several uncertainty functionalities of xDEM are being implemented directly in SciKit-GStat for spatial statistics
(e.g., fitting a sum of variogram models, pairwise subsampling for grid data). This will allow to simplify several
function inputs and outputs, by relying on a single {func}`~skgstat.Variogram` object.

This will trigger API changes in future package versions.
```

### Core routines for heteroscedasticity, spatial correlations, error propagation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.infer_heteroscedasticity_from_stable
    spatialstats.infer_spatial_correlation_from_stable
    spatialstats.spatial_error_propagation
```

### Sub-routines for heteroscedasticity

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.nd_binning
    spatialstats.interp_nd_binning
    spatialstats.two_step_standardization
```

### Sub-routines for spatial correlations

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.sample_empirical_variogram
    spatialstats.fit_sum_model_variogram
    spatialstats.correlation_from_variogram
```

### Sub-routines for error propagation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.number_effective_samples
```

### Empirical validation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.patches_method
```

### Plotting for uncertainty analysis

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.plot_variogram
    spatialstats.plot_1d_binning
    spatialstats.plot_2d_binning
```

## Stand-alone functions (moved)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    spatialstats.nmad
```


## Development classes (removal or re-factoring)

```{caution}
The {class}`xdem.dDEM` and {class}`xdem.DEMCollection` classes will be removed or re-factored in the near future.
```

### dDEM

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    dDEM
```

### DEMCollection

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEMCollection
```
