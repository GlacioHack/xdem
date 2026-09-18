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
    DEM.to_file
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

### Statistics

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.get_stats
```

### Terrain attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.slope
    DEM.aspect
    DEM.hillshade
    DEM.profile_curvature
    DEM.tangential_curvature
    DEM.planform_curvature
    DEM.flowline_curvature
    DEM.max_curvature
    DEM.min_curvature
    DEM.topographic_position_index
    DEM.terrain_ruggedness_index
    DEM.roughness
    DEM.rugosity
    DEM.fractal_roughness
    DEM.texture_shading
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

### Error structures and public workflows

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    ErrorMagnitude
    ErrorComponent
    ErrorStructure
    uncertainty.PropagationResult
    uncertainty.estimate_error_structure
    uncertainty.propagate_uncertainty
    uncertainty.propagate_uncertainty_coreg
    uncertainty.number_effective_samples
    uncertainty.spatial_error_propagation
    uncertainty.patches_method
```

### Shared fitting and analytical calculations

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    fit.interp_binning
    fit.get_perbin_binning
    uncertainty.estimation.two_step_standardization
    uncertainty.analytical.neff_exact
    uncertainty.analytical.neff_hugonnet_approx
    uncertainty.analytical.neff_circular_approx_theoretical
    uncertainty.analytical.neff_circular_approx_numerical
```

Generic grouping, cosampling, pair sampling, variogram fitting and their plots are provided by GeoUtils.
The entire `xdem.spatialstats` module and `DEM.estimate_uncertainty()` are deprecated; see
the [uncertainty migration guide](uncertainty_migration.md) for replacements and changes to inputs and outputs.

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
