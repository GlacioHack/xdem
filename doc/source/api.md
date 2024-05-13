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

### Opening or saving a DEM

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM
    DEM.info
    DEM.save
```

### Plotting a DEM

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM
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

## dDEM

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    dDEM
```

## DEMCollection

## dDEM

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEMCollection
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

    xdem.coreg.Coreg
    xdem.coreg.CoregPipeline
    xdem.coreg.BlockwiseCoreg
```

#### Fitting and applying transforms

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.Coreg.fit
    xdem.coreg.Coreg.apply
```

#### Other functionalities

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.Coreg.residuals
```

### Affine coregistration

#### Parent object (to define custom methods)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.AffineCoreg
```

#### Coregistration methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.VerticalShift
    xdem.coreg.NuthKaab
    xdem.coreg.ICP
    xdem.coreg.Tilt
```

#### Manipulating affine transforms

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.AffineCoreg.from_matrix
    xdem.coreg.AffineCoreg.to_matrix
    xdem.coreg.apply_matrix
    xdem.coreg.invert_matrix
```

### Bias-correction

#### Parent object (to define custom methods)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.BiasCorr
```

#### Bias-correction methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.Deramp
    xdem.coreg.DirectionalBias
    xdem.coreg.TerrainBias
```

## Terrain attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.terrain
```

## Volume integration methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.volume
```

## Fitting methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.fit
```

## Filtering methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.filters
```

## Spatial statistics methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.spatialstats
```

## Stand-alone functions (moved)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.spatialstats.nmad
```
