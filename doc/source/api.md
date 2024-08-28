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
Below, we only repeat the core attributes and methods of GeoUtils, see
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

### Coregistration

```{tip}
To build and pass your coregistration pipeline to {func}`~xdem.DEM.coregister_3d`, see the API of {ref}`api-geo-handle`.
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.coregister_3d
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
.. inheritance-diagram:: xdem.coreg.base xdem.coreg.affine xdem.coreg.biascorr
        :top-classes: xdem.Coreg
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

### Affine coregistration methods


**Generic parent class:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.AffineCoreg
```

**Convenience classes for specific coregistrations:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.VerticalShift
    xdem.coreg.NuthKaab
    xdem.coreg.ICP
```

### Bias-correction (including non-affine coregistration) methods

**Generic parent class:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.coreg.BiasCorr
```

**Convenience classes for specific corrections:**

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
