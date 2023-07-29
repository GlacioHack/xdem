(api)=

# API reference


This page provides a summary of xDEM’s API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

```{eval-rst}
.. currentmodule:: xdem
```

## DEM

```{eval-rst}
.. minigallery:: xdem.DEM
      :add-heading:
```

### Opening a DEM

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM
    DEM.info
```

### Create from an array

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.from_array
```

(api-dem-attrs)=

### Unique attributes

```{note}
A {class}`~xdem.DEM` inherits four unique attributes from {class}`~geoutils.Raster`, see [the dedicated section of GeoUtils' API](https://geoutils.readthedocs.io/en/latest/api.html#unique-attributes).
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.vcrs
```

### Derived attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.ccrs
```

### Vertical referencing

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.set_vcrs
    DEM.to_vcrs
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
.. inheritance-diagram:: xdem.coreg xdem.biascorr
        :top-classes: xdem.Coreg
```

### Coregistration object and pipeline

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.Coreg
    xdem.CoregPipeline
```

### Block-wise application of co-registrations

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.BlockwiseCoreg
```

### Affine coregistration methods


**Generic parent class:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.AffineCoreg
```

**Convenience classes for specific coregistrations:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.VerticalShift
    xdem.NuthKaab
    xdem.ICP
    xdem.Tilt
```

### Bias-correction (including non-affine coregistration) methods

**Generic parent class:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.BiasCorr
```

**Classes for any 1-, 2- and N-D biases:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.BiasCorr1D
    xdem.BiasCorr2D
    xdem.BiasCorrND
```

**Convenience classes for specific corrections:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.Deramp
    xdem.DirectionalBias
    xdem.TerrainBias
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
