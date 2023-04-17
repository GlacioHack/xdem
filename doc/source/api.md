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

    DEM.vref
```

### Derived attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DEM.ccrs
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

### Coregistration object and pipeline

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.Coreg
    xdem.CoregPipeline
```

### Rigid coregistration methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.BiasCorr
    xdem.NuthKaab
    xdem.ICP
    xdem.Deramp
```

### Spatial coregistration

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    xdem.BlockwiseCoreg
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



