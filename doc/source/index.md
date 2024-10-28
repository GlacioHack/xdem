---
title: xDEM
---

::::{grid}
:reverse:
:gutter: 2 1 1 1
:margin: 4 4 1 1

:::{grid-item}
:columns: 4

```{image} ./_static/xdem_logo_only.svg
    :width: 300px
    :class: only-light
```

```{image} ./_static/xdem_logo_only_dark.svg
    :width: 300px
    :class: only-dark
```
:::

:::{grid-item}
:columns: 8
:class: sd-fs-3
:child-align: center

xDEM aims at making the analysis of digital elevation models **easy**, **modular** and **robust**.

::::

```{important}
:class: margin
xDEM ``v0.1`` is released, with nearly all features planned 4 years ago 🎉! We are now adding an **Xarray accessor**, and re-structuring the "Uncertainty" features for 2025. Note the version you are
working on for reproducibility!
```

xDEM is **tailored to perform quantitative analysis that implicitly understands the intricacies of elevation data**,
both from a **georeferencing viewpoint** (vertical referencing, nodata values, projection, pixel interpretation) and
a **statistical viewpoint** (outlier robustness, specificities of 3D alignment and error structure).

It exposes **an intuitive object-based API to foster accessibility**, and strives **to be computationally scalable**
through Dask.

Additionally, through its sister-package [GeoUtils](https://geoutils.readthedocs.io/en/stable/), xDEM is built on top
of core geospatial packages (Rasterio, GeoPandas, PyProj) and numerical packages (NumPy, Xarray, SciPy) to provide
**consistent higher-level functionalities at the interface of DEMs and elevation point cloud objects**.

----------------

# Where to start?

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`edit_note;2em` About xDEM
:link: about-xdem
:link-type: ref

Learn more about why we developed xDEM.

+++
[Learn more »](about-xdem)
:::

:::{grid-item-card} {material-regular}`data_exploration;2em` Quick start
:link: quick-start
:link-type: ref

Run a short example of the package functionalities.

+++
[Learn more »](quick-start)
:::

:::{grid-item-card} {material-regular}`preview;2em` Features
:link: dem-class
:link-type: ref

Dive into the full documentation.

+++
[Learn more »](dem-class)
:::

::::

----------------


```{toctree}
:caption: Getting started
:maxdepth: 2

about_xdem
how_to_install
quick_start
citation
```

```{toctree}
:caption: Features
:maxdepth: 2

elevation_objects
vertical_ref
terrain
coregistration
biascorr
gapfill
uncertainty
```

```{toctree}
:caption: Resources
:maxdepth: 2

guides
cheatsheet
ecosystem
```

```{toctree}
:caption: Gallery of examples
:maxdepth: 2

basic_examples/index.rst
advanced_examples/index.rst
```

```{toctree}
:caption: Reference
:maxdepth: 2

logging
api
publis
background
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
