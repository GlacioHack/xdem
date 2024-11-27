# xDEM: robust analysis of DEMs in Python.

[![Documentation Status](https://readthedocs.org/projects/xdem/badge/?version=latest)](https://xdem.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/GlacioHack/xdem/actions/workflows/python-tests.yml/badge.svg)](https://github.com/GlacioHack/xdem/actions/workflows/python-tests.yml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xdem.svg)](https://anaconda.org/conda-forge/xdem)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/xdem.svg)](https://anaconda.org/conda-forge/xdem)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/xdem.svg)](https://anaconda.org/conda-forge/xdem)
[![PyPI version](https://badge.fury.io/py/xdem.svg)](https://badge.fury.io/py/xdem)
[![Coverage Status](https://coveralls.io/repos/github/GlacioHack/xdem/badge.svg?branch=main)](https://coveralls.io/github/GlacioHack/xdem?branch=main)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GlacioHack/xdem/main)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Formatted with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

**xDEM** is an open source project to develop a core Python package for the analysis of digital elevation models (DEMs).

It aims at **providing modular and robust tools for the most common analyses needed with DEMs**, including both geospatial
operations specific to DEMs and a wide range of 3D alignment and correction methods from published, peer-reviewed studies.
The core manipulation of DEMs (e.g., vertical alignment, terrain analysis) are **conveniently centered around a `DEM` class** (that, notably, re-implements all tools
of [gdalDEM](https://gdal.org/programs/gdaldem.html)). More complex pipelines (e.g., 3D rigid coregistration, bias corrections, filtering) are **built around
modular `Coreg`, `BiasCorr` classes that easily interface between themselves**. Finally, xDEM includes advanced
uncertainty analysis tools based on spatial statistics of [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/).

Additionally, xDEM inherits many convenient functionalities from [GeoUtils](https://github.com/GlacioHack/geoutils) such as
**implicit loading**, **numerical interfacing** and **convenient object-based geospatial methods** to easily perform
the most common higher-level tasks needed by geospatial users (e.g., reprojection, cropping, vector masking). Through [GeoUtils](https://github.com/GlacioHack/geoutils), xDEM
relies on [Rasterio](https://github.com/rasterio/rasterio), [GeoPandas](https://github.com/geopandas/geopandas) and [Pyproj](https://github.com/pyproj4/pyproj)
for georeferenced calculations, and on [NumPy](https://github.com/numpy/numpy) and [Xarray](https://github.com/pydata/xarray) for numerical analysis. It allows easy access to
the functionalities of these packages through interfacing or composition, and quick inter-operability through object conversion.

If you are looking for an accessible Python package to write the Python equivalent of your [GDAL](https://gdal.org/) command lines, or of your
[QGIS](https://www.qgis.org/en/site/) analysis pipeline **without a steep learning curve** on Python GIS syntax, xDEM is perfect for you! For more advanced
users, xDEM also aims at being efficient and scalable by supporting lazy loading and parallel computing (ongoing).

## Documentation

For a quick start, full feature description or search through the API, see xDEM's documentation at: https://xdem.readthedocs.io.

## Installation

### With `mamba`

```bash
mamba install -c conda-forge xdem
```
See [mamba's documentation](https://mamba.readthedocs.io/en/latest/) to install `mamba`, which will solve your environment much faster than `conda`.

### With `pip`

```bash
pip install xdem
```

## Citing methods implemented in the package

When using a method implemented in xDEM, please **cite both the package and the related study**:

Citing xDEM: [![Zenodo](https://zenodo.org/badge/doi/10.5281/zenodo.4809697.svg)](https://zenodo.org/doi/10.5281/zenodo.4809697)

Citing the related study:

- **Coregistration**:
  - Horizontal shift from aspect/slope relationship of *[Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011)*,
  - Iterative closest point (ICP) of *[Besl and McKay (1992)](http://dx.doi.org/10.1109/34.121791)*,
- **Bias correction**:
  - Along-track multi-sinusoidal noise by basin-hopping of *[Girod et al. (2017)](https://doi.org/10.3390/rs9070704)*,
- **Uncertainty analysis**:
  - Heteroscedasticity and multi-range correlations from stable terrain of *[Hugonnet et al. (2022)](https://doi.org/10.1109/JSTARS.2022.3188922)*,
- **Terrain attributes**:
  - Slope, aspect and hillshade of either *[Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918)* or *[Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)*,
  - Profile, plan and maximum curvature of *[Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)*,
  - Topographic position index of *[Weiss (2001)](http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf)*,
  - Terrain ruggedness index of either *[Riley et al. (1999)](http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf)* or *[Wilson et al. (2007)](http://dx.doi.org/10.1080/01490410701295962)*,
  - Roughness of *[Dartnell (2000)](http://dx.doi.org/10.14358/PERS.70.9.1081)*,
  - Rugosity of *[Jenness (2004)](https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2)*,
  - Fractal roughness of *[Taud et Parrot (2005)](https://doi.org/10.4000/geomorphologie.622)*.

## Contributing

We welcome new contributions, and will happily help you integrate your own DEM routines into xDEM!

After discussing a new feature or bug fix in an issue, you can open a PR to xDEM with the following steps:

1. Fork the repository, make a feature branch and push changes.
2. When ready, submit a pull request from the feature branch of your fork to `GlacioHack/xdem:main`.
3. The PR will be reviewed by at least one maintainer, discussed, then merged.

More details on [our contributing page](CONTRIBUTING.md).
