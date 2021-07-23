# xdem
Set of tools to manipulate Digital Elevation Models (DEMs)

More documentation to come!

[![Documentation Status](https://readthedocs.org/projects/xdem/badge/?version=latest)](https://xdem.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/GlacioHack/xdem/actions/workflows/python-package.yml/badge.svg)](https://github.com/GlacioHack/xdem/actions/workflows/python-package.yml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xdem.svg)](https://anaconda.org/conda-forge/xdem)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/xdem.svg)](https://anaconda.org/conda-forge/xdem)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/xdem.svg)](https://anaconda.org/conda-forge/xdem)

To cite this package: [![Zenodo](https://zenodo.org/badge/doi/10.5281/zenodo.4809697.svg)](https://zenodo.org/record/4809698)

## Installation

### With conda (recommended)
```bash
conda install --c conda-forge --strict-channel-priority xdem
```
The `--strict-channel-priority` flag seems essential for Windows installs to function correctly, and is recommended for UNIX-based systems as well.


### Installing with pip
**NOTE**: Setting up GDAL and PROJ may need some extra steps, depending on your operating system and configuration.
```bash
pip install xdem
```

### Installing for contributors
Recommended: Use conda for depencency solving.
```
$ git clone https://github.com/GlacioHack/xdem.git
$ cd ./xdem
$ conda env create -f dev-environment.yml
$ conda activate xdem
$ pip install -e .
```
After installing, we recommend to check that everything is working by running the tests:

```
$ pytest -rA
```

## Structure 

xdem are for now composed of three libraries:
- `coreg.py` with tools covering differet aspects of DEM coregistration
- `spatial_tools.py` for spatial operations on DEMs
- `dem.py` for DEM-specific operations, such as vertical datum correction.

## How to contribute

You can find ways to improve the libraries in the [issues](https://github.com/GlacioHack/xdem/issues) section. All contributions are welcome.
To avoid conflicts, it is suggested to use separate branches for each implementation. All changes must then be submitted to the dev branch using pull requests. Each PR must be reviewed by at least one other person.

Please see our [contribution page](CONTRIBUTING.md) for more detailed instructions.

### Documentation
See the documentation at https://xdem.readthedocs.io

### Testing - again please read!
These tools are only valuable if we can rely on them to perform exactly as we expect. So, we need testing. Please create tests for every function that you make, as much as you are able. Guidance/examples here for the moment: https://github.com/GeoUtils/georaster/blob/master/test/test_georaster.py
https://github.com/corteva/rioxarray/blob/master/test/integration/test_integration__io.py



### Examples

**Coregister a DEM to another DEM**
```python
import xdem

reference_dem = xdem.DEM("path/to/reference.tif")
dem_to_be_aligned = xdem.DEM("path/to/dem.tif")

nuth_kaab = xdem.coreg.NuthKaab()

nuth_kaab.fit(reference_dem.data, dem_to_be_aligned.data, transform=reference_dem.transform)


aligned_dem = xdem.DEM.from_array(
	nuth_kaab.apply(dem_to_be_aligned.data, transform=dem_to_be_aligned.transform),
	transform=dem_to_be_aligned.transform,
	crs=dem_to_be_aligned.crs
)

aligned_dem.save("path/to/coreg.tif")
```
This is an implementation of the [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011) approach.
[Please see the documentation](https://xdem.readthedocs.io/en/latest/coregistration.html) for more approaches.

**Subtract one DEM with another**
```python
import xdem

first_dem = xdem.DEM("path/to/first.tif")
second_dem = xdem.DEM("path/to/second.tif")

difference = first_dem - second_dem

difference.save("path/to/difference.tif")
```
By default, `second_dem` is reprojected to fit `first_dem`.
This can be switched with the keyword argument `reference="second"`.
The resampling method can also be changed (e.g. `resampling_method="nearest"`) from the default `"cubic_spline"`.

