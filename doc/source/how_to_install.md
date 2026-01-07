(how-to-install)=

# How to install

## Installing with ``mamba`` (recommended)

```bash
mamba install -c conda-forge xdem
```

```{tip}
Solving dependencies can take a long time with `conda`, `mamba` significantly speeds up the process. Install it with:

    conda install mamba -n base -c conda-forge

Once installed, the same commands can be run by simply replacing `conda` by `mamba`. More details available in the [mamba documentation](https://mamba.readthedocs.io/en/latest/).
```

## Installing with ``pip``

```bash
pip install xdem
```

```{warning}
Updating packages with `pip` (and sometimes `mamba`) can break your installation. If this happens, re-create an environment from scratch pinning directly all your other dependencies during initial solve (e.g., `mamba create -n xdem-env -c conda-forge xdem myotherpackage==1.0.0`).
```

## Installing for contributors

### With ``mamba``

```bash
git clone https://github.com/GlacioHack/xdem.git
mamba env create -f xdem/dev-environment.yml
```

### With ``pip``

Please note: pip installation is currently only possible under python3.10.

```bash
git clone https://github.com/GlacioHack/xdem.git
cd xdem
make install
```

After installing, you can check that everything is working by running the tests: `pytest`.

## Dependencies

xDEM's required dependencies are:

- [GeoUtils](https://geoutils.readthedocs.io/en/stable/) (version 0.2 and above),
- [Numba](https://numba.pydata.org/),

with themselves have dependencies on:

- [Rasterio](https://rasterio.readthedocs.io/en/stable/) (version 1.3 and above),
- [GeoPandas](https://geopandas.org/en/stable/) (version 0.12 and above),
- [SciPy](https://scipy.org/),
- [Xarray](https://xarray.dev/),
- [Rioxarray](https://corteva.github.io/rioxarray/stable/).

and second-order dependencies being notably [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [pyproj](https://pyproj4.github.io/pyproj/stable/) and [pyogrio](https://pyogrio.readthedocs.io/en/latest/).

Optional dependencies are:
- [Matplotlib](https://matplotlib.org/) for plotting,
- [LasPy](https://laspy.readthedocs.io/en/latest/) for reading and writing LAS/LAZ/COPC point cloud files,
- [Dask](https://www.dask.org/) for out-of-memory operations (coming soon),
- [tqdm](https://tqdm.github.io/) for displaying progress bars,
- [Cerberus](https://docs.python-cerberus.org/), [Pyyaml](https://pyyaml.org/) and [Weasyprint](https://weasyprint.org/) for the command-line interface and workflows.
