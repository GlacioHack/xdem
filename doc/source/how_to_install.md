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
make install
```

After installing, you can check that everything is working by running the tests: `pytest`.
