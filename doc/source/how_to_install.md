(how-to-install)=

# How to install

## Installing with ``mamba`` (recommended)

```bash
mamba install -c conda-forge xdem
```

```{important}
Solving dependencies can take a long time with `conda`, `mamba` significantly speeds up the process. Install it with:

    conda install mamba -n base -c conda-forge

Once installed, the same commands can be run by simply replacing `conda` by `mamba`. More details available in the [mamba documentation](https://mamba.readthedocs.io/en/latest/).
```

If running into the `sklearn` error `ImportError: dlopen: cannot load any more object with static TLS`, your system
needs to update its `glibc` (see details [here](https://github.com/scikit-learn/scikit-learn/issues/14485#issuecomment-822678559)).
If you have no administrator right on the system, you might be able to circumvent this issue by installing a working
environment with specific downgraded versions of `scikit-learn` and `numpy`:

```bash
mamba create -n xdem-env -c conda-forge xdem scikit-learn==0.20.3 numpy==1.19.*
```

On very old systems, if the above install results in segmentation faults, try setting more specifically
`numpy==1.19.2=py37h54aff64_0` (works with Debian 8.11, GLIBC 2.19).

## Installing with ``pip``

```bash
pip install xdem
```

```{warning}
Updating packages with `pip` (and sometimes `mamba`) can break your installation. If this happens, re-create an environment from scratch fixing directly all your dependencies.
```

## Installing for contributors

```bash
git clone https://github.com/GlacioHack/xdem.git
mamba env create -f xdem/dev-environment.yml
```

After installing, you can check that everything is working by running the tests: `pytest -rA`.
