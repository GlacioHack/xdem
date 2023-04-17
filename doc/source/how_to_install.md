(how-to-install)=

# How to install

## Installing with conda (recommended)

```bash
conda install -c conda-forge --strict-channel-priority xdem
```

**Notes**

- The `--strict-channel-priority` flag seems essential for Windows installs to function correctly, and is recommended for UNIX-based systems as well.

- Solving dependencies can take a long time with `conda`. To speed up this, consider installing `mamba`:

  ```bash
  conda install mamba -n base -c conda-forge
  ```

  Once installed, the same commands can be run by simply replacing `conda` by `mamba`. More details available through the [mamba project](https://github.com/mamba-org/mamba).

- If running into the `sklearn` error `ImportError: dlopen: cannot load any more object with static TLS`, your system
  needs to update its `glibc` (see details [here](https://github.com/scikit-learn/scikit-learn/issues/14485#issuecomment-822678559)).
  If you have no administrator right on the system, you might be able to circumvent this issue by installing a working
  environment with specific downgraded versions of `scikit-learn` and `numpy`:

  ```bash
  conda create -n xdem-env -c conda-forge xdem scikit-learn==0.20.3 numpy==1.19.*
  ```

  On very old systems, if the above install results in segmentation faults, try setting more specifically
  `numpy==1.19.2=py37h54aff64_0` (worked with Debian 8.11, GLIBC 2.19).

## Installing with pip

```bash
pip install xdem
```

**NOTE**: Setting up GDAL and PROJ may need some extra steps, depending on your operating system and configuration.

## Installing for contributors

Recommended: Use conda for dependency solving.

```shell
git clone https://github.com/GlacioHack/xdem.git
cd ./xdem
conda env create -f dev-environment.yml
conda activate xdem
pip install -e .
```

After installing, we recommend to check that everything is working by running the tests:

`pytest -rA`
