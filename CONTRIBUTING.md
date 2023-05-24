# How to contribute

## Overview: making a contribution

For more details, see the rest of this document.

1. Fork _GlacioHack/xdem_ and clone your fork repository locally.
2. Set up the development environment (section below).
3. Create a branch for the new feature or bug fix.
4. Make your changes, and add or modify related tests in _tests/_.
5. Commit, making sure to run `pre-commit` separately if not installed as git hook.
6. Push to your fork.
7. Open a pull request from GitHub to discuss and eventually merge.

## Development environment

GeoUtils currently supports only Python versions of 3.8 and higher, see `environment.yml` for detailed dependencies.

### Setup

Clone the git repo and create a `mamba` environment (see how to install `mamba` in the [mamba documentation](https://mamba.readthedocs.io/en/latest/)):

```bash
git clone https://github.com/GlacioHack/xdem.git
cd xdem
mamba env create -f dev-environment.yml  # Add '-n custom_name' if you want.
mamba activate xdem-dev  # Or any other name specified above
```

### Tests

At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, using `pytest` (see existing tests for examples).

To run the entire test suite, run `pytest` in the current directory:
```bash
pytest
```

### Formatting and linting

Install and run `pre-commit` (see [pre-commit documentation](https://pre-commit.com/)), which will use `.pre-commit-config.yaml` to verify spelling errors,
import sorting, type checking, formatting and linting.

You can then run pre-commit manually:
```bash
pre-commit run --all-files
```

Optionally, `pre-commit` can be installed as a git hook to ensure checks have to pass before committing.

## Rights

The license (see LICENSE) applies to all contributions.
