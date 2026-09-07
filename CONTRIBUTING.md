# How to contribute

We welcome new contributions to xDEM that is still very much in expansion!
Below is a guide to contributing to xDEM step by step, ensuring tests are passing and the documentation is updated.

## Overview: making a contribution

The technical steps to contributing to xDEM are:

1. Fork `GlacioHack/xdem` and clone your fork repository locally.
2. Set up the development environment **(see section "Setup" below)**,
3. Create a branch for the new feature or bug fix,
4. Make your changes,
5. Add or modify related tests in `tests/` **(see section "Tests" below)**,
6. Add or modify related documentation in `doc/` **(see section "Documentation" below)**,
7. Commit your changes,
8. Run `pre-commit` separately if not installed as git hook **(see section "Linting" below)**,
9. Push to your fork,
10. Open a pull request from GitHub to discuss and eventually merge.

## AI-assisted contribution policy

We welcome AI-assisted contributions under the following conditions:

- **Human communication.** Write the core PR description yourself, explaining the motivation, approach,
  and validation in your own words. AI may polish or translate your writing and help prepare supporting
  lists, tables, or other displays, but must not replace your own explanation. The same principle
  applies to comments during PR review, and opening an issue or discussion.

- **Human review and responsibility.** Personally review, understand, and validate the entire contribution,
  including code, tests, and documentation, before requesting review. Run relevant tests and accurately report
  what you checked. You remain responsible for the contribution and must be able to explain your choices and address
  review feedback.

- **References and licensing (copyright).** AI-generated code may reproduce or adapt existing copyrighted implementations
  without identifying their sources. For major features or conceptual changes to methods, algorithms, or
  design, make a reasonable search for relevant literature and existing implementations. Verify and cite relevant
  sources, and check for potential code reuse. Ensure that any reused or adapted material has a compatible license 
  with [xDEM's LICENSE](./LCENSE), and flag it to maintainers.

Maintainers may decline PRs whose descriptions appear predominantly AI-generated, or that do not demonstrate sufficient
human understanding and validation, or attention to copyright and source.

This AI policy was inspired from that of [NumPy](https://numpy.org/devdocs/dev/ai_policy.html) and [SciPy](https://docs.scipy.org/doc/scipy/dev/conduct/ai_policy.html) as of Sept 2026,
which was itself inspired by that of [SymPy](https://www.sympy.org/en/index.html).

## Development environment

xDEM currently supports Python versions of 3.10 to 3.14 (see `dev-environment.yml` for detailed dependencies), which are
tested in a continuous integration (CI) workflow running on GitHub Actions.

When you open a PR on xDEM, a single linting action and 9 test actions will automatically start, corresponding to all
supported Python versions (3.10, 3.11, 3.12, 3.13 and 3.14) and OS (Ubuntu, Mac, Windows). The coverage change of the tests will also
be reported by CoverAlls.

### Setup

#### With `mamba`
Clone the git repo and create a `mamba` environment (see how to install `mamba` in the [mamba documentation](https://mamba.readthedocs.io/en/latest/)):

```bash
git clone https://github.com/GlacioHack/xdem.git
cd xdem
mamba env create -f dev-environment.yml  # Add '-n custom_name' if you want.
mamba activate xdem-dev  # Or any other name specified above
```

#### With `pip`
```bash
git clone https://github.com/GlacioHack/xdem.git
cd xdem
make install
```

### Tests

At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, using `pytest` (see existing tests for examples).
The structure of test modules and functions in `tests/` largely mirrors that of the package modules and functions in `xdem/`.

To run the entire test suite, run `pytest` from the root of the repository:
```bash
pytest
```

Running `pytest` will trigger a script that automatically downloads test data from [https://github.com/GlacioHack/xdem-data](https://github.com/GlacioHack/xdem-data) used to run all tests.

RichDEM should only be used for testing purposes within the xDEM project. The functionality of xDEM must not depend on RichDEM.

### Documentation

If your changes need to be reflected in the documentation, update the related pages located in `doc/source/`. The documentation is written in MyST markdown syntax, similar to GitHub's default Markdown (see [MyST-NB](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html) for details).

To ensure that the documentation is building properly after your changes, if you are on Linux, you can run `pytest tests/test_doc.py`, which is equivalent to building directly calling `sphinx-build source/ build/html/` from the `doc/` folder. On Windows and Mac, the documentation is not maintained, so you can wait to open the PR for it to be checked on Linux by the CI.

### Formatting and linting

Install and run `pre-commit` from the root of the repository (such as with `mamba install pre-commit`, see [pre-commit documentation](https://pre-commit.com/) for details),
which will use `.pre-commit-config.yaml` to verify spelling errors, import sorting, type checking, formatting and linting:

```bash
pre-commit run --all
```

You can then commit and push those changes.
Optionally, `pre-commit` can be installed as a git hook to ensure checks have to pass before committing.

### Final steps

That's it! If the tests and documentation are passing, or if you need help to make those work, you can open a PR.

We'll receive word of your PR as soon as it is opened, and should follow up shortly to discuss the changes, and eventually give approval to merge. Thank you so much for contributing!

### Rights

The license (see LICENSE) applies to all contributions.
