% xdem documentation master file, created by
% sphinx-quickstart on Fri Mar 19 14:30:39 2021.
% You can adapt this file completely to your liking, but it should at least
% contain the root ``toctree`` directive.

# xDEM: Analysis of digital elevation models

xDEM aims at simplifying the analysis of digital elevation models (DEMs), including vertical referencing,
terrain attributes, co-registration, bias corrections, error statistics, and more.

:::{important}
xDEM is in early stages of development and its features might evolve rapidly. Note the version you are
working on for reproducibility!
We are working on making features fully consistent for the first long-term release `v0.1` (likely sometime in 2023).
:::

```{toctree}
:caption: Getting started
:maxdepth: 2

about_xdem
how_to_install
quick_start
```

```{toctree}
:caption: Background
:maxdepth: 2

intro_dems
intro_robuststats
intro_accuracy_precision
```

```{toctree}
:caption: Features
:maxdepth: 2

vertical_ref
terrain
coregistration
biascorr
filters
comparison
spatialstats
```

```{toctree}
:caption: Gallery of examples
:maxdepth: 2

basic_examples/index.rst
advanced_examples/index.rst
```

```{toctree}
:caption: API Reference
:maxdepth: 2

api.rst
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
