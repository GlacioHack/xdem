(about-xdem)=

# About xDEM


xDEM is a [Python](https://www.python.org/) package for the analysis of DEMs, with name standing for _cross-DEM analysis_[^sn1]  
and echoing its dependency on [xarray](https://docs.xarray.dev/en/stable/). It is designed for all Earth and planetary 
observation science, although our group currently has a strong focus on glaciological applications.

[^sn1]: The core features of xDEM rely on cross-analysis of surface elevation, for example for DEM alignment or error analysis.


```{epigraph}
The core mission of xDEM is to be **easy-of-use**, **modular**, **robust**, **reproducible** and **fully open**.

Additionally, xDEM aims to be **efficient**, **scalable** and **state-of-the-art**.
```

```{important}
:class: margin
xDEM is in early stages of development and its features might evolve rapidly. Note the version you are working on for 
**reproducibility**!
We are working on making features fully consistent for the first long-term release ``v0.1`` (likely sometime in 2023).
```

In details, those mean:

- **Ease-of-use:** all DEM basic operations or methods from published works should only require a few lines of code to be performed;

- **Modularity:** all DEM methods should be fully customizable, to allow both flexibility and inter-comparison;

- **Robustness:** all DEM methods should be tested within our continuous integration test-suite, to enforce that they always perform as expected;

- **Reproducibility:** all code should be version-controlled and release-based, to ensure consistency of dependant packages and works;

- **Open-source:** all code should be accessible and re-usable to anyone in the community, for transparency and open governance.

```{note}
:class: margin
Additional mission points, in particular **scalability**, are partly developed but not a priority until our first long-term release ``v0.1`` is reached. Those will be further developed specifically in a subsequent version ``v0.2``.
```

And, additionally:

- **Efficiency**: all methods should be optimized at the lower-level, to function with the highest performance offered by Python packages;

- **Scalability**: all methods should support both lazy processing and distributed parallelized processing, to work with high-resolution data on local machines as well as on HPCs;

- **State-of-the-art**: all methods should be at the cutting edge of remote sensing science, to provide users with the most reliable and up-to-date tools.


# The people behind xDEM

```{margin}
<sup>2</sup>More on our GlacioHack founder at [adehecq.github.io](https://adehecq.github.io/)!
```

xDEM was created during the [GlacioHack](https://github.com/GlacioHack) hackaton event, that was initiated by
Amaury Dehecq<sup>2</sup> and took place online on November 8, 2020.

```{margin}
<sup>3</sup>Check-out [glaciology.ch](https://glaciology.ch) on our founding group of VAW glaciology!
```

The initial core development of xDEM was performed by members of the Glaciology group of the Laboratory of Hydraulics, Hydrology and
Glaciology (VAW) at ETH Zürich<sup>3</sup>, with contributions by members of the University of Oslo, the University of Washington, and University
Grenobles Alpes.

We are not software developers but geoscientists, and we try our best to offer tools that can be useful to a larger group,
documented, reliable and maintained. All development and maintenance is made on a voluntary basis and we welcome
any new contributors. See some information on how to contribute in the dedicated page of our
[GitHub repository](https://github.com/GlacioHack/xdem/blob/main/CONTRIBUTING.md).


