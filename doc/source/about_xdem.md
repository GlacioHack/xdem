(about-xdem)=

# About xDEM

## What is xDEM?

xDEM is a Python package for the analysis of elevation data, and in particular that of digital elevation models (DEMs),
with name standing for _cross-DEM analysis_[^sn1] and echoing its dependency on [Xarray](https://docs.xarray.dev/en/stable/).

[^sn1]: Several core features of xDEM, in particular coregistration and uncertainty analysis, rely specifically on cross-analysis of elevation data over static surfaces.

## Why use xDEM?

xDEM implements a wide range of high-level operations required for analyzing elevation data in a consistent framework
tested to ensure the accuracy of these operations.

It has three main focus points:

1. Having an **easy and intuitive interface** based on the principle of least knowledge,
2. Providing **statistically robust methods** for reliable quantitative analysis,
3. Allowing **modular user input** to adapt to most applications.

Although modularity can sometimes hamper performance, we also aim to **preserve scalibility** as much as possible[^sn2].

[^sn2]: Out-of-memory, parallelizable computations relying on Dask are planned for 2025!

We particularly take to heart to verify the accuracy of our methods. For instance, our terrain attributes
which have their own modular Python-based implementation, are tested to match exactly
[gdaldem](https://gdal.org/programs/gdaldem.html) (slope, aspect, hillshade, roughness) and
[RichDEM](https://richdem.readthedocs.io/en/latest/) (curvatures).

## Who is behind xDEM?

xDEM was created by a group of researchers with expertise in elevation data analysis for change detection applied to glaciology.
Nowadays, its development is **jointly led by researchers in elevation data analysis** (including funding from NASA and SNSF) **and
engineers from CNES** (French Space Agency).

Most contributors and users are scientists or industrials working in **various fields of Earth observation**.


```{note}
:class: tip
:class: margin

xDEM is **merging efforts with CNES's [demcompare](https://github.com/CNES/demcompare)** to combine the best of both tools into one!
```

::::{grid}
:reverse:

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/nasa_logo.svg
    :width: 200px
    :class: dark-light
```

:::

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/snsf_logo.svg
    :width: 220px
    :class: only-light
```

```{image} ./_static/snsf_logo_dark.svg
    :width: 220px
    :class: only-dark
```

:::

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/cnes_logo.svg
    :width: 200px
    :class: only-light
```

```{image} ./_static/cnes_logo_dark.svg
    :width: 200px
    :class: only-dark
```

:::


::::

More details about the people behind xDEM, funding sources, and the package's objectives can be found on the **{ref}`credits` pages**.
