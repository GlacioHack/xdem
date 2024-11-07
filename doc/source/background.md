(background)=

# Background

Below, some more information on the people behind the package, and its mission.

## The people behind xDEM

### Creation

```{margin}
<sup>1</sup>More on our GlacioHack founder at [adehecq.github.io](https://adehecq.github.io/)!
```

xDEM was created during the **[GlacioHack](https://github.com/GlacioHack) hackathon**, that was initiated by
Amaury Dehecq<sup>1</sup> and took place online on November 8, 2020.

```{margin}
<sup>2</sup>Check-out [glaciology.ch](https://glaciology.ch) on our founding group of VAW glaciology!
```

The initial core development of xDEM was performed by members of the Glaciology group of the Laboratory of Hydraulics, Hydrology and
Glaciology (VAW) at ETH Zürich<sup>2</sup>, with contributions by members of the University of Oslo, the University of Washington, and University
Grenoble Alpes.

### Current team

The current lead development team includes **researchers in Earth observation and engineers from
[CNES](https://cnes.fr/en)** (French Space Agency). We specialize in elevation data, for analysis in Earth science or 
for operational use for 3D satellite missions.

Volunteers contributors span various backgrounds, in industry or research. We welcome
any new contributors! See how to contribute on [the dedicated page of our repository](https://github.com/GlacioHack/xdem/blob/main/CONTRIBUTING.md).

### Funding acknowledgments

Members of the lead development team acknowledge funding from:
- SNF (Swiss National Science Foundation), grant no. 184634, project by MeteoSwiss in the framework of the Global Climate Observing System Switzerland,
- NASA, grant no. xxx, Surface Topography and Vegetation project on the fusion of elevation data,
- NASA, grant no. xxx, ICESat-2 project on the efficient processing of elevation data in the cloud,
- CNES (French Space Agency): ?.

## Mission

```{epigraph}
The core mission of xDEM is to be **easy-of-use**, **modular** and **robust**.

It also attempts to be as **efficient**, **scalable** and **state-of-the-art** as possible.

Finally, as an open source package, it aspires to foster **reproducibility** and **open science**.
```

In details, those mean:

- **Ease-of-use:** all basic operations or methods from published works should only require a few lines of code to be performed;

- **Modularity:** all methods should be fully customizable, to allow both flexibility and inter-comparison;

- **Robustness:** all methods should be tested within our continuous integration test-suite, to enforce that they always perform as expected;

```{note}
:class: margin
**Scalability** is currently being improved towards a first major release ``v1.0``.
```

And, additionally:

- **Efficiency**: all methods should be optimized at the lower-level, to function with the highest performance offered by Python packages;

- **Scalability**: all methods should support both lazy processing and distributed parallelized processing, to work with high-resolution data on local machines as well as on HPCs;

- **State-of-the-art**: all methods should be at the cutting edge of remote sensing science, to provide users with the most reliable and up-to-date tools.

And finally:

- **Reproducibility:** all code should be version-controlled and release-based, to ensure consistency of dependent
  packages and works;

- **Open-source:** all code should be accessible and re-usable to anyone in the community, for transparency and open governance.
