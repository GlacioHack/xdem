(about-xdem)=

# About xDEM

## What is xDEM?

xDEM is a Python package for the analysis of elevation data, and in particular that of digital elevation models (DEMs), 
with name standing for _cross-DEM analysis_[^sn1] and echoing its dependency on [Xarray](https://docs.xarray.dev/en/stable/). 

[^sn1]: Several core features of xDEM, in particular coregistration and uncertainty analysis, rely specifically on cross-analysis of elevation data over static surfaces.

## Why use xDEM?

xDEM implements a wide range of high-level operations required for analyzing elevation data in a consistent framework that is 
extensively tested to ensure the accuracy of these operations, yet still allows for modular user input to facilitate all kinds of analysis.

It has three main focus points:

1. Having an **easy and intuitive interface** based on the principle of least knowledge,
2. Providing **statistically robust methods** for reliable quantitative analysis,
3. Allowing **modular user input** to adapt to most applications.

Although modularity can sometimes hamper performance, we also aim to **preserve scalibility** as much as possible[^sn2].

[^sn2]: Out-of-memory, parallelizable computations relying on Dask are planned for late 2024!

We particularly take to heart to verify the accuracy of our methods. For instance, our terrain attributes 
which have their own modular Python-based implementation, are tested to match exactly 
[gdalDEM](https://gdal.org/programs/gdaldem.html) (slope, aspect, hillshade, roughness) and 
[richDEM](https://richdem.readthedocs.io/en/latest/) (curvatures).

More details about the people behind xDEM and the package's objectives can be found on the **{ref}`background` reference 
page**.
