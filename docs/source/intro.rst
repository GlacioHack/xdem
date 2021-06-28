Introduction: why is it complex to assess DEM accuracy and precision?
=====================================================================

Digital Elevation Models are numerical representations of elevation. They are generated from different instruments (e.g., radiometer, radar, lidar), acquired in different conditions (e.g., ground, airborne, satellite), and using different post-processing techniques (e.g., stereophotogrammetry, interferometry).

While some complexities are specific to certain instruments and methods, all DEMs generally have:
    - an **arbitrary Ground Sampling Distance (GSD)** that does not necessarily represent their underlying spatial resolution,
    - an **georeferenced positioning subject to shifts, tilts or other deformations** due to inherent instrument errors, noise, or associated post-processing schemes,
    - a **large number of outliers** that can originate from various sources (e.g., photogrammetric blunders, clouds).

These factors lead to difficulties in assessing the accuracy and precision of DEMs, necessary to perform further analysis.

Accuracy and precision
**********************

Both accuracy and precision are important when analyzing DEMs:
    - the **accuracy** (systematic error) of a DEM describes how close a DEM is to the true location of measured elevations on the Earth's surface,
    - the **precision** (random error) of a DEM describes the typical spread of its error in measurement, independently of a possible bias from the true positioning.

Absolute or relative accuracy
*****************************

The measure of accuracy can be further split into two:
    - the **absolute accuracy** of a DEM is the average shift to the true positioning. Studies interested in analyzing features of a single DEM in relation to other georeferenced data might give great importance to this potential bias.
    - the **relative accuracy** of a DEM is the potential shifts, tilts, and deformations in relation to other elevation data that does not necessarily matches a given referencing. Studies interested in comparing DEMs between themselves might be only interested in this accuracy.


Dealing with DEM absolute accuracy
**********************************

Shifts due to poor absolute accuracy are common in elevation datasets, and can be easily corrected by performing a DEM co-registration to accurate, georeferenced point elevation data such as ICESat and ICESat-2.

For more details, see :ref:`coregistration` with point data.

Dealing with DEM relative accuracy
**********************************

As the **absolute accuracy** can be corrected a posteriori using reference elevation datasets, many analyses only focus on **relative accuracy**, i.e. the remaining biases between several DEMs co-registered relative one to another.
By harnessing the denser, nearly continuous sampling of raster DEMs (in opposition to the sparser sampling of higher-accuracy point elevation data), one can identify and correct other types of biases:
    - Terrain-related biases that can originate from the difference of resolution of DEMs, or instrument processing deformations.
    - Directional biases that can be linked to instrument noise, such as along-track oscillations observed in many widepsread DEM products (SRTM, ASTER, SPOT, Pléiades, etc).

Those biases can be tackled by iteratively combining co-registration and bias-correction methods (:ref:`coregistration`, TODO: ref bias corrections).

Dealing with DEM precision
**************************

While dealing with **accuracy** is quite straightforward as it consists of minimizing the differences (biases) between several datasets, assessing the **precision** of DEMs can be much more complex.
Measurement errors of a DEM cannot be quantified by a simple difference and require statistical inference.

The **precision** of DEMs has generally been reported by a single metric, for example: :math:`\pm` 2 m, but recent studies have shown the limitations of simplified metrics and provide more statistically advanced methods.
However, the lack of a simple implementation in a modern programming language makes these methods hard to reproduce and validate.
One of the goals of ``xdem`` is to simplify state-of-the-art statistical measures, to allow accurate DEM uncertainty estimation for everyone, regardless of one's statistical talent.

The tools for quantifying DEM precision are described in :ref:`spatialstats`.

