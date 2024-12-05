(ecosystem)=

# Ecosystem

xDEM is but a single tool among a large landscape of open tools for geospatial elevation analysis! Below is a list of
other **tools that you might find useful to combine with xDEM**, in particular for retrieving elevation data or to perform complementary analysis.

```{seealso}
Tools listed below only relate to elevation data. To analyze georeferenced rasters, vectors and point cloud data,
check out **xDEM's sister-package [GeoUtils](https://geoutils.readthedocs.io/)**.
```
## Python

Great Python tools for **pre-processing and retrieving elevation data**:
- [SlideRule](https://slideruleearth.io/) to pre-process and retrieve high-resolution elevation data in the cloud, including in particular [ICESat-2](https://icesat-2.gsfc.nasa.gov/) and [GEDI](https://gedi.umd.edu/),
- [pDEMtools](https://pdemtools.readthedocs.io/en/latest/) to pre-process and retrieve [ArcticDEM](https://www.pgc.umn.edu/data/arcticdem/) and [REMA](https://www.pgc.umn.edu/data/rema/) high-resolution DEMs available in polar regions,
- [icepyx](https://icepyx.readthedocs.io/en/latest/) to retrieve ICESat-2 data.

Complementary Python tools to **analyze elevation data** are for instance:
- [PDAL](https://pdal.io/en/latest/) for working with dense elevation point clouds,
- [demcompare](https://demcompare.readthedocs.io/en/stable/) to compare two DEMs together,
- [RichDEM](https://richdem.readthedocs.io/en/latest/) for in-depth terrain analysis, with a large range of method including many relevant to hydrology.

## Julia

If you are working in Julia, the [Geomorphometry](https://github.com/Deltares/Geomorphometry.jl) package provides a
wide range of terrain analysis for elevation data.

## R

If you are working in R, the [MultiscaleDTM](https://ailich.github.io/MultiscaleDTM/) package provides modular tools
for terrain analysis at multiple scales!

## Other community resources

Whether to retrieve data among their wide range of open datasets, or to dive into their other resources, be sure to check out the
amazing [OpenTopography](https://opentopography.org/) and [OpenAltimetry](https://openaltimetry.earthdatacloud.nasa.gov/data/) efforts!
