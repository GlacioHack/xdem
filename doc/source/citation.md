(citation)=

# Citing and method overview

When using a method implemented in xDEM, one should cite both the package and the original study behind the method (if there is any)!

## Citing xDEM

To cite the package, use the Zenodo DOI: [![Zenodo](https://zenodo.org/badge/doi/10.5281/zenodo.4809697.svg)](https://zenodo.org/doi/10.5281/zenodo.4809697).

## Method overview

For citation and other purposes, here's an overview of all methods implemented in the package and their reference, if it exists.
More details are available on each feature page!

### Terrain attributes

```{list-table}
   :widths: 1 2
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Reference
   * - Slope, aspect and hillshade
     - [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918) or [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107)
   * - Curvatures
     - [Zevenbergen and Thorne (1987)](http://dx.doi.org/10.1002/esp.3290120107) and [Moore et al. (1991)](https://doi.org/10.1002/hyp.3360050103)
   * - Topographic position index
     - [Weiss (2001)](http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf)
   * - Terrain ruggedness index
     - [Riley et al. (1999)](http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf) or [Wilson et al. (2007)](http://dx.doi.org/10.1080/01490410701295962)
   * - Roughness
     - [Dartnell (2000)](https://environment.sfsu.edu/node/11292)
   * - Rugosity
     - [Jenness (2004)](<https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2>)
   * - Fractal roughness
     - [Taud and Parrot (2005)](https://doi.org/10.4000/geomorphologie.622)
```

### Coregistration

```{list-table}
   :widths: 1 2
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Reference
   * - Nuth and Kääb
     - [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011)
   * - Dh minimization
     - N/A
   * - Least Z-difference
     - [Rosenholm and Torlegård (1988)](https://www.asprs.org/wp-content/uploads/pers/1988journal/oct/1988_oct_1385-1389.pdf)
   * - Iterative closest point
     - [Besl and McKay (1992)](https://doi.org/10.1117/12.57955), [Chen and Medioni (1992)](https://doi.org/10.1016/0262-8856(92)90066-C)
   * - Coherent point drift
     - [Myronenko and Song (2010)](https://doi.org/10.1109/TPAMI.2010.46)
   * - Vertical shift
     - N/A
```

### Bias-correction

```{list-table}
   :widths: 1 2
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Reference
   * - Deramp
     - N/A
   * - Directional bias (sinusoids)
     - [Girod et al. (2017)](https://doi.org/10.3390/rs9070704)
   * - Terrain bias (curvature)
     - [Gardelle et al. (2012)](https://doi.org/10.3189/2012JoG11J175)
   * - Terrain bias (elevation)
     - [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011)
   * - Vertical shift
     - N/A
```

### Gap-filling

```{list-table}
   :widths: 1 2
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Reference
   * - Bilinear
     - N/A
   * - Local and regional hypsometric
     - [Arendt et al. (2002)](https://doi.org/10.1126/science.1072497), [McNabb et al. (2019)](https://tc.copernicus.org/articles/13/895/2019/)
```


### Uncertainty analysis

```{list-table}
   :widths: 1 1
   :header-rows: 1
   :stub-columns: 1

   * - Method
     - Reference
   * - R2009 (nested ranges, circular approx.)
     - [Rolstad et al. (2009)](http://dx.doi.org/10.3189/002214309789470950)
   * - H2022 (heterosc., nested ranges, spatial propag.)
     - [Hugonnet et al. (2022)](http://dx.doi.org/10.1109/JSTARS.2022.3188922)
```
