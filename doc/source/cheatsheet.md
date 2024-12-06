---
file_format: mystnb
mystnb:
  execution_timeout: 60
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
(cheatsheet)=

# Cheatsheet: How to correct... ?

In elevation data analysis, the problem generally starts with identifying what correction method to apply when
observing a specific pattern of error in your own data.

Below, we summarize a cheatsheet that links what method is likely to correct a pattern of error you can visually
identify on **a map of elevation differences with another elevation dataset (looking at static surfaces)**!

## Cheatsheet

The patterns of errors categories listed in this spreadsheet **are linked to visual examples further below** that
you can use to compare to your own elevation differences.

```{list-table}
   :widths: 1 2 2 2
   :header-rows: 1
   :stub-columns: 1

   * - Pattern
     - Description
     - Cause and correction
     - Notes
   * - {ref}`sharp-landforms`
     - Positive and negative errors that are larger near high slopes and symmetric with opposite slope orientation, making landforms appear visually.
     - Likely horizontal shift due to geopositioning errors, use a {ref}`coregistration` such as {class}`~xdem.coreg.NuthKaab`.
     - Even a tiny horizontal misalignment can be visually identified! To not confuse with {ref}`peak-cavity`.
   * - {ref}`peak-cavity`
     - Positive and negative errors, with one sign located exclusively near peaks and the other exclusively near cavities.
     - Likely resolution-type errors, use a {ref}`biascorr` such as {class}`~xdem.coreg.TerrainBias`.
     - Can be over-corrected, sometimes better to simply ignore during analysis. Or avoid by downsampling all elevation data to the lowest resolution, rather than upsampling to the highest.
   * - {ref}`smooth-large-field`
     - Smooth offsets varying at scale of 10 km+, often same sign (either positive or negative).
     - Likely wrong {ref}`vertical-ref`, can set and transform with {func}`~xdem.DEM.set_vcrs` and {func}`~xdem.DEM.to_vcrs`.
     - Vertical references often only exists in a user guide, they are not coded in the raster CRS and need to be set manually.
   * - {ref}`ramp-or-dome`
     - Ramping errors, often near the edge of the data extent, sometimes with a center dome.
     - Likely ramp/rotations due to camera errors, use either a {ref}`coregistration` such as {class}`~xdem.coreg.ICP` or a {ref}`biascorr` such as {class}`~xdem.coreg.Deramp`.
     - Can sometimes be more rigorously fixed ahead of DEM generation with bundle adjustment.
   * - {ref}`undulations`
     - Positive and negative errors undulating patterns at one or several frequencies well larger than pixel size.
     - Likely jitter-type errors, use a {ref}`biascorr` such as {class}`~xdem.coreg.DirectionalBias`.
     - Can sometimes be more rigorously fixed ahead of DEM generation with jitter correction.
   * - {ref}`point-alternating`
     - Point data errors that alternate between negative and positive, higher on steeper slopes.
     - Likely wrong point-raster comparison, use [point interpolation or reduction on the raster instead](https://geoutils.readthedocs.io/en/stable/raster_vector_point.html#rasterpoint-operations) such as {func}`~xdem.DEM.interp_points`.
     - Rasterizing point data introduces spatially correlated random errors, instead it is recommended to interpolate raster data at the point coordinates.
```

## Visual patterns of errors

```{important}
The patterns of errors below are **created synthetically to examplify the effect of a single source of error**.
In your own elevation differences, those will be mixed together and with random errors inherent to the data.

For examples on real data, see the **{ref}`examples-basic` and {ref}`examples-advanced` gallery examples**!
```

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9  # Default 10 is a bit too big for coregistration plots
```

It is often crucial to relate the location of your errors on static surfaces to the terrain distribution
(in particular, its slope and curvature), which can usually be inferred visually from a hillshade.


```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example file"
:  code_prompt_hide: "Hide the code for opening example file"

import xdem
import geoutils as gu
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Open an example DEM
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
```

```{code-cell} ipython3
hs = dem.hillshade()
hs.plot(cmap="Greys_r", cbar_title="Hillshade")
```

(sharp-landforms)=
### Sharp landforms

Example of sharp landforms appearing with a horizontal shift due to geolocation errors. We here translate the DEM
horizontally by 1/10th of a pixel, for a pixel resolution of 20 m.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code to simulate horizontal shift errors"
:  code_prompt_hide: "Hide code to simulate horizontal shift errors"

# Simulate a translation of 1/10th of a pixel
x_shift = 0.1
y_shift = 0.1
dem_shift = dem.translate(x_shift, y_shift, distance_unit="pixel")

# Resample and plot the elevation differences of the horizontal shift
dh = dem - dem_shift.reproject(dem)
dh.plot(cmap='RdYlBu', vmin=-3, vmax=3, cbar_title="Elevation differences of\nhorizontal shift (m)")
```

(peak-cavity)=
### Peak cuts and cavity fills

Example of peak cutting and cavity filling errors. We here downsampled our DEM from 20 m to 100 m to simulate a lower
native resolution, then upsample it again to 20 m, to show the errors affect areas near high curvatures such as
peaks and cavities.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code to simulate resolution errors"
:  code_prompt_hide: "Hide code to simulate resolution errors"

# Downsample DEM (bilinear)
dem_100m = dem.reproject(res=100)

# Upsample (bilinear again) and compare
dh = dem - dem_100m.reproject(dem)
dh.plot(cmap='RdYlBu', vmin=-40, vmax=40, cbar_title="Elevation differences of\nresolution change (m)")
```

(smooth-large-field)=
### Smooth-field offset

Example of smooth large offset field created by a wrong vertical CRS. We here show the difference due to the EGM96
geoid added on top of the ellipsoid.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code to simulate vertical referencing errors"
:  code_prompt_hide: "Hide code to simulate vertical referencing errors"

# Set current vertical CRS as ellipsoid
dem.set_vcrs("EGM96")
# Transform vertical reference to geoid
trans_dem = dem.to_vcrs("Ellipsoid")

# Plot the elevation differences of the vertical transformation
dh = dem - trans_dem
dh.plot(cmap='RdYlBu', cbar_title="Elevation differences of\nvertical transform (m)")
```

(ramp-or-dome)=
### Ramp or dome

Example of ramp created by a rotation due to camera errors. We here show just a slight rotation of 0.02 degrees.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code to simulate rotation errors"
:  code_prompt_hide: "Hide code to simulate rotation errors"

# Apply a rotation of 0.02 degrees
rotation = np.deg2rad(0.02)
# Affine matrix for 3D transformation
matrix = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(rotation), -np.sin(rotation), 0],
        [0, np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 0, 1],
    ]
)
# Select a centroid
centroid = [dem.bounds.left + 5000, dem.bounds.top - 2000, np.median(dem) + 100]
# We rotate the elevation data
dem_rotated = xdem.coreg.apply_matrix(dem, matrix, centroid=centroid)

# Plot the elevation differences of the rotation
dh = dem - dem_rotated
dh.plot(cmap='RdYlBu', cbar_title="Elevation differences of\nrotation (m)")
```


(undulations)=
### Undulations

Example of undulations resembling jitter errors. We here artificially create a sinusoidal signal at a certain angle.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code to simulate undulation errors"
:  code_prompt_hide: "Hide code to simulate undulation errors"

# Get rotated coordinates along an angle
angle = -20
xx = gu.raster.get_xy_rotated(dem, along_track_angle=angle)[0]

# One sinusoid: amplitude, phases and frequencies
params = np.array([(2, 3000, np.pi)]).flatten()

# Create a sinusoidal bias and add to the DEM
from xdem.fit import sumsin_1d
synthetic_bias_arr = sumsin_1d(xx.flatten(), *params).reshape(np.shape(dem.data))

# Plot the elevation differences of the undulations
synthetic_bias = dem.copy(new_array=synthetic_bias_arr)
synthetic_bias.plot(cmap='RdYlBu', vmin=-3, vmax=3, cbar_title="Elevation differences of\nundulations (m)")
```

(point-alternating)=
### Point alternating

An example of alternating point errors created by wrong point-raster comparison by rasterization of the points,
which are especially large around steep slopes.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code to simulate point-raster comparison errors"
:  code_prompt_hide: "Hide code to simulate point-raster comparison errors"

# Simulate swath coordinates of an elevation point cloud
x = np.linspace(dem.bounds.left, dem.bounds.right, 100)
y = np.linspace(dem.bounds.top - 5000, dem.bounds.bottom + 5000, 100)

# Interpolate DEM at these coordinates to build the point cloud
# (to approximate the real elevation at these coordinates,
# which has negligible impact compared to rasterization)
z = dem.interp_points((x,y))
epc = gu.Vector(gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x, y=y, crs=dem.crs), data={"z": z}))

# Rasterize point cloud back on the DEM grid
epc_rast = epc.rasterize(dem, in_value=z, out_value=np.nan)

# For easier plotting, convert the valid dh values to points
dh = dem - epc_rast
dh_pc = dh.to_pointcloud(data_column_name="dh")

# Plot the elevation differences of the rasterization on top of a hillshade
hs.plot(cmap="Greys_r", add_cbar=False)
dh_pc.plot(column="dh", cmap='RdYlBu', vmin=-10, vmax=10, legend=True, cbar_title="Elevation differences of\npoint-raster differencing (m)")
```
