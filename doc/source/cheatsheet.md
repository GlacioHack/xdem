(cheatsheet)=

# Cheatsheet: How to correct... ?

In elevation data analysis, the problem generally starts with identifying what correction method to apply when 
observing a specific pattern of error in your own data.

Below, we summarize a cheatsheet that links what method is likely to correct a pattern of error you can visually 
identify on **static surfaces of a map of elevation differences with another elevation dataset**!

## Cheatsheet

The patterns of errors categories listed in this spreadsheet **are linked to visual example further below**, so that 
you can use them as a reference to compare to your own elevation differences.

```{list-table}
   :widths: 1 2 2 2
   :header-rows: 1
   :stub-columns: 1

   * - Pattern
     - Description
     - Cause and correction
     - Notes
   * - {ref}`sharp-landforms`
     - Positive and negative errors that are larger near high slopes, making landforms appear visually. 
     - Likely horizontal shift due to geopositioning errors, use a {ref}`coregistration` such as {class}`~xdem.coreg.NuthKaab`.
     - Even a tiny horizontal misalignment (1/10th of a pixel) can be visually identified!
   * - {ref}`smooth-large-field`
     - Smooth offsets varying at scale of 10 km+, often same sign (either positive or negative).
     - Likely wrong {ref}`vertical-ref`, can set and transform with {func}`~xdem.DEM.set_vcrs` and {func}`~xdem.DEM.to_vcrs`.
     - Vertical references often only exists in a user guide, they are not coded in the raster CRS and need to be set manually.
   * - {ref}`ramp-or-dome`
     - Ramping errors, often near the edge of the data extent, sometimes with a center dome.
     - Likely ramp/rotations due to camera errors, use either a {ref}`coregistration` such as {class}`~xdem.coreg.ICP` or a {ref}`bias-correction` such as {class}`~xdem.coreg.Deramp`.
     - Can sometimes be more rigorously fixed ahead of DEM generation with bundle adjustment.
   * - {ref}`undulations`
     - Positive and negative errors undulating patterns at one or several frequencies well larger than pixel size.
     - Likely jitter-type errors, use a {ref}`bias-correction` such as {class}`~xdem.coreg.DirectionalBias`.
     - Can sometimes be more rigorously fixed ahead of DEM generation with jitter correction.
   * - {ref}`point-oscillation`
     - Point data errors that oscillate between negative and positive.
     - Likely wrong point-raster comparison, use [point interpolation or reduction on the raster instead](https://geoutils.readthedocs.io/en/stable/raster_vector_point.html#rasterpoint-operations).
     - Rasterizing point data introduces spatially correlated random errors, instead it is recommended to interpolate raster data at the point coordinates.
```

## Visual patterns of errors

(sharp-landforms)=
### Sharp landforms

```{code-cell} ipython3
# Simulate a translation
x_shift = 5
y_shift = 5
dem_shift = dem.translate(x_shift, y_shift)

# Resample and plot
dh = dem - dem_shift.reproject(dem)
dh.plot(cmap='RdYlBu', vmin=-5, vmax=5, ax=ax[1], cbar_title="Elevation differences (m)")
```

(smooth-large-field)=
### Smooth large-scale offset field

(ramp-or-dome)=
### Ramp or dome

(undulations)=
### Undulations

(point-oscillation)=
### Point oscillation

