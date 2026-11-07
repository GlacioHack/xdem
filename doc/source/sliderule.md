---
file_format: mystnb
mystnb:
  execution_timeout: 300
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
(pair-reference-data)=

# Pair with reference data

Most analysis of **xDEM relies on independent, high-precision elevation data to use as reference**, whether for
coregistration, bias corrections or uncertainty analysis.

[SlideRule](https://slideruleearth.io/) provides access to elevation archives such as ICESat-2 and GEDI.
Here, we use **saved ICESat-2 ATL06 points** near Longyearbyen to coregister a DEM and estimate its error structure.
The example downloads the saved file from [xdem-data](https://github.com/GlacioHack/xdem-data),
so **SlideRule is not required** to run it or build this page.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9  # Default 10 is a bit too big for coregistration plots
```

## Loading reference data

The saved file contains 176,022 ATL06 land elevation points from 2019–2020, with finite elevations and
`atl06_quality_summary == 0`. The query and a script to retrieve the points are in
[xdem-data's download script](https://github.com/GlacioHack/xdem-data/blob/main/download_data.py).
SlideRule is needed only to repeat that query.

We open the file as an {class}`~xdem.EPC`, selecting `h_li` as the elevation column.
ICESat-2 heights are relative to the ellipsoid, whereas the example DEM heights are relative to EGM96.
We first convert the DEM to ellipsoidal heights, then project the point coordinates to the DEM CRS.
See {ref}`vertical-ref` for details on vertical referencing and geoid grids.

```{code-cell} ipython3
import geoutils as gu
import numpy as np
import xdem

# Open example DEM
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"), vcrs="EGM96")
dem = dem.to_vcrs("Ellipsoid")

# Open saved ICESat-2 points and project their coordinates
points = xdem.EPC(xdem.examples.get_path("longyearbyen_epc"), data_name="h_li")
points = points.reproject(crs=dem.crs)

# Exclude glaciers, whose elevation changes are not DEM errors
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
inlier_mask = ~glacier_outlines.create_mask(dem)
```

## Coregistration

Running the coregistration is as simple as passing the elevation point cloud to
{meth}`~xdem.coreg.Coreg.fit_and_apply`.

```{code-cell} ipython3
# Run a translation coregistration
nk = xdem.coreg.NuthKaab()
aligned_dem = nk.fit_and_apply(
    reference_elev=points, to_be_aligned_elev=dem,
    inlier_mask=inlier_mask, random_state=42,
)

# Print the estimated translation parameters
for name in ["shift_x", "shift_y", "shift_z"]:
    print(f"{name}: {nk.meta['outputs']['affine'][name]:.2f} m")
```

Let's visualize the improvement in elevation differences after coregistration, using only points on static surfaces.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Derive hillshade for background
hs = dem.hillshade()
static_points = inlier_mask.interp_at_points(points, method="nearest", as_array=True) == 1

# Get point differences before and after
comparison = points.gdf.loc[static_points].copy()
comparison["dh_before"] = points.data[static_points] - dem.interp_at_points(points, as_array=True)[static_points]
comparison["dh_after"] = points.data[static_points] - aligned_dem.interp_at_points(points, as_array=True)[static_points]

# Plot before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, column, title in zip(axes, ["dh_before", "dh_after"], ["Before coregistration", "After coregistration"]):
    hs.plot(ax=ax, cmap="Greys_r", add_cbar=False)
    gu.Vector(comparison).plot(
        column=column, cmap="RdYlBu", vmin=-10, vmax=10, ax=ax,
        markersize=0.5, cbar_title="ICESat-2 − DEM (m)",
    )
    ax.set_title(title)
plt.show()

for column in ["dh_before", "dh_after"]:
    values = comparison[column].to_numpy()
    print(f"{column}: median = {np.nanmedian(values):.2f} m, NMAD = {gu.stats.nmad(values):.2f} m")
```

We can print a coregistration summary using {meth}`~xdem.coreg.Coreg.info`.

```{code-cell} ipython3
# Show full coregistration information
nk.info()
```
