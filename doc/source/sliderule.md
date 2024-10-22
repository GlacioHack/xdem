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

# Pair with SlideRule for reference

Most analysis of **xDEM relies on independent, high-precision elevation data to use as reference**, whether for 
coregistration, bias-corrections or uncertainty analysis.

[SlideRule](https://slideruleearth.io/) provides the ideal way to retrieve such data by accessing big data archives 
of high-precision elevations, such as ICESat-2 or GEDI, efficiently in the cloud.

Below, a short example to coregister and perform uncertainty analysis of a DEM with ICESat-2 ATL06.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9  # Default 10 is a bit too big for coregistration plots
```

```{code-cell} ipython3
from sliderule import sliderule, icesat2
import geoutils as gu
import xdem
import numpy as np

# Open example DEM
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))

# Derive inlier mask as glaciers
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
inlier_mask = ~glacier_outlines.create_mask(dem)

# Define region of interest as DEM footprint
bounds = list(dem.get_bounds_projected(4326))
region = sliderule.toregion(bounds)["poly"]

# Initiliaze SlideRule client
sliderule.init("slideruleearth.io")

# Define desired parameters for ICESat-2 ATL06
params = {
    "poly": region,
    "srt": icesat2.SRT_LAND,  # Surface-type
    "cnf": icesat2.CNF_SURFACE_HIGH,  # Confidence level
    "ats": 20.0,  # Minimum along-track spread
    "cnt": 10,  # Minimum count
}

# Request ATL06 subsetting in parallel
gdf = icesat2.atl06sp(params)
gdf = gdf[np.isfinite(gdf["h_li"])] # Keep valid data
gdf = gdf[gdf["atl06_quality_summary"]==0]  # Keep very high-confidence data
```

```{code-cell} ipython3
# Run a translation coregistration
nk = xdem.coreg.NuthKaab()
aligned_dem = nk.fit_and_apply(reference_elev=gdf, to_be_aligned_elev=dem, inlier_mask=inlier_mask, z_name="h_li")
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

hs = dem.hillshade()
# Plot before and after
import matplotlib.pyplot as plt
gdf = gdf.to_crs(dem.crs)
f, ax = plt.subplots(1, 2)
ax[0].set_title("Before translations")
hs.plot(cmap="Greys_r", add_cbar=False)
z_pc = dem.interp_points((gdf.geometry.x.values, gdf.geometry.y.values))
gdf["dh_bef"] = gdf["h_li"] - z_pc
gdf.plot(column="dh_bef", cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[0], markersize=0.5)
ax[1].set_title("After translations")
hs.plot(cmap="Greys_r", add_cbar=False)
z_pc_aligned = aligned_dem.interp_points((gdf.geometry.x.values, gdf.geometry.y.values))
gdf["dh_aft"] = gdf["h_li"] - z_pc_aligned
gdf.plot(column="dh_aft", cmap='RdYlBu', vmin=-30, vmax=30, ax=ax[1], markersize=0.5)
_ = ax[1].set_yticklabels([])
plt.savefig("/home/atom/ongoing/test.png", dpi=400)
```

```{code-cell} ipython3
# Run uncertainty analysis
sig_dem, rho_sig = aligned_dem.estimate_uncertainty(other_elev=gdf, z_name="h_li")
```