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

# Finding reference data

Most analyses in xDEM require **independent, high-precision elevation data** as reference for
3D coregistration, bias corrections or uncertainty analysis.

Many datasets can serve as reference, such as **[ICESat-2](https://icesat-2.gsfc.nasa.gov/)** and
**[GEDI](https://gedi.umd.edu/)** for their broad coverage, or [local lidar surveys](https://portal.opentopography.org/datasets).
[SlideRule](https://slideruleearth.io/) is one of the few packages offering **efficient cloud access** across these datasets.

Below, we first list **potential reference datasets** and how to access them. We then examplify how to retrieve ICESat-2 points
with SlideRule, then **pair its outputs with xDEM** for coregistration and uncertainty analysis.

```{code-cell} ipython3
:tags: [remove-cell]

%matplotlib inline

# To get a good resolution for displayed figures
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9  # Default 10 is a bit too big for coregistration plots
```

## Choosing a reference dataset

Several open **lidar, altimetry and DEM datasets** can serve as reference data:

```{table}
:widths: auto
:width: 100%

| Dataset | Coverage and type | Access |
| --- | --- | --- |
| Airborne lidar | Local/regional surveys | [OpenTopography](https://portal.opentopography.org/datasets), [USGS](https://www.usgs.gov/tools/lidarexplorer) |
| USGS 3DEP DEM | 1-m terrain models (U.S.) | [SlideRule](https://docs.slideruleearth.io/user-guide/raster-sampling), [USGS](https://www.usgs.gov/tools/lidarexplorer) |
| ICESat-2 | Global along-track altimetry | [SlideRule](https://docs.slideruleearth.io/user-guide/icesat2), [icepyx](https://icepyx.readthedocs.io/en/latest/) |
| GEDI | Low- to mid-latitude altimetry | [SlideRule](https://docs.slideruleearth.io/user-guide/gedi), [Earthdata](https://search.earthdata.nasa.gov/) |
| ArcticDEM/REMA | 2-m polar stereo DEMs | [SlideRule](https://docs.slideruleearth.io/user-guide/raster-sampling), [PGC](https://www.pgc.umn.edu/guides/stereo-derived-elevation-models/pgc-dem-products-arcticdem-rema-and-earthdem/), [pDEMtools](https://pdemtools.readthedocs.io/en/latest/) |
```

When datasets have **enough static surfaces in common**, use any acquisition date for the reference.
If static surfaces are scarce, **acquisitions close in time** might be necessary to allow dynamic surfaces to serve as an error proxy.
See {ref}`static-surfaces` for details.

## Retrieving reference data

We retrieve reference data using **[SlideRule](https://slideruleearth.io/)** for a chosen **area and time range**.

Here, we choose all ICESat-2 ATL06 elevations from 2019–2020 intersecting the area of our example DEM in Svalbard, with quality filtering,
totalling around 176,000 points.

```{note}
The SlideRule code snippet below is not run when building the documentation and **may be out-of-date**.
Refer to the [SlideRule documentation](https://docs.slideruleearth.io/) for current usage!

Note that this snippet is also **optional for running the example** further below, as its output is stored and downloaded as an {ref}`xDEM data example <data-example>`.
```

```python
import numpy as np
import xdem
from sliderule import icesat2, sliderule

# Connect to SlideRule
sliderule.init("slideruleearth.io")

# Define the query area from the DEM bounds
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
region = sliderule.toregion(list(dem.get_bbox_projected(out_crs=4326)))["poly"]

# Select high-confidence land returns during 2019–2020
params = {
    "poly": region,
    "srt": icesat2.SRT_LAND, "cnf": icesat2.CNF_SURFACE_HIGH,
    "ats": 20.0, "cnt": 10,
    "t0": "2019-01-01T00:00:00Z", "t1": "2021-01-01T00:00:00Z",
}

# Retrieve elevations and remove invalid or low-quality estimates
gdf = icesat2.atl06sp(params)
gdf = gdf[np.isfinite(gdf["h_li"]) & (gdf["atl06_quality_summary"] == 0)]

# Save the point cloud
gdf.to_file("EPC_IS2.gpkg")
```

## Preparing datasets

Before comparison, both datasets need a **common horizontal and vertical reference**.

We open the saved ICESat-2 points as an {ref}`EPC <epc-class>` and convert the example DEM from EGM96
to ellipsoidal heights. The points are then projected to the DEM CRS. See {ref}`vertical-ref` for details.

```{note}
If datasets are not in the same projection, later steps will still run, **reprojecting on the reference** by default!
```

```{code-cell} ipython3
import geoutils as gu
import numpy as np
import xdem

# Open example DEM
dem = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"), vcrs="EGM96")
dem = dem.to_vcrs("Ellipsoid")

# Open saved ICESat-2 points and project their coordinates
points = xdem.EPC(xdem.examples.get_path("longyearbyen_epc"), data_name="h_li")
points = points.reproject(crs=dem.crs)

# Prepare a glacier mask, as elevation changes are not representative of DEM errors
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
inlier_mask = ~glacier_outlines.create_mask(dem)
```

We can view the DEM, ICESat-2 elevations and glacier outlines later used to select static surfaces:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
dem.plot(ax=ax, cmap="Greys_r", cbar_title="DEM elevation (m)", title="DEM, ICESat-2 and glacier outlines")
points.plot(
    ref=dem, ax=ax, cmap="terrain", markersize=0.5, edgecolor="black", linewidth=0.1, label="ICESat-2",
    cbar_title="ICESat-2 elevation (m)", cax=ax.inset_axes((1.28, 0, 0.05, 1)),
)
glacier_outlines.plot(ref=dem, ax=ax, fc="none", ec="black", linewidth=0.7)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles + [Patch(fc="none", ec="black", label="Glacier outlines")], markerscale=4)
ax.set(xlabel="", ylabel="")
plt.show()
```

## Coregistration

Coregistration corrects **systematic rigid offsets** between the DEM and reference points.
Here, we align the DEM on static surfaces with {meth}`~xdem.DEM.coregister_3d` and the {ref}`Least Z-difference method <lzd>`.

```{code-cell} ipython3
# Run LZD coregistration
lzd = xdem.coreg.LZD(trim_residuals=True)
aligned_dem = dem.coregister_3d(
    points, lzd, inlier_mask=inlier_mask, random_state=42,
)

# Print the estimated translation parameters
for name in ["shift_x", "shift_y", "shift_z"]:
    print(f"{name}: {lzd.meta['outputs']['affine'][name]:.2f} m")
```

Let's compare elevation differences on static surfaces before and after coregistration:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show plotting code"
:  code_prompt_hide: "Hide plotting code"

# Derive hillshade for background
hs = dem.hillshade()
static_points = points[inlier_mask.interp_at_points(points, method="nearest", as_array=True) == 1]

# Get point differences before and after
dh_before = static_points - dem.interp_at_points(static_points)
dh_after = static_points - aligned_dem.interp_at_points(static_points)

# Remove outliers from both plots
is_inlier = np.ones(static_points.point_count, dtype=bool)
for differences in (dh_before.data, dh_after.data):
    is_inlier &= np.abs(differences - np.nanmedian(differences)) < 3 * gu.stats.nmad(differences)
dh_before, dh_after = dh_before[is_inlier], dh_after[is_inlier]

# Plot before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, dh, title in zip(axes, (dh_before, dh_after), ("Before coregistration", "After coregistration")):
    hs.plot(ax=ax, cmap="Greys_r", add_cbar=False, title=title)
    dh.plot(
        ax=ax, cmap="RdYlBu", vmin=-10, vmax=10, markersize=0.5,
        cbar_title="ICESat-2 − DEM (m)", legend_kwds={"extend": "both"},
    )
    ax.set(xlabel="", ylabel="")
plt.show()

for name, dh in (("dh_before", dh_before), ("dh_after", dh_after)):
    print(f"{name}: median = {np.nanmedian(dh.data):.2f} m, NMAD = {gu.stats.nmad(dh.data):.2f} m")
```

We can use {ref}`info() <coreg-meta>` to print a coregistration **summary**:

```{code-cell} ipython3
# Show full coregistration information
lzd.info()
```

See {ref}`coregistration` for more details.

## Uncertainty analysis

Uncertainty analysis estimates **error magnitude and spatial correlation** from differences on static surfaces.

For this, we simply use {meth}`~xdem.DEM.estimate_error_structure` from the aligned DEM, defining err the short-range error component
varies with slope, while the long-range error component has constant magnitude.

Importantly, we assume **errors from ICESat-2 are negligible** compared to DEM errors.

```{code-cell} ipython3
errors = aligned_dem.estimate_error_structure(
    points, mask=inlier_mask, precoreg=False, other_precision="negligible",
    predictors=("slope",),
    components={
        "short_range": {"magnitude": "heteroscedastic", "correlation": "stable"},
        "long_range": {"magnitude": "constant", "correlation": "gaussian"},
    },
    n_pairs=50_000, n_lags=15, random_state=42,
)
errors.plot(magnitude_kwargs={"xlabel": "Slope (°)", "ylabel": "Elevation error (1σ, m)"})
plt.show()
```

Snow depth and other surface changes could still affect the error proxy, even after quality filtering and using default robust estimators such as the NMAD and Dowd's.

We can then predict error magnitude across the DEM from its slope:

```{code-cell} ipython3
predictors = {"slope": aligned_dem.slope()}
elevation_error = errors.predict_magnitude(predictors, like=aligned_dem)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
elevation_error.plot(ax=ax, cmap="Purples", vmin=0, cbar_title="Elevation error (1σ, m)")
glacier_outlines.plot(ref=elevation_error, ax=ax, fc="none", ec="black", linewidth=0.7)
ax.set(title="DEM error estimated from ICESat-2", xlabel="", ylabel="")
plt.show()
```

The map extrapolates errors from static surfaces to terrain with similar slopes, including glaciers.

**Random error realizations** can account for the spatial correlation missing from the magnitude map: nearby errors vary together.

To examplify those, we draw three realizations below with {meth}`~xdem.DEM.random_field`, which can be used for
**numerical uncertainty propagation** through any analysis.

```{code-cell} ipython3
field_dem = aligned_dem.reproject(res=100)

field_predictors = {"slope": predictors["slope"].reproject(field_dem)}
fields = field_dem.random_field(errors, predictors=field_predictors, n_fields=3, backend="gstools", random_state=42)
perturbed_dems = [field_dem + field for field in fields]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True, layout="constrained")
for index, (field, ax) in enumerate(zip(fields, axes), start=1):
    field.plot(
        ax=ax, cmap="RdYlBu", vmin=-5, vmax=5, cbar_title="Error (m)", title=f"Realization {index}",
    )
plt.show()
```

**Analytical propagation** is also supported for many methods, and generally more efficient.

For instance, we can propagate uncertainty during coregistration, which also **improves the fit**:

```{code-cell} ipython3
coreg_uncertainty = xdem.uncertainty.propagate_uncertainty_coreg(
    static_points[is_inlier], dem, xdem.coreg.LZD(only_translation=True, subsample=200),
    error_structure=errors, error_predictors={"tba": {"slope": dem.slope()}},
    method="analytical", random_state=42,
)
# We use ±2σ, roughly a 95% confidence interval for Gaussian errors
for name, axis in zip(("tx", "ty", "tz"), "XYZ"):
    value, sigma = coreg_uncertainty.estimate[name], coreg_uncertainty.std[name]
    print(f"Translation in {axis}: {value:.2f} ± {2 * sigma:.2f} m")
```

See {ref}`uncertainty-coreg` in the {ref}`uncertainty` guide for more details.
