"""Plot an example of local hypsometric interpolation at Scott Turnerbreen, Svalbard."""

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np

import xdem

dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
outlines_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

ddem = xdem.dDEM(dem_2009 - dem_1990, start_time=np.datetime64("1990-08-01"), end_time=np.datetime64("2009-08-01"))

ddem.data /= 2009 - 1990

scott_1990 = outlines_1990.query("NAME == 'Scott Turnerbreen'")
mask = scott_1990.create_mask(ddem)

ddem_bins = xdem.volume.hypsometric_binning(ddem[mask], dem_2009[mask])
stds = xdem.volume.hypsometric_binning(ddem[mask], dem_2009[mask], aggregation_function=np.std)

plt.figure(figsize=(8, 8))
plt.grid(zorder=0)
plt.plot(ddem_bins["value"], ddem_bins.index.mid, linestyle="--", zorder=1)

plt.barh(
    y=ddem_bins.index.mid,
    width=stds["value"],
    left=ddem_bins["value"] - stds["value"] / 2,
    height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
    zorder=2,
    edgecolor="black",
)
for bin in ddem_bins.index:
    plt.vlines(ddem_bins.loc[bin, "value"], bin.left, bin.right, color="black", zorder=3)

plt.xlabel("Elevation change (m / a)")
plt.twiny()
plt.barh(
    y=ddem_bins.index.mid,
    width=ddem_bins["count"] / ddem_bins["count"].sum(),
    left=0,
    height=(ddem_bins.index.left - ddem_bins.index.right) * 1,
    zorder=2,
    alpha=0.2,
)
plt.xlabel("Normalized area distribution (hypsometry)")

plt.ylabel("Elevation (m a.s.l.)")

plt.tight_layout()
plt.show()
