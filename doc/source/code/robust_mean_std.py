"""Plot example of NMAD/median as robust estimators for guide page."""

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np

# Create example distribution
dh_inliers = np.random.default_rng(42).normal(loc=-5, scale=3, size=10**6)

# Add outliers
dh_outliers = np.concatenate(
    (
        np.repeat(-34, 600),
        np.repeat(-33, 1800),
        np.repeat(-32, 3600),
        np.repeat(-31, 8500),
        np.repeat(-30, 15000),
        np.repeat(-29, 9000),
        np.repeat(-28, 3800),
        np.repeat(-27, 1900),
        np.repeat(-26, 700),
    )
)
dh_all = np.concatenate((dh_inliers, dh_outliers))

# Get traditional and robust statistics on all data
mean_dh = np.nanmean(dh_all)
median_dh = np.nanmedian(dh_all)

std_dh = np.nanstd(dh_all)
nmad_dh = gu.stats.nmad(dh_all)

# Get traditional and robust statistics on inlier data
mean_dh_in = np.nanmean(dh_inliers)
median_dh_in = np.nanmedian(dh_inliers)

std_dh_in = np.nanstd(dh_inliers)
nmad_dh_in = gu.stats.nmad(dh_inliers)

# Plot
fig, ax = plt.subplots()
h1 = ax.hist(dh_inliers, bins=np.arange(-40, 25), density=False, color="gray", label="Inlier data")
h2 = ax.hist(dh_outliers, bins=np.arange(-40, 25), density=False, color="red", label="Outlier data")

max_count = max(h1[0])
ax.vlines(x=[mean_dh_in, median_dh_in], ymin=0, ymax=max_count, colors=["tab:gray", "black"])
ax.vlines(
    x=[mean_dh_in - std_dh_in, mean_dh_in + std_dh_in, median_dh_in - nmad_dh_in, median_dh_in + nmad_dh_in],
    ymin=0,
    ymax=max_count,
    colors=["gray", "gray", "black", "black"],
    linestyles="dashed",
)

ax.vlines(x=[mean_dh, median_dh], ymin=0, ymax=max_count, colors=["red", "darkred"])
ax.vlines(
    x=[mean_dh - std_dh, mean_dh + std_dh, median_dh - nmad_dh, median_dh + nmad_dh],
    ymin=0,
    ymax=max_count,
    colors=["red", "red", "darkred", "darkred"],
    linestyles="dashed",
)

ax.set_xlim((-40, 25))
ax.set_xlabel("Elevation differences (m)")
ax.set_ylabel("Count")

from matplotlib.patches import Rectangle

handles = [
    Rectangle((0, 0), 1, 1, color=h1[-1][0].get_facecolor(), alpha=1),
    Rectangle((0, 0), 1, 1, color=h2[-1][0].get_facecolor(), alpha=1),
]
labels = ["Inlier data", "Outlier data"]

data_legend = ax.legend(handles=handles, labels=labels, loc="upper right")
ax.add_artist(data_legend)

# Legends
p1 = plt.plot([], [], color="red", label=f"Mean: {np.round(mean_dh, 2)} m")
p2 = plt.plot([], [], color="red", linestyle="dashed", label=f"±STD: {np.round(std_dh, 2)} m")
p3 = plt.plot([], [], color="darkred", label=f"Median: {np.round(median_dh, 2)} m")
p4 = plt.plot([], [], color="darkred", linestyle="dashed", label=f"±NMAD: {np.round(nmad_dh, 2)} m")
first_legend = ax.legend(handles=[p[0] for p in [p1, p2, p3, p4]], loc="center right", title="All data")
ax.add_artist(first_legend)

p1 = plt.plot([], [], color="gray", label=f"Mean: {np.round(mean_dh_in, 2)} m")
p2 = plt.plot([], [], color="gray", linestyle="dashed", label=f"±STD: {np.round(std_dh_in, 2)} m")
p3 = plt.plot([], [], color="black", label=f"Median: {np.round(median_dh_in, 2)} m")
p4 = plt.plot([], [], color="black", linestyle="dashed", label=f"±NMAD: {np.round(nmad_dh_in, 2)} m")
second_legend = ax.legend(handles=[p[0] for p in [p1, p2, p3, p4]], loc="center left", title="Inlier data")
ax.add_artist(second_legend)

ax.set_title("Effect of outliers on estimating\ncentral tendency and dispersion")
