"""Plot example of Dowd variogram as robust estimator for guide page."""

import matplotlib.pyplot as plt
import numpy as np
from skgstat import OrdinaryKriging, Variogram

import xdem

# Inspired by test_variogram in skgstat
# Generate some random but spatially correlated data with a range of ~20
np.random.seed(42)
c = np.random.default_rng(41).random((50, 2)) * 60
np.random.seed(42)
v = np.random.default_rng(42).normal(10, 4, 50)

V = Variogram(c, v).describe()
V["effective_range"] = 20
OK = OrdinaryKriging(V, coordinates=c, values=v)

c = np.array(np.meshgrid(np.arange(60), np.arange(60).T)).reshape(2, 60 * 60).T
dh = OK.transform(c)
dh = dh.reshape((60, 60))

# Add outliers
dh_outliers = dh.copy()
dh_outliers[0:6, 0:6] = -20

# Derive empirical variogram for Dowd and Matheron
df_inl_matheron = xdem.spatialstats.sample_empirical_variogram(
    dh, estimator="matheron", gsd=1, random_state=42, subsample=2000
)
df_inl_dowd = xdem.spatialstats.sample_empirical_variogram(dh, estimator="dowd", gsd=1, random_state=42, subsample=2000)

df_all_matheron = xdem.spatialstats.sample_empirical_variogram(
    dh_outliers, estimator="matheron", gsd=1, random_state=42, subsample=2000
)
df_all_dowd = xdem.spatialstats.sample_empirical_variogram(
    dh_outliers, estimator="dowd", gsd=1, random_state=42, subsample=2000
)

fig, ax = plt.subplots()

ax.plot(df_inl_matheron.lags, df_inl_matheron.exp, color="black", marker="x")
ax.plot(df_inl_dowd.lags, df_inl_dowd.exp, color="black", linestyle="dashed", marker="x")
ax.plot(df_all_matheron.lags, df_all_matheron.exp, color="red", marker="x")
ax.plot(df_all_dowd.lags, df_all_dowd.exp, color="red", linestyle="dashed", marker="x")


p1 = plt.plot([], [], color="darkgrey", label="Matheron", marker="x")
p2 = plt.plot([], [], color="darkgrey", linestyle="dashed", label="Dowd", marker="x")
first_legend = ax.legend(handles=[p[0] for p in [p1, p2]], loc="lower right")
ax.add_artist(first_legend)

p1 = plt.plot([], [], color="black", label="Inlier data")
p2 = plt.plot([], [], color="red", label="Inlier data + outlier data \n(1% of data replaced by 10 NMAD)")
second_legend = ax.legend(handles=[p[0] for p in [p1, p2]], loc="upper left")
ax.add_artist(second_legend)

ax.set_xlabel("Spatial lag (m)")
ax.set_ylabel("Variance of elevation changes (m²)")
ax.set_ylim((0, 15))
ax.set_xlim((0, 40))

ax.set_title("Effect of outliers on estimating\nspatial correlation")
