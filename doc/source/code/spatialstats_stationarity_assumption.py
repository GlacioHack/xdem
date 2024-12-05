"""Documentation plot illustrating stationarity of mean and variance"""

import matplotlib.pyplot as plt
import numpy as np

import xdem

# Example x vector
x = np.linspace(0, 1, 200)

sig = 0.2
rng = np.random.default_rng(42)
y_rand1 = rng.normal(0, sig, size=len(x))
y_rand2 = rng.normal(0, sig, size=len(x))
y_rand3 = rng.normal(0, sig, size=len(x))


y_mean = np.array([0.5 * xval - 0.25 if xval > 0.5 else 0.5 * (1 - xval) - 0.25 for xval in x])

fac_y_std = 0.5 + 2 * x


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 4))

# Stationary mean and variance
ax1.plot(x, y_rand1, color="tab:blue", linewidth=0.5)
ax1.hlines(0, xmin=0, xmax=1, color="black", label="Mean")
ax1.hlines(
    [-2 * sig, 2 * sig],
    xmin=0,
    xmax=1,
    colors=["tab:gray", "tab:gray"],
    label="Standard deviation",
    linestyles="dashed",
)
ax1.set_xlim((0, 1))
ax1.set_title("Stationary mean\nStationary variance")
# ax1.legend()
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.set_ylim((-1, 1))
ax1.set_xticks([])
ax1.set_yticks([])
ax1.plot(1, 0, ">k", transform=ax1.transAxes, clip_on=False)
ax1.plot(0, 1, "^k", transform=ax1.transAxes, clip_on=False)

# Non-stationary mean and stationary variance
ax2.plot(x, y_rand2 + y_mean, color="tab:olive", linewidth=0.5)
ax2.plot(x, y_mean, color="black", label="Mean")
ax2.plot(x, y_mean + 2 * sig, color="tab:gray", label="Dispersion (2$\\sigma$)", linestyle="dashed")
ax2.plot(x, y_mean - 2 * sig, color="tab:gray", linestyle="dashed")
ax2.set_xlim((0, 1))
ax2.set_title("Non-stationary mean\nStationary variance")
ax2.legend(loc="lower center")
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_ylim((-1, 1))
ax2.plot(1, 0, ">k", transform=ax2.transAxes, clip_on=False)
ax2.plot(0, 1, "^k", transform=ax2.transAxes, clip_on=False)

# Stationary mean and non-stationary variance
ax3.plot(x, y_rand3 * fac_y_std, color="tab:orange", linewidth=0.5)
ax3.hlines(0, xmin=0, xmax=1, color="black", label="Mean")
ax3.plot(x, 2 * sig * fac_y_std, color="tab:gray", linestyle="dashed")
ax3.plot(x, -2 * sig * fac_y_std, color="tab:gray", linestyle="dashed")
ax3.set_xlim((0, 1))
ax3.set_title("Stationary mean\nNon-stationary variance")
# ax1.legend()
ax3.spines["right"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylim((-1, 1))
ax3.plot(1, 0, ">k", transform=ax3.transAxes, clip_on=False)
ax3.plot(0, 1, "^k", transform=ax3.transAxes, clip_on=False)

plt.tight_layout()
plt.show()
