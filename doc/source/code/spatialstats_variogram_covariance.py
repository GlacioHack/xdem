"""Documentation plot illustrating the link between variogram and covariance"""

import matplotlib.pyplot as plt
import numpy as np
from skgstat.models import exponential

# Example of variogram and covariance relationship with an exponential model form
fig, ax = plt.subplots()
x = np.linspace(0, 100, 100)
ax.plot(x, exponential(x, 15, 10), color="tab:blue", linewidth=2)
ax.plot(x, 10 - exponential(x, 15, 10), color="black", linewidth=2)
ax.hlines(10, xmin=0, xmax=100, linestyles="dashed", colors="tab:red")
ax.text(75, exponential(75, 15, 10) - 1, "Semi-variogram $\\gamma(l)$", ha="center", va="top", color="tab:blue")
ax.text(
    75,
    10 - exponential(75, 15, 10) + 1,
    "Covariance $C(l) = \\sigma^{2} - \\gamma(l)$",
    ha="center",
    va="bottom",
    color="black",
)
ax.text(75, 11, "Variance $\\sigma^{2}$", ha="center", va="bottom", color="tab:red")
ax.set_xlim((0, 100))
ax.set_ylim((0, 12))
ax.set_xlabel("Spatial lag $l$")
ax.set_ylabel("Variance of elevation differences (m²)")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.tight_layout()
plt.show()
