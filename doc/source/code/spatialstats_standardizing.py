"""Documentation plot illustrating standardization of a distribution"""

import matplotlib.pyplot as plt
import numpy as np

# Example x vector
mu = 15
sig = 5
rng = np.random.default_rng(42)
y = rng.normal(mu, sig, size=300)

fig, ax1 = plt.subplots(figsize=(8, 3))

# Original histogram
ax1.hist(y, color="tab:blue", edgecolor="white", linewidth=0.5, alpha=0.7)
ax1.vlines(mu, ymin=0, ymax=90, color="tab:blue", linestyle="dashed", lw=2)
ax1.vlines([mu - 2 * sig, mu + 2 * sig], ymin=0, ymax=90, colors=["tab:blue", "tab:blue"], linestyles="dotted", lw=2)
ax1.annotate(
    "Original\ndata $x$\n$\\mu_{x} = 15$\n$\\sigma_{x} = 5$",
    xy=(mu + 0.5, 85),
    xytext=(mu + 5, 110),
    arrowprops=dict(color="tab:blue", width=0.5, headwidth=8),
    color="tab:blue",
    fontweight="bold",
    ha="left",
)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_yticks([])
ax1.set_ylim((0, 130))

# Standardized histogram
ax1.hist((y - mu) / sig, color="tab:olive", edgecolor="white", linewidth=0.5, alpha=0.7)
ax1.vlines(0, ymin=0, ymax=90, color="tab:olive", linestyle="dashed", lw=2)
ax1.vlines([-2, 2], ymin=0, ymax=90, colors=["tab:olive", "tab:olive"], linestyles="dotted", lw=2)
ax1.annotate(
    "Standardized\ndata $z$\n$\\mu_{z} = 0$\n$\\sigma_{z} = 1$",
    xy=(-0.3, 85),
    xytext=(-5, 110),
    arrowprops=dict(color="tab:olive", width=0.5, headwidth=8),
    color="tab:olive",
    fontweight="bold",
    ha="left",
)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_yticks([])
ax1.set_ylim((0, 130))

ax1.annotate(
    "",
    xy=(0, 65),
    xytext=(mu, 65),
    arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", fc="w"),
    color="black",
)
ax1.text(
    mu / 2,
    90,
    "Standardization:\n$z = \\frac{x - \\mu}{\\sigma}$",
    color="black",
    ha="center",
    fontsize=14,
    fontweight="bold",
)
ax1.plot([], [], color="tab:gray", linestyle="dashed", label="Mean")
ax1.plot([], [], color="tab:gray", linestyle="dotted", label="Standard\ndeviation (2$\\sigma$)")
ax1.legend(loc="center right")
