"""Documentation plot illustrating the link between variogram and covariance"""
import matplotlib.pyplot as plt
import numpy as np
import skgstat.models as models

# Example variogram function
def variogram_exp(h):
    val = models.exponential(h, 15, 10)
    return val

fig, ax = plt.subplots()
x = np.linspace(0,100,100)
ax.plot(x, variogram_exp(x), color='tab:blue', linewidth=2)
ax.plot(x, 10 - variogram_exp(x), color='black', linewidth=2)
ax.hlines(10, xmin=0, xmax=100, linestyles='dashed', colors='tab:red')
ax.text(75, variogram_exp(75)-1, 'Semi-variogram $\\gamma(l)$', ha='center', va='top', color='tab:blue')
ax.text(75, 10 - variogram_exp(75) + 1, 'Covariance $C(l) = \\sigma^{2} - \\gamma(l)$', ha='center', va='bottom', color='black')
ax.text(75, 11, 'Variance $\\sigma^{2}$', ha='center', va='bottom', color='tab:red')
ax.set_xlim((0, 100))
ax.set_ylim((0, 12))
ax.set_xlabel('Spatial lag $l$')
ax.set_ylabel('Variance of elevation differences (m²)')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.show()