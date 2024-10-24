(static-surfaces)=

# Static surfaces as error proxy

Below, a short guide explaining the use of static surfaces as an error proxy for quantitative elevation analysis.

## The great benefactor of elevation analysis

Elevation data benefits from an uncommon asset, which is that **large proportions of planetary surface elevations
usually remain virtually unchanged through time** (at least, within decadal time scales). Those static surfaces,
sometimes also referred to as "stable terrain", generally refer to bare-rock, grasslands, and are often isolated by
excluding dynamic surfaces such as glaciers, snow, forests and cities. If small proportions of static surfaces are
not masked, they are generally filtered out by robust estimators (see {ref}`robust-estimators`).

:::{figure} imgs/stable_terrain_diagram.png
:width: 100%

Source: [Hugonnet et al. (2022)](https://doi.org/10.1109/jstars.2022.3188922).
:::

## Use for coregistration and further uncertainty analysis

Elevation data can rarely be compared to simultaneous acquisitions to assess its sources of error. This is
where **static surfaces come to the rescue, and can act as an error proxy**. By assuming no changes happened on these
surfaces, and that they have the same error structure as other surfaces, it becomes possible to perform
coregistration, bias-correction and further uncertainty analysis!

Below, we summarize the basic principles of how using static surfaces allows to perform coregistration and uncertainty analysis, and the related limitations.

### For coregistration and bias-correction (systematic errors)

**Static surfaces $S$ are key to a coregistration or bias correction transformation $C$** for which it is assumed that,
for two sets of elevation data $h_{1}$ and $h_{2}$, we have:

$$
(h_{1} - C(h_{2}))_{S} \approx 0
$$

and aim to find the best transformation $C$ to minimize this problem.

The above relation is not generally true for every pixel or footprint, however, due to random errors that
exist in all data. Consequently, we can only write:

$$
\textrm{mean} (h_{1} - C(h_{2}))_{S \gg r^{2}} \approx 0
$$

where $r$ is the correlation range of random errors, and $S \gg r^{2}$ assumes that static surfaces cover a domain much
larger than this correlation range. If static surfaces cover too small an area, coregistration will naturally become
less reliable.

```{note}
One of the objectives of xDEM is to allow to use knowledge on random errors to refine
coregistration for limited static surface areas, stay tuned!
```

### For further uncertainty analysis (random errors)

**Static surfaces are also essential for uncertainty analysis aiming to infer the random errors of elevation
data** but, in this case, we have to consider the effect of random errors from both sets of elevation data.

We first assume that elevation $h_{2}$ is now largely free of systematic errors after performing coregistration and
bias corrections $C$. The analysis of elevation differences $dh$ on static surfaces $S$ will represent the mixed random
errors of the two sets of data, that we can assume are statistically independent (if indeed acquired separately), which yields:

$$
\sigma_{dh, S} = \sigma_{h_{\textrm{1}} - h_{\textrm{2}}} = \sqrt{\sigma_{h_{\textrm{1}}}^{2} + \sigma_{h_{\textrm{2}}}^{2}}
$$

where $\sigma$ is the random error at any data point.

If one set of elevation data is known to be of much higher-precision, one can assume that the analysis of differences
will represent only the precision of the rougher DEM. For instance, $\sigma_{h_{1}} = 3 \sigma_{h_{2}}$ implies that more than
95% of $\sigma_{dh}$ comes from $\sigma_{h_{1}}$ from the above equation.

More generally:

$$
\sigma_{dh, S} = \sigma_{h_{\textrm{higher precision}} - h_{\textrm{lower precision}}} \approx \sigma_{h_{\textrm{lower precision}}}
$$

And the same applies to the spatial correlation of these random errors:

$$
\rho_{dh, S}(d) = \rho_{h_{\textrm{higher precision}} - h_{\textrm{lower precision}}}(d) \approx \rho_{h_{\textrm{lower precision}}}(d)
$$

where $\rho(d)$ is the spatial correlation, and $d$ is the spatial lag (distance between data points).

----------------

:::{admonition} References and more reading
:class: tip

Static surfaces can be used as a **proxy for assessing systematic and random errors**, which directly relates to
what is commonly referred to as accuracy and precision of elevation data, detailed in the **next guide page on {ref}`accuracy-precision`**.

See the **{ref}`spatial-stats` guide page** for more details on spatial statistics applied to uncertainty quantification.

**References:** [Hugonnet et al. (2022)](https://doi.org/10.1109/jstars.2022.3188922), Uncertainty analysis of digital elevation models by spatial inference from stable terrain.
:::
