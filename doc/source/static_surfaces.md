(static-surfaces)=

# Static surfaces as error proxy

## The great benefactor of quantitative elevation analysis 

Elevation data benefits from an uncommon asset, which is that large proportions of planetary surface elevations 
remain virtually unchanged through time. Those static surfaces, sometimes also referred to as "stable terrain", represents 
the surfaces that have supposedly not been subject to any elevation change. They generally refer to bare-rock, grasslands, 
often isolated by excluding dynamic surfaces such as glaciers, snow and forests. If small proportions of static surfaces 
are not masked, they are generally filtered out by robust estimators (see {ref}`robust-stats`).

## Use for coregistration and further uncertainty analysis

Elevation data can rarely be compared to simultaneous acquisitions to assess systematic errors (assessed by coregistration) and 
random errors (assessed by further uncertainty analysis). This is where static surfaces come to the rescue, and can act as an error 
proxy. By assuming no changes on these surfaces, and that they have the same error structure as dynamic surfaces, it becomes 
possible to perform coregistration, bias correction and further uncertainty analysis.

### For coregistration and bias correction (systematic errors)



### For further uncertainty analysis (random errors)

To statistically infer the random error of elevation data, it is compared against independent elevation observations.

Significant measurement errors can originate from both sets of elevation observations, and the analysis of differences will represent the mixed precision of the two.
As there is no reason for a dependency between the elevation data sets, the analysis of elevation differences yields:

$$
\sigma_{dh} = \sigma_{h_{\textrm{precision1}} - h_{\textrm{precision2}}} = \sqrt{\sigma_{h_{\textrm{precision1}}}^{2} + \sigma_{h_{\textrm{precision2}}}^{2}}
$$

If the other elevation data is known to be of higher-precision, one can assume that the analysis of differences will represent only the precision of the rougher DEM.

$$
\sigma_{dh} = \sigma_{h_{\textrm{higher precision}} - h_{\textrm{lower precision}}} \approx \sigma_{h_{\textrm{lower precision}}}
$$
