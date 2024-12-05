---
file_format: mystnb
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
(elevation-point-cloud)=

# The elevation point cloud ({class}`~xdem.EPC`)

In construction, planned for 2025.

However, **elevation point clouds are already supported for coregistration and bias correction** by passing a {class}`geopandas.GeoDataFrame`
associated to an elevation column name argument `z_name` to {func}`~xdem.coreg.Coreg.fit_and_apply`.
