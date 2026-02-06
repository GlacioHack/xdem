---
file_format: mystnb
mystnb:
  execution_timeout: 150
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
(cli-accuracy)=

# Accuracy workflow

The `accuracy` workflow of xDEM performs an **accuracy assessment of an elevation dataset**.

This assessment relies on analyzing the elevation differences to a secondary elevation dataset on static surfaces, as an error proxy
to perform coregistration and bias-correction (systematic errors) and to perform uncertainty quantification (structured random errors).

:::{admonition} More reading
:class: tip

For scientific background on this workflow, we recommend reading the **{ref}`static-surfaces`**, **{ref}`accuracy-precision`** and **{ref}`spatial-stats` guide pages**.

:::

```{caution}
This workflow is still in development and its interface may thus change rapidly. It currently includes only co-registration, and we are adding support for uncertainty quantification.
```

## Basic usage

Below is an example of basic usage for the `accuracy` workflow, including how to build your **configuration file**, and how to run `xdem accuracy` and interpret its **logging output and report**.

### Configuration file

The configuration file of the `accuracy` workflow contains four categories: `inputs`, `outputs`, `coregistration` and `statistics`.
Only the **paths to the two elevation datasets** in the `inputs` section are **required** parameters. All others can be left out, in which case they default to pre-defined parameters.

By default, the `accuracy` workflow **reprojects on the reference elevation dataset**, performs a **{ref}`nuthkaab` coregistration (horizontal and vertical translations) on all terrain**, computes **15 different statistics**, and saves **level-1 (intermediate) outputs in `./outputs`** .

In the example of configuration file below, we define:
- The **paths to the two elevation datasets** which are **required**,
- The **path to a vector of unstable surfaces** to exclude terrain during the analysis (polygon interior is excluded),
- The **path to an output directory** where the results will be written,
- The **name of the coregistration** method to run, and the subsample size to use,
- The **specific list of statistics** to compute after/before coregistration.

```{code-cell} bash
:tags: [remove-cell]
cd _workflows/
```

```{literalinclude} _workflows/accuracy_config.yaml
:language: yaml
```

For details on the individual parameters, see {ref}`params-accuracy` further below. For generic information on the YAML configuration file, see the {ref}`cli` page.

```{tip}
To display a template of all available configuration options for the YAML file, use the `--template-config` argument.
```

### Running the workflow

Now that we have this configuration file, we run the workflow.

```{code-cell} python
:tags: [hide-output]
:mystnb:
:  code_prompt_show: "Show logging output"
:  code_prompt_hide: "Hide logging output"

!xdem accuracy --config accuracy_config.yaml
```

The logging output is printed in the terminal, showing the different steps. For instance, we can see that the coregistration converged in three iterations.

Finally, a report is created (both in HTML and PDF formats) in the output directory.

We can visualize the report of our workflow above:

```{raw} html
<iframe src="_static/outputs_accuracy/report.pdf" width="100%" height="800"></iframe>
```

## Workflow details

This section describes in detail the steps for the `accuracy` workflow, including a summary chart and all parameters of its CLI interface.

### Chart of steps

The `accuracy` workflow, including its **inputs**, **outputs**, **processing steps** and **output detail level**, are described on the following chart:

:::{figure} imgs/accuracy_workflow_pipeline.png
:width: 100%
:::

(params-accuracy)=
### Configuration parameters

The parameters to pass to the `accuracy` workflow are divided into four categories:
- The `inputs` define file opening and pre-processing, including **two required paths to elevation data**, but also optional masking, CRS and nodata over-riding, and downsampling factor,
- The `outputs` define file writing and report generation, with various **levels** of detail for the produced outputs,
- The `coregistration` define steps for coregistration, directly **interfacing with the {ref}`coregistration` module** of xDEM,
- The `statistics` define steps for computing statistics before/after coregistration, directly **interfacing with the [Statistics](https://geoutils.readthedocs.io/en/stable/stats.html) module** of GeoUtils.

These categories and detailed parameter values are further detailed below:

:::::::{tab-set}
::::::{tab-item} `inputs`

**Required:** Yes

Elevation input information, split between reference and to-be-aligned elevation data.

:::::{tab-set}
::::{tab-item} `reference_elev`

:::{table} Inputs parameters for `reference_elev`
:widths: 20, 40, 20, 10, 10

| Name               | Description                              | Type       | Default | Required |
|-------------------|------------------------------------------|-----------|--------------|---------|
| `path_to_elev`     | Path to reference elevation              | str       |              | Yes     |
| `force_source_nodata` | No data elevation                        | int       |              | No      |
| `path_to_mask`     | Path to mask associated to the elevation | str |              | No      |
| `from_vcrs`        | Original vcrs                            | int, str  | None         | No      |
| `to_vcrs`          | Destination vcrs                         | int, str  | None         | No      |
| `downsample`       | Downsampling elevation factor >= 1       | int, float| 1            | No      |
:::

:::{note}
For transforming between vertical CRS with ``from_vcrs``/``to_vcrs`` please refer to {ref}`vertical-ref`.
The ``downsample`` parameter allows the user to resample the elevation by a round factor.
The default value of 1 means no downsampling.

And, if you want to test the CLI with xDEM example data, they can also refer to data alias.
Please refer to {ref}`data-example` to have more information.
:::

:::{code-block} yaml
inputs:
    reference_elev:
        path_to_elev: "path_to/reference_elev.tif"
        force_source_nodata: -32768
        from_vcrs: None
        to_vcrs: None
    to_be_aligned_elev:
        path_to_elev: "path_to/to_be_aligned_elev.tif"
        path_to_mask: "path_to/mask.tif"
:::

::::

::::{tab-item} `to_be_aligned_elev`

:::{table} Inputs parameters for `to_be_aligned_elev`
:widths: 20, 40, 20, 10, 10

| Name                  | Description                        | Type       | Default | Required |
|-----------------------|-----------------------------------|-----------|--------------|---------|
| `path_to_elev`         | Path to to-be-aligned elevation   | str       |              | Yes     |
| `force_source_nodata`  | No data elevation                 | int       |              | No      |
| `path_to_mask`         | Path to mask associated to the elevation | str |              | No      |
| `from_vcrs`            | Original vcrs                     | int, str  | None         | No      |
| `to_vcrs`              | Destination vcrs                  | int, str  | None         | No      |
| `downsample`           | Downsampling elevation factor >= 1 | int, float| 1           | No      |
:::

:::{note}
For transforming between vertical CRS with ``from_vcrs``/``to_vcrs`` please refer to {ref}`vertical-ref`.
The ``downsample`` parameter allows the user to resample the elevation by a round factor.
The default value of 1 means no downsampling.

And, if you want to test the CLI with xDEM example data, they can also refer to data alias.
Please refer to {ref}`data-example` to have more information.
:::

:::{code-block} yaml
inputs:
    reference_elev:
        path_to_elev: "path_to/reference_elev.tif"
        force_source_nodata: -32768
        from_vcrs: None
        to_vcrs: None
    to_be_aligned_elev:
        path_to_elev: "path_to/to_be_aligned_elev.tif"
        path_to_mask: "path_to/mask.tif"
:::

::::

:::::

::::::

::::::{tab-item} `coregistration`

**Required:** No

Coregistration step details.

You can create a pipeline with up to three coregistration steps by using the keys
``step_one``, ``step_two`` and ``step_three``.

Available coregistration method see {ref}`coregistration`.

:::{note}
By default, coregistration is carried out using the {ref}`nuthkaab` method.
To disable coregistration, set ``process: False`` in the configuration file.
:::

:::{table} Inputs parameters for coregistration
:widths: 30, 20, 10, 10, 10, 10, 10

| Parameter name           | Subparameter name        | Description                       | Type  | Default    | Available                       | Required |
|--------------------------|-------------------------|----------------------------------|-------|----------------|-------------------------------------|---------|
| `step_[one, two, three]` | `method`                | Name of coregistration method    | str   | NuthKaab        | Every available coregistration method | No      |
|                          | `extra_information`     | Extra parameters fitting with the method | dict  |                |                                     | No      |
| `sampling_grid`          |                         | Destination elevation for reprojection | str   | reference_elev | reference_elev or to_be_aligned_elev | No      |
| `process`                |                         | Activate the coregistration       | bool  | True           | True or False                        | No      |
:::

:::{note}
The data provided in ``extra_information`` is not checked for errors before executing the code.
Its use is entirely the responsibility of the user.
:::

:::{code-block} yaml
coregistration:
  step_one:
    method: "NuthKaab"
    extra_information: {"max_iterations": 10}
  step_two:
    method: "DHMinimize"
  sampling_grid: "reference_elev"
  process: True
:::

Other example:

:::{code-block} yaml
coregistration:
  step_one:
    method: "VerticalShift"
  sampling_grid: "reference_elev"
  process: True
:::

::::::

::::::{tab-item} `statistics`

**Required:** No

Statistics step information. This section relates to the computed statistics:

1. If no block is specified, all available statistics are calculated by default:

   [mean, median, max, min, sum, sum of squares, 90th percentile, LE90, nmad, rmse, std,
   valid count, total count, percentage valid points, inter quartile range]

2. If a block is specified but no statistics are provided, then no statistics will be computed.

3. If a block is specified and some statistics are provided, then only these statistics are computed.

:::{code-block} yaml
statistics:
  - min
  - max
  - mean
:::

If a mask is provided, the statistics are also computed inside the mask.

::::::

::::::{tab-item} `outputs`

**Required:** No

Outputs information. Operates by levels:

1. **Level 1** → aligned elevation only
2. **Level 2** → more detailed output

:::{table} Outputs parameters
:widths: 20, 40, 10, 10, 10, 10

| Name           | Description                   | Type | Default  | Available                          | Required |
|----------------|-------------------------------|------|---------------|----------------------------------------|---------|
| `path`         | Path for outputs               | str  | outputs/      |                                        | No      |
| `level`        | Level for detailed outputs     | int  | 1             | 1 or 2                                 | No      |
| `output_grid`  | Grid for outputs resampling    | str  | reference_elev | reference_elev or to_be_aligned_elev   | No      |
:::

:::{code-block} yaml
outputs:
    level: 1
    path: "path_to/outputs"
    output_grid: "reference_elev"
:::

Tree of outputs for level 1 (including coregistration step):

:::{code-block} text
- root
  ├─ tables
  │   └─ aligned_elev_stats.csv
  ├─ plots
  │   ├─ diff_elev_after_coreg_map.png
  │   ├─ diff_elev_before_coreg_map.png
  │   ├─ diff_elev_before_after_hist.png
  │   ├─ reference_elev_map.png
  │   ├─ masked_elev_map.png (if mask_elev is given in input)
  │   └─ to_be_aligned_elev_map.png
  ├─ rasters
  │   └─ aligned_elev.tif
  ├─ report.html
  ├─ report.pdf
  └─ used_config.yaml
:::

Tree of outputs for level 2 (including coregistration step):

:::{code-block} text
- root
  ├─ tables
  │   ├─ aligned_elev_stats.csv
  │   ├─ diff_elev_after_coreg_stats.csv
  │   ├─ diff_elev_before_coreg_stats.csv
  │   ├─ reference_elev_stats.csv
  │   └─ to_be_aligned_elev_stats.csv
  ├─ plots
  │   ├─ diff_elev_after_coreg_map.png
  │   ├─ diff_elev_before_coreg_map.png
  │   ├─ diff_elev_before_after_hist.png
  │   ├─ reference_elev_map.png
  │   ├─ masked_elev_map.png (if mask_elev is given in input)
  │   └─ to_be_aligned_elev_map.png
  ├─ rasters
  │   ├─ aligned_elev.tif
  │   ├─ diff_elev_after_coreg.tif
  │   ├─ diff_elev_before_coreg.tif
  │   └─ to_be_aligned_elev_reprojected.tif
  ├─ report.html
  ├─ report.pdf
  └─ used_config.yaml
:::

::::::

:::::::
