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

(cli-topo)=
# Topography workflow

The `topo` workflow of xDEM performs a **topographical summary of an elevation dataset**.

This summary derives a series of terrain attributes (e.g. slope, hillshade, aspect, etc.) with statistics (e.g. mean, max, min, etc.).


```{caution}
This workflow is still in development and its interface may thus change rapidly. It currently includes only classical terrain attributes.
```

## Basic usage

Below is an example of basic usage for the `topo` workflow, including how to build your **configuration file**, and how to run `xdem topo` and interpret its **logging output and report**.

### Configuration file

The configuration file of the `topo` workflow contains four categories: `inputs`, `outputs`, `statistics` and `terrain_attributes`.
Only the **path to the elevation dataset** in the `inputs` section is a **required** parameter. All others can be left out, in which case they default to pre-defined parameters.

By default, the `topo` workflow derives **slope, aspect and max. curvature**, computes **15 different statistics**, and saves **level-1 (intermediate) outputs in `./outputs`** .

In the example of configuration file below, we define:
- The **path to the elevation dataset** which is **required**,
- The **path to a mask**, to exclude terrain during the analysis,
- The **path to an output directory** where the results will be written,
- The **specific list of terrain attributes** to derive,
- The **specific list of statistics** to compute after/before coregistration.

```{code-cell} bash
:tags: [remove-cell]
cd _workflows/
```
```{literalinclude} _workflows/topo_config.yaml
:language: yaml
```
For details on the individual parameters, see {ref}`params-topo` further below. For generic information on the YAML configuration file, see the {ref}`cli` page.


```{tip}
To display a template of all available configuration options for the YAML file, use the `--display_template_config` argument
```

### Running the workflow

Now that we have this configuration file, we run the workflow.

```{code-cell} python
:tags: [hide-output]
:mystnb:
:  code_prompt_show: "Show logging output"
:  code_prompt_hide: "Hide logging output"

!xdem topo --config topo_config.yaml

print("lol8")
```

The logging output is printed in the terminal, showing the different steps.

```{code-cell} python
:tags: [remove-cell]

# Copy output folder to build directory to be able to embed HTML directly below
import os
import shutil
from pathlib import Path

# Source + destination
src = Path("outputs_topo")
dst = Path("../../build/_workflows/outputs_topo")

# Ensure clean copy (important for incremental builds)
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)

# Copy entire directory tree
shutil.copytree(src, dst)
```

Finally, a report is created (both in HTML and PDF formats) in the output directory.

We can visualize the report of our workflow above:

```{raw} html
<iframe src="_workflows/outputs_topo/report.html" width="100%" height="800"></iframe>
```

## Workflow details

This section describes in detail the steps for the `topo` workflow, including a summary chart and all parameters of its CLI interface.

### Chart of steps

The `topo` workflow is described by the following chart:

:::{figure} imgs/topo_workflow_pipeline.png
:width: 100%
:::

(params-topo)=
### Configuration parameters

```{eval-rst}
.. tabs::

   .. tab:: inputs

     **Required:** Yes

     Elevation input information.

     .. csv-table:: Inputs parameters for elevation
        :header: "Name", "Description", "Type", "Default value", "Required"
        :widths: 20, 40, 20, 10, 10

        "``path_to_elev``", "Path to reference elevation", "str", "", "Yes"
        "``force_source_nodata``", "No data elevation", "int", "", "No"
        "``path_to_mask``", "Path to mask associated to the elevation", "str", "", "No"
        "``from_vcrs``", "Original vcrs", "int, str", None, "No"
        "``to_vcrs``", "Destination vcrs", "int, str", None, "No"
        "``downsample``", "Downsampling elevation factor >= 1", "int, float", 1, "No"

     .. note::
        For transforming between vertical CRS with ``from_vcrs``/``to_vcrs`` please refer to :doc:`vertical_ref`.
        The ``downsample`` parameter allows the user to resample the elevation by a round factor. The default value of 1 means no downsampling.


     .. code-block:: yaml

         inputs:
           reference_elev:
             path_to_elev: "path_to/reference_elev.tif"
             force_source_nodata: -32768
             from_vcrs: None
             to_vcrs: None

   .. tab:: statistics

      **Required:** No

      Statistics step information. This section relates to the computed statistics:
        1. If no block is specified, all available statistics are calculated by default.
        [mean, median, max, min, sum, sum of squares, 90th percentile, LE90, nmad, rmse, std, valid count, total count,
        percentage valid points, inter quartile range]

        2. If a block is specified but no statistics are provided, then no statistics will be computed.

        3. If a block is specified and some statistics are provided, then only these statistics are computed.

      .. code-block:: yaml

         statistics:
           - min
           - max
           - mean

      If a mask is provided, the statistics are also computed inside the mask.

   .. tab:: terrain attributes

      **Required:** No

      List or set of dictionaries for extra information.

      .. note::
          - If no block is specified, slope, aspect, and curvature attributes are calculated by default.
          - If a block is specified but no information is provided, then no attributes will be calculated.

      .. code-block:: yaml

         terrain_attributes:
           - hillshade
           - slope

      or

      .. code-block:: yaml

         terrain_attributes:
           hillshade:
               extra_information:
           slope:
              extra_information:
           aspect:
            extra_information:
                degrees: False

      .. note::
        The data provided in extra_information is not checked for errors before executing the code.
        Its use is entirely the responsibility of the user.

   .. tab:: outputs

    **Required:** No

    Outputs information.

    Operates by levels:

    1. Level 1 → aligned elevation only
    2. Level 2 → more detailed output

    .. csv-table:: Outputs parameters
       :header: "Name", "Description", "Type", "Default value", "Available Value", "Required"
       :widths: 20, 40, 10, 10, 10, 10

       "``path``", "Path for outputs", "str", "outputs", "", "No"
       "``level``", "Level for detailed outputs", "int", "1", "1 or 2", "No"

    .. code-block:: yaml

      outputs:
          level : 1
          path : "path_to/outputs"

    Tree of outputs for level 1

    .. code-block:: text

      - root
        ├─ tables
        │   ├─ elev_stats.csv
        │   └─ elev_with_mask_stats.csv
        ├─ plots
        │   ├─ elev_map.png
        │   ├─ masked_elev_map.png (if mask_elev is given in input)
        │   └─ terrain_attributes_map.png
        ├─ rasters
        ├─ report.html
        ├─ report.pdf
        └─ used_config.yaml

    Tree of outputs for level 2

    .. code-block:: text

      - root
        ├─ tables
        │   ├─ elev_stats.csv
        │   └─ elev_with_mask_stats.csv
        ├─ plots
        │   ├─ elev_map.png
        │   ├─ masked_elev_map.png (if mask_elev is given in input)
        │   └─ terrain_attributes_map.png
        ├─ rasters
        │   ├─ aspect.tif
        │   ├─ curvature.tif
        │   ├─ hillshade.tif
        │   ├─ rugosity.tif
        │   ├─ slope.tif
        │   ├─ terrain_ruggedness_index.tif
        │   └─ ...
        ├─ report.html
        ├─ report.pdf
        └─ used_config.yaml
```
