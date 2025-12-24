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

## Summary

The `accuracy` workflow of xDEM performs an **accuracy assessment of an elevation dataset**.

This assessment relies on analyzing the elevation differences to a secondary elevation dataset on static surfaces, as an error proxy 
to perform coregistration and bias-correction (systematic errors) and to perform uncertainty quantification (structured random errors).

:::{admonition} More reading
:class: tip

For scientific background on this workflow, we recommend reading the **{ref}`static-surfaces`**, **{ref}`accuracy-precision`** and **{ref}`spatial-stats` guide pages**.

:::

```{caution}
This workflow is still in development, and currently includes only co-registration. We are adding support for uncertainty quantification.
```

## Basic usage

### Configuration file

We use the following example configuration file:

```{code-cell} bash
:tags: [remove-cell]
cd _workflows/
```

```{literalinclude} _workflows/accuracy_config.yaml
:language: yaml
```

```{tip}

To display a template of all available configuration options for the YAML file, use the `--display_template_config` argument
```

And run the workflow:

```{code-cell} python
:tags: [hide-output]
:mystnb:
:  code_prompt_show: "Show logging output"
:  code_prompt_hide: "Hide logging output"

!xdem accuracy --config accuracy_config.yaml

print("lol31")
```

```{code-cell} python
:tags: [remove-cell]

# Copy output folder to build directory to be able to embed HMTL directly below
import os
import shutil
from pathlib import Path

# Source + destination
src = Path("outputs_accuracy")
dst = Path("../../build/_workflows/outputs_accuracy")

# Ensure clean copy (important for incremental builds)
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)

# Copy entire directory tree
shutil.copytree(src, dst)
```

### Generated report

```{raw} html
<iframe src="_workflows/outputs_accuracy/report.html" width="100%" height="600"></iframe>
```

## Workflow description

### Chart of steps

Here is a chart summarizing of the accuracy worfklow:

:::{figure} imgs/accuracy_workflow_pipeline.png
:width: 100%
:::

### Configuration parameters

The paramaters to pass to the `accuracy` workflow are divided into four categories:
- The `inputs` define file opening and pre-processing,
- The `outputs` define file writing and report generation,
- The `coregistration` define steps for coregistration,
- The `statistics` define steps for computing statistics before/after coregistration.

These parameters are detailed below:

```{eval-rst}
.. tabs::

   .. tab:: inputs

      **Required:** Yes

      Elevation input information.

      .. tabs::

        .. tab:: reference_elev

            .. csv-table:: Inputs parameters for reference_elev
               :header: "Name", "Description", "Type", "Default value", "Required"
               :widths: 20, 40, 20, 10, 10

               "path_to_elev", "Path to reference elevation", "str", "", "Yes"
               "force_source_nodata", "No data elevation", "int", "", "No"
               "path_to_mask", "Path to mask associated to the elevation", "str", "", "No"
               "from_vcrs", "Original vcrs", "str, int", None, "No"
               "to_vcrs", "Destination vcrs", "str, int", None, "No"
               "downsample", "Downsampling elevation factor >= 1", "int, float", 1, "No"

            .. note:: For setting the vcrs please refer to :doc:`vertical_ref`.
            .. note:: Take care that the path_to_elev and path_to_mask point to existing data.

            .. note:: The downsample parameter allows the user to resample the elevation by a round factor.
                      The default value of 1 means no downsampling.

        .. tab:: to_be_aligned_elev

            .. csv-table:: Inputs parameters for to_be_aligned_elev
               :header: "Name", "Description", "Type", "Default value", "Required"
               :widths: 20, 40, 20, 10, 10

               "path_to_elev", "Path to to_be_aligned elevation", "str", "", "Yes"
               "force_source_nodata", "No data elevation", "int", "", "No"
               "path_to_mask", "Path to mask associated to the elevation", "str", "", "No"
               "from_vcrs", "Original vcrs", "int, str", None, "No"
               "to_vcrs", "Destination vcrs", "int, str", None, "No"
               "downsample", "Downsampling elevation factor >= 1", "int, float", 1, "No"

            .. note:: For setting the vcrs please refer to :doc:`vertical_ref`
            .. note:: Take care that the path_to_elev and path_to_mask point to existing data.

            .. note:: The downsample parameter allows the user to resample the elevation by a round factor.
                      The default value of 1 means no downsampling.

      .. code-block:: yaml

        inputs:
            reference_elev:
                path_to_elev: "path_to/reference_elev.tif"
                force_source_nodata: -32768
                from_vcrs: None
                to_vcrs: None
            to_be_aligned_elev:
                path_to_elev: "path_to/to_be_aligned_elev.tif"
                path_to_mask: "path_to/mask.tif"

   .. tab:: coregistration

      **Required:** No

      Coregistration step details.

      You can create a pipeline with up to three coregistration steps by using the keys
      ``step_one``, ``step_two`` and ``step_three``.

      Available coregistration method see : :ref:`coregistration`

      .. note::
        By default, coregistration is carried out using the Nuth and Kääb method.
        To disable coregistration, set `process: False` in the configuration file.

      .. tabs::

        .. csv-table:: Inputs parameters for coregistration
           :header: "Parameter namer", "Subparameter name", "Description", "Type", "Default value", "Available Value", "Required"
           :widths: 30, 20, 10, 10, 10, 10, 10

           "step_[one | two | three]", "method", "Name of coregistration method", "str", "NuthKaab", "Every available coregistration method", "No"
           "", "extra_information", "Extra parameters fitting with the method", "dict", "", "", "No"
           "sampling_grid", "", "Destination elevation for reprojection", "str", "reference_elev", "reference_elev or to_be_aligned_elev", "No"
           "process", "", "Activate the coregistration", "bool", "True", "True or False", "No"

        .. note::

            The data provided in extra_information is not checked for errors before executing the code.
            Its use is entirely the responsibility of the user.

      .. code-block:: yaml

        coregistration:
          step_one:
            method: "NuthKaab"
            extra_information : {"max_iterations": 10}
          step_two:
            method: "DHMinimize"
          sampling_grid: "reference_elev"
          process: True

      other example :

      .. code-block:: yaml

        coregistration:
          step_one:
            method: "VerticalShift"
          sampling_grid: "reference_elev"
          process: True

   .. tab:: statistics

      **Required:** No

      Statistics step information. This section relates to the computed statistics:

      1. If no block is specified, all available statistics are calculated by default:

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

   .. tab:: outputs

     **Required:** No

     Outputs information.
     Operates by levels:

     1. **Level 1** → aligned elevation only
     2. **Level 2** → more detailed output

     .. csv-table:: Outputs parameters
       :header: "Name", "Description", "Type", "Default value", "Available Value", "Required"
       :widths: 20, 40, 10, 10, 10, 10

       "path", "Path for outputs", "str", "outputs", "", "No"
       "level", "Level for detailed outputs", "int", "1", "1 or 2", "No"
       "output_grid", "Grid for outputs resampling", "str", "reference_elev", "reference_elev or to_be_aligned_elev", "No"

     .. code-block:: yaml

       outputs:
           level : 1
           path : "path_to/outputs"
           output_grid: "reference_elev"

     Tree of outputs for level 1 (including coregistration step)

     .. code-block:: text

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

     Tree of outputs for level 2 (including coregistration step)

     .. code-block:: text

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
```
