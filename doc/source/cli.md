(cli)=

# Command Line Interface (CLI)

To simplify the use of xdem and provide a universal tool to assess one or multiple DEMs,
we have decided to implement a Command Line Interface (CLI).
To support this, we offer a set of workflows that can be easily run using a configuration file.
Users can also create their own workflows and submit them for inclusion.

```{note}
**workflow definition** : combinations of various xDEM and GeoUtils features tailored to specific applications
```
## Global information

All workflows follow the same command structure:
```{code}
xdem workflow_name --config config_file.yaml
```

The configuration YAML file can contain only the required input parameters for the workflow.
xDEM then automatically fills in the rest with suggested settings. Users are free to edit the
configuration to run only the parts they need.


It is possible to display a template of all possible configuration options for the file using the following command:

```{code}
xdem workflow_name --generate-config
```

```{note}
At the end of the execution, several output files are saved to disk, including an HTML report
and its corresponding PDF version.
```


## Workflow diff-analysis

### Summary

The diff-analysis workflow allows the user to generate various outputs to facilitate
the analysis of differences between two elevation datasets. An optional coregistration step can also be included.

:::{figure} imgs/diffanalysis_pipeline.png
:width: 100%
:::

### Available command line

Run the workflow

```{code}
xdem diff-analysis --config config_file.yaml
```
Preview available parameters

```{code}
xdem diff-analysis --generate-config
```

### Detailed description of input parameters :

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

               "path_to_elev", "Path to reference DEM", "str", "", "Yes"
               "no_data", "No data DEM", "int", "", "No"
               "path_to_mask", "Path to mask associated to the DEM", "str", "", "No"
               "from_vcrs", "Original vcrs", "dict", "{'common': 'EGM98'}", "No"
               "to_vcrs", "Destination vcrs", "dict", "{'common': 'EGM98'}", "No"

            .. csv-table:: Zoom on from_vcrs and to_vcrs configuration
               :header: "Key", "Type", "Allowed Values", "Example"
               :widths: 15, 15, 40, 30

               "common", "str", "EGM96, EGM08, Ellipsoid", "{'common': 'EGM98'}"
               "proj_grid", "str", "Grid file name from https://cdn.proj.org/", "{'proj_grid': 'us_noaa_geoid06_ak.tif'}"
               "epsg_code", "int", "EPSG code of the vertical CRS", "{'epsg_code': 5773}"

        .. tab:: to_be_aligned_elev

            .. csv-table:: Inputs parameters for to_be_aligned_elev
               :header: "Name", "Description", "Type", "Default value", "Required"
               :widths: 20, 40, 20, 10, 10

               "path_to_elev", "Path to reference DEM", "str", "", "Yes"
               "no_data", "No data DEM", "int", "", "No"
               "path_to_mask", "Path to mask associated to the DEM", "str", "", "No"
               "from_vcrs", "Original vcrs", "dict", "{'common': 'EGM98'}", "No"
               "to_vcrs", "Destination vcrs", "dict", "{'common': 'EGM98'}", "No"

            .. csv-table:: Zoom on from_vcrs and to_vcrs configuration
               :header: "Key", "Type", "Allowed Values", "Example"
               :widths: 15, 15, 40, 30

               "common", "str", "EGM96, EGM08, Ellipsoid", "{'common': 'EGM98'}"
               "proj_grid", "str", "Grid file name from https://cdn.proj.org/", "{'proj_grid': 'us_noaa_geoid06_ak.tif'}"
               "epsg_code", "int", "EPSG code of the vertical CRS", "{'epsg_code': 5773}"

      .. code-block:: yaml

        inputs:
            reference_elev:
                path_to_elev: "path_to/ref_dem.tif"
                nodata: -32768
                from_vcrs: {"common": "EGM96"}
                to_vcrs: {"common": "Ellipsoid"}
            to_be_aligned_elev:
                path_to_elev: "path_to/to_be_aligned_dem.tif"
                path_to_mask: "path_to/mask.tif

   .. tab:: coregistration

      **Required:** No

      Coregistration step details. You can create a pipeline with up to three coregistration steps by
      using the keys step_one, step_two, and step_three.

      .. tabs::

        .. tab:: step_one/two/three

            .. csv-table:: Inputs parameters for reference_elev
               :header: "Name", "Description", "Type", "Default value","Available Value", "Required"
               :widths: 20, 40, 10, 10, 10, 10

               "method", "Name of coregistration method", "str", "NuthKaab", "Every available coregistration method", No"
               "extra_information", "Extra parameters fitting with the method", "dict", "", "", No"

        .. note::

            The data provided in extra_information is not validated beforehand.
            Its use is entirely the responsibility of the user.

        .. tab:: sampling source

            .. csv-table:: Inputs parameters sampling source
               :header: "Name", "Description", "Type", "Default value", "Available Value", "Required"
               :widths: 20, 40, 10, 10, 10, 10

               "sampling_source", "Destination dem for reprojection", "str", "reference_dem", "reference_dem or to_be_aligned_elev", "No"

        .. tab:: process

            .. csv-table:: Inputs parameters sampling source
               :header: "Name", "Description", "Type", "Default value", "Available Value", "Required"
               :widths: 20, 40, 10, 10, 10, 10

               "process", "Activate the coregistration", "bool", "True", "True or False", "No"

      .. code-block:: yaml

        coregistration:
          step_one:
            method: "NuthKaab"
            extra_informations : {"max_iterations": 10}
          step_two:
            method: "DHMinimize"

          sampling_source: "reference_elev"
          process: True

   .. tab:: statistics

      **Required:** No

      Statistics step information. This section relates to the computed statistics:
        1. If no block is specified, all available statistics are calculated by default.
        2. If a block is specified but no statistics are provided, then no statistics will be computed.

      .. code-block:: yaml

        statistics:
          - min
          - max
          - mean

   .. tab:: outputs

     **Required:** No

     Outputs information, fonctionne par niveau, si niveau 1 alors nous uniquement le dem aligné
     si niveau 2

     .. csv-table:: Outputs parameters
       :header: "Name", "Description", "Type", "Default value", "Available Value", "Required"
       :widths: 20, 40, 10, 10, 10, 10

       "path", "Path for outputs", "str", "outputs", "", "No"
       "level", "Level for detailed outputs", "int", "1", "1 or 2", "No"

     .. code-block:: yaml

       outputs:
           level : 1
           path : "path_to/outputs"

     Tree of outputs for level 1

     .. code-block:: text

      - root
        ├─ csv
        │   └─ aligned_elev_stats.csv
        ├─ png
        │   ├─ diff_elev_after_coreg_map.png
        │   ├─ diff_elev_before_coreg_map.png
        │   ├─ diff_elev_before_after_hist.png
        │   ├─ reference_elev_map.png
        │   └─ to_be_aligned_elev_map.png
        ├─ raster
        │   └─ aligned_elev.tif
        ├─ report.html
        ├─ report.pdf
        └─ used_config.yaml

     Tree of outputs for level 2

     .. code-block:: text

      - root
        ├─ csv
        │   ├─ aligned_elev_stats.csv
        │   ├─ diff_elev_after_coreg_stats.csv
        │   ├─ diff_elev_before_coreg_stats.csv
        │   ├─ reference_elev_stats.csv
        │   └─ to_be_aligned_elev_stats.csv
        ├─ png
        │   ├─ diff_elev_after_coreg_map.png
        │   ├─ diff_elev_before_coreg_map.png
        │   ├─ diff_elev_before_after_hist.png
        │   ├─ reference_elev_map.png
        │   └─ to_be_aligned_elev_map.png
        ├─ raster
        │   ├─ aligned_elev.tif
        │   ├─ diff_elev_after_coreg.tif
        │   ├─ diff_elev_before_coreg.tif
        │   └─ to_be_aligned_elev_reprojected.tif
        ├─ report.html
        ├─ report.pdf
        └─ used_config.yaml
```



## Workflow Topo-Summmary

### Summary

The Topo-Summmary workflow aims to provide the user with various information about their DEM,
such as terrain attributes and sets of statistics.

:::{figure} imgs/pipeline_toposummary.png
:width: 100%
:::


### Available command line

Run the workflow

```{code}
xdem topo_summary --config config_file.yaml
```

Preview available parameters

```{code}
xdem diff-analysis --generate-config
```

### Detailed description of input parameters :

```{eval-rst}
.. tabs::

   .. tab:: inputs

     **Required:** Yes

     Elevation input information.

     .. csv-table:: Inputs parameters for elevation
        :header: "Name", "Description", "Type", "Default value", "Required"
        :widths: 20, 40, 20, 10, 10

        "path_to_elev", "Path to reference elevation", "str", "", "Yes"
        "no_data", "No data elevation", "int", "", "No"
        "path_to_mask", "Path to mask associated to the elevation", "str", "", "No"
        "from_vcrs", "Original vcrs", "dict", """{'common': 'EGM98'}""", "No"
        "to_vcrs", "Destination vcrs", "dict", """{'common': 'EGM98'}""", "No"

     .. csv-table:: Zoom on from_vcrs and to_vcrs configuration
       :header: "Key", "Type", "Description / Allowed Values", "Example"
       :widths: 15, 15, 40, 30

       "common", "str", "EGM96, EGM08, Ellipsoid", """{'common': 'EGM98'}"""
       "proj_grid", "str", "Grid file name from https://cdn.proj.org/", """{'proj_grid': 'us_noaa_geoid06_ak.tif'}"""
       "epsg_code", "int", "EPSG code of the vertical CRS", """{'epsg_code': 5773}"""

     .. code-block:: yaml

         inputs:
           reference_elev:
             path_to_elev: "path_to/ref_dem.tif"
             nodata: -32768
             from_vcrs: {"common": "EGM96"}
             to_vcrs: {"common": "Ellipsoid"}

   .. tab:: statistics

      **Required:** No

      Statistics step information. This section relates to the computed statistics:
        1. If no block is specified, all available statistics are calculated by default.
        2. If a block is specified but no statistics are provided, then no statistics will be computed.

      .. code-block:: yaml

         statistics:
           - min
           - max
           - mean

   .. tab:: terrain attributes

      **Required:** No

      List or set of dictionaries for extra information.
        – If no block is specified, slope, aspect, and max_curvature attributes are calculated by default.
        – If a block is specified but no information is provided, then no attributes will be calculated.

      .. code-block:: yaml

         terrain_attributes:
           - hillshade
           - slope

      OR

      .. code-block:: yaml

         terrain_attributes:
           hillshade:
               extra_informations:
           slope:
              extra_informations:
           aspect:
            extra_informations:
                degrees: False

      .. note::
        The data provided in extra_information is not validated beforehand.
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

       "path", "Path for outputs", "str", "outputs", "", "No"
       "level", "Level for detailed outputs", "int", "1", "1 or 2", "No"

    .. code-block:: yaml

      outputs:
          level : 1
          path : "path_to/outputs"

    Tree of outputs for level 1

    .. code-block:: text

      - root
        ├─ csv
        │   ├─ elev_stats.csv
        │   └─ elev_with_mask_stats.csv
        ├─ png
        │   ├─ elev_map.png
        │   └─ terrain_attributes_map.png
        ├─ raster
        ├─ report.html
        ├─ report.pdf
        └─ used_config.yaml

    Tree of outputs for level 2

    .. code-block:: text

      - root
        ├─ csv
        │   ├─ elev_stats.csv
        │   └─ elev_with_mask_stats.csv
        ├─ png
        │   ├─ elev_map.png
        │   └─ terrain_attributes_map.png
        ├─ raster
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
