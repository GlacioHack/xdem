(cli-topo)=

# Topo workflow

## Summary

The topo workflow is designed to provide users with comprehensive information about their elevation model,
including terrain attributes (i.e. slope, hillshade, aspect, etc.) and statistical analyses (i.e. mean, max, min, etc.)



:::{figure} imgs/topo_workflow_pipeline.png
:width: 100%
:::

## Available command line

Run the workflow :

```{code}
xdem topo --config config_file.yaml
```

To display a template of all available configuration options for the YAML file, use the following command:

```{code}
xdem topo --display_template_config
```

## Detailed description of input parameters

```{eval-rst}
.. tabs::

   .. tab:: inputs

     **Required:** Yes

     Elevation input information.

     .. csv-table:: Inputs parameters for elevation
        :header: "Name", "Description", "Type", "Default value", "Required"
        :widths: 20, 40, 20, 10, 10

        "path_to_elev", "Path to reference elevation", "str", "", "Yes"
        "force_source_nodata", "No data elevation", "int", "", "No"
        "path_to_mask", "Path to mask associated to the elevation", "str", "", "No"
        "from_vcrs", "Original vcrs", "int, str", None, "No"
        "to_vcrs", "Destination vcrs", "int, str", None, "No"

     .. note:: For setting the vcrs please refer to :doc:`vertical_ref`

     .. code-block:: yaml

         inputs:
           reference_elev:
             path_to_elev: "path_to/ref_dem.tif"
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

       "path", "Path for outputs", "str", "outputs", "", "No"
       "level", "Level for detailed outputs", "int", "1", "1 or 2", "No"

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

## Report example

```{eval-rst}
    .. raw:: html

        <meta charset='UTF-8'><title>Topographic summary results</title></head>

        <h2>Elevation Model</h2>
        <img src='_static/elevation (m).png' alt='Image PNG' style='max-width: 100%; height: auto;'>
        <h2>Masked elevation Model</h2>
        <img src='_static/masked_elevation.png' alt='Image PNG' style='max-width: 100%; height: auto;'>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Information about inputs</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>reference_elev</td><td>{'path_to_elev': '/xdem/examples/data/Longyearbyen/data/DEM_1990.tif', 'path_to_mask': '/xdem/examples/data/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp', 'force_source_nodata': -9999, 'from_vcrs': 'EGM96', 'to_vcrs': 'EGM96'}</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>elevation information</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>Driver</td><td>GTiff</td></tr>
        <tr><td>Filename</td><td>/xdem/examples/data/Longyearbyen/data/DEM_1990.tif</td></tr>
        <tr><td>Grid size</td><td>us_nga_egm96_15.tif</td></tr>
        <tr><td>Number of band</td><td>(1,)</td></tr>
        <tr><td>Data types</td><td>float32</td></tr>
        <tr><td>Nodata Value</td><td>-9999</td></tr>
        <tr><td>Pixel interpretation</td><td>Area</td></tr>
        <tr><td>Pixel size</td><td>(20.0, 20.0)</td></tr>
        <tr><td>Width</td><td>1332</td></tr>
        <tr><td>Height</td><td>985</td></tr>
        <tr><td>Transform</td><td>| 20.00, 0.00, 502810.00|
        | 0.00,-20.00, 8674030.00|
        | 0.00, 0.00, 1.00|</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Global statistics</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>mean</td><td>381.32</td></tr>
        <tr><td>median</td><td>365.23</td></tr>
        <tr><td>max</td><td>1022.29</td></tr>
        <tr><td>min</td><td>8.38</td></tr>
        <tr><td>sum</td><td>500301600.0</td></tr>
        <tr><td>sumofsquares</td><td>268858540032.0</td></tr>
        <tr><td>90thpercentile</td><td>727.55</td></tr>
        <tr><td>le90</td><td>766.79</td></tr>
        <tr><td>nmad</td><td>291.27</td></tr>
        <tr><td>rmse</td><td>452.68</td></tr>
        <tr><td>std</td><td>243.95</td></tr>
        <tr><td>standarddeviation</td><td>243.95</td></tr>
        <tr><td>validcount</td><td>1312020</td></tr>
        <tr><td>totalcount</td><td>1312020</td></tr>
        <tr><td>percentagevalidpoints</td><td>100.0</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Mask statistics</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>mean</td><td>592.41</td></tr>
        <tr><td>median</td><td>591.89</td></tr>
        <tr><td>max</td><td>1011.9</td></tr>
        <tr><td>min</td><td>139.9</td></tr>
        <tr><td>sum</td><td>98813536.0</td></tr>
        <tr><td>sumofsquares</td><td>62425526272.0</td></tr>
        <tr><td>90thpercentile</td><td>788.29</td></tr>
        <tr><td>le90</td><td>498.87</td></tr>
        <tr><td>nmad</td><td>163.87</td></tr>
        <tr><td>rmse</td><td>611.76</td></tr>
        <tr><td>std</td><td>152.66</td></tr>
        <tr><td>standarddeviation</td><td>152.66</td></tr>
        <tr><td>validcount</td><td>1312020</td></tr>
        <tr><td>totalcount</td><td>1312020</td></tr>
        <tr><td>percentagevalidpoints</td><td>100.0</td></tr>
        </table>
        </div>
        <h2>Terrain attributes</h2>
        <img src='_static/terrain_attributes.png' alt='Image PNG' style='max-width: 100%; height: auto;'>
```
