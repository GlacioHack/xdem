(cli-accuracy)=

# Accuracy workflow

## Summary

The accuracy workflow is designed to help users analyze differences between two elevation datasets by
generating various outputs. It also includes optional coregistration for improved alignment.

:::{figure} imgs/accuracy_workflow_pipeline.png
:width: 100%
:::

## Available command line

Run the workflow:

```{code}
xdem accuracy --config config_file.yaml
```
To display a template of all available configuration options for the YAML file, use the following command:

```{code}
xdem accuracy --display_template_config
```

## Detailed description of input parameters

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

            .. note:: For setting the vcrs please refer to :doc:`vertical_ref`

        .. tab:: to_be_aligned_elev

            .. csv-table:: Inputs parameters for to_be_aligned_elev
               :header: "Name", "Description", "Type", "Default value", "Required"
               :widths: 20, 40, 20, 10, 10

               "path_to_elev", "Path to to_be_aligned elevation", "str", "", "Yes"
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
            to_be_aligned_elev:
                path_to_elev: "path_to/to_be_aligned_dem.tif"
                path_to_mask: "path_to/mask.tif"

   .. tab:: coregistration

      **Required:** No

      Coregistration step details. You can create a pipeline with up to three coregistration steps by
      using the keys step_one, step_two, and step_three.
      Available coregistration : see coregistration information <coregistration.md>`

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

## Report example

```{eval-rst}
    .. raw:: html

        <meta charset='UTF-8'><title>Qualify elevation results</title></head>
        <h2>Digital Elevation Model</h2>
        <div style='display: flex; gap: 10px;'>
          <img src='_static/reference_elev_map.png' alt='Image PNG' style='max-width: 100%; height: auto; width: 40%;'>
          <img src='_static/to_be_aligned_elev_map.png' alt='Image PNG' style='max-width: 100%; height: auto; width: 40%;'>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Information about inputs</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>reference_elev</td><td>{'path_to_elev': '/xdem/examples/data/Longyearbyen/data/DEM_2009_ref.tif', 'from_vcrs': 'EGM96', 'to_vcrs': 'EGM96'}</td></tr>
        <tr><td>to_be_aligned_elev</td><td>{'path_to_elev': '/xdem/examples/data/Longyearbyen/data/DEM_1990.tif', 'path_to_mask': '/xdem/examples/data/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp', 'from_vcrs': 'EGM96', 'to_vcrs': 'EGM96'}</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Coregistration user configuration</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>step_one</td><td>{'method': 'NuthKaab'}</td></tr>
        <tr><td>sampling_grid</td><td>reference_elev</td></tr>
        <tr><td>process</td><td>True</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>NuthKaab inputs</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>random</td><td>{'subsample': 500000.0, 'random_state': None}</td></tr>
        <tr><td>fitorbin</td><td>{'fit_or_bin': 'bin_and_fit', 'fit_func': <function _nuth_kaab_fit_func at 0x72164be14f70>, 'fit_optimizer': <function curve_fit at 0x721659466170>, 'bin_sizes': 72, 'bin_statistic': <function nanmedian at 0x72168b9c20f0>, 'nd': 1, 'bias_var_names': ['aspect']}</td></tr>
        <tr><td>iterative</td><td>{'max_iterations': 10, 'tolerance': 0.0}</td></tr>
        <tr><td>specific</td><td>{}</td></tr>
        <tr><td>affine</td><td>{'apply_vshift': True}</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>NuthKaab outputs</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>affine</td><td>{'shift_x': 9.19, 'shift_y': 2.79, 'shift_z': -1.99}</td></tr>
        <tr><td>random</td><td>{'subsample_final': 500000}</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Statistics on reference elevation</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>Mean</td><td>378.05</td></tr>
        <tr><td>Median</td><td>360.65</td></tr>
        <tr><td>Maximum</td><td>1022.21</td></tr>
        <tr><td>Minimum</td><td>8.05</td></tr>
        <tr><td>Sum</td><td>496010624.0</td></tr>
        <tr><td>Sum of square</td><td>265449963520.0</td></tr>
        <tr><td>90th percentile</td><td>724.54</td></tr>
        <tr><td>LE90</td><td>766.59</td></tr>
        <tr><td>NMAD</td><td>290.22</td></tr>
        <tr><td>RMSE</td><td>449.8</td></tr>
        <tr><td>STD</td><td>243.72</td></tr>
        <tr><td>Standard deviation</td><td>243.72</td></tr>
        <tr><td>Valid count</td><td>1312020</td></tr>
        <tr><td>Total count</td><td>1312020</td></tr>
        <tr><td>Percentage valid points</td><td>100.0</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Statistics on to be aligned elevation</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>Mean</td><td>381.32</td></tr>
        <tr><td>Median</td><td>365.23</td></tr>
        <tr><td>max</td><td>1022.29</td></tr>
        <tr><td>Minimum</td><td>8.38</td></tr>
        <tr><td>SUM</td><td>500301600.0</td></tr>
        <tr><td>Sum of squares</td><td>268858540032.0</td></tr>
        <tr><td>90th percentile</td><td>727.55</td></tr>
        <tr><td>LE90</td><td>766.79</td></tr>
        <tr><td>NMAD</td><td>291.27</td></tr>
        <tr><td>RMSE</td><td>452.68</td></tr>
        <tr><td>STD</td><td>243.95</td></tr>
        <tr><td>Standard deviation</td><td>243.95</td></tr>
        <tr><td>Valid count</td><td>1312020</td></tr>
        <tr><td>Total count</td><td>1312020</td></tr>
        <tr><td>Percentage valid points</td><td>100.0</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Statistics on alti diff before coregistration</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>Mean</td><td>3.27</td></tr>
        <tr><td>Median</td><td>2.77</td></tr>
        <tr><td>Maximum</td><td>51.44</td></tr>
        <tr><td>Minimum</td><td>-54.51</td></tr>
        <tr><td>Sum</td><td>4290968.0</td></tr>
        <tr><td>Sum of squares</td><td>62515040.0</td></tr>
        <tr><td>90th percentile</td><td>9.18</td></tr>
        <tr><td>LE 90</td><td>18.37</td></tr>
        <tr><td>NMAD</td><td>3.81</td></tr>
        <tr><td>RMSE</td><td>6.9</td></tr>
        <tr><td>STD</td><td>6.08</td></tr>
        <tr><td>Standard deviation</td><td>6.08</td></tr>
        <tr><td>Valid count</td><td>1312020</td></tr>
        <tr><td>Total count</td><td>1312020</td></tr>
        <tr><td>Percentage valid points</td><td>100.0</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Statistics on alti diff after coregistration</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>Mean</td><td>1.18</td></tr>
        <tr><td>Median</td><td>0.36</td></tr>
        <tr><td>max</td><td>50.31</td></tr>
        <tr><td>Minimum</td><td>-49.91</td></tr>
        <tr><td>Sum</td><td>1546964.0</td></tr>
        <tr><td>Sum of squares</td><td>42468744.0</td></tr>
        <tr><td>90th percentile</td><td>6.03</td></tr>
        <tr><td>LE90</td><td>16.12</td></tr>
        <tr><td>NMAD</td><td>2.83</td></tr>
        <tr><td>RMSE</td><td>5.69</td></tr>
        <tr><td>STD</td><td>5.57</td></tr>
        <tr><td>Standard deviation</td><td>5.57</td></tr>
        <tr><td>Valid count</td><td>1312020</td></tr>
        <tr><td>Total count</td><td>1312020</td></tr>
        <tr><td>Percentage valid points</td><td>100.0</td></tr>
        </table>
        </div>
        <div style='clear: both; margin-bottom: 30px;'>
        <h2>Statistics aligned elevation</h2>
        <table border='1' cellspacing='0' cellpadding='5'>
        <tr><th>Information</th><th>Value</th></tr>
        <tr><td>Mean</td><td>379.33</td></tr>
        <tr><td>Median</td><td>363.24</td></tr>
        <tr><td>max</td><td>1020.3</td></tr>
        <tr><td>Minimum</td><td>6.38</td></tr>
        <tr><td>Sum</td><td>497688448.0</td></tr>
        <tr><td>Sum of squares</td><td>266870931456.0</td></tr>
        <tr><td>90th percentile</td><td>725.56</td></tr>
        <tr><td>LE90</td><td>766.79</td></tr>
        <tr><td>NMAD</td><td>291.27</td></tr>
        <tr><td>RMSE</td><td>451.0</td></tr>
        <tr><td>STD</td><td>243.95</td></tr>
        <tr><td>Standard deviation</td><td>243.95</td></tr>
        <tr><td>Valid count</td><td>1312020</td></tr>
        <tr><td>Total count</td><td>1312020</td></tr>
        <tr><td>Percentage valid points</td><td>100.0</td></tr>
        </table>
        </div>
        <h2>Altitude differences</h2>
        <div style='display: flex; gap: 10px;'>
          <img src='_static/diff_elev_before_coreg.png' alt='Image PNG' style='max-width: 40%; height: auto; width: 50%;'>
          <img src='_static/diff_elev_after_coreg.png' alt='Image PNG' style='max-width: 40%; height: auto; width: 50%;'>
        </div>
        <h2>Differences histogram</h2>
        <img src='_static/elev_diff_histo.png' alt='Image PNG' style='max-width: 40%; height: auto;'>
        </div>
```
