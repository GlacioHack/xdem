(cli)=

# Command Line Interface (CLI)

To simplify the use of xdem and provide a universal tool to assess one or multiple DEMs,
we have decided to implement a Command Line Interface (CLI).
To support this, we offer a set of workflows that can be easily run using a configuration file.
Users can also create their own workflows and submit them for inclusion.

```{note}
**workflow** : combinations of various xDEM and GeoUtils features tailored to specific applications
```
## Global information

All workflows follow the same command structure:
```{code}
xdem workflow_name config_file.yaml
```

The configuration file can contain only the required input parameters for the workflow.
xDEM then automatically fills in the rest with suggested settings. Users are free to edit the
configuration to run only the parts they need.
At the end of the execution, several output files are saved to disk, including an HTML report
and its corresponding PDF version.


## Workflow Compare
The Compare workflow aims to provide the user blabla sur le workflow

```{code}
xdem compare config_file.yaml
```

Each tab is a YAML key.

```{eval-rst}
.. tabs::

   .. tab:: inputs

      **Required:** Yes

      DEM input information.

      .. tabs::

        .. tab:: reference_elev

            .. csv-table:: Inputs parameters for reference_elev
               :header: "Name", "Description", "Type", "Default value", "Required"
               :widths: 20, 40, 20, 10, 10

               "reference_elev", "Path to reference DEM", "str", "", "Yes"
               "no_data", "No data DEM", "int", "", "No"
               "mask", "Path to mask associated to the DEM", "str", "", "No"
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

               "to_be_aligned_elev", "Path to reference DEM", "str", "", "Yes"
               "no_data", "No data DEM", "int", "", "No"
               "mask", "Path to mask associated to the DEM", "str", "", "No"
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
                dem: "path_to/ref_dem.tif"
                nodata: -32768
                from_vcrs: {"common": "EGM96"}
                to_vcrs: {"common": "Ellipsoid"}
            to_be_aligned_elev:
                dem: "path_to/to_be_aligned_dem.tif"
                mask: "path_to/mask.tif

   .. tab:: coregistration

      **Required:** No

      Coregistration step information. Il est possible de créer une pipeline de maximum trois étapes de coregistration
      grâce à l'utilisation des clefs step_one, step_two et step_three

      .. tabs::

        .. tab:: step_one/two/three

            .. csv-table:: Inputs parameters for reference_elev
               :header: "Name", "Description", "Type", "Default value","Available Value", "Required"
               :widths: 20, 40, 10, 10, 10, 10

               "method", "Name of coregistration method", "str", "NuthKaab", "Every available coregistration method", No"
               "extra_information", "Extra parameters fitting with the method", "dict", "", "", No"

        .. tab:: sampling source

            .. csv-table:: Inputs parameters sampling source
               :header: "Name", "Description", "Type", "Default value", "Available Value", "Required"
               :widths: 20, 40, 10, 10, 10, 10

               "sampling_source", "Destination dem for reprojection", "str", "reference_dem", "reference_dem or to_be_aligned_elev", "No"

      .. code-block:: yaml

        coregistration:
          step_one:
            method: "NuthKaab"
            extra_informations : {"max_iterations": 10}
          step_two:
            method: "DHMinimize"

          sampling_source: "reference_elev"

   .. tab:: statistics

      **Required:** No

      Statistics step information. Blabla sur les stats, il faut une liste.
      Si le block n'est pas indiqué nous avons par défaut touts les stats de calculées
      Si le block est indiqué mais qu'aucune stats n'est entré alors aucun stats n'est calculée

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
          │   └─ aligned_dem.csv
          ├─ png
          │   ├─ Altitude_difference_after_coregistration.png
          │   ├─ Altitude_difference_before_coregistration.png
          │   ├─ histo_diff.png
          │   ├─ Reference_elevation.png
          │   └─ To_be_aligned_elevation.png
          ├─ raster
          │   └─ aligned_dem.tif
          ├─ report.html
          ├─ report.pdf
          └─ used_config.yaml

     Tree of outputs for level 2

     .. code-block:: text

        - root
          ├─ csv
          │   ├─ align_dem.csv
          │   ├─ alti_diff_after_stats.csv
          │   ├─ alti_diff_before_stats.csv
          │   ├─ reference_stats.csv
          │   └─ to_be_aligned_stats.csv
          ├─ png
          │   ├─ Altitude_difference_after_coregistration.png
          │   ├─ Altitude_difference_before_coregistration.png
          │   ├─ histo_diff.png
          │   ├─ Reference_elevation.png
          │   └─ To_be_aligned_elevation.png
          ├─ raster
          │   ├─ aligned_dem.tif
          │   ├─ diff_after.tif
          │   ├─ diff_before.tif
          │   └─ reference_elev_reprojected.tif
          ├─ report.html
          ├─ report.pdf
          └─ used_config.yaml
```



## Workflow Topo-Summmary

The Topo-Summmary workflow aims to provide the user with various information about their DEM,
such as terrain attributes and sets of statistics.

```{code}
xdem topo_summary config_file.yaml
```

```{eval-rst}
.. tabs::

   .. tab:: inputs

     **Required:** Yes

     DEM input information.


     .. csv-table:: Inputs parameters for DEM
        :header: "Name", "Description", "Type", "Default value", "Required"
        :widths: 20, 40, 20, 10, 10

        "reference_elev", "Path to reference DEM", "str", "", "Yes"
        "no_data", "No data DEM", "int", "", "No"
        "mask", "Path to mask associated to the DEM", "str", "", "No"
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
           dem: "path_to/ref_dem.tif"
           nodata: -32768
           from_vcrs: {"common": "EGM96"}
           to_vcrs: {"common": "Ellipsoid"}

   .. tab:: statistics

      **Required:** No

      Statistics step information. Blabla sur les stats, il faut une liste.
      Si le block n'est pas indiqué, nous avons par défaut toutes les stats calculées.
      Si le block est indiqué mais qu'aucune stat n’est entrée, alors aucune stat n’est calculée.

      .. code-block:: yaml

         statistics:
           - min
           - max
           - mean

   .. tab:: terrain attributes

      **Required:** No

      liste ou ensemble de dict pour les extras infos .
      Si le block n'est pas indiqué, nous avons par les 6 par défaut de la doc.
      Si le block est indiqué mais qu'aucune info n’est entrée, alors aucun attributes n’est calculé.

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

   .. tab:: outputs

    **Required:** No

    Outputs information.
    Fonctionne par niveau :
    - niveau 1 → uniquement le DEM aligné
    - niveau 2 → plus détaillé

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
         │   ├─ stats_dem.csv
         │   └─ stats_dem_mask.csv
         ├─ png
         │   ├─ dem.png
         │   └─ terrain_attributes.png
         ├─ raster
         ├─ report.html
         ├─ report.pdf
         └─ used_config.yaml

    Tree of outputs for level 2

    .. code-block:: text

       - root
         ├─ csv
         │   ├─ stats_dem.csv
         │   └─ stats_dem_mask.csv
         ├─ png
         │   ├─ dem.png
         │   └─ terrain_attributes.png
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



## Workflow VolumeChange

```{important}
This version is currently just a basic working framework.
```
