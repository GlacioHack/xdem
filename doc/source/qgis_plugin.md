(qgis_plugin)=

# QGIS plugin
`````{grid}

````{grid-item}
:columns: 8
xDEM offers a **graphical user interface (GUI)** in the form of a QGIS plugin. This interface provides access to most of xDEM's features without needing any programming skills.
````

````{grid-item}
:columns: 4
```{figure} imgs/qgis_plugin/qgis_logo.png
:width: 60%
```
````

`````

**QGIS** is an open source geographic information system (GIS). This software allows users to view and process geospatial data in order to extract information from it. Available on all platforms (Linux, Windows, macOS), QGIS offers a wide range of processing options. To supplement the basic functionality, extensions can be added, known as plugins. These are small modules that connect to the main software and communicate with it through a python API.

```{important}
The QGIS plugin is a recent feature! Its interface and install process are subject to change depending on feedback for future releases.
```

## Installation
The plugin is available on the official QGIS repository, here are the installation steps:
1. In QGIS go to `Plugins` > `Manage and Install Plugins...` > `Not installed`.
2. Search for xDEM.
3. Click on `Install Plugin`.

If you need a specific version, you can download it directly from the [QGIS website](https://plugins.qgis.org/plugins/).
1. Search for xDEM.
2. Download the .zip file for the desired version.
3. In QGIS, go to `Plugins` > `Manage and Install Plugins...` > `Install plugin from zip`.

It will take a few minutes for the dependencies to install properly, **do not force QGIS to close.**

Once installation is complete, xDEM will appear in the **processing toolbox** <img width="22" height="22" alt="Image" src="https://github.com/user-attachments/assets/cf19c59d-50bb-46b3-98a6-dad66e099b06" />.

```{note}
The plugin is available on Linux and Windows. However, configurations are extremely diverse, whether in terms of the QGIS version or Python version, so compatibility issues may occur.
```

## Getting started
Most of xDEM's algorithms are integrated into the plugin, all available in the Processing Toolbox, located on the menu bar at the top of the software interface.

:::{figure} imgs/qgis_plugin/toolbox_button.png
:width: 80%
:::

Once the toolbox is open, xDEM algorithms appears at the bottom, in a dedecated section.

:::{figure} imgs/qgis_plugin/toolbox.png
:width: 30%
:::

To get started with the plugin's features, here are two examples, a coregistration followed by deriving the slope. The dataset used is the same as that used in the other xdem examples, which is Longyearbyen, available for download in [xdem-data repository](https://github.com/GlacioHack/xdem-data).

### Coregistration
The coregistration methods that are available in the Python API ({ref}`supported_coreg_method`) are also included in the plugin, each with its own dedicated interface.
Inputs can be layers that are already loaded in QGIS or files directly from disk.
Outputs can be saved to temporary files (the default) or written on disk.
Here is the co-registration interface for the Least Z-difference method ({ref}`lzd`).

:::{figure} imgs/qgis_plugin/coreg_interface.png
:width: 80%
:::

```{note}
For the advanced parameters, the default settings are the same as those in the API.
Blockwise mode is also available ({ref}`blockwise`), it is normally accessed via the {class}`~xdem.coreg.BlockwiseCoreg` object and is provided here as the advanced parameter `blocksize`.
```

Once the processing is complete, the `log` section provides information about the coregistration's metadata.

:::{figure} imgs/qgis_plugin/coreg_log.png
:width: 70%
:::

This data can then be saved as a text file using the button in the lower right corner.

### Terrain attributes
The {ref}`terrain-attributes` specific to the {class}`xdem.DEM` object are also included.
Here is an example of the slope processing interface configured with the method [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918).

:::{figure} imgs/qgis_plugin/slope_interface.png
:width: 80%
:::

For all of the plugin's algorithms, once processing is complete, the output file is opened in the current project. This feature can be disabled thank to the checkbox `Open output file after running algorithm`, if visualization in QGIS is not necessary or if the dataset is too large.

:::{figure} imgs/qgis_plugin/slope_result.png
:width: 80%
:::

## Workflows
The two workflows are implemented in xDEM via the {ref}`cli` have also been implemented in the plugin.
To use them, you don’t need a configuration file, unlike the CLI, everything is directly configurable in the QGIS interface.

:::{figure} imgs/qgis_plugin/topo_workflow_interface.png
:width: 80%
:::
The process generates several output files, exactly the same set you’d obtain with {ref}`cli`: a folder containing rasters, PNG plots, statistical tables, and both HTML and PDF reports. When the process is complete, the PDF report can be open in your default browser thanks to the checkbox `Open PDF report`.

## Pipeline building
QGIS offers a feature for creating pipelines through a graphical interface called the **Model Designer**.
This tool is accessible via the `Processing` section of the menu bar.
Its detailed functionality is described in the [QGIS documentation](https://docs.qgis.org/3.44/en/docs/user_manual/processing/modeler.html).

:::{figure} imgs/qgis_plugin/model_designer_button.png
:width: 50%
:::

Thanks to this feature, it is possible to chain together xDEM processing steps and, for example, create coregistration pipelines or even combine them with native QGIS features. Here is an example of a coregistration pipeline, followed by a calculation of elevation difference.

:::{figure} imgs/qgis_plugin/coreg_pipeline.png
:width: 80%
:::

This pipeline combines two methods, {ref}`icp`, and {ref}`nuthkaab`, then the difference is calculated using the `QGIS Raster Calculator`.
In terms of outputs, two are generated, the aligned DEM and the final elevation difference.
