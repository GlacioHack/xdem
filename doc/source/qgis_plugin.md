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

**QGIS** is an open source GIS, geographic information system. This software allows users to view and process geospatial data in order to extract information from it. Available on all platforms, QGIS offers a wide range of processing options. To supplement the basic functionality, extensions can be added, known as plugins. These are small modules that connect to the main software and communicate with it through a python API.

```{important}
The QGIS plugin is a recent feature! Its interface and install process are subject to change depending on feedback for future releases.
```

## Installation
The plugin is available on the official QGIS repository, here are the installation steps:

1. In QGIS go to `Plugins` > `Manage and Install Plugins...` > `Not installed`
2. Search for xDEM
3. Click on `Install Plugin`

It will take a few minutes for the dependencies to install properly, **do not force QGIS to close.**

Once installation is complete, xDEM will appear in the processing toolbox.

```{note}
The plugin is available on Linux and Windows. However, configurations are extremely diverse, whether in terms of the QGIS version or Python version, so compatibility issues may occur.
```

## Getting started
There are 26 algorithms available in the plugins, all available in the Processing Toolbox, located on the menu bar at the top of the software interface.

:::{figure} imgs/qgis_plugin/toolbox_button.png
:width: 80%
:::

Once the toolbox is open, xDEM algorithms appears at the bottom, next to [GDAL](https://gdal.org/en/stable/).

:::{figure} imgs/qgis_plugin/toolbox.png
:width: 30%
:::

To get started with the plugin's features here is an example of a coregistration followed by a slope terrain attributes. The dataset used is the same as that used in the other xdem examples, which is Longyearbyen, available for download in [xdem data](https://github.com/GlacioHack/xdem-data).

### Coregistration
The coregistration methods included in the Python API are available in the plugin, each with its own dedicated interface.
Input can be layers present in QGIS or files directly from the disk. As for the outputs, they can either be saved to temporary files (by default) or saved to disk. Here is the co-registration interface for the [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011) method.

:::{figure} imgs/qgis_plugin/coreg_interface.png
:width: 80%
:::

```{note}
For the advanced parameters, the default settings are the same as those in the API.
```

Once the processing is complete, the `log` section provides information about the coregistration's metadata.

:::{figure} imgs/qgis_plugin/coreg_log.png
:width: 70%
:::

This data can then be saved as a text file using the button in the lower right corner.

### Terrain attributes
The terrain attributes specific to the `xdem.DEM` object are also included, with their advanced parameters.
Here is an example of the slope processing interface configured with the method [Horn (1981)](http://dx.doi.org/10.1109/PROC.1981.11918).

:::{figure} imgs/qgis_plugin/slope_interface.png
:width: 80%
:::

For all of the plugin's algorithms, once processing is complete, the result is opened in the current project, this feature can be disabled if visualization in QGIS is not necessary, or if the dataset is too large.

:::{figure} imgs/qgis_plugin/slope_result.png
:width: 80%
:::

## Workflows
The two workflows available in xdem via the {ref}`cli` have also been implemented in the plugin.
To use them, there is no need to use a configuration file as is usually the case, everything is directly configurable in the QGIS interface.

:::{figure} imgs/qgis_plugin/topo_workflow_interface.png
:width: 80%
:::

As for the output, it is identical to what is included in the {ref}`cli`, a folder containing rasters, plots, and statistical tables as well as the HTML and PDF reports, this report can be opened in the default browser once the process is complete.

## Pipeline building
QGIS offers a feature for creating pipelines through a graphical interface called the **Model Disigner**.
This tool is accessible via the `Processing` section of the menu bar.
Its detailed functionality is described in the [QGIS documentation](https://docs.qgis.org/3.44/en/docs/user_manual/processing/modeler.html).

:::{figure} imgs/qgis_plugin/model_designer_button.png
:width: 50%
:::

Thanks to this feature, it is possible to chain together xDEM processing steps and, for example, create coregistration pipelines or even combine them with native QGIS features. Here is an example of a coregistration pipeline, followed by a calculation of elevation difference.

:::{figure} imgs/qgis_plugin/coreg_pipeline.png
:width: 80%
:::

This pipeline combines two methods, ICP [Besl and McKay (1992)](https://doi.org/10.1117/12.57955), [Chen and Medioni (1992)](https://doi.org/10.1016/0262-8856(92)90066-C), and [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011), then the difference is calculated using the `QGIS Raster Calculator`.
In terms of outputs, two are generated, the aligned DEM and the final elevation difference.
