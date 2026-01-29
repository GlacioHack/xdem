---
file_format: mystnb
mystnb:
  execution_timeout: 60
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
(data)=

# Data examples

xDEM uses and proposes several data examples to manipulate and test the features and workflows.

## Description

The south of Longyearbyen, capital of Svalbard, a Norwegian archipelago, is the most used dataset in xDEM.
It is composed of:

| Alias                             |         Filename          |    Type     |                                                 Description |
|-----------------------------------|:-------------------------:|:-----------:|------------------------------------------------------------:|
| `"longyearbyen_ref_dem"`          |     DEM_2009_ref.tif      |   Raster    |                                     DEM of the area in 2010 |
| `"longyearbyen_ref_tba"`          |       DEM_1990.tif        |   Raster    |                                     DEM of the area in 1990 |
| `"longyearbyen_glacier_outlines"` | CryoClim_GAO_SJ_1990.shp  |   Vector    |                    Glacier outlines in the Svalbard in 1990 |
| `"longyearbyen_glacier_outlines_2010"` | CryoClim_GAO_SJ_2010.shp  |   Vector    |                    Glacier outlines in the Svalbard in 2010 |
| `"longyearbyen_epc"`              |        EPC_IS.gpkg        | Point Cloud | Land elevation over the aera measured between 2019 and 2021 |

```{note}
So, `"longyearbyen_ref_dem"` (for "reference" dem) can be compared to `"longyearbyen_ref_tba"` (for "to-be-aligned" dem).
```

Another data is available over the Giza pyramid complex:

| Alias         | Filename |  Type  |             Description |
|---------------|:--------:|:------:|------------------------:|
| `"gizeh_dem"` | DSM.tif  | Raster | DEM of the area in 2013 |


```{note}
:class: margin
All of these data can be used under open licence and if you need other information
about all of these, you can read the [README.md](https://github.com/GlacioHack/xdem-data/blob/main/README.md)
of the [xdem-data github project](https://github.com/GlacioHack/xdem-data) where they are stored.
```

## Access to data

### Python

If you want to use one of the example data, you can run this function with the corresponding data alias:

```{code-cell} ipython3
import xdem

# Download the 2010 raster DEM Longyearbyen dataset in output_dir and return its path
path = xdem.examples.get_path("longyearbyen_ref_dem")
```

It downloads the entire dataset of the alias if it was not already available and returns its absolute file path.

Also, you can download all the data, no matter the area by running:

```{code-cell} ipython3
import xdem

# Download the all of the xDEM data (Longyearbyen and Gizeh dataset) and return the output directory path
output_dir = xdem.examples.get_all_data()
```

In both case, you can specify the `output_dir` where you want that data to be download or searched:

```{code-cell}
import xdem
output_dir = "/dir/my_output_dir"
path = xdem.examples.get_path("longyearbyen_ref_dem", output_dir=output_dir)
output_dir = xdem.examples.get_all_data(output_dir=output_dir)
```

### Command line interface

To experiment the {ref}`cli`, you can also use the alias to refer data examples in the configuration file:

```{code-cell}
inputs:
  reference_elev:
    path_to_elev: "longyearbyen_ref_dem"
    path_to_mask: "longyearbyen_glacier_outlines_2010"
...
```

Workflows will automatically download and manage the needed data.

### Bash

To download the data samples, you can run:

```bash
mkdir data_examples
tar -xvz -C data_examples  --wildcards  "*/data" --strip-components 2 -f <(wget -q -O - https://github.com/marinebcht/xdem-data/archive/1338aaa06842f9e6b3182069a295618eb54914a5.tar.gz)
```
