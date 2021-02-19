# DemUtils
Set of tools to manipulate Digital Elevation Models (DEMs)

More documentation to come!


## Installation ##

```
$ git clone https://github.com/GlacioHack/DemUtils.git
$ cd ./DemUtils
$ conda create -f environment.yml
$ conda activate glacio
```

## Structure 

DemUtils are for now composed of two libraries:
- `coreg.py` with tools covering differet aspects of DEM coregistration
- `spatial_tools.py` for spatial operations on DEMs

## How to contribute

You can find ways to improve the libraries in the [issues](https://github.com/GlacioHack/DemUtils/issues) section. All contributions are welcome.
To avoid conflicts, it is suggested to use separate branches for each implementation. All changes must then be submitted to the dev branch using pull requests. Each PR must be reviewed by at least one other person.

### Documentation - please read ! ###
In the interest of keeping the documentation simple, please write all docstring in reStructuredText (https://docutils.sourceforge.io/rst.html) format - eventually, we will try to set up auto-documentation using sphinx and readthedocs, and this will help in that task.

### Testing - again please read!
These tools are only valuable if we can rely on them to perform exactly as we expect. So, we need testing. Please create tests for every function that you make, as much as you are able. Guidance/examples here for the moment: https://github.com/GeoUtils/georaster/blob/master/test/test_georaster.py
https://github.com/corteva/rioxarray/blob/master/test/integration/test_integration__io.py
