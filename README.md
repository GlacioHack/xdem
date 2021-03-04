# xdem
Set of tools to manipulate Digital Elevation Models (DEMs)

More documentation to come!


## Installation ##

```
$ git clone https://github.com/GlacioHack/xdem.git
$ cd ./xdem
$ conda create -f environment.yml
$ conda activate glacio
```
or
```bash
pip install git+https://github.com/GlacioHack/xdem.git
```

## Structure 

xdem are for now composed of three libraries:
- `coreg.py` with tools covering differet aspects of DEM coregistration
- `spatial_tools.py` for spatial operations on DEMs
- `dem.py` for DEM-specific operations, such as vertical datum correction.

## How to contribute

You can find ways to improve the libraries in the [issues](https://github.com/GlacioHack/xdem/issues) section. All contributions are welcome.
To avoid conflicts, it is suggested to use separate branches for each implementation. All changes must then be submitted to the dev branch using pull requests. Each PR must be reviewed by at least one other person.

### Documentation - please read ! ###
In the interest of keeping the documentation simple, please write all docstring in reStructuredText (https://docutils.sourceforge.io/rst.html) format - eventually, we will try to set up auto-documentation using sphinx and readthedocs, and this will help in that task.

### Testing - again please read!
These tools are only valuable if we can rely on them to perform exactly as we expect. So, we need testing. Please create tests for every function that you make, as much as you are able. Guidance/examples here for the moment: https://github.com/GeoUtils/georaster/blob/master/test/test_georaster.py
https://github.com/corteva/rioxarray/blob/master/test/integration/test_integration__io.py



### Examples

**Coregister a DEM to another DEM**
```python
import xdem

reference_dem = "path/to/reference.tif"
dem_to_be_aligned = "path/to/dem.tif"
mask = "path/to/mask.shp"  # This is optional. Could for example be glacier outlines.

aligned_dem, error = xdem.coreg.coregister(reference_dem, dem_to_be_aligned, mask=mask)

aligned_dem.save("path/to/coreg.tif")
```
The default coregistration method is a [Nuth and Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011) implementation, but this can be changed with the keyword argument `method=...`, e.g. to `"icp"`.
The currently supported methods are: `"nuth_kaab"`, `"icp"` and `"deramp"`.

**Subtract one DEM with another**
```python
import xdem

first_dem = "path/to/first.tif"
second_dem = "path/to/second.tif"

difference = xdem.spatial_tools.subtract_rasters(first_dem, second_dem)

difference.save("path/to/difference.tif")
```
By default, `second_dem` is reprojected to fit `first_dem`.
This can be switched with the keyword argument `reference="second"`.
The resampling method can also be changed (e.g. `resampling_method="nearest"`) from the default `"cubic_spline"`.

