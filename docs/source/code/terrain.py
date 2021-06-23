"""Generate terrain attribute examples."""
import xdem

dem = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])

slope = xdem.terrain.slope(dem.data, resolution=dem.res)

aspect = xdem.terrain.aspect(dem.data)

hillshade = xdem.terrain.hillshade(dem.data, resolution=dem.res, azimuth=315., altitude=45.)

slope, aspect, hillshade = xdem.terrain.get_terrain_attribute(
    dem.data,
    attribute=["slope", "aspect", "hillshade"],
    resolution=dem.res
)
