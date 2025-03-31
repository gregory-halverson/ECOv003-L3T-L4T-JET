import pytest

# List of dependencies
dependencies = [
    "affine",
    "astropy",
    "breathing_earth_system_simulator",
    "colored_logging",
    "ECOv002_CMR",  
    "ECOv002_granules",
    "FLiESANN",
    "gedi_canopy_height",
    "geopandas",
    "GEOS5FP",
    "h5py",
    "koppengeiger",
    "matplotlib",
    "MCD12C1_2019_v006",
    "MOD16_JPL",
    "MODISCI",
    "msgpack",
    "msgpack_numpy",
    "netCDF4",
    "numpy",
    "pandas",
    "pycksum",
    "pykdtree",
    "pyproj",
    "pyresample",
    "rasterio",
    "rasters",
    "scipy",
    "sentinel_tiles",
    "shapely",
    "six",
    "soil_capacity_wilting",
    "solar_apparent_time",
    "sun_angles",
    "tensorflow",
    "untangle",
    "xmltodict"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
