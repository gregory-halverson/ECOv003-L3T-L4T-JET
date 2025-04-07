import pytest

# List of dependencies
dependencies = [
    "BESS_JPL",
    "check_distribution",
    "colored_logging",
    "ECOv002_granules",
    "FLiESANN",
    "FLiESLUT",
    "GEOS5FP",
    "koppengeiger",
    "MCD12C1_2019_v006",
    "numpy",
    "pandas",
    "PMJPL",
    "PTJPLSM",
    "dateutil",
    "rasters",
    "sklearn",
    "STIC_JPL",
    "sun_angles",
    "untangle",
    "verma_net_radiation"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
