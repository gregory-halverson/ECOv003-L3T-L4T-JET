import logging  # Used for logging messages and tracking execution.
import posixpath  # For manipulating POSIX-style paths (e.g., for file operations).
import shutil  # For high-level file operations, like zipping directories.
import socket  # For network-related operations, potentially for checking server reachability.
import sys  # Provides access to system-specific parameters and functions, used for command-line arguments and exit.
import warnings  # For issuing warnings.
from datetime import datetime  # For working with dates and times.
from os import makedirs  # For creating directories.
from os.path import join, abspath, dirname, expanduser, exists, basename  # For path manipulation (joining, absolute paths, directory names, user home, existence check, base name).
from shutil import which  # For finding the path to an executable.
from uuid import uuid4  # For generating unique identifiers.
from pytictoc import TicToc  # A simple timer for measuring code execution time.
import numpy as np  # Fundamental package for numerical computation, especially with arrays.
import pandas as pd  # For data manipulation and analysis, especially with tabular data (DataFrames).
import sklearn  # Scikit-learn, a machine learning library.
import sklearn.linear_model  # Specifically for linear regression models.
from dateutil import parser  # For parsing dates and times from various formats.
from AquaSEBS import AquaSEBS
import colored_logging as cl  # Custom module for colored console logging.

import rasters as rt  # Custom or external library for raster data processing.
from rasters import Raster, RasterGrid, RasterGeometry  # Specific classes from the rasters library for handling raster data, grids, and geometries.
from rasters import linear_downscale  # Functions for downscaling of rasters.

from check_distribution import check_distribution  # Custom module for checking and potentially visualizing data distributions.

from solar_apparent_time import UTC_offset_hours_for_area, solar_hour_of_day_for_area, solar_day_of_year_for_area  # Custom modules for solar time calculations.

from koppengeiger import load_koppen_geiger  # Custom module for loading Köppen-Geiger climate data.
import FLiESANN  # Custom module for the FLiES-ANN (Forest Light Environmental Simulator - Artificial Neural Network) model.
from GEOS5FP import GEOS5FP, FailedGEOS5FPDownload  # Custom module for interacting with GEOS-5 FP atmospheric data, including an exception for download failures.
from sun_angles import calculate_SZA_from_DOY_and_hour  # Custom module for calculating Solar Zenith Angle (SZA).

from MCD12C1_2019_v006 import load_MCD12C1_IGBP  # Custom module for loading MODIS Land Cover Type (IGBP classification) data.
from FLiESANN import FLiESANN  # Re-importing FLiESANN, potentially the main class.

from MODISCI import MODISCI
from BESS_JPL import BESS_JPL  # Custom module for the BESS-JPL (Breathing Earth System Simulator - Jet Propulsion Laboratory) model.
from PMJPL import PMJPL  # Custom module for the PMJPL (Penman-Monteith Jet Propulsion Laboratory) model.
from STIC_JPL import STIC_JPL  # Custom module for the STIC-JPL (Surface Temperature Initiated Closure - Jet Propulsion Laboratory) model.
from PTJPLSM import PTJPLSM  # Custom module for the PTJPLSM (Priestley-Taylor Jet Propulsion Laboratory - Soil Moisture) model.
from verma_net_radiation import verma_net_radiation, daylight_Rn_integration_verma  # Custom modules for net radiation calculation using Verma's model and daily integration.
from sun_angles import SHA_deg_from_DOY_lat, sunrise_from_SHA, daylight_from_SHA  # Additional solar angle calculations.

from ECOv003_L3T_L4T_JET.write_ECOv003_products import write_ECOv003_products

from ECOv003_granules import write_L3T_JET  # Functions for writing ECOSTRESS Level 3/4 products.
from ECOv003_granules import write_L3T_ETAUX
from ECOv003_granules import write_L4T_ESI
from ECOv003_granules import write_L4T_WUE

from ECOv003_granules import L2TLSTE, L2TSTARS, L3TJET, L3TSM, L3TSEB, L3TMET, L4TESI, L4TWUE  # Product classes or constants from ECOv003_granules.

from ECOv002_granules import L2TLSTE as ECOv002L2TLSTE  # Importing L2TLSTE from ECOv002_granules with an alias to avoid naming conflicts.
from ECOv002_granules import L2TSTARS as ECOv002L2TSTARS  # Importing L2TSTARS from ECOv002_granules with an alias to avoid naming conflicts.

from ECOv003_granules import ET_COLORMAP, SM_COLORMAP, WATER_COLORMAP, CLOUD_COLORMAP, RH_COLORMAP, GPP_COLORMAP  # Colormaps for visualization.

from ECOv003_exit_codes import * # Import all custom exit codes.

from .version import __version__  # Import the package version.
from .constants import * # Import all constants used in the package.
from .runconfig import read_runconfig, ECOSTRESSRunConfig  # Modules for reading and handling run configuration.

from .generate_L3T_L4T_JET_runconfig import generate_L3T_L4T_JET_runconfig  # Module for generating run configuration files.
from .L3TL4TJETConfig import L3TL4TJETConfig  # Specific run configuration class for L3T/L4T JET.

from .NDVI_to_FVC import NDVI_to_FVC  # Module for converting NDVI to Fractional Vegetation Cover.

from .sharpen_meteorology_data import sharpen_meteorology_data  # Module for sharpening meteorological data.
from .sharpen_soil_moisture_data import sharpen_soil_moisture_data  # Module for sharpening soil moisture data.

from .exceptions import *

from .version import __version__

logger = logging.getLogger(__name__)  # Get a logger instance for this module.

def L3T_L4T_JET(
        runconfig_filename: str,
        upsampling: str = None,
        downsampling: str = None,
        Rn_model_name: str = RN_MODEL_NAME,
        include_SEB_diagnostics: bool = INCLUDE_SEB_DIAGNOSTICS,
        include_JET_diagnostics: bool = INCLUDE_JET_DIAGNOSTICS,
        zero_COT_correction: bool = ZERO_COT_CORRECTION,
        sharpen_meteorology: bool = SHARPEN_METEOROLOGY,
        sharpen_soil_moisture: bool = SHARPEN_SOIL_MOISTURE,
        strip_console: bool = STRIP_CONSOLE,
        save_intermediate: bool = SAVE_INTERMEDIATE,
        show_distribution: bool = SHOW_DISTRIBUTION,
        floor_Topt: bool = FLOOR_TOPT,
        overwrite: bool = False) -> int:
    """
    Processes ECOSTRESS L2T LSTE and L2T STARS granules to produce L3T and L4T JET products (ECOSTRESS Collection 3).

    This function orchestrates the entire processing workflow, including reading run configuration,
    loading input data, performing meteorological and soil moisture sharpening, running
    evapotranspiration and gross primary productivity models (FLiES-ANN, BESS-JPL, STIC-JPL, PMJPL, PTJPLSM),
    calculating daily integrated products, and writing the output granules.

    Args:
        runconfig_filename: Path to the XML run configuration file.
        upsampling: Upsampling method for spatial resampling (e.g., 'average', 'linear'). Defaults to 'average'.
        downsampling: Downsampling method for spatial resampling (e.g., 'linear', 'average'). Defaults to 'linear'.
        Rn_model_name: Model to use for net radiation ('verma', 'BESS'). Defaults to RN_MODEL_NAME.
        include_SEB_diagnostics: Whether to include Surface Energy Balance diagnostics in the output. Defaults to INCLUDE_SEB_DIAGNOSTICS.
        include_JET_diagnostics: Whether to include JET diagnostics in the output. Defaults to INCLUDE_JET_DIAGNOSTICS.
        zero_COT_correction: Whether to set Cloud Optical Thickness to zero for correction. Defaults to ZERO_COT_CORRECTION.
        sharpen_meteorology: Whether to sharpen meteorological variables using a regression model. Defaults to SHARPEN_METEOROLOGY.
        sharpen_soil_moisture: Whether to sharpen soil moisture using a regression model. Defaults to SHARPEN_SOIL_MOISTURE.
        strip_console: Whether to strip console output from the logger. Defaults to STRIP_CONSOLE.
        save_intermediate: Whether to save intermediate processing steps. Defaults to SAVE_INTERMEDIATE.
        show_distribution: Whether to show distribution plots of intermediate and final products. Defaults to SHOW_DISTRIBUTION.
        floor_Topt: Whether to floor the optimal temperature (Topt) in the models. Defaults to FLOOR_TOPT.
        overwrite: Whether to overwrite existing output files. If False, skips processing if all output files exist. Defaults to False.

    Returns:
        An integer representing the exit code of the process.
    """
    exit_code = SUCCESS_EXIT_CODE

    if upsampling is None:
        upsampling = "average"

    if downsampling is None:
        downsampling = "linear"

    try:
        runconfig = L3TL4TJETConfig(runconfig_filename)
        working_directory = runconfig.working_directory
        granule_ID = runconfig.granule_ID
        log_filename = join(working_directory, "log", f"{granule_ID}.log")
        cl.configure(filename=log_filename, strip_console=strip_console)
        timer = TicToc()
        timer.tic()
        logger.info(f"started L3T L4T JET run at {cl.time(datetime.utcnow())} UTC")
        logger.info(f"L3T_L4T_JET PGE ({cl.val(runconfig.PGE_version)})")
        logger.info(f"L3T_L4T_JET run-config: {cl.file(runconfig_filename)}")

        L3T_JET_granule_ID = runconfig.L3T_JET_granule_ID
        logger.info(f"L3T JET granule ID: {cl.val(L3T_JET_granule_ID)}")

        L3T_JET_directory = runconfig.L3T_JET_directory
        logger.info(f"L3T JET granule directory: {cl.dir(L3T_JET_directory)}")
        L3T_JET_zip_filename = runconfig.L3T_JET_zip_filename
        logger.info(f"L3T JET zip file: {cl.file(L3T_JET_zip_filename)}")
        L3T_JET_browse_filename = runconfig.L3T_JET_browse_filename
        logger.info(f"L3T JET preview: {cl.file(L3T_JET_browse_filename)}")

        L3T_ETAUX_directory = runconfig.L3T_ETAUX_directory
        logger.info(f"L3T ETAUX granule directory: {cl.dir(L3T_ETAUX_directory)}")
        L3T_ETAUX_zip_filename = runconfig.L3T_ETAUX_zip_filename
        logger.info(f"L3T ETAUX zip file: {cl.file(L3T_ETAUX_zip_filename)}")
        L3T_ETAUX_browse_filename = runconfig.L3T_ETAUX_browse_filename
        logger.info(f"L3T ETAUX preview: {cl.file(L3T_ETAUX_browse_filename)}")

        L4T_ESI_granule_ID = runconfig.L4T_ESI_granule_ID
        logger.info(f"L4T ESI PT-JPL granule ID: {cl.val(L4T_ESI_granule_ID)}")
        L4T_ESI_directory = runconfig.L4T_ESI_directory
        logger.info(f"L4T ESI PT-JPL granule directory: {cl.dir(L4T_ESI_directory)}")
        L4T_ESI_zip_filename = runconfig.L4T_ESI_zip_filename
        logger.info(f"L4T ESI PT-JPL zip file: {cl.file(L4T_ESI_zip_filename)}")
        L4T_ESI_browse_filename = runconfig.L4T_ESI_browse_filename
        logger.info(f"L4T ESI PT-JPL preview: {cl.file(L4T_ESI_browse_filename)}")

        L4T_WUE_granule_ID = runconfig.L4T_WUE_granule_ID
        logger.info(f"L4T WUE granule ID: {cl.val(L4T_WUE_granule_ID)}")
        L4T_WUE_directory = runconfig.L4T_WUE_directory
        logger.info(f"L4T WUE granule directory: {cl.dir(L4T_WUE_directory)}")
        L4T_WUE_zip_filename = runconfig.L4T_WUE_zip_filename
        logger.info(f"L4T WUE zip file: {cl.file(L4T_WUE_zip_filename)}")
        L4T_WUE_browse_filename = runconfig.L4T_WUE_browse_filename
        logger.info(f"L4T WUE preview: {cl.file(L4T_WUE_browse_filename)}")

        required_files = [
            L3T_JET_zip_filename,
            L3T_JET_browse_filename,
            L3T_ETAUX_zip_filename,
            L3T_ETAUX_browse_filename,
            L4T_ESI_zip_filename,
            L4T_ESI_browse_filename,
            L4T_WUE_zip_filename,
            L4T_WUE_browse_filename
        ]

        some_files_missing = False

        for filename in required_files:
            if exists(filename):
                logger.info(f"found product file: {cl.file(filename)}")
            else:
                logger.info(f"product file not found: {cl.file(filename)}")
                some_files_missing = True

        if not some_files_missing and not overwrite:
            logger.info("L3T_L4T_JET output already found (use --overwrite flag to regenerate)")
            return SUCCESS_EXIT_CODE
        elif not some_files_missing and overwrite:
            logger.info("L3T_L4T_JET output already found but overwrite flag is set, proceeding with processing")

        logger.info(f"working_directory: {cl.dir(working_directory)}")
        output_directory = runconfig.output_directory
        logger.info(f"output directory: {cl.dir(output_directory)}")
        sources_directory = runconfig.sources_directory
        logger.info(f"sources directory: {cl.dir(sources_directory)}")
        GEOS5FP_directory = runconfig.GEOS5FP_directory
        logger.info(f"GEOS-5 FP directory: {cl.dir(GEOS5FP_directory)}")
        static_directory = runconfig.static_directory
        logger.info(f"static directory: {cl.dir(static_directory)}")
        GEDI_directory = runconfig.GEDI_directory
        logger.info(f"GEDI directory: {cl.dir(GEDI_directory)}")
        MODISCI_directory = runconfig.MODISCI_directory
        logger.info(f"MODIS CI directory: {cl.dir(MODISCI_directory)}")
        MCD12_directory = runconfig.MCD12_directory
        logger.info(f"MCD12C1 IGBP directory: {cl.dir(MCD12_directory)}")
        soil_grids_directory = runconfig.soil_grids_directory
        logger.info(f"SoilGrids directory: {cl.dir(soil_grids_directory)}")
        logger.info(f"log: {cl.file(log_filename)}")
        orbit = runconfig.orbit
        logger.info(f"orbit: {cl.val(orbit)}")
        scene = runconfig.scene
        logger.info(f"scene: {cl.val(scene)}")
        tile = runconfig.tile
        logger.info(f"tile: {cl.val(tile)}")
        build = runconfig.build
        logger.info(f"build: {cl.val(build)}")
        product_counter = runconfig.product_counter
        logger.info(f"product counter: {cl.val(product_counter)}")
        L2T_LSTE_filename = runconfig.L2T_LSTE_filename
        logger.info(f"L2T_LSTE file: {cl.file(L2T_LSTE_filename)}")
        L2T_STARS_filename = runconfig.L2T_STARS_filename
        logger.info(f"L2T_STARS file: {cl.file(L2T_STARS_filename)}")

        if not exists(L2T_LSTE_filename):
            raise InputFilesInaccessible(f"L2T LSTE file does not exist: {L2T_LSTE_filename}")

        # Check the basename of the file to determine collection, not the full path
        L2T_LSTE_basename = basename(L2T_LSTE_filename)
        if "ECOv003" in L2T_LSTE_basename:
            L2T_LSTE_granule = L2TLSTE(L2T_LSTE_filename)
        elif "ECOv002" in L2T_LSTE_basename:
            L2T_LSTE_granule = ECOv002L2TLSTE(L2T_LSTE_filename)
        else:
            raise ValueError(f"collection not recognized in L2T LSTE filename: {L2T_LSTE_filename}")

        if not exists(L2T_STARS_filename):
            raise InputFilesInaccessible(f"L2T STARS file does not exist: {L2T_STARS_filename}")

        # Check the basename of the file to determine collection, not the full path
        L2T_STARS_basename = basename(L2T_STARS_filename)
        if "ECOv003" in L2T_STARS_basename:
            L2T_STARS_granule = L2TSTARS(L2T_STARS_filename)
        elif "ECOv002" in L2T_STARS_basename:
            L2T_STARS_granule = ECOv002L2TSTARS(L2T_STARS_filename)
        else:
            raise ValueError(f"collection not recognized in L2T STARS filename: {L2T_STARS_filename}")

        metadata = L2T_STARS_granule.metadata_dict
        metadata["StandardMetadata"]["PGEVersion"] = __version__
        metadata["StandardMetadata"]["PGEName"] = "L3T_L4T_JET"
        metadata["StandardMetadata"]["ProcessingLevelID"] = "L3T"
        metadata["StandardMetadata"]["SISName"] = "Level 3 Product Specification Document"
        metadata["StandardMetadata"]["SISVersion"] = "Preliminary"
        metadata["StandardMetadata"]["AuxiliaryInputPointer"] = "AuxiliaryNWP"

        geometry = L2T_LSTE_granule.geometry
        time_UTC = L2T_LSTE_granule.time_UTC
        logger.info(f"overpass time: {cl.time(time_UTC)} UTC")
        date_UTC = time_UTC.date()
        logger.info(f"overpass date: {cl.time(date_UTC)} UTC")
        time_solar = L2T_LSTE_granule.time_solar
        logger.info(
            f"orbit {cl.val(orbit)} scene {cl.val(scene)} tile {cl.place(tile)} overpass time: {cl.time(time_UTC)} UTC ({cl.time(time_solar)} solar)")
        timestamp = f"{time_UTC:%Y%m%dT%H%M%S}"

        hour_of_day = solar_hour_of_day_for_area(time_UTC=time_UTC, geometry=geometry)
        day_of_year = solar_day_of_year_for_area(time_UTC=time_UTC, geometry=geometry)

        logger.info("reading surface temperature from L2T LSTE product")
        ST_K = L2T_LSTE_granule.ST_K
        ST_C = ST_K - 273.15
        check_distribution(ST_C, "ST_C", date_UTC=date_UTC, target=tile)

        logger.info(f"reading elevation from L2T LSTE: {L2T_LSTE_granule.product_filename}")
        elevation_km = L2T_LSTE_granule.elevation_km
        check_distribution(elevation_km, "elevation_km", date_UTC=date_UTC, target=tile)

        emissivity = L2T_LSTE_granule.emissivity
        water_mask = L2T_LSTE_granule.water

        logger.info("reading cloud mask from L2T LSTE product")
        cloud_mask = L2T_LSTE_granule.cloud
        check_distribution(cloud_mask, "cloud_mask", date_UTC=date_UTC, target=tile)

        logger.info("reading NDVI from L2T STARS product")
        NDVI = L2T_STARS_granule.NDVI
        check_distribution(NDVI, "NDVI", date_UTC=date_UTC, target=tile)

        logger.info("reading albedo from L2T STARS product")
        albedo = L2T_STARS_granule.albedo
        check_distribution(albedo, "albedo", date_UTC=date_UTC, target=tile)

        percent_cloud = 100 * np.count_nonzero(cloud_mask) / cloud_mask.size
        metadata["ProductMetadata"]["QAPercentCloudCover"] = percent_cloud

        GEOS5FP_connection = GEOS5FP(
            download_directory=GEOS5FP_directory
        )

        MODISCI_connection = MODISCI(directory=MODISCI_directory)

        SZA_deg = calculate_SZA_from_DOY_and_hour(
            lat=geometry.lat,
            lon=geometry.lon,
            DOY=day_of_year,
            hour=hour_of_day
        )

        check_distribution(SZA_deg, "SZA", date_UTC=date_UTC, target=tile)

        if np.all(SZA_deg >= SZA_DEGREE_CUTOFF):
            raise DaytimeFilter(f"solar zenith angle exceeds {SZA_DEGREE_CUTOFF} for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        logger.info("retrieving GEOS-5 FP aerosol optical thickness raster")
        AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry)
        check_distribution(AOT, "AOT", date_UTC=date_UTC, target=tile)

        logger.info("generating GEOS-5 FP cloud optical thickness raster")
        COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry)
        check_distribution(COT, "COT", date_UTC=date_UTC, target=tile)

        logger.info("generating GEOS5-FP water vapor raster in grams per square centimeter")
        vapor_gccm = GEOS5FP_connection.vapor_gccm(time_UTC=time_UTC, geometry=geometry)
        check_distribution(vapor_gccm, "vapor_gccm", date_UTC=date_UTC, target=tile)

        logger.info("generating GEOS5-FP ozone raster in grams per square centimeter")
        ozone_cm = GEOS5FP_connection.ozone_cm(time_UTC=time_UTC, geometry=geometry)
        check_distribution(ozone_cm, "ozone_cm", date_UTC=date_UTC, target=tile)

        logger.info(f"running Forest Light Environmental Simulator for {cl.place(tile)} at {cl.time(time_UTC)} UTC")

        doy_solar = time_solar.timetuple().tm_yday
        KG_climate = load_koppen_geiger(albedo.geometry)

        if zero_COT_correction:
            COT = COT * 0.0
            
        elevation_m = elevation_km * 1000

        FLiES_results = FLiESANN(
            albedo=albedo,
            geometry=geometry,
            time_UTC=time_UTC,
            day_of_year=doy_solar,
            hour_of_day=hour_of_day,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            elevation_m=elevation_m,
            SZA_deg=SZA_deg,
            KG_climate=KG_climate,
            GEOS5FP_connection=GEOS5FP_connection,
        )
        
        # Updated variable names to match new FLiESANN results dictionary

        SWin_TOA_Wm2 = FLiES_results["SWin_TOA_Wm2"]        # Previously "Ra"
        SWin_FLiES_ANN_raw = FLiES_results["SWin_Wm2"]      # Previously "Rg"
        UV_Wm2 = FLiES_results["UV_Wm2"]              # Previously "UV"
        PAR_Wm2 = FLiES_results["PAR_Wm2"]            # Previously "VIS"
        NIR_Wm2 = FLiES_results["NIR_Wm2"]            # Previously "NIR"
        PAR_diffuse_Wm2 = FLiES_results["PAR_diffuse_Wm2"] # Previously "VISdiff"
        NIR_diffuse_Wm2 = FLiES_results["NIR_diffuse_Wm2"] # Previously "NIRdiff"
        PAR_direct_Wm2 = FLiES_results["PAR_direct_Wm2"]  # Previously "VISdir"
        NIR_direct_Wm2 = FLiES_results["NIR_direct_Wm2"]  # Previously "NIRdir"

        albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry)
        RVIS_NWP = GEOS5FP_connection.ALBVISDR(time_UTC=time_UTC, geometry=geometry)
        albedo_visible = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
        check_distribution(albedo_visible, "albedo_visible")
        RNIR_NWP = GEOS5FP_connection.ALBNIRDR(time_UTC=time_UTC, geometry=geometry)
        albedo_NIR = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
        check_distribution(albedo_NIR, "albedo_NIR")
        PAR_direct_Wm2 = PAR_direct_Wm2
        check_distribution(PAR_direct_Wm2, "PAR_direct_Wm2")

        coarse_geometry = geometry.rescale(GEOS_IN_SENTINEL_COARSE_CELL_SIZE)

        # Use raw FLiES-ANN output directly without bias correction
        SWin_Wm2 = SWin_FLiES_ANN_raw

        check_distribution(SWin_Wm2, "SWin_FLiES_ANN", date_UTC=date_UTC, target=tile)

        # Use FLiES-ANN solar radiation exclusively
        SWin = SWin_Wm2
        SWin = rt.where(np.isnan(ST_K), np.nan, SWin)

        if np.all(np.isnan(SWin)) or np.all(SWin == 0):
            raise BlankOutput(f"blank solar radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        # Sharpen meteorological variables if enabled.
        if sharpen_meteorology:
            try:
                Ta_C, RH, Ta_C_smooth = sharpen_meteorology_data(
                    ST_C=ST_C,
                    NDVI=NDVI,
                    albedo=albedo,
                    geometry=geometry,
                    coarse_geometry=coarse_geometry,
                    time_UTC=time_UTC,
                    date_UTC=date_UTC,
                    tile=tile,
                    orbit=orbit,
                    scene=scene,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    GEOS5FP_connection=GEOS5FP_connection
                )
            except Exception as e:
                logger.error(e)
                logger.warning("unable to sharpen meteorology")
                Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
                Ta_C_smooth = Ta_C
                RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
        else:
            Ta_C = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
            Ta_C_smooth = Ta_C
            RH = GEOS5FP_connection.RH(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)

        # Sharpen soil moisture if enabled.
        if sharpen_soil_moisture:
            try:
                SM = sharpen_soil_moisture_data(
                    ST_C=ST_C,
                    NDVI=NDVI,
                    albedo=albedo,
                    water_mask=water_mask,
                    geometry=geometry,
                    coarse_geometry=coarse_geometry,
                    time_UTC=time_UTC,
                    date_UTC=date_UTC,
                    tile=tile,
                    orbit=orbit,
                    scene=scene,
                    upsampling=upsampling,
                    downsampling=downsampling,
                    GEOS5FP_connection=GEOS5FP_connection
                )
            except Exception as e:
                logger.error(e)
                logger.warning("unable to sharpen soil moisture")
                SM = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
        else:
            SM = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)

        # Calculate Saturated Vapor Pressure (SVP_Pa) and Actual Vapor Pressure (Ea_Pa, Ea_kPa).
        SVP_Pa = 0.6108 * np.exp((17.27 * Ta_C) / (Ta_C + 237.3)) * 1000  # [Pa]
        Ea_Pa = RH * SVP_Pa
        Ea_kPa = Ea_Pa / 1000
        Ta_K = Ta_C + 273.15

        logger.info(f"running Breathing Earth System Simulator for {cl.place(tile)} at {cl.time(time_UTC)} UTC")

        BESS_results = BESS_JPL(
            ST_C=ST_C,
            NDVI=NDVI,
            albedo=albedo,
            elevation_m=elevation_m,
            geometry=geometry,
            time_UTC=time_UTC,
            hour_of_day=hour_of_day,
            day_of_year=day_of_year,
            GEOS5FP_connection=GEOS5FP_connection,
            MODISCI_connection=MODISCI_connection,
            Ta_C=Ta_C,
            RH=RH,
            SWin_Wm2=SWin_Wm2,
            PAR_diffuse_Wm2=PAR_diffuse_Wm2,
            PAR_direct_Wm2=PAR_direct_Wm2,
            NIR_diffuse_Wm2=NIR_diffuse_Wm2,
            NIR_direct_Wm2=NIR_direct_Wm2,
            UV_Wm2=UV_Wm2,
            albedo_visible=albedo_visible,
            albedo_NIR=albedo_NIR,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            KG_climate=KG_climate,
            SZA_deg=SZA_deg,
            GEDI_download_directory=GEDI_directory,
            upscale_to_daylight=True
        )

        Rn_BESS_Wm2 = BESS_results["Rn_Wm2"]
        check_distribution(Rn_BESS_Wm2, "Rn_BESS_Wm2", date_UTC=date_UTC, target=tile)
        G_BESS_Wm2 = BESS_results["G_Wm2"]
        check_distribution(Rn_BESS_Wm2, "Rn_BESS_Wm2", date_UTC=date_UTC, target=tile)
        
        LE_BESS_Wm2 = BESS_results["LE_Wm2"]
        check_distribution(LE_BESS_Wm2, "LE_BESS_Wm2", date_UTC=date_UTC, target=tile)
        
        # FIXME BESS needs to generate ET_daylight_kg
        ET_daylight_BESS_kg = BESS_results["ET_daylight_kg"]

        ## an need to revise evaporative fraction to take soil heat flux into account
        EF_BESS = rt.where((LE_BESS_Wm2 == 0) | ((Rn_BESS_Wm2 - G_BESS_Wm2) == 0), 0, LE_BESS_Wm2 / (Rn_BESS_Wm2 - G_BESS_Wm2))
        
        Rn_daily_BESS = daylight_Rn_integration_verma(
            Rn_Wm2=Rn_BESS_Wm2,
            time_UTC=time_UTC,
            geometry=geometry
        )

        LE_daily_BESS = rt.clip(EF_BESS * Rn_daily_BESS, 0, None)

        if water_mask is not None:
            LE_BESS_Wm2 = rt.where(water_mask, np.nan, LE_BESS_Wm2)

        check_distribution(LE_BESS_Wm2, "LE_BESS_Wm2", date_UTC=date_UTC, target=tile)
        
        GPP_inst_umol_m2_s = BESS_results["GPP"]
        
        if water_mask is not None:
            GPP_inst_umol_m2_s = rt.where(water_mask, np.nan, GPP_inst_umol_m2_s)

        check_distribution(GPP_inst_umol_m2_s, "GPP_inst_umol_m2_s", date_UTC=date_UTC, target=tile)

        if np.all(np.isnan(GPP_inst_umol_m2_s)):
            raise BlankOutput(f"blank GPP output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        NWP_filenames = sorted([posixpath.basename(filename) for filename in GEOS5FP_connection.filenames])
        AuxiliaryNWP = ",".join(NWP_filenames)
        metadata["ProductMetadata"]["AuxiliaryNWP"] = AuxiliaryNWP

        verma_results = verma_net_radiation(
            SWin_Wm2=SWin_Wm2,
            albedo=albedo,
            ST_C=ST_C,
            emissivity=emissivity,
            Ta_C=Ta_C,
            RH=RH
        )

        Rn_verma_Wm2 = verma_results["Rn_Wm2"]

        if Rn_model_name == "verma":
            Rn_Wm2 = Rn_verma_Wm2
        elif Rn_model_name == "BESS":
            Rn_Wm2 = Rn_BESS_Wm2
        else:
            raise ValueError(f"unrecognized net radiation model: {Rn_model_name}")

        if np.all(np.isnan(Rn_Wm2)) or np.all(Rn_Wm2 == 0):
            raise BlankOutput(f"blank net radiation output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        STIC_results = STIC_JPL(
            geometry=geometry,
            time_UTC=time_UTC,
            Rn_Wm2=Rn_Wm2,
            RH=RH,
            Ta_C=Ta_C_smooth,
            ST_C=ST_C,
            albedo=albedo,
            emissivity=emissivity,
            NDVI=NDVI,
            max_iterations=3,
            upscale_to_daylight=True
        )

        LE_STIC_Wm2 = STIC_results["LE_Wm2"]
        check_distribution(LE_STIC_Wm2, "LE_STIC_Wm2", date_UTC=date_UTC, target=tile)
        
        ET_daylight_STIC_kg = STIC_results["ET_daylight_kg"]
        check_distribution(ET_daylight_STIC_kg, "ET_daylight_STIC_kg", date_UTC=date_UTC, target=tile)
        
        LE_canopy_STIC_Wm2 = STIC_results["LE_canopy_Wm2"]
        check_distribution(LE_canopy_STIC_Wm2, "LE_canoy_STIC_Wm2", date_UTC=date_UTC, target=tile)
        
        G_STIC_Wm2 = STIC_results["G_Wm2"]
        check_distribution(G_STIC_Wm2, "G_STIC_Wm2", date_UTC=date_UTC, target=tile)

        LE_canopy_fraction_STIC = rt.clip(rt.where((LE_canopy_STIC_Wm2 == 0) | (LE_STIC_Wm2 == 0), 0, LE_canopy_STIC_Wm2 / LE_STIC_Wm2), 0, 1)
        check_distribution(LE_canopy_fraction_STIC, "LE_canopy_fraction_STIC", date_UTC=date_UTC, target=tile)

        ## FIXME need to revise evaporative fraction to take soil heat flux into account
        EF_STIC = rt.where((LE_STIC_Wm2 == 0) | ((Rn_Wm2 - G_STIC_Wm2) == 0), 0, LE_STIC_Wm2 / (Rn_Wm2 - G_STIC_Wm2))

        PTJPLSM_results = PTJPLSM(
            geometry=geometry,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            Rn_Wm2=Rn_Wm2,
            Ta_C=Ta_C,
            RH=RH,
            soil_moisture=SM,
            field_capacity_directory=soil_grids_directory,
            wilting_point_directory=soil_grids_directory,
            canopy_height_directory=GEDI_directory,
            upscale_to_daylight=True
        )

        LE_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_Wm2"], 0, None)
        check_distribution(LE_PTJPLSM_Wm2, "LE_PTJPLSM_Wm2", date_UTC=date_UTC, target=tile)
        
        ET_daylight_PTJPLSM_kg = PTJPLSM_results["ET_daylight_kg"]
        check_distribution(ET_daylight_PTJPLSM_kg, "ET_daylight_PTJPLSM_kg", date_UTC=date_UTC, target=tile)
        
        G_PTJPLSM = PTJPLSM_results["G_Wm2"]
        check_distribution(G_PTJPLSM, "G_PTJPLSM", date_UTC=date_UTC, target=tile)

        EF_PTJPLSM = rt.where((LE_PTJPLSM_Wm2 == 0) | ((Rn_Wm2 - G_PTJPLSM) == 0), 0, LE_PTJPLSM_Wm2 / (Rn_Wm2 - G_PTJPLSM))
        check_distribution(EF_PTJPLSM, "EF_PTJPLSM", date_UTC=date_UTC, target=tile)

        if np.all(np.isnan(LE_PTJPLSM_Wm2)):
            raise BlankOutput(
                f"blank PT-JPL-SM instantaneous ET output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        if np.all(np.isnan(LE_PTJPLSM_Wm2)):
            raise BlankOutput(
                f"blank daily ET output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        LE_canopy_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_canopy_Wm2"], 0, None)
        check_distribution(LE_canopy_PTJPLSM_Wm2, "LE_canopy_PTJPLSM_Wm2", date_UTC=date_UTC, target=tile)

        LE_canopy_fraction_PTJPLSM = rt.clip(LE_canopy_PTJPLSM_Wm2 / LE_PTJPLSM_Wm2, 0, 1)
        check_distribution(LE_canopy_fraction_PTJPLSM, "LE_canopy_fraction_PTJPLSM", date_UTC=date_UTC, target=tile)

        if water_mask is not None:
            LE_canopy_fraction_PTJPLSM = rt.where(water_mask, np.nan, LE_canopy_fraction_PTJPLSM)
        
        LE_soil_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_soil_Wm2"], 0, None)
        check_distribution(LE_soil_PTJPLSM_Wm2, "LE_soil_PTJPLSM_Wm2", date_UTC=date_UTC, target=tile)

        LE_soil_fraction_PTJPLSM = rt.clip(LE_soil_PTJPLSM_Wm2 / LE_PTJPLSM_Wm2, 0, 1)
        
        if water_mask is not None:
            LE_soil_fraction_PTJPLSM = rt.where(water_mask, np.nan, LE_soil_fraction_PTJPLSM)
        
        check_distribution(LE_soil_fraction_PTJPLSM, "LE_soil_fraction_PTJPLSM", date_UTC=date_UTC, target=tile)
        
        LE_interception_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_interception_Wm2"], 0, None)
        check_distribution(LE_interception_PTJPLSM_Wm2, "LE_interception_PTJPLSM_Wm2", date_UTC=date_UTC, target=tile)

        LE_interception_fraction_PTJPLSM = rt.clip(LE_interception_PTJPLSM_Wm2 / LE_PTJPLSM_Wm2, 0, 1)
        
        if water_mask is not None:
            LE_interception_fraction_PTJPLSM = rt.where(water_mask, np.nan, LE_interception_fraction_PTJPLSM)
        
        check_distribution(LE_interception_fraction_PTJPLSM, "LE_interception_fraction_PTJPLSM", date_UTC=date_UTC, target=tile)
        
        PET_instantaneous_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["PET_Wm2"], 0, None)
        check_distribution(PET_instantaneous_PTJPLSM_Wm2, "PET_instantaneous_PTJPLSM_Wm2", date_UTC=date_UTC, target=tile)

        ESI_PTJPLSM = rt.clip(LE_PTJPLSM_Wm2 / PET_instantaneous_PTJPLSM_Wm2, 0, 1)

        if water_mask is not None:
            ESI_PTJPLSM = rt.where(water_mask, np.nan, ESI_PTJPLSM)

        check_distribution(ESI_PTJPLSM, "ESI_PTJPLSM", date_UTC=date_UTC, target=tile)

        if np.all(np.isnan(ESI_PTJPLSM)):
            raise BlankOutput(f"blank ESI output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

        PMJPL_results = PMJPL(
            geometry=geometry,
            time_UTC=time_UTC,
            ST_C=ST_C,
            emissivity=emissivity,
            NDVI=NDVI,
            albedo=albedo,
            Ta_C=Ta_C,
            RH=RH,
            elevation_km=elevation_km,
            Rn_Wm2=Rn_Wm2,
            GEOS5FP_connection=GEOS5FP_connection,
            upscale_to_daylight=True
        )

        LE_PMJPL_Wm2 = PMJPL_results["LE_Wm2"]
        check_distribution(LE_PMJPL_Wm2, "LE_PMJPL_Wm2", date_UTC=date_UTC, target=tile)
        
        ET_daylight_PMJPL_kg = PMJPL_results["ET_daylight_kg"]
        check_distribution(ET_daylight_PMJPL_kg, "ET_daylight_PMJ", date_UTC=date_UTC, target=tile)
        
        G_PMJPL_Wm2 = PMJPL_results["G_Wm2"]
        check_distribution(G_PMJPL_Wm2, "G_PMJPL_Wm2", date_UTC=date_UTC, target=tile)

        # FIXME get rid of the instantaneous latent heat flux aggregation
        LE_instantaneous_Wm2 = rt.Raster(
            np.nanmedian([np.array(LE_PTJPLSM_Wm2), np.array(LE_BESS_Wm2), np.array(LE_PMJPL_Wm2), np.array(LE_STIC_Wm2)], axis=0),
            geometry=geometry)

        windspeed_mps = GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
        check_distribution(windspeed_mps, "windspeed_mps", date_UTC=date_UTC, target=tile)
        
        SWnet_Wm2 = SWin_Wm2 * (1 - albedo)
        check_distribution(SWnet_Wm2, "SWnet_Wm2", date_UTC=date_UTC, target=tile)

        # Adding debugging statements for input rasters before the AquaSEBS call
        logger.info("checking input distributions for AquaSEBS")
        check_distribution(ST_C, "ST_C", date_UTC=date_UTC, target=tile)
        check_distribution(emissivity, "emissivity", date_UTC=date_UTC, target=tile)
        check_distribution(albedo, "albedo", date_UTC=date_UTC, target=tile)
        check_distribution(Ta_C, "Ta_C", date_UTC=date_UTC, target=tile)
        check_distribution(RH, "RH", date_UTC=date_UTC, target=tile)
        check_distribution(windspeed_mps, "windspeed_mps", date_UTC=date_UTC, target=tile)
        check_distribution(SWnet_Wm2, "SWnet", date_UTC=date_UTC, target=tile)
        check_distribution(Rn_Wm2, "Rn_Wm2", date_UTC=date_UTC, target=tile)
        check_distribution(SWin_Wm2, "SWin_Wm2", date_UTC=date_UTC, target=tile)

        # FIXME AquaSEBS need to do daylight upscaling
        AquaSEBS_results = AquaSEBS(
            WST_C=ST_C,
            emissivity=emissivity,
            albedo=albedo,
            Ta_C=Ta_C,
            RH=RH,
            windspeed_mps=windspeed_mps,
            SWnet=SWnet_Wm2,
            Rn_Wm2=Rn_Wm2,
            SWin_Wm2=SWin_Wm2,
            geometry=geometry,
            time_UTC=time_UTC,
            water=water_mask,
            GEOS5FP_connection=GEOS5FP_connection,
            upscale_to_daylight=True
        )

        for key, value in AquaSEBS_results.items():
            check_distribution(value, key)

        # FIXME need to revise how the water surface evaporation is inserted into the JET product

        LE_AquaSEBS_Wm2 = AquaSEBS_results["LE_Wm2"]
        check_distribution(LE_AquaSEBS_Wm2, "LE_AquaSEBS_Wm2", date_UTC=date_UTC, target=tile)
        
        LE_instantaneous_Wm2 = rt.where(water_mask, LE_AquaSEBS_Wm2, LE_instantaneous_Wm2)
        check_distribution(LE_instantaneous_Wm2, "LE_instantaneous_Wm2", date_UTC=date_UTC, target=tile)
        
        ET_daylight_AquaSEBS_kg = AquaSEBS_results["ET_daylight_kg"]
        check_distribution(ET_daylight_AquaSEBS_kg, "ET_daylight_AquaSEBS_kg", date_UTC=date_UTC, target=tile)
        
        ## FIXME need to revise evaporative fraction to take soil heat flux into account
        EF_PMJPL = rt.where((LE_PMJPL_Wm2 == 0) | ((Rn_Wm2 - G_PMJPL_Wm2) == 0), 0, LE_PMJPL_Wm2 / (Rn_Wm2 - G_PMJPL_Wm2))
        check_distribution(EF_PMJPL, "EF_PMJPL", date_UTC=date_UTC, target=tile)

        ## FIXME need to revise evaporative fraction to take soil heat flux into account
        EF = rt.where((LE_instantaneous_Wm2 == 0) | (Rn_Wm2 == 0), 0, LE_instantaneous_Wm2 / Rn_Wm2)
        check_distribution(EF, "EF", date_UTC=date_UTC, target=tile)

        SHA_deg = SHA_deg_from_DOY_lat(day_of_year, geometry.lat)
        check_distribution(SHA_deg, "SHA_deg", date_UTC=date_UTC, target=tile)
        sunrise_hour = sunrise_from_SHA(SHA_deg)
        check_distribution(sunrise_hour, "sunrise_hour", date_UTC=date_UTC, target=tile)
        daylight_hours = daylight_from_SHA(SHA_deg)
        check_distribution(daylight_hours, "daylight_hours", date_UTC=date_UTC, target=tile)

        Rn_daylight_Wm2 = daylight_Rn_integration_verma(
            Rn_Wm2=Rn_Wm2,
            time_UTC=time_UTC,
            geometry=geometry
        )

        Rn_daylight_Wm2 = rt.clip(Rn_daylight_Wm2, 0, None)
        check_distribution(Rn_daylight_Wm2, "Rn_daylight_Wm2", date_UTC=date_UTC, target=tile)
        
        LE_daylight_Wm2 = rt.clip(EF * Rn_daylight_Wm2, 0, None)
        check_distribution(LE_daylight_Wm2, "LE_daylight_Wm2", date_UTC=date_UTC, target=tile)

        daylight_seconds = daylight_hours * 3600.0
        check_distribution(daylight_seconds, "daylight_seconds", date_UTC=date_UTC, target=tile)

        # # replace the daily ET calculations here with an aggregatation of the daylight upscaled values produced by the model packages
        # ET_daylight_kg = np.clip(LE_daylight_Wm2 * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)
        # check_distribution(ET_daily_kg, "ET_daily_kg", date_UTC=date_UTC, target=tile)

        # ET_daylight_BESS_kg = np.clip(LE_daily_BESS * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)
        # LE_daylight_STIC_Wm2 = rt.clip(EF_STIC * Rn_daylight_Wm2, 0, None)
        # ET_daylight_STIC_Wm2 = np.clip(LE_daylight_STIC_Wm2 * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)
        # LE_daylight_PTJPLSM_Wm2 = rt.clip(EF_PTJPLSM * Rn_daylight_Wm2, 0, None)
        # ET_daylight_PTJPLSM_kg = np.clip(LE_daylight_PTJPLSM_Wm2 * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)
        # LE_daylight_PMJPL_Wm2 = rt.clip(EF_PMJPL * Rn_daylight_Wm2, 0, None)
        # ET_daylight_PMJPL_kg = np.clip(LE_daylight_PMJPL_Wm2 * daylight_seconds / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM, 0, None)

        # ET_daylight_uncertainty = rt.Raster(
        #     np.nanstd([np.array(LE_PTJPLSM_Wm2), np.array(LE_BESS_Wm2), np.array(LE_PMJPL_Wm2), np.array(LE_STIC_Wm2)], axis=0),
        #     geometry=geometry).mask(~water_mask)
        
        ET_daylight_kg = np.nanmedian([
            np.array(ET_daylight_PTJPLSM_kg),
            np.array(ET_daylight_BESS_kg),
            np.array(ET_daylight_PMJPL_kg),
            np.array(ET_daylight_STIC_kg)
        ], axis=0)
        
        if isinstance(geometry, RasterGeometry):
            ET_daylight_kg = rt.Raster(ET_daylight_kg, geometry=geometry)
        
        # overlay water surface evaporation on top of daylight evapotranspiration aggregate
        ET_daylight_kg = rt.where(np.isnan(ET_daylight_AquaSEBS_kg), ET_daylight_kg, ET_daylight_AquaSEBS_kg)
        check_distribution(ET_daylight_kg, "ET_daylight_kg", date_UTC=date_UTC, target=tile)

        ET_uncertainty = np.nanstd([
            np.array(ET_daylight_PTJPLSM_kg),
            np.array(ET_daylight_BESS_kg),
            np.array(ET_daylight_PMJPL_kg),
            np.array(ET_daylight_STIC_kg)
        ], axis=0)
        
        if isinstance(geometry, RasterGeometry):
            ET_uncertainty = rt.Raster(ET_uncertainty, geometry=geometry)

        GPP_inst_g_m2_s = GPP_inst_umol_m2_s / 1000000 * 12.011
        ET_canopy_inst_kg_m2_s = LE_canopy_PTJPLSM_Wm2 / LATENT_VAPORIZATION_JOULES_PER_KILOGRAM
        WUE = GPP_inst_g_m2_s / ET_canopy_inst_kg_m2_s
        WUE = rt.where(np.isinf(WUE), np.nan, WUE)
        WUE = rt.clip(WUE, 0, 10)

        metadata["StandardMetadata"]["CollectionLabel"] = "ECOv003"

        write_ECOv003_products(
            runconfig=runconfig,
            metadata=metadata,
            LE_PTJPLSM_Wm2=LE_PTJPLSM_Wm2,
            ET_daylight_PTJPLSM_kg=ET_daylight_PTJPLSM_kg,
            LE_STIC_Wm2=LE_STIC_Wm2,
            ET_daylight_STIC_kg=ET_daylight_STIC_kg,
            LE_BESS_Wm2=LE_BESS_Wm2,
            ET_daylight_BESS_kg=ET_daylight_BESS_kg,
            LE_PMJPL_Wm2=LE_PMJPL_Wm2,
            ET_daylight_PMJPL_kg=ET_daylight_PMJPL_kg,
            ET_daylight_kg=ET_daylight_kg,
            ET_uncertainty=ET_uncertainty,
            LE_canopy_fraction_PTJPLSM=LE_canopy_fraction_PTJPLSM,
            LE_canopy_fraction_STIC=LE_canopy_fraction_STIC,
            LE_soil_fraction_PTJPLSM=LE_soil_fraction_PTJPLSM,
            LE_interception_fraction_PTJPLSM=LE_interception_fraction_PTJPLSM,
            water_mask=water_mask,
            cloud_mask=cloud_mask,
            Ta_C=Ta_C,
            RH=RH,
            Rn_Wm2=Rn_Wm2,
            SWin=SWin,
            SM=SM,
            ESI_PTJPLSM=ESI_PTJPLSM,
            PET_instantaneous_PTJPLSM_Wm2=PET_instantaneous_PTJPLSM_Wm2,
            WUE=WUE,
            GPP_inst_g_m2_s=GPP_inst_g_m2_s
        )

        logger.info(f"finished L3T L4T JET run in {cl.time(timer.tocvalue())} seconds")

    except (BlankOutput, BlankOutputError) as exception:
        logger.exception(exception)
        exit_code = BLANK_OUTPUT

    except (FailedGEOS5FPDownload, ConnectionError, LPDAACServerUnreachable) as exception:
        logger.exception(exception)
        exit_code = AUXILIARY_SERVER_UNREACHABLE

    except ECOSTRESSExitCodeException as exception:
        logger.exception(exception)
        exit_code = exception.exit_code

    return exit_code


def main(argv=sys.argv):
    """
    Main function to parse command line arguments and run the L3T_L4T_JET process.

    Args:
        argv: Command line arguments. Defaults to sys.argv.

    Returns:
        An integer representing the exit code.
    """
    if len(argv) == 1 or "--version" in argv:
        print(f"L3T/L4T JET PGE ({__version__})")
        print(f"usage: ECOv003-L3T-L4T-JET RunConfig.xml [--overwrite] [--strip-console] [--save-intermediate] [--show-distribution]")
        print(f"  --overwrite: Overwrite existing output files if they exist")
        print(f"  --strip-console: Strip console output from logger")
        print(f"  --save-intermediate: Save intermediate processing steps")
        print(f"  --show-distribution: Show distribution plots")

        if "--version" in argv:
            return SUCCESS_EXIT_CODE
        else:
            return RUNCONFIG_FILENAME_NOT_SUPPLIED

    strip_console = "--strip-console" in argv
    save_intermediate = "--save-intermediate" in argv
    show_distribution = "--show-distribution" in argv
    overwrite = "--overwrite" in argv
    runconfig_filename = str(argv[1])

    exit_code = L3T_L4T_JET(
        runconfig_filename=runconfig_filename,
        strip_console=strip_console,
        save_intermediate=save_intermediate,
        show_distribution=show_distribution,
        overwrite=overwrite
    )

    logger.info(f"L3T/L4T JET exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
