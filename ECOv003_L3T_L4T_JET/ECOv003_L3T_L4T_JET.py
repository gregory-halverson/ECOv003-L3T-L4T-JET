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

from .read_ECOv003_inputs import read_ECOv003_inputs
from .read_ECOv003_configuration import read_ECOv003_configuration  # Module for reading ECOv003 input data.
from .JET import JET

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
        # Read and process configuration
        config = read_ECOv003_configuration(
            runconfig_filename=runconfig_filename,
            strip_console=strip_console,
            overwrite=overwrite
        )

        # Check if we should exit early (output already exists and no overwrite)
        if config['should_exit']:
            return config['exit_code']

        # Unpack configuration variables
        runconfig = config['runconfig']
        timer = config['timer']
        working_directory = config['working_directory']
        granule_ID = config['granule_ID']
        log_filename = config['log_filename']
        L3T_JET_granule_ID = config['L3T_JET_granule_ID']
        L3T_JET_directory = config['L3T_JET_directory']
        L3T_JET_zip_filename = config['L3T_JET_zip_filename']
        L3T_JET_browse_filename = config['L3T_JET_browse_filename']
        L3T_ETAUX_directory = config['L3T_ETAUX_directory']
        L3T_ETAUX_zip_filename = config['L3T_ETAUX_zip_filename']
        L3T_ETAUX_browse_filename = config['L3T_ETAUX_browse_filename']
        L4T_ESI_granule_ID = config['L4T_ESI_granule_ID']
        L4T_ESI_directory = config['L4T_ESI_directory']
        L4T_ESI_zip_filename = config['L4T_ESI_zip_filename']
        L4T_ESI_browse_filename = config['L4T_ESI_browse_filename']
        L4T_WUE_granule_ID = config['L4T_WUE_granule_ID']
        L4T_WUE_directory = config['L4T_WUE_directory']
        L4T_WUE_zip_filename = config['L4T_WUE_zip_filename']
        L4T_WUE_browse_filename = config['L4T_WUE_browse_filename']
        output_directory = config['output_directory']
        sources_directory = config['sources_directory']
        GEOS5FP_directory = config['GEOS5FP_directory']
        static_directory = config['static_directory']
        GEDI_directory = config['GEDI_directory']
        MODISCI_directory = config['MODISCI_directory']
        MCD12_directory = config['MCD12_directory']
        soil_grids_directory = config['soil_grids_directory']
        orbit = config['orbit']
        scene = config['scene']
        tile = config['tile']
        build = config['build']
        product_counter = config['product_counter']
        L2T_LSTE_filename = config['L2T_LSTE_filename']
        L2T_STARS_filename = config['L2T_STARS_filename']
        L2T_LSTE_granule = config['L2T_LSTE_granule']
        geometry = config['geometry']
        time_UTC = config['time_UTC']
        date_UTC = config['date_UTC']
        timestamp = config['timestamp']

        # Call read_ECOv003_inputs to process all input data
        inputs = read_ECOv003_inputs(
            L2T_LSTE_filename=L2T_LSTE_filename,
            L2T_STARS_filename=L2T_STARS_filename,
            orbit=orbit,
            scene=scene,
            tile=tile,
            GEOS5FP_directory=GEOS5FP_directory,
            MODISCI_directory=MODISCI_directory,
            time_UTC=time_UTC,
            date_UTC=date_UTC,
            geometry=geometry,
            zero_COT_correction=zero_COT_correction,
            sharpen_meteorology=sharpen_meteorology,
            sharpen_soil_moisture=sharpen_soil_moisture,
            upsampling=upsampling,
            downsampling=downsampling
        )

        # Unpack results from read_ECOv003_inputs
        metadata = inputs['metadata']
        ST_K = inputs['ST_K']
        ST_C = inputs['ST_C']
        elevation_km = inputs['elevation_km']
        elevation_m = inputs['elevation_m']
        emissivity = inputs['emissivity']
        water_mask = inputs['water_mask']
        cloud_mask = inputs['cloud_mask']
        NDVI = inputs['NDVI']
        albedo = inputs['albedo']
        GEOS5FP_connection = inputs['GEOS5FP_connection']
        MODISCI_connection = inputs['MODISCI_connection']
        SZA_deg = inputs['SZA_deg']
        AOT = inputs['AOT']
        COT = inputs['COT']
        vapor_gccm = inputs['vapor_gccm']
        ozone_cm = inputs['ozone_cm']
        hour_of_day = inputs['hour_of_day']
        day_of_year = inputs['day_of_year']
        time_solar = inputs['time_solar']
        KG_climate = inputs['KG_climate']
        coarse_geometry = inputs['coarse_geometry']
        Ta_C = inputs['Ta_C']
        Ta_C_smooth = inputs['Ta_C_smooth']
        RH = inputs['RH']
        SM = inputs['SM']
        SVP_Pa = inputs['SVP_Pa']
        Ea_Pa = inputs['Ea_Pa']
        Ea_kPa = inputs['Ea_kPa']
        Ta_K = inputs['Ta_K']

        # Replace the science code with a call to the JET function
        results = JET(
            ST_C=ST_C,
            NDVI=NDVI,
            emissivity=emissivity,
            albedo=albedo,
            geometry=geometry,
            time_UTC=time_UTC,
            day_of_year=day_of_year,
            hour_of_day=hour_of_day,
            COT=COT,
            AOT=AOT,
            vapor_gccm=vapor_gccm,
            ozone_cm=ozone_cm,
            elevation_m=elevation_m,
            SZA_deg=SZA_deg,
            KG_climate=KG_climate,
            GEOS5FP_connection=GEOS5FP_connection,
            MODISCI_connection=MODISCI_connection,
            Ta_C=Ta_C,
            RH=RH,
            soil_moisture=SM,
            soil_grids_directory=soil_grids_directory,
            GEDI_directory=GEDI_directory,
            water_mask=water_mask,
            Rn_model_name=Rn_model_name,
            downsampling=downsampling,
            orbit=orbit,
            scene=scene,
            tile=tile,
            date_UTC=date_UTC
        )

        # Extract all variables from the results dictionary
        SWin_TOA_Wm2 = results["SWin_TOA_Wm2"]
        SWin_FLiES_ANN_raw = results["SWin_FLiES_ANN_raw"]
        UV_Wm2 = results["UV_Wm2"]
        PAR_Wm2 = results["PAR_Wm2"]
        NIR_Wm2 = results["NIR_Wm2"]
        PAR_diffuse_Wm2 = results["PAR_diffuse_Wm2"]
        NIR_diffuse_Wm2 = results["NIR_diffuse_Wm2"]
        PAR_direct_Wm2 = results["PAR_direct_Wm2"]
        NIR_direct_Wm2 = results["NIR_direct_Wm2"]
        LE_PTJPLSM_Wm2 = results["LE_PTJPLSM_Wm2"]
        ET_daylight_PTJPLSM_kg = results["ET_daylight_PTJPLSM_kg"]
        LE_STIC_Wm2 = results["LE_STIC_Wm2"]
        ET_daylight_STIC_kg = results["ET_daylight_STIC_kg"]
        LE_BESS_Wm2 = results["LE_BESS_Wm2"]
        ET_daylight_BESS_kg = results["ET_daylight_BESS_kg"]
        LE_PMJPL_Wm2 = results["LE_PMJPL_Wm2"]
        ET_daylight_PMJPL_kg = results["ET_daylight_PMJPL_kg"]
        ET_daylight_kg = results["ET_daylight_kg"]
        ET_uncertainty = results["ET_uncertainty"]
        LE_canopy_fraction_PTJPLSM = results["LE_canopy_fraction_PTJPLSM"]
        LE_canopy_fraction_STIC = results["LE_canopy_fraction_STIC"]
        LE_soil_fraction_PTJPLSM = results["LE_soil_fraction_PTJPLSM"]
        LE_interception_fraction_PTJPLSM = results["LE_interception_fraction_PTJPLSM"]
        Rn_Wm2 = results["Rn_Wm2"]
        SWin = results["SWin_Wm2"]
        ESI_PTJPLSM = results["ESI_PTJPLSM"]
        PET_instantaneous_PTJPLSM_Wm2 = results["PET_instantaneous_PTJPLSM_Wm2"]
        WUE = results["WUE"]
        GPP_inst_g_m2_s = results["GPP_inst_g_m2_s"]
        AuxiliaryNWP = results["AuxiliaryNWP"]

        metadata["StandardMetadata"]["CollectionLabel"] = "ECOv003"
        metadata["ProductMetadata"]["AuxiliaryNWP"] = AuxiliaryNWP

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
