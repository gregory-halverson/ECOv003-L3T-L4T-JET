"""
JET (Joint Evapotranspiration) Module

This module contains the main JET science function that orchestrates the calculation of 
evapotranspiration using multiple models (FLiES-ANN, BESS-JPL, STIC-JPL, PTJPLSM, PMJPL, AquaSEBS).
"""

import logging
import posixpath
from datetime import datetime
from typing import Union, Dict
import numpy as np
import rasters as rt
from rasters import Raster, RasterGeometry

from GEOS5FP import GEOS5FPConnection
from MODISCI import MODISCI

from AquaSEBS import AquaSEBS
from BESS_JPL import BESS_JPL
from PMJPL import PMJPL
from PTJPLSM import PTJPLSM
from STIC_JPL import STIC_JPL
from FLiESANN import FLiESANN
from verma_net_radiation import verma_net_radiation, daylight_Rn_integration_verma
from check_distribution import check_distribution

from .constants import LATENT_VAPORIZATION_JOULES_PER_KILOGRAM
from .exceptions import BlankOutput

logger = logging.getLogger(__name__)


def JET(
        ST_C: Union[Raster, np.ndarray, float],
        emissivity: Union[Raster, np.ndarray, float],
        NDVI: Union[Raster, np.ndarray, float],
        albedo: Union[Raster, np.ndarray, float],
        geometry: RasterGeometry,
        time_UTC: datetime,
        COT: Union[Raster, np.ndarray, float],
        AOT: Union[Raster, np.ndarray, float],
        vapor_gccm: Union[Raster, np.ndarray, float],
        ozone_cm: Union[Raster, np.ndarray, float],
        elevation_m: Union[Raster, np.ndarray, float],
        SZA_deg: Union[Raster, np.ndarray, float],
        KG_climate: Union[Raster, np.ndarray, str],
        Ta_C: Union[Raster, np.ndarray, float],
        RH: Union[Raster, np.ndarray, float],
        soil_moisture: Union[Raster, np.ndarray, float],
        MODISCI_connection: MODISCI,
        soil_grids_directory: str,
        GEDI_directory: str,
        Rn_model_name: str,
        downsampling: str,
        GEOS5FP_connection: GEOS5FPConnection = None,
        water_mask: Union[Raster, np.ndarray, bool] = None) -> Dict[str, Union[Raster, np.ndarray]]:
    """
    Main science function for JET (JPL Evapotranspiration Ensemble).
    
    This function orchestrates the calculation of evapotranspiration using multiple models
    including FLiES-ANN for solar radiation, BESS-JPL for GPP and ET, STIC-JPL for ET partitioning,
    PTJPLSM for ET with soil moisture, PMJPL for Penman-Monteith ET, and AquaSEBS for water surface
    evaporation.
    
    Args:
        albedo: Surface albedo (0-1)
        geometry: Raster geometry object defining the spatial grid
        time_UTC: UTC time string
        hour_of_day: Hour of day (0-23)
        COT: Cloud optical thickness
        AOT: Aerosol optical thickness
        vapor_gccm: Water vapor column (g/cm²)
        ozone_cm: Ozone column (cm)
        elevation_m: Elevation in meters
        SZA_deg: Solar zenith angle in degrees
        KG_climate: Köppen-Geiger climate classification
        GEOS5FP_connection: Connection to GEOS-5 FP data
        MODISCI_connection: Connection to MODIS clumping index data
        Ta_C: Air temperature in Celsius
        RH: Relative humidity (0-1)
        ST_C: Surface temperature in Celsius
        NDVI: Normalized Difference Vegetation Index
        emissivity: Surface emissivity (0-1)
        soil_moisture: Volumetric soil moisture (m³/m³)
        water_mask: Boolean mask for water bodies
        soil_grids_directory: Directory containing soil grids data
        GEDI_directory: Directory containing GEDI canopy height data
        Rn_model_name: Net radiation model name ('verma' or 'BESS')
        downsampling: Resampling method for downsampling
        day_of_year: Day of year (1-366)
        date_UTC: UTC date string
        tile: Tile identifier
        orbit: Orbit number
        scene: Scene number
        
    Returns:
        Dictionary containing all output variables including ET, GPP, WUE, and intermediate results
        
    Raises:
        BlankOutput: If critical output variables are all NaN or zero
    """
        # Create GEOS5FP connection if not provided
    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FPConnection()
    
    # Run FLiES-ANN
    logger.info(f"running Forest Light Environmental Simulator at {time_UTC} UTC")
    
    FLiES_results = FLiESANN(
        albedo=albedo,
        geometry=geometry,
        time_UTC=time_UTC,
        COT=COT,
        AOT=AOT,
        vapor_gccm=vapor_gccm,
        ozone_cm=ozone_cm,
        elevation_m=elevation_m,
        SZA_deg=SZA_deg,
        KG_climate=KG_climate,
        GEOS5FP_connection=GEOS5FP_connection,
    )
    
    # Extract FLiES-ANN results with updated variable names
    SWin_TOA_Wm2 = FLiES_results["SWin_TOA_Wm2"]
    SWin_FLiES_ANN_raw = FLiES_results["SWin_Wm2"]
    UV_Wm2 = FLiES_results["UV_Wm2"]
    PAR_Wm2 = FLiES_results["PAR_Wm2"]
    NIR_Wm2 = FLiES_results["NIR_Wm2"]
    PAR_diffuse_Wm2 = FLiES_results["PAR_diffuse_Wm2"]
    NIR_diffuse_Wm2 = FLiES_results["NIR_diffuse_Wm2"]
    PAR_direct_Wm2 = FLiES_results["PAR_direct_Wm2"]
    NIR_direct_Wm2 = FLiES_results["NIR_direct_Wm2"]

    # Calculate partitioned albedo
    albedo_NWP = GEOS5FP_connection.ALBEDO(time_UTC=time_UTC, geometry=geometry)
    RVIS_NWP = GEOS5FP_connection.ALBVISDR(time_UTC=time_UTC, geometry=geometry)
    albedo_visible = rt.clip(albedo * (RVIS_NWP / albedo_NWP), 0, 1)
    check_distribution(albedo_visible, "albedo_visible")
    RNIR_NWP = GEOS5FP_connection.ALBNIRDR(time_UTC=time_UTC, geometry=geometry)
    albedo_NIR = rt.clip(albedo * (RNIR_NWP / albedo_NWP), 0, 1)
    check_distribution(albedo_NIR, "albedo_NIR")
    check_distribution(PAR_direct_Wm2, "PAR_direct_Wm2")

    # Use raw FLiES-ANN output directly without bias correction
    SWin_Wm2 = SWin_FLiES_ANN_raw
    check_distribution(SWin_Wm2, "SWin_FLiES_ANN")

    # Use FLiES-ANN solar radiation exclusively
    SWin = SWin_Wm2
    SWin = rt.where(np.isnan(ST_C), np.nan, SWin)

    # Check for blank output
    if np.all(np.isnan(SWin)) or np.all(SWin == 0):
        raise BlankOutput(
            f"blank solar radiation output at {time_UTC} UTC")

    logger.info(f"running Breathing Earth System Simulator at {time_UTC} UTC")

    BESS_results = BESS_JPL(
        ST_C=ST_C,
        NDVI=NDVI,
        albedo=albedo,
        elevation_m=elevation_m,
        geometry=geometry,
        time_UTC=time_UTC,
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
    check_distribution(Rn_BESS_Wm2, "Rn_BESS_Wm2")
    G_BESS_Wm2 = BESS_results["G_Wm2"]
    check_distribution(Rn_BESS_Wm2, "Rn_BESS_Wm2")
    
    LE_BESS_Wm2 = BESS_results["LE_Wm2"]
    check_distribution(LE_BESS_Wm2, "LE_BESS_Wm2")
    
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

    check_distribution(LE_BESS_Wm2, "LE_BESS_Wm2")
    
    GPP_inst_umol_m2_s = BESS_results["GPP"]
    
    if water_mask is not None:
        GPP_inst_umol_m2_s = rt.where(water_mask, np.nan, GPP_inst_umol_m2_s)

    check_distribution(GPP_inst_umol_m2_s, "GPP_inst_umol_m2_s")

    if np.all(np.isnan(GPP_inst_umol_m2_s)):
        raise BlankOutput(f"blank GPP output at {time_UTC} UTC")

    NWP_filenames = sorted([posixpath.basename(filename) for filename in GEOS5FP_connection.filenames])
    AuxiliaryNWP = ",".join(NWP_filenames)
    

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
        raise BlankOutput(f"blank net radiation output at {time_UTC} UTC")

    STIC_results = STIC_JPL(
        geometry=geometry,
        time_UTC=time_UTC,
        Rn_Wm2=Rn_Wm2,
        RH=RH,
        Ta_C=Ta_C,
        ST_C=ST_C,
        albedo=albedo,
        emissivity=emissivity,
        NDVI=NDVI,
        max_iterations=3,
        upscale_to_daylight=True
    )

    LE_STIC_Wm2 = STIC_results["LE_Wm2"]
    check_distribution(LE_STIC_Wm2, "LE_STIC_Wm2")
    
    ET_daylight_STIC_kg = STIC_results["ET_daylight_kg"]
    check_distribution(ET_daylight_STIC_kg, "ET_daylight_STIC_kg")
    
    LE_canopy_STIC_Wm2 = STIC_results["LE_canopy_Wm2"]
    check_distribution(LE_canopy_STIC_Wm2, "LE_canoy_STIC_Wm2")
    
    G_STIC_Wm2 = STIC_results["G_Wm2"]
    check_distribution(G_STIC_Wm2, "G_STIC_Wm2")

    LE_canopy_fraction_STIC = rt.clip(rt.where((LE_canopy_STIC_Wm2 == 0) | (LE_STIC_Wm2 == 0), 0, LE_canopy_STIC_Wm2 / LE_STIC_Wm2), 0, 1)
    check_distribution(LE_canopy_fraction_STIC, "LE_canopy_fraction_STIC")

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
        soil_moisture=soil_moisture,
        field_capacity_directory=soil_grids_directory,
        wilting_point_directory=soil_grids_directory,
        canopy_height_directory=GEDI_directory,
        upscale_to_daylight=True
    )

    LE_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_Wm2"], 0, None)
    check_distribution(LE_PTJPLSM_Wm2, "LE_PTJPLSM_Wm2")
    
    ET_daylight_PTJPLSM_kg = PTJPLSM_results["ET_daylight_kg"]
    check_distribution(ET_daylight_PTJPLSM_kg, "ET_daylight_PTJPLSM_kg")
    
    G_PTJPLSM = PTJPLSM_results["G_Wm2"]
    check_distribution(G_PTJPLSM, "G_PTJPLSM")

    EF_PTJPLSM = rt.where((LE_PTJPLSM_Wm2 == 0) | ((Rn_Wm2 - G_PTJPLSM) == 0), 0, LE_PTJPLSM_Wm2 / (Rn_Wm2 - G_PTJPLSM))
    check_distribution(EF_PTJPLSM, "EF_PTJPLSM")

    if np.all(np.isnan(LE_PTJPLSM_Wm2)):
        raise BlankOutput(
            f"blank PT-JPL-SM instantaneous ET output for at {time_UTC} UTC")

    if np.all(np.isnan(LE_PTJPLSM_Wm2)):
        raise BlankOutput(
            f"blank daily ET output for at {time_UTC} UTC")

    LE_canopy_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_canopy_Wm2"], 0, None)
    check_distribution(LE_canopy_PTJPLSM_Wm2, "LE_canopy_PTJPLSM_Wm2")

    LE_canopy_fraction_PTJPLSM = rt.clip(LE_canopy_PTJPLSM_Wm2 / LE_PTJPLSM_Wm2, 0, 1)
    check_distribution(LE_canopy_fraction_PTJPLSM, "LE_canopy_fraction_PTJPLSM")

    if water_mask is not None:
        LE_canopy_fraction_PTJPLSM = rt.where(water_mask, np.nan, LE_canopy_fraction_PTJPLSM)
    
    LE_soil_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_soil_Wm2"], 0, None)
    check_distribution(LE_soil_PTJPLSM_Wm2, "LE_soil_PTJPLSM_Wm2")

    LE_soil_fraction_PTJPLSM = rt.clip(LE_soil_PTJPLSM_Wm2 / LE_PTJPLSM_Wm2, 0, 1)
    
    if water_mask is not None:
        LE_soil_fraction_PTJPLSM = rt.where(water_mask, np.nan, LE_soil_fraction_PTJPLSM)
    
    check_distribution(LE_soil_fraction_PTJPLSM, "LE_soil_fraction_PTJPLSM")
    
    LE_interception_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["LE_interception_Wm2"], 0, None)
    check_distribution(LE_interception_PTJPLSM_Wm2, "LE_interception_PTJPLSM_Wm2")

    LE_interception_fraction_PTJPLSM = rt.clip(LE_interception_PTJPLSM_Wm2 / LE_PTJPLSM_Wm2, 0, 1)
    
    if water_mask is not None:
        LE_interception_fraction_PTJPLSM = rt.where(water_mask, np.nan, LE_interception_fraction_PTJPLSM)
    
    check_distribution(LE_interception_fraction_PTJPLSM, "LE_interception_fraction_PTJPLSM")
    
    PET_instantaneous_PTJPLSM_Wm2 = rt.clip(PTJPLSM_results["PET_Wm2"], 0, None)
    check_distribution(PET_instantaneous_PTJPLSM_Wm2, "PET_instantaneous_PTJPLSM_Wm2")

    ESI_PTJPLSM = rt.clip(LE_PTJPLSM_Wm2 / PET_instantaneous_PTJPLSM_Wm2, 0, 1)

    if water_mask is not None:
        ESI_PTJPLSM = rt.where(water_mask, np.nan, ESI_PTJPLSM)

    check_distribution(ESI_PTJPLSM, "ESI_PTJPLSM")

    if np.all(np.isnan(ESI_PTJPLSM)):
        raise BlankOutput(f"blank ESI output for at {time_UTC} UTC")

    # TODO update PM-JPL to take elevation in meters so all models use the same units

    elevation_km = elevation_m / 1000.0

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
    check_distribution(LE_PMJPL_Wm2, "LE_PMJPL_Wm2")
    
    ET_daylight_PMJPL_kg = PMJPL_results["ET_daylight_kg"]
    check_distribution(ET_daylight_PMJPL_kg, "ET_daylight_PMJ")
    
    G_PMJPL_Wm2 = PMJPL_results["G_Wm2"]
    check_distribution(G_PMJPL_Wm2, "G_PMJPL_Wm2")

    # FIXME get rid of the instantaneous latent heat flux aggregation
    LE_instantaneous_Wm2 = rt.Raster(
        np.nanmedian([np.array(LE_PTJPLSM_Wm2), np.array(LE_BESS_Wm2), np.array(LE_PMJPL_Wm2), np.array(LE_STIC_Wm2)], axis=0),
        geometry=geometry)

    windspeed_mps = GEOS5FP_connection.wind_speed(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
    check_distribution(windspeed_mps, "windspeed_mps")
    
    SWnet_Wm2 = SWin_Wm2 * (1 - albedo)
    check_distribution(SWnet_Wm2, "SWnet_Wm2")

    # Adding debugging statements for input rasters before the AquaSEBS call
    logger.info("checking input distributions for AquaSEBS")
    check_distribution(ST_C, "ST_C")
    check_distribution(emissivity, "emissivity")
    check_distribution(albedo, "albedo")
    check_distribution(Ta_C, "Ta_C")
    check_distribution(RH, "RH")
    check_distribution(windspeed_mps, "windspeed_mps")
    check_distribution(SWnet_Wm2, "SWnet")
    check_distribution(Rn_Wm2, "Rn_Wm2")
    check_distribution(SWin_Wm2, "SWin_Wm2")

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
    check_distribution(LE_AquaSEBS_Wm2, "LE_AquaSEBS_Wm2")
    
    LE_instantaneous_Wm2 = rt.where(water_mask, LE_AquaSEBS_Wm2, LE_instantaneous_Wm2)
    check_distribution(LE_instantaneous_Wm2, "LE_instantaneous_Wm2")
    
    ET_daylight_AquaSEBS_kg = AquaSEBS_results["ET_daylight_kg"]
    check_distribution(ET_daylight_AquaSEBS_kg, "ET_daylight_AquaSEBS_kg")

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
    check_distribution(ET_daylight_kg, "ET_daylight_kg")

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

    results = {
        'SWin_TOA_Wm2': SWin_TOA_Wm2,
        'SWin_FLiES_ANN_raw': SWin_FLiES_ANN_raw,
        'SWin_Wm2': SWin_Wm2,
        'UV_Wm2': UV_Wm2,
        'PAR_Wm2': PAR_Wm2,
        'NIR_Wm2': NIR_Wm2,
        'PAR_diffuse_Wm2': PAR_diffuse_Wm2,
        'NIR_diffuse_Wm2': NIR_diffuse_Wm2,
        'PAR_direct_Wm2': PAR_direct_Wm2,
        'NIR_direct_Wm2': NIR_direct_Wm2,
        'Rn_BESS_Wm2': Rn_BESS_Wm2,
        'LE_BESS_Wm2': LE_BESS_Wm2,
        'ET_daylight_BESS_kg': ET_daylight_BESS_kg,
        'EF_BESS': EF_BESS,
        'Rn_daily_BESS': Rn_daily_BESS,
        'LE_daily_BESS': LE_daily_BESS,
        'GPP_inst_umol_m2_s': GPP_inst_umol_m2_s,
        'Rn_verma_Wm2': Rn_verma_Wm2,
        'Rn_Wm2': Rn_Wm2,
        'LE_STIC_Wm2': LE_STIC_Wm2,
        'ET_daylight_STIC_kg': ET_daylight_STIC_kg,
        'LE_canopy_STIC_Wm2': LE_canopy_STIC_Wm2,
        'G_STIC_Wm2': G_STIC_Wm2,
        'LE_canopy_fraction_STIC': LE_canopy_fraction_STIC,
        'EF_STIC': EF_STIC,
        'LE_PTJPLSM_Wm2': LE_PTJPLSM_Wm2,
        'ET_daylight_PTJPLSM_kg': ET_daylight_PTJPLSM_kg,
        'G_PTJPLSM': G_PTJPLSM,
        'EF_PTJPLSM': EF_PTJPLSM,
        'LE_canopy_PTJPLSM_Wm2': LE_canopy_PTJPLSM_Wm2,
        'LE_canopy_fraction_PTJPLSM': LE_canopy_fraction_PTJPLSM,
        'LE_soil_PTJPLSM_Wm2': LE_soil_PTJPLSM_Wm2,
        'LE_soil_fraction_PTJPLSM': LE_soil_fraction_PTJPLSM,
        'LE_interception_PTJPLSM_Wm2': LE_interception_PTJPLSM_Wm2,
        'LE_interception_fraction_PTJPLSM': LE_interception_fraction_PTJPLSM,
        'PET_instantaneous_PTJPLSM_Wm2': PET_instantaneous_PTJPLSM_Wm2,
        'ESI_PTJPLSM': ESI_PTJPLSM,
        'LE_PMJPL_Wm2': LE_PMJPL_Wm2,
        'ET_daylight_PMJPL_kg': ET_daylight_PMJPL_kg,
        'G_PMJPL_Wm2': G_PMJPL_Wm2,
        'LE_instantaneous_Wm2': LE_instantaneous_Wm2,
        'windspeed_mps': windspeed_mps,
        'SWnet_Wm2': SWnet_Wm2,
        'LE_AquaSEBS_Wm2': LE_AquaSEBS_Wm2,
        'ET_daylight_AquaSEBS_kg': ET_daylight_AquaSEBS_kg,
        'ET_daylight_kg': ET_daylight_kg,
        'ET_uncertainty': ET_uncertainty,
        'GPP_inst_g_m2_s': GPP_inst_g_m2_s,
        'ET_canopy_inst_kg_m2_s': ET_canopy_inst_kg_m2_s,
        'WUE': WUE,
        'AuxiliaryNWP': AuxiliaryNWP
    }

    return results
