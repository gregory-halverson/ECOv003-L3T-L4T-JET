import logging
from ECOv003_granules import write_L3T_JET, write_L3T_ETAUX, write_L4T_ESI, write_L4T_WUE

def write_ECOv003_products(runconfig, metadata, 
                          LE_PTJPLSM_Wm2, ET_daylight_PTJPLSM_kg, LE_STIC_Wm2, ET_daylight_STIC_kg, 
                          LE_BESS_Wm2, ET_daylight_BESS_kg, LE_PMJPL_Wm2, ET_daylight_PMJPL_kg, 
                          ET_daylight_kg, ET_uncertainty, LE_canopy_fraction_PTJPLSM, LE_canopy_fraction_STIC, 
                          LE_soil_fraction_PTJPLSM, LE_interception_fraction_PTJPLSM, water_mask, cloud_mask, 
                          Ta_C, RH, Rn_Wm2, SWin, SM, ESI_PTJPLSM, PET_instantaneous_PTJPLSM_Wm2, WUE, GPP_inst_g_m2_s):
    """
    Writes the ECOv003 products to their respective files.

    Args:
        runconfig: The run configuration object containing file paths and settings.
        metadata: Metadata dictionary for the products.
        LE_PTJPLSM_Wm2: Latent heat flux from PTJPLSM model.
        ET_daylight_PTJPLSM_kg: Daylight evapotranspiration from PTJPLSM model.
        LE_STIC_Wm2: Latent heat flux from STIC model.
        ET_daylight_STIC_kg: Daylight evapotranspiration from STIC model.
        LE_BESS_Wm2: Latent heat flux from BESS model.
        ET_daylight_BESS_kg: Daylight evapotranspiration from BESS model.
        LE_PMJPL_Wm2: Latent heat flux from PMJPL model.
        ET_daylight_PMJPL_kg: Daylight evapotranspiration from PMJPL model.
        ET_daylight_kg: Total daylight evapotranspiration.
        ET_uncertainty: Uncertainty in evapotranspiration.
        LE_canopy_fraction_PTJPLSM: Canopy latent heat fraction from PTJPLSM model.
        LE_canopy_fraction_STIC: Canopy latent heat fraction from STIC model.
        LE_soil_fraction_PTJPLSM: Soil latent heat fraction from PTJPLSM model.
        LE_interception_fraction_PTJPLSM: Interception latent heat fraction from PTJPLSM model.
        water_mask: Water mask layer.
        cloud_mask: Cloud mask layer.
        Ta_C: Air temperature in Celsius.
        RH: Relative humidity.
        Rn_Wm2: Net radiation.
        SWin: Shortwave incoming radiation.
        SM: Soil moisture.
        ESI_PTJPLSM: Evaporative stress index from PTJPLSM model.
        PET_instantaneous_PTJPLSM_Wm2: Instantaneous potential evapotranspiration from PTJPLSM model.
        WUE: Water use efficiency.
        GPP_inst_g_m2_s: Instantaneous gross primary productivity.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    write_L3T_JET(
        L3T_JET_zip_filename=runconfig.L3T_JET_zip_filename,
        L3T_JET_browse_filename=runconfig.L3T_JET_browse_filename,
        L3T_JET_directory=runconfig.L3T_JET_directory,
        orbit=runconfig.orbit,
        scene=runconfig.scene,
        tile=runconfig.tile,
        time_UTC=runconfig.time_UTC,
        build=runconfig.build,
        product_counter=runconfig.product_counter,
        LE_instantaneous_PTJPLSM_Wm2=LE_PTJPLSM_Wm2,
        ET_daylight_PTJPLSM_kg=ET_daylight_PTJPLSM_kg,
        LE_instantaneous_STICJPL_Wm2=LE_STIC_Wm2,
        ET_daylight_STICJPL_kg=ET_daylight_STIC_kg,
        LE_instantaneous_BESSJPL_Wm2=LE_BESS_Wm2,
        ET_daylight_BESSJPL_kg=ET_daylight_BESS_kg,
        LE_instantaneous_PMJPL_Wm2=LE_PMJPL_Wm2,
        ET_daylight_PMJPL_kg=ET_daylight_PMJPL_kg,
        ET_daylight_kg=ET_daylight_kg,
        ET_daylight_uncertainty_kg=ET_uncertainty,
        LE_canopy_fraction_PTJPLSM=LE_canopy_fraction_PTJPLSM,
        LE_canopy_fraction_STIC=LE_canopy_fraction_STIC,
        LE_soil_fraction_PTJPLSM=LE_soil_fraction_PTJPLSM,
        LE_interception_fraction_PTJPLSM=LE_interception_fraction_PTJPLSM,
        water_mask=water_mask,
        cloud_mask=cloud_mask,
        metadata=metadata
    )

    write_L3T_ETAUX(
        L3T_ETAUX_zip_filename=runconfig.L3T_ETAUX_zip_filename,
        L3T_ETAUX_browse_filename=runconfig.L3T_ETAUX_browse_filename,
        L3T_ETAUX_directory=runconfig.L3T_ETAUX_directory,
        orbit=runconfig.orbit,
        scene=runconfig.scene,
        tile=runconfig.tile,
        time_UTC=runconfig.time_UTC,
        build=runconfig.build,
        product_counter=runconfig.product_counter,
        Ta_C=Ta_C,
        RH=RH,
        Rn=Rn_Wm2,
        Rg=SWin,
        SM=SM,
        water_mask=water_mask,
        cloud_mask=cloud_mask,
        metadata=metadata
    )

    write_L4T_ESI(
        L4T_ESI_zip_filename=runconfig.L4T_ESI_zip_filename,
        L4T_ESI_browse_filename=runconfig.L4T_ESI_browse_filename,
        L4T_ESI_directory=runconfig.L4T_ESI_directory,
        orbit=runconfig.orbit,
        scene=runconfig.scene,
        tile=runconfig.tile,
        time_UTC=runconfig.time_UTC,
        build=runconfig.build,
        product_counter=runconfig.product_counter,
        ESI=ESI_PTJPLSM,
        PET=PET_instantaneous_PTJPLSM_Wm2,
        water_mask=water_mask,
        cloud_mask=cloud_mask,
        metadata=metadata
    )

    write_L4T_WUE(
        L4T_WUE_zip_filename=runconfig.L4T_WUE_zip_filename,
        L4T_WUE_browse_filename=runconfig.L4T_WUE_browse_filename,
        L4T_WUE_directory=runconfig.L4T_WUE_directory,
        orbit=runconfig.orbit,
        scene=runconfig.scene,
        tile=runconfig.tile,
        time_UTC=runconfig.time_UTC,
        build=runconfig.build,
        product_counter=runconfig.product_counter,
        WUE=WUE,
        GPP=GPP_inst_g_m2_s,
        water_mask=water_mask,
        cloud_mask=cloud_mask,
        metadata=metadata
    )

    logger.info("Finished writing ECOv003 products.")