from datetime import datetime, date

import sklearn

import numpy as np
import pandas as pd
import rasters as rt
from rasters import Raster, RasterGeometry

from check_distribution import check_distribution

from GEOS5FP import GEOS5FP

import logging

from .exceptions import BlankOutputError

logger = logging.getLogger(__name__)

def sharpen_meteorology_data(
    ST_C: Raster,
    NDVI: Raster,
    albedo: Raster,
    geometry: RasterGeometry,
    coarse_geometry: RasterGeometry,
    time_UTC: datetime,
    date_UTC: date,
    tile: str,
    orbit: str,
    scene: str,
    upsampling: str,
    downsampling: str,
    GEOS5FP_connection: GEOS5FP
) -> tuple[Raster, Raster, Raster]:
    """
    Sharpens air temperature and relative humidity using a linear regression model.

    Args:
        ST_C: Surface temperature in Celsius (fine resolution).
        NDVI: Normalized Difference Vegetation Index (fine resolution).
        albedo: Albedo (fine resolution).
        geometry: Fine-resolution raster geometry.
        coarse_geometry: Coarse-resolution raster geometry.
        time_UTC: UTC timestamp of the overpass.
        date_UTC: UTC date of the overpass.
        tile: Granule tile ID.
        orbit: Orbit ID.
        scene: Scene ID.
        upsampling: Upsampling method for spatial resampling.
        downsampling: Downsampling method for spatial resampling.
        GEOS5FP_connection: An instance of the GEOS5FP connection.

    Returns:
        A tuple containing:
            - Ta_C: Sharpened air temperature in Celsius.
            - RH: Sharpened relative humidity.
            - Ta_C_smooth: Coarse-resolution air temperature from GEOS-5 FP (for gap-filling).

    Raises:
        BlankOutput: If the sharpened air temperature or humidity output is blank.
    """
    # Resample surface temperature, NDVI, and albedo to coarse geometry for training.
    ST_C_coarse = ST_C.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(ST_C_coarse, "ST_C_coarse", date_UTC, tile)
    NDVI_coarse = NDVI.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(NDVI_coarse, "NDVI_coarse", date_UTC, tile)
    albedo_coarse = albedo.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(albedo_coarse, "albedo_coarse", date_UTC, tile)

    # Retrieve coarse-resolution meteorological data from GEOS-5 FP.
    Ta_C_coarse = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling)
    check_distribution(Ta_C_coarse, "Ta_C_coarse", date_UTC, tile)
    Td_C_coarse = GEOS5FP_connection.Td_C(time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling)
    check_distribution(Td_C_coarse, "Td_C_coarse", date_UTC, tile)
    SM_coarse = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling)
    check_distribution(SM_coarse, "SM_coarse", date_UTC, tile)

    # Create a Pandas DataFrame of coarse samples for regression.
    coarse_samples = pd.DataFrame({
        "Ta_C": np.array(Ta_C_coarse).ravel(),
        "Td_C": np.array(Td_C_coarse).ravel(),
        "SM": np.array(SM_coarse).ravel(),
        "ST_C": np.array(ST_C_coarse).ravel(),
        "NDVI": np.array(NDVI_coarse).ravel(),
        "albedo": np.array(albedo_coarse).ravel()
    }).dropna()  # Remove rows with NaN values.

    if len(coarse_samples) == 0:
        raise ValueError(f"unable to retrieve coarse samples for downscaling orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

    # --- Sharpen Air Temperature (Ta_C) ---
    Ta_C_model = sklearn.linear_model.LinearRegression()
    Ta_C_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["Ta_C"])
    Ta_C_intercept = Ta_C_model.intercept_
    ST_C_Ta_C_coef, NDVI_Ta_C_coef, albedo_Ta_C_coef = Ta_C_model.coef_
    logger.info(
        f"air temperature regression: Ta_C = {Ta_C_intercept:0.2f} + {ST_C_Ta_C_coef:0.2f} * ST_C + {NDVI_Ta_C_coef:0.2f} * NDVI + {albedo_Ta_C_coef:0.2f} * albedo")

    # Predict air temperature at fine resolution using the regression model.
    Ta_C_prediction = ST_C * ST_C_Ta_C_coef + NDVI * NDVI_Ta_C_coef + albedo * albedo_Ta_C_coef + Ta_C_intercept
    check_distribution(Ta_C_prediction, "Ta_C_prediction", date_UTC, tile)

    # Upsample the predicted air temperature to coarse resolution to calculate bias.
    logger.info(
        f"up-sampling predicted air temperature from {int(Ta_C_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
    Ta_C_prediction_coarse = Ta_C_prediction.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(Ta_C_prediction_coarse, "Ta_C_prediction_coarse", date_UTC, tile)
    Ta_C_bias_coarse = Ta_C_prediction_coarse - Ta_C_coarse  # Calculate the bias at coarse resolution.
    check_distribution(Ta_C_bias_coarse, "Ta_C_bias_coarse", date_UTC, tile)

    # Downsample the bias to fine resolution and apply bias correction.
    logger.info(
        f"down-sampling air temperature bias from {int(Ta_C_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
    Ta_C_bias_smooth = Ta_C_bias_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(Ta_C_bias_smooth, "Ta_C_bias_smooth", date_UTC, tile)
    logger.info("bias-correcting air temperature")
    Ta_C = Ta_C_prediction - Ta_C_bias_smooth  # Apply bias correction.
    check_distribution(Ta_C, "Ta_C", date_UTC, tile)

    # Get smooth (downsampled GEOS5FP) air temperature for gap-filling.
    Ta_C_smooth = GEOS5FP_connection.Ta_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
    check_distribution(Ta_C_smooth, "Ta_C_smooth", date_UTC, tile)
    logger.info("gap-filling air temperature")
    Ta_C = rt.where(np.isnan(Ta_C), Ta_C_smooth, Ta_C)  # Fill NaNs with smooth data.
    check_distribution(Ta_C, "Ta_C", date_UTC, tile)

    # Mask out values that are not covered by the original ECOSTRESS pass
    LST_mask = np.isnan(ST_C)
    Ta_C[LST_mask] = np.nan
    check_distribution(Ta_C, "Ta_C", date_UTC, tile)
    Ta_C_smooth[LST_mask] = np.nan
    check_distribution(Ta_C_smooth, "Ta_C_smooth", date_UTC, tile)

    # Calculate and log errors for quality assessment.
    logger.info(
        f"up-sampling final air temperature from {int(Ta_C.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
    Ta_C_final_coarse = Ta_C.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(Ta_C_final_coarse, "Ta_C_final_coarse", date_UTC, tile)
    Ta_C_error_coarse = Ta_C_final_coarse - Ta_C_coarse
    check_distribution(Ta_C_error_coarse, "Ta_C_error_coarse", date_UTC, tile)
    logger.info(
        f"down-sampling air temperature error from {int(Ta_C_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
    Ta_C_error = Ta_C_error_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(Ta_C_error, "Ta_C_error", date_UTC, tile)

    # Check for blank air temperature output after sharpening.
    if np.all(np.isnan(Ta_C)):
        raise BlankOutputError(
            f"blank air temperature output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

    # --- Sharpen Dew-point Temperature (Td_C) ---
    Td_C_model = sklearn.linear_model.LinearRegression()
    Td_C_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["Td_C"])
    Td_C_intercept = Td_C_model.intercept_
    ST_C_Td_C_coef, NDVI_Td_C_coef, albedo_Td_C_coef = Td_C_model.coef_

    logger.info(
        f"dew-point temperature regression: Td_C = {Td_C_intercept:0.2f} + {ST_C_Td_C_coef:0.2f} * ST_C + {NDVI_Td_C_coef:0.2f} * NDVI + {albedo_Td_C_coef:0.2f} * albedo")
    Td_C_prediction = ST_C * ST_C_Td_C_coef + NDVI * NDVI_Td_C_coef + albedo * albedo_Td_C_coef + Td_C_intercept
    check_distribution(Td_C_prediction, "Td_C_prediction", date_UTC, tile)
    logger.info(
        f"up-sampling predicted dew-point temperature from {int(Td_C_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
    Td_C_prediction_coarse = Td_C_prediction.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(Td_C_prediction_coarse, "Td_C_prediction_coarse", date_UTC, tile)
    Td_C_bias_coarse = Td_C_prediction_coarse - Td_C_coarse
    check_distribution(Td_C_bias_coarse, "Td_C_bias_coarse", date_UTC, tile)
    logger.info(
        f"down-sampling dew-point temperature bias from {int(Td_C_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
    Td_C_bias_smooth = Td_C_bias_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(Td_C_bias_smooth, "Td_C_bias_smooth", date_UTC, tile)
    logger.info("bias-correcting dew-point temperature")
    Td_C = Td_C_prediction - Td_C_bias_smooth
    check_distribution(Td_C, "Td_C", date_UTC, tile)
    Td_C_smooth = GEOS5FP_connection.Td_C(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
    check_distribution(Td_C_smooth, "Td_C_smooth", date_UTC, tile)
    logger.info("gap-filling dew-point temperature")
    Td_C = rt.where(np.isnan(Td_C), Td_C_smooth, Td_C)
    check_distribution(Td_C, "Td_C", date_UTC, tile)

    # Mask out values that are not covered by the original ECOSTRESS pass
    Td_C[LST_mask] = np.nan
    check_distribution(Td_C, "Td_C", date_UTC, tile)

    # Calculate and log errors for quality assessment.
    logger.info(
        f"up-sampling final dew-point temperature from {int(Td_C.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method")
    Td_C_final_coarse = Td_C.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(Td_C_final_coarse, "Td_C_final_coarse", date_UTC, tile)
    Td_C_error_coarse = Td_C_final_coarse - Td_C_coarse
    check_distribution(Td_C_error_coarse, "Td_C_error_coarse", date_UTC, tile)
    logger.info(
        f"down-sampling dew-point temperature error from {int(Td_C_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method")
    Td_C_error = Td_C_error_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(Td_C_error, "Td_C_error", date_UTC, tile)

    # Calculate Relative Humidity (RH) from air and dew-point temperatures.
    RH = rt.clip(np.exp((17.625 * Td_C) / (243.04 + Td_C)) / np.exp((17.625 * Ta_C) / (243.04 + Ta_C)), 0, 1)

    # Check for blank humidity output after sharpening.
    if np.all(np.isnan(RH)):
        raise BlankOutput(
            f"blank humidity output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC")

    return Ta_C, RH, Ta_C_smooth
