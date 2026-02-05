"""
Air Temperature Uncertainty Quantification Module

This module provides functions to generate uncertainty estimates for air temperature
predictions using Gradient Boosting quantile regression models.

Functions:
    calculate_air_temperature_uncertainty_layer: Generate uncertainties for input data
    format_uncertainty_layer_as_dataframe: Convert output to DataFrame format
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple


def calculate_air_temperature_uncertainty_layer(
    input_features: Union[np.ndarray, pd.DataFrame],
    quantile_models: Dict[float, object],
    feature_names: Optional[list] = None,
    return_all_quantiles: bool = False,
) -> Dict:
    """
    Generate an air temperature uncertainty quantification data layer.
    
    Uses trained Gradient Boosting quantile regression models to produce point 
    predictions and uncertainty estimates for air temperature error across the
    range of input atmospheric and surface conditions.
    
    Parameters
    ----------
    input_features : array-like or DataFrame, shape (n_samples, n_features)
        Input variables for prediction. Should contain the same features and 
        format as used during model training.
        
    quantile_models : dict
        Dictionary of trained GradientBoostingRegressor models, keyed by 
        quantile value (float 0-1). Should include keys for at minimum:
        {0.05, 0.25, 0.50, 0.75, 0.95}
        
    feature_names : list, optional
        Names of input features for metadata. If None and input_features is
        a DataFrame, column names are used.
        
    return_all_quantiles : bool, default False
        If True, include predictions at all quantile levels in output.
        If False, return only aggregated uncertainty metrics.
    
    Returns
    -------
    uncertainty_layer : dict
        Dictionary with keys:
        
        - 'median_prediction': ndarray
          Point estimate (50th percentile) of air temperature error
          
        - 'lower_bound': ndarray
          5th percentile (lower bound of 90% prediction interval)
          
        - 'upper_bound': ndarray
          95th percentile (upper bound of 90% prediction interval)
          
        - 'interval_width': ndarray
          Width of 90% prediction interval (upper_bound - lower_bound)
          
        - 'iqr_lower': ndarray
          25th percentile (lower bound of interquartile range)
          
        - 'iqr_upper': ndarray
          75th percentile (upper bound of interquartile range)
          
        - 'iqr_width': ndarray
          Interquartile range width (75th - 25th percentile)
          
        - 'metadata': dict
          Information about the layer including:
          - n_samples: number of input samples
          - n_features: number of input features
          - feature_names: list of feature names
          - quantiles_used: list of quantile levels in quantile_models
          - model_type: 'Gradient Boosting Quantile Regression'
          - prediction_interval: confidence level of main PI ('90%')
          - iqr_interval: confidence level of IQR ('50%')
          
        If return_all_quantiles=True, also includes:
        
        - 'quantile_predictions': dict
          Predictions at all quantile levels keyed by quantile value
    
    Raises
    ------
    ValueError
        If input_features is not 2-dimensional
    KeyError
        If required quantile levels (0.05, 0.25, 0.50, 0.75, 0.95) 
        are missing from quantile_models
    
    Examples
    --------
    Generate uncertainty layer for new data:
    
    >>> import pandas as pd
    >>> from air_temperature_uncertainty import calculate_air_temperature_uncertainty_layer
    >>> 
    >>> # Load your input features
    >>> new_features = pd.read_csv('input_features.csv')
    >>> 
    >>> # Generate uncertainty estimates
    >>> uncertainty = calculate_air_temperature_uncertainty_layer(
    ...     input_features=new_features,
    ...     quantile_models=trained_models  # dict of pre-trained models
    ... )
    >>> 
    >>> # Extract predictions
    >>> median = uncertainty['median_prediction']
    >>> lower_95 = uncertainty['lower_bound']
    >>> upper_95 = uncertainty['upper_bound']
    
    Notes
    -----
    The uncertainty bounds represent the expected range of air temperature 
    error predictions. When applied to actual JET air temperature predictions,
    these errors can be used to:
    
    1. Characterize prediction uncertainty in absolute terms
    2. Flag high-uncertainty predictions for manual review
    3. Provide ensemble-like uncertainty estimates
    4. Enable probabilistic decision-making in downstream applications
    
    The heteroscedastic nature of the intervals (width varies with inputs)
    means uncertainty naturally increases for challenging prediction scenarios.
    """
    
    # Validate and convert input
    if isinstance(input_features, pd.DataFrame):
        X_input = input_features.values
        if feature_names is None:
            feature_names = input_features.columns.tolist()
    else:
        X_input = np.asarray(input_features)
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_input.shape[1])]
    
    # Check dimensionality
    if X_input.ndim != 2:
        raise ValueError(
            f"Input features must be 2-dimensional, got shape {X_input.shape}"
        )
    
    # Check for required quantile levels
    required_quantiles = {0.05, 0.25, 0.50, 0.75, 0.95}
    available_quantiles = set(quantile_models.keys())
    if not required_quantiles.issubset(available_quantiles):
        missing = required_quantiles - available_quantiles
        raise KeyError(
            f"Missing required quantile models: {missing}. "
            f"Available: {available_quantiles}"
        )
    
    # Generate predictions at each quantile
    quantile_predictions = {}
    for q in sorted(quantile_models.keys()):
        quantile_predictions[q] = quantile_models[q].predict(X_input)
    
    # Extract key quantiles
    pred_05 = quantile_predictions[0.05]
    pred_25 = quantile_predictions[0.25]
    pred_50 = quantile_predictions[0.50]
    pred_75 = quantile_predictions[0.75]
    pred_95 = quantile_predictions[0.95]
    
    # Calculate interval widths (measures of uncertainty)
    interval_width_90 = pred_95 - pred_05
    interval_width_50 = pred_75 - pred_25
    
    # Assemble output
    uncertainty_layer = {
        "median_prediction": pred_50,
        "lower_bound": pred_05,
        "upper_bound": pred_95,
        "interval_width": interval_width_90,
        "iqr_lower": pred_25,
        "iqr_upper": pred_75,
        "iqr_width": interval_width_50,
        "metadata": {
            "n_samples": X_input.shape[0],
            "n_features": X_input.shape[1],
            "feature_names": feature_names,
            "quantiles_used": sorted(available_quantiles),
            "model_type": "Gradient Boosting Quantile Regression",
            "prediction_interval": "90%",
            "iqr_interval": "50%",
        },
    }
    
    # Optionally include full quantile predictions
    if return_all_quantiles:
        uncertainty_layer["quantile_predictions"] = quantile_predictions
    
    return uncertainty_layer


def format_uncertainty_layer_as_dataframe(
    uncertainty_layer: Dict,
) -> pd.DataFrame:
    """
    Convert uncertainty layer dictionary to a formatted DataFrame.
    
    Parameters
    ----------
    uncertainty_layer : dict
        Output dictionary from calculate_air_temperature_uncertainty_layer()
    
    Returns
    -------
    df_output : DataFrame
        DataFrame with uncertainty estimates as columns:
        - median_error: 50th percentile error prediction
        - error_lower_95ci: 5th percentile (lower bound)
        - error_upper_95ci: 95th percentile (upper bound)
        - uncertainty_90pi_width: Width of 90% prediction interval
        - error_iqr_lower: 25th percentile
        - error_iqr_upper: 75th percentile
        - uncertainty_50iqr_width: Width of interquartile range
    
    Examples
    --------
    >>> df_unc = format_uncertainty_layer_as_dataframe(uncertainty_layer)
    >>> df_unc.to_csv('air_temperature_uncertainty.csv', index=False)
    """
    
    df_output = pd.DataFrame(
        {
            "median_error": uncertainty_layer["median_prediction"],
            "error_lower_95ci": uncertainty_layer["lower_bound"],
            "error_upper_95ci": uncertainty_layer["upper_bound"],
            "uncertainty_90pi_width": uncertainty_layer["interval_width"],
            "error_iqr_lower": uncertainty_layer["iqr_lower"],
            "error_iqr_upper": uncertainty_layer["iqr_upper"],
            "uncertainty_50iqr_width": uncertainty_layer["iqr_width"],
        }
    )
    
    return df_output


def calculate_coverage_metrics(
    observed_values: np.ndarray,
    uncertainty_layer: Dict[str, np.ndarray],
    confidence_levels: Tuple[float, ...] = (0.50, 0.90),
) -> Dict[str, float]:
    """
    Evaluate calibration of uncertainty estimates.
    
    Calculates the percentage of observed values that fall within prediction
    intervals at specified confidence levels.
    
    Parameters
    ----------
    observed_values : ndarray
        True observed error values
        
    uncertainty_layer : dict
        Output from calculate_air_temperature_uncertainty_layer()
        
    confidence_levels : tuple of float
        Confidence levels to evaluate. Default (0.50, 0.90) evaluates both
        IQR (50%) and 90% prediction intervals.
    
    Returns
    -------
    coverage_metrics : dict
        Dictionary with confidence levels as keys and calculated coverage
        percentages as values. Values should be close to the specified
        confidence level for well-calibrated models.
        Example: {0.50: 51.2, 0.90: 89.8}
    
    Notes
    -----
    A well-calibrated model will have:
    - coverage[0.50] ≈ 50% (within IQR)
    - coverage[0.90] ≈ 90% (within 90% PI)
    """
    
    coverage_metrics = {}
    
    for conf in confidence_levels:
        if conf == 0.50:
            # Use IQR bounds
            in_interval = (observed_values >= uncertainty_layer["iqr_lower"]) & (
                observed_values <= uncertainty_layer["iqr_upper"]
            )
        elif conf == 0.90:
            # Use 90% PI bounds
            in_interval = (observed_values >= uncertainty_layer["lower_bound"]) & (
                observed_values <= uncertainty_layer["upper_bound"]
            )
        else:
            # For other confidence levels, would need additional quantile models
            raise NotImplementedError(
                f"Coverage calculation for confidence level {conf} requires "
                "additional quantile models"
            )
        
        coverage = np.mean(in_interval) * 100
        coverage_metrics[conf] = coverage
    
    return coverage_metrics
