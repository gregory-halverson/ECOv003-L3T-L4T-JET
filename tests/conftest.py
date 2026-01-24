"""
Pytest configuration file.

This file configures pytest behavior including warning filters for external libraries.
"""
import warnings

# Apply warning filters at module import time
# These warnings are expected from external libraries and safe to ignore
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*invalid value encountered in divide.*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*invalid value encountered in.*",
)
