from typing import Union
from rasters import Raster
from datetime import date
import numpy as np
from logging import getLogger
import colored_logging as cl

logger = getLogger(__name__)

class BlankOutputError(Exception):
    pass

def check_distribution(
        image: Raster,
        variable: str,
        date_UTC: Union[date, str],
        target: str):
    unique = np.unique(image)
    nan_proportion = np.count_nonzero(np.isnan(image)) / np.size(image)

    if len(unique) < 10:
        logger.info(f"variable {cl.name(variable)} ({image.dtype}) on {cl.time(f'{date_UTC:%Y-%m-%d}')} at {cl.place(target)} with {cl.val(unique)} unique values")

        for value in unique:
            if np.isnan(value):
                count = np.count_nonzero(np.isnan(image))
            else:
                count = np.count_nonzero(image == value)

            if value == 0 or np.isnan(value):
                logger.info(f"* {cl.colored(value, 'red')}: {cl.colored(count, 'red')}")
            else:
                logger.info(f"* {cl.val(value)}: {cl.val(count)}")
    else:
        minimum = np.nanmin(image)

        if minimum < 0:
            minimum_string = cl.colored(f"{minimum:0.3f}", "red")
        else:
            minimum_string = cl.val(f"{minimum:0.3f}")

        maximum = np.nanmax(image)

        if maximum <= 0:
            maximum_string = cl.colored(f"{maximum:0.3f}", "red")
        else:
            maximum_string = cl.val(f"{maximum:0.3f}")

        if nan_proportion > 0.5:
            nan_proportion_string = cl.colored(f"{(nan_proportion * 100):0.2f}%", "yellow")
        elif nan_proportion == 1:
            nan_proportion_string = cl.colored(f"{(nan_proportion * 100):0.2f}%", "red")
        else:
            nan_proportion_string = cl.val(f"{(nan_proportion * 100):0.2f}%")

        message = "variable " + cl.name(variable) + \
            " on " + cl.time(f"{date_UTC:%Y-%m-%d}") + \
            " at " + cl.place(target) + \
            " min: " + minimum_string + \
            " mean: " + cl.val(f"{np.nanmean(image):0.3f}") + \
            " max: " + maximum_string + \
            " nan: " + nan_proportion_string + f" ({cl.val(image.nodata)})"

        if np.all(image == 0):
            message += " all zeros"
            logger.warning(message)
        else:
            logger.info(message)

    if nan_proportion == 1:
        raise BlankOutputError(f"variable {variable} on {date_UTC:%Y-%m-%d} at {target} is a blank image")
