import numpy as np
from scipy import signal
from itertools import product
from typing import List, Union, Callable, Optional

def find_nearest_idx(array, value):
    # I hope to the great kookaburra above that this doesn't already exist in numpy...

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def normalize(data: np.ndarray, norm_type: str = "min-max") -> np.ndarray:
    """
    Normalizes the input numerical data array based on the specified normalization
    type. This function supports "min-max" scaling, mean-based scaling, and
    z-score normalization. The function is designed to handle NaN values in the
    input data during the computation by using `numpy`'s nan-aware functions.

    The type of normalization to be applied is determined by the `norm_type`
    parameter, which defaults to "min-max". For invalid or unsupported
    values of `norm_type`, a ValueError is raised.

    :param data: The numerical input data array to be normalized. The array
        can contain NaN values, which will be accounted for during
        calculations.
    :type data: np.ndarray

    :param norm_type: The type of normalization to be applied. Supported
        values include:
            - "min-max" for scaling data to a range using minimum and
              maximum values,
            - "mean" for centering data around the mean,
            - "z-score" for standardizing data in terms of standard
              deviation.
        Defaults to "min-max".
    :type norm_type: str

    :return: The normalized data array with the selected transformation
        applied. The resulting shape matches the input data.
    :rtype: np.ndarray

    :raises ValueError: If `norm_type` does not match any of the
        supported normalization types ("min-max", "mean", or
        "z-score").
    """
    if norm_type == "max":
        return data / np.nanmax(data)
    if norm_type == "min-max":
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    elif norm_type == "mean":
        return (data - np.nanmean(data)) / (np.nanmax(data) - np.nanmin(data))
    elif norm_type == "z-score":
        return (data - np.nanmean(data)) / np.nanstd(data)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")



def calculate_am_xm(v_in_f: np.ndarray, v_out_f: np.ndarray, meas_type="amp"):
    meas_type = meas_type.lower()
    if meas_type not in ["amp", "phase"]:
        raise ValueError(f"Unrecognized measurement type: {meas_type}, must be 'amp' or 'phase'")

    v_in_t = np.fft.ifft(v_in_f)
    v_out_t = np.fft.ifft(v_out_f)

    corr = signal.correlate(v_in_t, v_out_t)
    lags = signal.correlation_lags(len(v_in_t), len(v_out_t))
    lag = lags[np.argmax(np.abs(corr))]

    v_out_t = np.roll(v_out_t, -lag)

    pin_t = np.abs(v_in_t**2)
    pout_t = np.abs(v_out_t**2)
    gain_t = pout_t / pin_t
    am_am = 10*np.log10(np.abs(gain_t))
    am_pm = np.angle(gain_t, deg=True)
    if meas_type == "amp":
        return pin_t, am_am
    else:
        return pin_t, am_pm
