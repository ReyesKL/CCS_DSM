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



def calculate_am_xm(a1_f: np.ndarray, b2_f: np.ndarray, meas_type="amp", Z0=50):
    meas_type = meas_type.lower()
    if meas_type not in ["amp", "phase"]:
        raise ValueError(f"Unrecognized measurement type: {meas_type}, must be 'amp' or 'phase'")

    a1_t = np.fft.ifft(a1_f)
    b2_t = np.fft.ifft(b2_f)

    corr = signal.correlate(a1_t, b2_t)
    lags = signal.correlation_lags(len(a1_t), len(b2_t))
    lag = lags[np.argmax(np.abs(corr))]

    b2_t = np.roll(b2_t, -lag)

    # get the currents and voltages
    v1 = (np.conj(Z0) * a1_t) / np.sqrt(np.abs(np.real(Z0)))
    i1 = (a1_t) / np.sqrt(np.abs(np.real(Z0)))

    v2 = (Z0 * b2_t) / np.sqrt(np.abs(np.real(Z0)))
    i2 = (b2_t) / np.sqrt(np.abs(np.real(Z0)))

    # get pin and pout (assuming the system is matched)
    pin = (v1 * np.conj(i1)) / 2
    pout = (v2 * np.conj(i2)) / 2

    gain = pout/pin

    if meas_type == "amp":
        return pin, np.abs(gain)
    else:
        return pin, np.angle(gain, deg=True)