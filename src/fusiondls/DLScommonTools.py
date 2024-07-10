import pickle as pkl

import numpy as np
from scipy import interpolate

from .typing import FloatArray, PathLike


def make_arrays(
    scan2d: list[list[dict]],
    list_BxBt_scales: list[float],
    list_Lc_scales: list[float],
    new: bool = True,
    cvar: str = "ne",
    cut: bool = True,
) -> dict[str, FloatArray]:
    """Calculate 2D arrays of detachment window and threshold improvement

    Converts between nested lists of dicts and dict of arrays.

    Parameters
    ----------
    scan2d:
        Nested lists of ``BxBt``, ``Lc`` results
    list_BxBt_scales:
        Array of flux expansion values
    list_Lc_scales:
        Array of connection length values
    new:
        Use new format of ``scan2d``
    cvar:
        Name of control variable
    cut:
        FIXME

    Returns
    -------
    dict:
        Dictionary of 2D arrays of results
    """

    if new:  # New format for 2D scans

        def flatten(key: str):
            return np.array([[col[key] for col in row] for row in scan2d])

        def cut_func(predicate, array):
            return np.where(predicate, array, np.nan)

        window = flatten("window")
        window_ratio = flatten("window_ratio")
        threshold = flatten("threshold")

        if cut:
            if cvar == "q":
                predicate = window_ratio <= 1
            elif cvar == "ne":
                predicate = window_ratio >= 1
            else:
                raise ValueError(
                    f"Expected one of ('q', 'ne') for 'cvar', got '{cvar}'"
                )

            threshold = cut_func(predicate, threshold)
            window = cut_func(predicate, window)
            window_ratio = cut_func(predicate, window_ratio)

        index_1_BxBt = np.where(list_BxBt_scales == 1)
        index_1_Lc = np.where(list_Lc_scales == 1)
        window_norm = window[index_1_BxBt, index_1_Lc]
        window_ratio_norm = window_ratio[index_1_BxBt, index_1_Lc]
        threshold_norm = threshold[index_1_BxBt, index_1_Lc]

        arr = {
            "window_norm": (window - window_norm) / abs(window_norm),
            "window_ratio_norm": (window_ratio - window_ratio_norm)
            / abs(window_ratio_norm),
            "threshold_norm": (threshold / threshold_norm) - 1,
            "window_base": window_norm,
            "window_ratio_base": window_ratio_norm,
            "threshold_base": threshold_norm,
        }
    else:
        arr_window = []
        arr_threshold = []
        arr_window_ratio = []
        arr = {}

        for i in range(len(list_BxBt_scales)):
            arr_window.append(scan2d[i]["window"])
            arr_threshold.append(scan2d[i]["threshold"])
            arr_window_ratio.append(scan2d[i]["window_ratio"])

        arr["window"] = np.array(arr_window)
        arr["threshold"] = np.array(arr_threshold)
        arr["window_ratio"] = np.array(arr_window_ratio)

        index_1_BxBt = np.where(list_BxBt_scales == 1)
        index_1_Lc = np.where(list_Lc_scales == 1)
        window_norm = arr["window"][index_1_BxBt, index_1_Lc]
        threshold_norm = arr["threshold"][index_1_BxBt, index_1_Lc]

        arr["window_norm"] = (arr["window"] - window_norm) / abs(window_norm) - 1
        arr["threshold_norm"] = arr["threshold"] / threshold_norm - 1

    return arr


def make_window_band(
    d: dict[str, FloatArray],
    o: dict[str, FloatArray],
    spol_middle: float,
    size: float = 0.05,
    q: bool = False,
) -> dict[str, FloatArray]:
    """Make detachment window band with a middle at the provided SPol coordinate

    The default window size is 5%

    Parameters
    ----------
    d:
        Profiles dictionary
    o:
        Results dictionary
    spol_middle:
        Middle of window band in :math:`S_{pol}`
    size:
        Window band size
    q:
        True if control variable is heat flux

    """

    band = {}
    crel = np.array(o["crel"]) if q is False else 1 / np.array(o["crel"])
    splot = np.array(o["Splot"])
    spolplot = np.array(o["SpolPlot"])
    Btot = d["Btot"]
    Btot_grad = np.gradient(Btot)

    spar_from_crel = interpolate.UnivariateSpline(crel, splot, k=5)
    spol_from_crel = interpolate.UnivariateSpline(crel, spolplot, k=5)
    crel_from_spol = interpolate.UnivariateSpline(spolplot, crel, k=5)

    c_middle = crel_from_spol(spol_middle)

    band["C"] = [None] * 3
    band["C"][0] = c_middle * (1 - size)
    band["C"][1] = c_middle
    band["C"][2] = c_middle * (1 + size)

    for param in ["Spar", "Spol", "index", "R", "Z", "Btot"]:
        band[param] = np.array([float] * 3)

    for i in range(3):
        band["Spar"][i] = spar_from_crel(band["C"][i])
        band["Spol"][i] = spol_from_crel(band["C"][i])
        band["index"][i] = np.argmin(np.abs(d["S"] - band["Spar"][i]))
        band["R"][i] = d["R"][band["index"][i]]
        band["Z"][i] = d["Z"][band["index"][i]]
        band["Btot"][i] = d["Btot"][band["index"][i]]

    band["width_pol"] = band["Spol"][2] - band["Spol"][0]
    band["width_par"] = band["Spar"][2] - band["Spar"][0]
    band["Btot_avg"] = np.mean(Btot[band["index"][0] : band["index"][2]])
    band["Btot_grad_avg"] = np.mean(Btot_grad[band["index"][0] : band["index"][2]])

    return band


def file_write(data: dict[str, FloatArray], filename: PathLike):
    """Writes an object to a pickle file"""
    with open(filename, "wb") as f:
        pkl.dump(data, f)


def file_read(filename: PathLike) -> dict[str, FloatArray]:
    """Reads a pickle file and returns it"""
    with open(filename, "rb") as f:
        return pkl.load(f)


def pad_profile(S, data):
    """
    DLS terminates the domain at the front meaning downstream domain is ignored.
    This adds zeros to a result array data to fill those with zeros according to
    the distance array S.
    """

    intended_length = len(S)
    actual_length = len(data)

    return np.insert(data, 0, np.zeros(intended_length - actual_length))
