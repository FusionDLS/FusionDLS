import pickle as pkl
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy import interpolate
from typing_extensions import Self

from .typing import FloatArray, PathLike, Scalar


def scale_BxBt(
    Btot: FloatArray,
    Xpoint: int,
    scale_factor: Optional[Scalar] = None,
    BxBt: Optional[Scalar] = None,
) -> FloatArray:
    r"""Scale a :math:`B_\mathrm{total}` profile to have an arbitrary
    flux expansion (ratio of :math:`B_\mathrm{X-point}` to
    :math:`B_\mathrm{target}`)

    Specify either a scale factor (``scale_factor``) or required flux
    expansion (``BxBt``).

    TODO: MAKE SURE BPOL IS SCALED TOO

    Parameters
    ----------
    Btot:
        :math:`B_\mathrm{total}` array
    Xpoint:
        Index of X-point location in ``Btot`` array
    scale_factor:
        Multiplicative factor applied to initial ``Btot``
    BxBt:
        Desired flux expansion
    """

    if scale_factor is None and BxBt is None:
        raise ValueError("Missing required argument: one of `scale_factor` or `BxBt`")
    if scale_factor is not None and BxBt is not None:
        raise ValueError(
            "Exactly one of `scale_factor` or `BxBt` must be supplied (both given)"
        )

    Bt_base = Btot[0]
    Bx_base = Btot[Xpoint]
    BxBt_base = Bx_base / Bt_base

    if scale_factor is not None:
        BxBt = BxBt_base * scale_factor

    # Keep Bx the same, scale Bt.
    # Calc new Bt based on desired BtBx
    Bt_new = 1 / (BxBt / Bx_base)

    Btot_new = Btot * (Bx_base - Bt_new) / (Bx_base - Bt_base)

    # Translate to keep the same Bx as before
    transl_factor = Btot_new[Xpoint] - Bx_base
    Btot_new -= transl_factor

    # Replace upstream of the Xpoint with the old data
    # So that we are only scaling downstream of Xpoint
    Btot_new[Xpoint:] = Btot[Xpoint:]

    return Btot_new


def scale_Lc(
    S_base: FloatArray,
    Spol_base: FloatArray,
    Xpoint: int,
    scale_factor: Optional[Scalar] = None,
    Lc: Optional[Scalar] = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Scale :math:`S_\parallel` and :math:`S_{pol}` profiles for
    arbitrary connection length, :math:`L_c`

    Specify either a scale factor (``scale_factor``) or required
    connection length (``Lc``)

    TODO: IMPLEMENT SPOL SCALING

    Parameters
    ----------
    S_base:
        :math:`S_\parallel` array
    Spol_base:
        :math:`S_{pol}` array
    Xpoint:
        Index of X-point location in ``S_base`` and ``Spol_base`` arrays
    scale_factor:
        Multiplicative factor applied to initial ``S_base`` and ``Spol_base``
    Lc:
        Desired connection length
    """

    if scale_factor is None and Lc is None:
        raise ValueError("Missing required argument: one of `scale_factor` or `Lc`")
    if scale_factor is not None and Lc is not None:
        raise ValueError(
            "Exactly one of `scale_factor` or `Lc` must be supplied (both given)"
        )

    Lc_base = S_base[Xpoint]

    # Having Lc non-zero
    if Lc is not None:
        scale_factor = Lc / Lc_base

    # This is only required to keep mypy happy
    assert scale_factor is not None

    # Scale up to get correct length
    S_new = S_base * scale_factor
    Spol_new = Spol_base * scale_factor

    # Align Xpoints
    S_new += S_base[Xpoint] - S_new[Xpoint]
    Spol_new += Spol_base[Xpoint] - Spol_new[Xpoint]

    # Make both the same upstream of Xpoint
    S_new[Xpoint:] = S_base[Xpoint:]
    Spol_new[Xpoint:] = Spol_base[Xpoint:]

    # Offset to make both targets at S = 0
    S_new -= S_new[0]
    Spol_new -= Spol_new[0]

    return S_new, Spol_new


def scale_Lm(
    S_base: FloatArray,
    Spol_base: FloatArray,
    Xpoint: int,
    scale_factor: Optional[Scalar] = None,
    Lm: Optional[Scalar] = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Scale :math:`S_\parallel` and :math:`S_{pol}` profiles above X-point
    for arbitrary midplane length, :math:`L_m`

    Specify either a scale factor (``scale_factor``) or required
    midplane length (``Lm``)

    TODO: IMPLEMENT SPOL SCALING

    Parameters
    ----------
    S_base:
        :math:`S_\parallel` array
    Spol_base:
        :math:`S_{pol}` array
    Xpoint:
        Index of X-point location in ``S_base`` and ``Spol_base`` arrays
    scale_factor:
        Multiplicative factor applied to initial ``S_base`` and ``Spol_base``
    Lm:
        Desired midplane length
    """

    if scale_factor is None and Lm is None:
        raise ValueError("Missing required argument: one of `scale_factor` or `Lm`")
    if scale_factor is not None and Lm is not None:
        raise ValueError(
            "Exactly one of `scale_factor` or `Lm` must be supplied (both given)"
        )

    Lm_base = S_base[-1]

    if Lm is not None:
        scale_factor = Lm / Lm_base

    # This is only required to keep mypy happy
    assert scale_factor is not None

    # Scale up to get correct length
    S_new = S_base * scale_factor
    Spol_new = Spol_base * scale_factor

    # Align Xpoints
    S_new += S_base[Xpoint] - S_new[Xpoint]
    Spol_new += Spol_base[Xpoint] - Spol_new[Xpoint]

    # Make both the same upstream of Xpoint
    S_new[:Xpoint] = S_base[:Xpoint]
    Spol_new[:Xpoint] = Spol_base[:Xpoint]

    # Offset to make both targets at S = 0
    S_new -= S_new[0]
    Spol_new -= Spol_new[0]

    return S_new, Spol_new


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


@dataclass
class MagneticGeometry:
    r"""Magnetic geometry for a diverator leg

    Attributes
    ----------
    Bpol:
        Poloidal magnetic field
    Btot:
        Total magnetic field
    R:
        Radial coordinate in metres
    Z:
        Vertical coordinate in metres
    S:
        $S_\parallel$, distance from the target
    Spol:
        $S_{poloidal}$
    zl:

    Xpoint:
        Index of the X-point in the leg arrays
    Bx:
        Value of the magnetic field at the X-point
    Sx:
        Value of $S$ at the X-point
    Spolx:
        Value of $S_{pol}$ at the X-point
    zx:

    R_full:
        2D array of major radius
    Z_full:
        2D array of vertical coordinate
    R_ring:
        Major radius of full SOL ring
    Z_ring:
        Vertical coordinate of full SOL ring
    """

    Bpol: FloatArray
    Btot: FloatArray
    R: FloatArray
    Z: FloatArray
    S: FloatArray
    Spol: FloatArray
    zl: FloatArray
    Xpoint: int
    Bx: float
    Sx: float
    Spolx: float
    zx: float
    R_full: FloatArray
    Z_full: FloatArray
    R_ring: FloatArray
    Z_ring: FloatArray

    @classmethod
    def from_pickle(cls, filename: str, design: str, side: str) -> Self:
        """Read a particular design and side from a pickle balance file."""

        with open(filename, "rb") as f:
            eqb = pkl.load(f)

        return cls(**eqb[design][side])

    @classmethod
    def read_design(cls, filename: str, design: str) -> dict[str, Self]:
        """Read all divertor legs for a single design from a pickle balance file."""

        with open(filename, "rb") as f:
            eqb = pkl.load(f)

        return {side: cls(**data) for side, data in eqb[design].items()}

    def scale_flux_expansion(
        self,
        *,
        scale_factor: Optional[Scalar] = None,
        expansion: Optional[Scalar] = None,
    ) -> Self:
        r"""Scale a :math:`B_\mathrm{total}` profile to have an arbitrary
        flux expansion (ratio of :math:`B_\mathrm{X-point}` to
        :math:`B_\mathrm{target}`), return a new `MagneticGeometry`.

        Specify either a scale factor (``scale_factor``) or required flux
        expansion (``expansion``).

        Parameters
        ----------
        scale_factor:
            Multiplicative factor applied to initial ``Btot``
        expansion:
            Desired flux expansion
        """

        new_data = asdict(self)
        new_data["Btot"] = scale_BxBt(self.Btot, self.Xpoint, scale_factor, expansion)

        return MagneticGeometry(**new_data)

    def scale_connection_length(
        self,
        *,
        scale_factor: Optional[Scalar] = None,
        connection_length: Optional[Scalar] = None,
    ) -> Self:
        r"""Scale :math:`S_\parallel` and :math:`S_{pol}` profiles for
        arbitrary connection length, :math:`L_c`, return a new `MagneticGeometry`.

        Specify either a scale factor or required connection length.

        Parameters
        ----------
        scale_factor:
            Multiplicative factor applied to initial ``S_base`` and ``Spol_base``
        connection_length:
            Desired connection length

        """

        new_data = asdict(self)
        new_data["S"], new_data["Spol"] = scale_Lc(
            self.S, self.Spol, self.Xpoint, scale_factor, connection_length
        )

        return MagneticGeometry(**new_data)

    def scale_midplane_length(
        self,
        *,
        scale_factor: Optional[Scalar] = None,
        midplane_length: Optional[Scalar] = None,
    ) -> Self:
        r"""Scale :math:`S_\parallel` and :math:`S_{pol}` profiles for
        arbitrary midplane length, :math:`L_m`, return a new `MagneticGeometry`.

        Specify either a scale factor or required midplane length.

        Parameters
        ----------
        scale_factor:
            Multiplicative factor applied to initial ``S_base`` and ``Spol_base``
        midplane_length:
            Desired midplane length

        """

        new_data = asdict(self)
        new_data["S"], new_data["Spol"] = scale_Lm(
            self.S, self.Spol, self.Xpoint, scale_factor, midplane_length
        )

        return MagneticGeometry(**new_data)
