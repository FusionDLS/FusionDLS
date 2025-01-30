import copy
import pickle
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import interpolate
from scipy.integrate import trapezoid
from typing_extensions import Never, Self

from .typing import FloatArray, PathLike, Scalar


@dataclass
class Profile(MutableMapping):
    r"""Magnetic profile for a single diverger leg.

    Contains methods to calculate basic statistics as well as to modify the
    profile, e.g. to change its connection length or total flux expansion.
    """

    Bpol: FloatArray
    """Poloidal magnetic field"""

    Btot: FloatArray
    """Total magnetic field"""

    R: FloatArray
    """Radial coordinate in metres"""

    Z: FloatArray
    """Vertical coordinate in metres"""

    S: FloatArray
    r""":math:`S_\parallel`, distance from the target"""

    Spol: FloatArray
    r""":math:`S_{poloidal}`"""

    Xpoint: int
    """Index of the X-point in the leg arrays"""

    zl: FloatArray | None = None
    """Additional Z coordinate array"""

    R_full: FloatArray | None = None
    """2D array of major radius"""

    Z_full: FloatArray | None = None
    """2D array of vertical coordinate"""

    R_ring: FloatArray | None = None
    """Major radius of full SOL ring"""

    Z_ring: FloatArray | None = None
    """Vertical coordinate of full SOL ring"""

    name: str = "base"
    """Name of the profile"""

    # Methods for constructing a profile

    @classmethod
    def from_pickle(cls, filename: PathLike, design: str, side: str) -> Self:
        """Read a particular design and side from a pickle balance file."""
        with open(filename, "rb") as f:
            eqb = pickle.load(f)
        return cls(**cls._drop_properties(eqb[design][side]))

    @classmethod
    def read_design(cls, filename: PathLike, design: str) -> dict[str, Self]:
        """Read all divertor legs for a single design from a pickle balance file."""
        with open(filename, "rb") as f:
            eqb = pickle.load(f)
        return {
            side: cls(**cls._drop_properties(data))
            for side, data in eqb[design].items()
        }

    @classmethod
    def _drop_properties(cls, data: dict) -> dict:
        """Helper function to remove dict keys that are now properties"""
        return {
            k: v
            for k, v in data.items()
            if not (k in cls.__dict__ and isinstance(cls.__dict__[k], property))
        }

    def copy(self) -> Self:
        """Return a deep copy of the profile"""
        return copy.deepcopy(self)

    # Components to implement MutableMapping
    # These give the profile dictionary-like behaviour

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Unknown key: {key}")

    def __delitem__(self, _) -> Never:
        raise NotImplementedError("Deleting attributes is not allowed")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    # Properties and simple functions
    # These implement simple utilities for the profile

    @property
    def R_leg(self) -> FloatArray:
        """Major radius of the leg"""
        return self.R[: self.Xpoint + 1]

    @property
    def Z_leg(self) -> FloatArray:
        """Vertical coordinate of the leg"""
        return self.Z[: self.Xpoint + 1]

    @property
    def Sx(self) -> float:
        """Parallel coordinate at the X-point"""
        return self.S[self.Xpoint]

    @property
    def Spolx(self) -> float:
        """Poloidal coordinate at the X-point"""
        return self.Spol[self.Xpoint]

    @property
    def Bx(self) -> float:
        """Total magnetic field at the X-point"""
        return self.Btot[self.Xpoint]

    @property
    def zx(self) -> float:
        """Vertical coordinate at the X-point"""
        if self.zl is None:
            raise AttributeError("No Z profile provided")
        return self.zl[self.Xpoint]

    @property
    def connection_length(self) -> float:
        """Return connection length of profile"""
        return self.S[-1] - self.S[0]

    @property
    def upstream_length(self) -> float:
        """Return connection length upstream of the X-point"""
        return self.connection_length - self.Sx

    @property
    def total_flux_expansion(self) -> float:
        """Return total flux expansion of profile"""
        return self.Btot[self.Xpoint] / self.Btot[0]

    @property
    def average_frac_gradB(self) -> float:
        """Return the average fractional Btot gradient below the X-point"""
        return ((np.gradient(self.Btot, self.Spol) / self.Btot)[: self.Xpoint]).mean()

    @property
    def gradB_integral(self) -> float:
        """Return the integral of the fractional Btot gradient below the X-point"""
        return trapezoid(
            (np.gradient(self.Btot, self.Spol) / self.Btot)[: self.Xpoint],
            self.Spol[: self.Xpoint],
        )

    @property
    def gradB_average(self) -> float:
        """Return the integral of the fractional Btot gradient below the X-point"""
        return np.mean((np.gradient(self.Btot, self.Spol) / self.Btot)[: self.Xpoint])

    @property
    def Bpitch_integral(self) -> float:
        """Return the integral of the pitch angle Bpol/Btot below the X-point"""
        return trapezoid(
            (self.Bpol / self.Btot)[: self.Xpoint], self.Spol[: self.Xpoint]
        )

    @property
    def Bpitch_average(self) -> np.floating:
        """Return the integral of the pitch angle Bpol/Btot below the X-point"""
        return np.mean((self.Bpol / self.Btot)[: self.Xpoint])

    @property
    def average_B_ratio(self) -> float:
        """Return the average Btot below X-point"""
        return self.Btot[self.Xpoint] / (self.Btot[: self.Xpoint]).mean()

    def B(self, s: float) -> float:
        try:
            return self._B(s)
        except AttributeError:
            self._B = interpolate.interp1d(self.S, self.Btot, kind="cubic")
            return self._B(s)

    # Scaling and morphing functions

    def scale_flux_expansion(
        self,
        *,
        scale_factor: Scalar | None = None,
        expansion: Scalar | None = None,
        name: str | None = None,
    ) -> Self:
        r"""Scale a :math:`B_\mathrm{total}` profile to have an arbitrary
        flux expansion (ratio of :math:`B_\mathrm{X-point}` to
        :math:`B_\mathrm{target}`), return a new `Profile`.

        Specify either a scale factor (``scale_factor``) or required flux
        expansion (``expansion``) as a keyword argument.

        Parameters
        ----------
        scale_factor
            Multiplicative factor applied to initial ``Btot``
        expansion
            Desired flux expansion
        name
            Name of the new profile, uses old name by default.

        Example
        -------

        >>> new_profile = profile.scale_flux_expansion(scale_factor=2)

        """
        if scale_factor is None and expansion is None:
            raise ValueError(
                "Missing required argument: one of `scale_factor` or `expansion`"
            )
        if scale_factor is not None and expansion is not None:
            raise ValueError(
                "Exactly one of `scale_factor` or `expansion` must be supplied"
            )

        Bt_base = self.Btot[0]
        Bx_base = self.Bx
        BxBt_base = Bx_base / Bt_base

        if scale_factor is not None:
            expansion = BxBt_base * scale_factor

        # Keep type checkers happy...
        assert expansion is not None

        # Keep Bx the same, scale Bt.
        # Calc new Bt based on desired BtBx
        Bt_new = 1 / (expansion / Bx_base)

        Btot_new = self.Btot * (Bx_base - Bt_new) / (Bx_base - Bt_base)

        # Translate to keep the same Bx as before
        transl_factor = Btot_new[self.Xpoint] - Bx_base
        Btot_new -= transl_factor

        # Replace upstream of the Xpoint with the old data
        # So that we are only scaling downstream of Xpoint
        Btot_new[self.Xpoint :] = self.Btot[self.Xpoint :]

        if name is None:
            name = self.name

        return replace(self, Btot=Btot_new, name=name)

    def scale_connection_length(
        self,
        *,
        scale_factor: Scalar | None = None,
        connection_length: Scalar | None = None,
        name: str | None = None,
    ) -> Self:
        r"""Scale :math:`S_\parallel` and :math:`S_{pol}` profiles for arbitrary
        connection length, :math:`L_c`, return a new `Profile`.

        Specify either a scale factor or required connection length.

        Not that this will not modify :math:`R` and `:math:`Z` coordinates, so
        these will be invalid in the new profile.

        Parameters
        ----------
        scale_factor
            Multiplicative factor applied to initial ``S`` and ``Spol``
        connection_length
            Desired connection length
        name
            Name of the new profile, uses old name by default.
        """

        if scale_factor is None and connection_length is None:
            raise ValueError(
                "Missing required argument: one of `scale_factor` or `connection_length`"
            )
        if scale_factor is not None and connection_length is not None:
            raise ValueError(
                "Exactly one of `scale_factor` or `connection_length` must be supplied"
            )

        if connection_length is not None:
            scale_factor = connection_length / self.connection_length

        # Keep type checkers happy...
        assert scale_factor is not None

        # After scaling, the upstream portion will be kept the same, but the
        # total connection length must be scaled correctly. The scale factor
        # must therefore be adjusted such that only scaling the leg portion will
        # result in the correct connection length.
        # len_total = len_upstream + len_leg
        # len_new = len_total * scale_factor = len_upstream + len_leg_new
        # len_leg_new = len_total * scale_factor - len_upstream
        # adjusted_scale_factor = len_leg_new / len_leg
        adjusted_scale_factor = (
            self.connection_length * scale_factor - self.upstream_length
        ) / self.Sx

        # Scale up to get correct length
        S_new = self.S * adjusted_scale_factor
        Spol_new = self.Spol * adjusted_scale_factor

        # Align Xpoints
        S_new += self.S[self.Xpoint] - S_new[self.Xpoint]
        Spol_new += self.Spol[self.Xpoint] - Spol_new[self.Xpoint]

        # Make both the same upstream of self.Xpoint
        S_new[self.Xpoint :] = self.S[self.Xpoint :]
        Spol_new[self.Xpoint :] = self.Spol[self.Xpoint :]

        # Offset to make both targets at S = 0
        S_new -= S_new[0]
        Spol_new -= Spol_new[0]

        if name is None:
            name = self.name

        return replace(self, S=S_new, Spol=Spol_new, name=name)

    def offset_control_points(
        self,
        offsets: list[dict[str, float]],
        factor: float = 1.0,
        constant_pitch: bool = False,
        Bpol_shift: dict[str, float] | None = None,
        name: str | None = None,
    ) -> Self:
        """
        Take profile and add control points ``[x, y]``, then perform cord spline
        interpolation to get interpolated profile in ``[xs, ys]``.

        Offsets are a list of dictionaries, each defining a point along the leg
        to shift vertically or horizontally::

            [dict(pos = 1, offsety = -0.1, offsetx = 0.2), ...]

        Where ``pos`` is the fractional poloidal position along the field line
        starting at the target, and ``offsety`` and ``offsetx`` are vertical and
        horizontal offsets in [m].

        NOTE: ``pos`` is defined with 0 at the target, but should be defined
        starting from the X-point, i.e. ``pos`` should start at 1 and then
        decrease towards 0.

        Following the offsets, ``S``, ``Spol``, ``Bpol`` and ``Btot`` are
        recalculated.

        Currently only supports changing topology below the X-point.

        Parameters
        ----------
        offsets
            Each dictionary contains either positions or offsets and a position
            along the field line of a control point, starting at the X-point
            (``pos=1``).
        factor
            Factor to scale the effect of point shifting, where 0 results in no
            change, 1 results in the profile shifted according to offsets, and
            0.5 shifts the profile halfway.
        constant_pitch
            If ``True``, keep the same magnetic pitch when recalculating the
            profile. If ``False``, keep same Bpol profile.  Keeping magnetic
            pitch the same can lead to high Bpol causing unrealistically high
            Btot near the target for short divertor leg designs where Btor is
            low. This can lead to a region of flux compression.
        Bpol_shift
            Dict containing:

            - ``"width"``: gaussian width in m
            - ``"pos"``: position in m poloidal from the target
            - ``"height"``: height in Bpol units
        name
            Name of the new profile, uses old name by default.
        """
        if Bpol_shift is not None:
            keys = {"width", "pos", "height"}
            if not all((k := key) in Bpol_shift for key in keys):
                raise ValueError(f"Missing key {k} in Bpol_shift")

        # Control points defining profile
        R_control, Z_control = self._shift_points(
            self.R_leg, self.Z_leg, offsets, factor=factor
        )

        # Calculate the new leg RZ from the spline
        # Note xs and ys are backwards
        dist = get_cord_distance(self.R_leg, self.Z_leg)  # Distances along old leg
        spl = cord_spline(R_control[::-1], Z_control[::-1], return_spline=True)

        # New leg interpolated onto same points as old leg
        R_leg_spline, Z_leg_spline = spl(dist)

        # Calculate total RZ by adding upstream
        R_new = np.concatenate((R_leg_spline, self.R[self["Xpoint"] + 1 :]))
        Z_new = np.concatenate((Z_leg_spline, self.Z[self["Xpoint"] + 1 :]))

        # Calculate existing toroidal field (1/R)
        Btor = np.sqrt(self.Btot**2 - self.Bpol**2)  # Toroidal field
        Bpitch = self.Bpol / self.Btot

        # Save existing parameters
        Btor_leg = Btor[: self.Xpoint + 1]
        Bpol_leg = self.Bpol[: self.Xpoint + 1]
        Bpitch_leg = Bpitch[: self.Xpoint + 1]

        # Calculate new S poloidal from R,Z
        Spol_new = returnll(R_new, Z_new)

        # Calculate toroidal field (1/R)
        Btor_leg_new = Btor_leg * (self.R_leg / R_leg_spline)

        # Calculate total field
        # Either keep same Bpitch or same Bpol
        if constant_pitch:
            Btot_leg_new = np.sqrt(Btor_leg_new**2 / (1 - Bpitch_leg**2))
            Bpol_leg_new = np.sqrt(Btot_leg_new**2 - Btor_leg_new**2)

            Bpol_new = np.concatenate(
                [
                    Bpol_leg_new,
                    self.Bpol[self.Xpoint + 1 :],
                ]
            )

            # Convolve Bpol with a gaussian of a width, position and height
            if Bpol_shift is not None:
                width = Bpol_shift["width"]
                pos = Bpol_shift["pos"]
                height = Bpol_shift["height"]
                weight = (
                    width
                    * np.sqrt(2 * np.pi)
                    * np.exp(-0.5 * ((Spol_new - pos) / width) ** 2)
                )
                weight = weight / np.max(weight) * height
                Bpol_new -= weight

        else:
            Btot_leg_new = np.sqrt(Btor_leg_new**2 + Bpol_leg**2)
            Bpol_new = self.Bpol

        Btot_new = np.concatenate((Btot_leg_new, self.Btot[self["Xpoint"] + 1 :]))

        # Calculate parallel connection length
        S_new = returnS(R_new, Z_new, Btot_new, Bpol_new)

        if name is None:
            name = self.name

        return replace(
            self,
            R=R_new,
            Z=Z_new,
            Bpol=Bpol_new,
            Btot=Btot_new,
            S=S_new,
            Spol=Spol_new,
            name=name,
        )

    @staticmethod
    def _shift_points(
        R: FloatArray,
        Z: FloatArray,
        offsets: list[dict[str, float]],
        factor: float = 1.0,
    ) -> tuple[NDArray, NDArray]:
        """
        Make control points on a field line according to points of index in list i.

        Parameters
        ----------
        R
            R coordinates of field line.
        Z
            Z coordinates of field line.
        offsets
            Each dictionary contains either positions or offsets and a position
            along the field line of a control point. See ``offset_control_points()``.
        factor
            Factor to scale the effect of point shifting, where 0 = no change,
            1 = profile shifted according to offsets, 0.5 = profile shifted halfway.
        """

        #        XPOINT ---------   TARGET
        spl = cord_spline(R, Z, return_spline=True)
        x, y = [], []
        keys = {"pos", "offsetx", "offsety", "posx", "posy"}

        for point in offsets:
            position = point["pos"]

            if "offsetx" in point and "posx" in point:
                raise ValueError("Offset and position cannot be set simultaneously")
            if "offsety" in point and "posy" in point:
                raise ValueError("Offset and position cannot be set simultaneously")
            for key in point:
                if key not in keys:
                    msg = f"Unknown key {key}, should be one of {keys}"
                    raise ValueError(msg)

            # RZ coordinates of existing point
            Rs, Zs = spl(position)

            offsetx = point.get("offsetx", 0)
            offsety = point.get("offsety", 0)

            # If position specified, overwrite offsets with a calculation
            if "posx" in point:
                offsetx = point["posx"] - Rs
            if "posy" in point:
                offsety = point["posy"] - Zs

            if ("posx" in point or "posy" in point) and factor != 1:
                raise Exception("Factor scaling not supported when passing position")

            offsetx *= factor
            offsety *= factor

            x.append(Rs + offsetx)
            y.append(Zs + offsety)

        return np.array(x), np.array(y)

    def get_offsets_strike_point(
        self, positions: FloatArray, R_strike: float, Z_strike: float
    ) -> list[dict[str, float]]:
        """Calculate offsets to allow the creation of a new field line Profile
        based on the strike point coordinates only.

        Returns a list of dictionaries containing the calculated offsets for
        each position.  Each dictionary has the following keys:

        - ``"pos"``: The original position.
        - ``"posx"``: The new radial coordinate after applying the offset.
        - ``"posy"``: The new vertical coordinate after applying the offset.

        Useful for parameter scans.

        Parameters
        ----------
        positions
            Positions along the leg where offsets need to be calculated.  These
            are the interpolation spline control points. You need at least 3
            between the X-point (pos = 1) and target (pos = 0). They need to be
            defined from the X-point first (i.e. starting at 1 and decreasing
            towards 0). Recommend to use several points near the target to
            ensure the profile doesn't curl due to the interpolation.
        R_strike
            Desired radial coordinate of the strike point.
        Z_strike
            Desired vertical coordinate of the strike point.
        """

        Z_Xpoint = self.Z[self.Xpoint]
        R_Xpoint = self.R[self.Xpoint]

        R_strike_original = self.R[0]
        Z_strike_original = self.Z[0]

        R = np.zeros_like(positions)
        Z = np.zeros_like(positions)
        spl = cord_spline(self.R_leg, self.Z_leg, return_spline=True)

        for i, pos in enumerate(positions):
            R[i], Z[i] = spl(pos)

        strikeOffsetR = R_strike - R_strike_original
        strikeOffsetZ = Z_strike - Z_strike_original

        Rdist = (R_Xpoint - R) / (R_Xpoint - R[-1])
        Zdist = (Z_Xpoint - Z) / (Z_Xpoint - Z[-1])
        Rnew = R + strikeOffsetR * Rdist
        Znew = Z + strikeOffsetZ * Zdist

        offsets = []
        for i, pos in enumerate(positions):
            offsets.append(
                {
                    "pos": pos,
                    "posx": Rnew[i],
                    "posy": Znew[i],
                }
            )
        return offsets

    def plot_topology(self):
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))

        basestyle = {"c": "black"}
        xstyle = {"marker": "+", "linewidth": 2, "s": 150, "c": "r", "zorder": 100}

        ax = axes[0, 0]
        ax.set_title(r"Fractional $B_{tot}$ gradient")
        ax.plot(
            self["Spol"],
            np.gradient(self["Btot"], self["Spol"]) / self["Btot"],
            **basestyle,
        )
        ax.scatter(
            self["Spol"][self["Xpoint"]],
            (np.gradient(self["Btot"], self["Spol"]) / self["Btot"])[self["Xpoint"]],
            **xstyle,
        )
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{tot}$ $[T]$")

        ax = axes[1, 0]
        ax.set_title(r"$B_{tot}$")
        ax.plot(self["Spol"], self["Btot"], **basestyle)
        ax.scatter(self["Spol"][self["Xpoint"]], self["Btot"][self["Xpoint"]], **xstyle)
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{tot}$ $[T]$")

        ax = axes[0, 1]
        ax.set_title(r"Field line pitch $B_{pol}/B_{tot}$")
        ax.plot(self["Spol"], self["Bpol"] / self["Btot"], **basestyle)
        ax.scatter(
            self["Spol"][self["Xpoint"]],
            (self["Bpol"] / self["Btot"])[self["Xpoint"]],
            **xstyle,
        )
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{pol} \ / B_{tot}$ ")

        ax = axes[1, 1]
        ax.set_title(r"$B_{pol}$")
        ax.plot(self["Spol"], self["Bpol"], **basestyle)
        ax.scatter(
            self["Spol"][self["Xpoint"]], (self["Bpol"])[self["Xpoint"]], **xstyle
        )
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{\theta}$ $[T]$")

        fig.tight_layout()

    def plot(
        self,
        mode: str = "Btot",
        ax: Axes | None = None,
        legend: bool = False,
        parallel: bool = True,
        full_RZ: bool = False,
        label: str = "",
        color: str = "teal",
        **kwargs,
    ):
        """
        Parameters
        ----------
        mode
            What to plot:

            - ``"Btot"``: total B profile
            - ``"RZ"``: RZ space leg profile (excl. above X-point)
            - ``"Spar_Spol"``: Parallel vs poloidal connection length
            - ``"magnetic_pitch"``: Bpol / Btot
        ax
            Matplotlib axis to plot on (optional)
        legend
            Whether to include a legend for when no axis is provided
        parallel
            If true, plot parallel connection length, else poloidal
        full_RZ
            If false, exclude region above X-point in RZ plot
        color
            Color of the plot
        kwargs
            Keyword arguments to pass to plot
        """
        if ax is None:
            _fig, ax = plt.subplots()

        if parallel:
            x = self.S
            ax.set_xlabel(r"$s_{\parallel}$ (m from target)")
        else:
            x = self.Spol
            ax.set_xlabel(r"$s_{\theta}$ (m from target)")

        if mode == "Btot":
            ax.plot(x, self.Btot, color=color, label=label, **kwargs)
            ax.set_ylabel(r"$B_{tot}$ (T)")
        elif mode == "Bpol":
            ax.plot(x, self.Bpol, color=color, label=label, **kwargs)
            ax.set_ylabel(r"$B_{pol}$ (T)")
        elif mode == "RZ":
            if full_RZ:
                ax.plot(
                    self.R,
                    self.Z,
                    color=color,
                    label=label,
                    **kwargs,
                )
            else:
                ax.plot(
                    self.R[: self.Xpoint + 1],
                    self.Z[: self.Xpoint + 1],
                    color=color,
                    label=label,
                    **kwargs,
                )
            ax.set_xlabel("R (m)")
            ax.set_ylabel("Z (m)")
        elif mode == "Spar_Spol":
            ax.plot(self.Spol, self.S, color=color, label=label, **kwargs)
            ax.set_ylabel(r"$S_{\parallel}$")
        elif mode == "magnetic_pitch":
            ax.plot(x, self.Bpol / self.Btot, color=color, label=label, **kwargs)
            ax.set_ylabel(r"$B_{pol} / B_{tot}$")
        else:
            raise ValueError(
                f"Mode {mode} not recognised. Try Btot, RZ, magnetic_pitch or Spar_Spol"
            )

        if legend is True and ax is None:
            ax.legend()

    def plot_control_points(
        self,
        linesettings=None,
        markersettings=None,
        ylim=(None, None),
        xlim=(None, None),
        dpi=100,
        ax=None,
        color="limegreen",
    ):
        if markersettings is None:
            markersettings = {}
        if linesettings is None:
            linesettings = {}
        if ax is None:
            _fig, ax = plt.subplots(dpi=dpi)
            ax.plot(
                self["R"],
                self["Z"],
                linewidth=3,
                marker="o",
                markersize=0,
                color="black",
                alpha=1,
            )
            ax.set_xlabel(r"$R\ (m)$")
            ax.set_ylabel(r"$Z\ (m)$")

            if ylim != (None, None):
                ax.set_ylim(ylim)
            if xlim != (None, None):
                ax.set_xlim(xlim)

            ax.set_title("RZ Space")
            ax.grid(alpha=0.3, color="k")
            ax.set_aspect("equal")

        default_line_args = {"c": color, "alpha": 0.7, "zorder": 100}
        default_marker_args = {
            "c": color,
            "marker": "+",
            "linewidth": 15,
            "s": 3,
            "zorder": 100,
        }

        line_args = {**default_line_args, **linesettings}
        marker_args = {**default_marker_args, **markersettings}

        ax.plot(
            self["R_leg_spline"], self["Z_leg_spline"], **line_args, label=self.name
        )
        ax.scatter(self["R_control"], self["Z_control"], **marker_args)

        ax.set_xlabel(r"$R\ (m)$")
        ax.set_ylabel(r"$Z\ (m)$")

        pad = 0.2

        selector = slice(None, self["Xpoint"])

        R_leg_original = self["R_original"][selector]
        Z_leg_original = self["Z_original"][selector]

        Rmax = R_leg_original.max()
        Rmin = R_leg_original.min()
        Zmax = Z_leg_original.max()
        Zmin = Z_leg_original.min()

        Rspan = Rmax - Rmin
        Zspan = Zmax - Zmin

        ax.set_xlim(Rmin - Rspan * pad, Rmax + Rspan * pad)
        ax.set_ylim(Zmin - Zspan * pad, Zmax + Zspan * pad)

        if ylim != (None, None):
            ax.set_ylim(ylim)
        if xlim != (None, None):
            ax.set_xlim(xlim)

        ax.grid(alpha=0.3, color="k")
        ax.set_aspect("equal")


class Morph:
    """
    This class creates new field line profiles by interpolating between
    any two profiles. You provide a start and end profile and it will return
    intermediate ones according to a morph factor.
    """

    def __init__(self, R, Z, Xpoint, Btot, Bpol, S, Spol):
        """
        Class is initialised with the properties of the base (start) profile.
        Needs to be refactored to accept a Profile class.
        """
        self.R = R
        self.Z = Z
        self.Xpoint = Xpoint

        # Split into leg up to and incl. xpoint
        self.R_leg = R[: Xpoint + 1]
        self.Z_leg = Z[: Xpoint + 1]

        self.Btot = Btot
        self.Bpol = Bpol
        self.S = S
        self.Spol = Spol

    def set_start_profile(self, profile, offsets):
        """
        Sets the start profile based on what the class was initialised with.
        You must provide offsets which is a dictionary of the spline control
        points and their offsets. For the start profile, the offsets dictionary
        should contain just the control points with no offsets.
        See the function shift_offsets() for what the offsets should look like.
        """
        self.start = self.make_profile_spline(profile, offsets)
        self.start["R_leg"] = self.R_leg
        self.start["Z_leg"] = self.Z_leg
        self.start["R"] = self.R
        self.start["Z"] = self.Z
        self.start["S"] = self.S
        self.start["Spol"] = self.Spol
        self.start["Btot"] = self.Btot
        self.start["Bpol"] = self.Bpol
        self.start["Xpoint"] = self.Xpoint

    def set_end_profile(self, offsets):
        """
        Sets the end profile based on the offsets dictionary with the original
        control point coordinates and their desired offsets.
        """
        self.end = self.make_profile_spline(offsets)
        self.end = self._populate_profile(self.end)

    def generate_profiles(self, factors):
        """
        Make a series of profiles according to provided factors
        where factor = 0 corresponds to start, factor = 1
        corresponds to end and factor = 0.5 corresponds to halfway.
        """
        profiles = {}
        for i in factors:
            profiles[i] = self.morph_between(i)

        self.profiles = profiles

    def morph_between(self, factor):
        prof = {}
        prof["x"] = self.start["x"] + factor * (self.end["x"] - self.start["x"])
        prof["y"] = self.start["y"] + factor * (self.end["y"] - self.start["y"])
        prof["xs"], prof["ys"] = cord_spline(prof["x"], prof["y"])  # Interpolate
        return self._populate_profile(prof)

    def _populate_profile(self, prof):
        """
        Add the rest of the profile to the leg above the X-point
        Add Bpol and Btot along entire leg
        Returns new modified profile
        """

        start = self.start
        prof["Xpoint"] = start["Xpoint"]

        ## Add leg above X-point
        # xs and ys are backwards
        dist = get_cord_distance(
            start["R_leg"], start["Z_leg"]
        )  # Distances along old leg
        spl = cord_spline(
            prof["xs"][::-1], prof["ys"][::-1], return_spline=True
        )  # Spline interp for new leg
        R_leg_new, Z_leg_new = spl(
            dist
        )  # New leg interpolated onto same points as old leg

        prof["R"] = np.concatenate(
            [
                R_leg_new,
                start["R"][start["Xpoint"] + 1 :],
            ]
        )

        prof["Z"] = np.concatenate(
            [
                Z_leg_new,
                start["Z"][start["Xpoint"] + 1 :],
            ]
        )

        ## Poloidal dist and field
        prof["Spol"] = returnll(prof["R"], prof["Z"])
        prof["Bpol"] = start["Bpol"].copy()  # Assume same poloidal field as start

        ## Total field
        Btor = np.sqrt(start["Btot"] ** 2 - start["Bpol"] ** 2)  # Toroidal field
        Btor_leg = Btor[: start["Xpoint"] + 1]
        Btor_leg_new = Btor_leg * (start["R_leg"] / R_leg_new)

        Bpol_leg = start["Bpol"][: start["Xpoint"] + 1]
        Btot_leg_new = np.sqrt(Btor_leg_new**2 + Bpol_leg**2)

        prof["Btot"] = np.concatenate(
            [
                Btot_leg_new,
                start["Btot"][start["Xpoint"] + 1 :],
            ]
        )

        prof["S"] = returnS(prof["R"], prof["Z"], prof["Btot"], prof["Bpol"])

        return prof

    def plot_profile(self, prof, dpi=100, ylim=(None, None), xlim=(None, None)):
        _fig, ax = plt.subplots(dpi=dpi)

        s = self.start
        p = prof

        ax.plot(s["xs"], s["ys"], c="forestgreen", zorder=100, alpha=1)
        ax.scatter(
            s["x"], s["y"], c="limegreen", zorder=100, marker="+", linewidth=15, s=3
        )
        ax.plot(p["xs"], p["ys"], c="deeppink", zorder=100, alpha=0.4)
        ax.scatter(p["x"], p["y"], c="red", zorder=100, marker="x")

        ax.plot(
            s["R"],
            s["Z"],
            linewidth=3,
            marker="o",
            markersize=0,
            color="black",
            alpha=1,
        )

        ax.set_xlabel(r"$R\ (m)$", fontsize=15)
        ax.set_ylabel(r"$Z\ (m)$")

        if ylim != (None, None):
            ax.set_ylim(ylim)
        if xlim != (None, None):
            ax.set_xlim(xlim)

        ax.set_title("RZ Space")
        ax.grid(alpha=0.3, color="k")
        ax.set_aspect("equal")


def compare_profile_topologies(base_profile, profiles):
    """
    Do a bunch of plots to compare the properties of two profiles
    """

    d = base_profile

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    markers = ["o", "v"]

    profstyle = {"alpha": 0.3}

    basestyle = {"c": "black"}
    xstyle = {"marker": "+", "linewidth": 2, "s": 150, "c": "r", "zorder": 100}

    S_pol_xpoint_max = max(p["Spol"][p["Xpoint"]] for p in profiles)

    Spol_shift_base = S_pol_xpoint_max - d["Spol"][d["Xpoint"]]

    ax = axes[0, 0]
    ax.set_title(r"Fractional $B_{tot}$ gradient")

    ax.plot(
        d["Spol"] + Spol_shift_base,
        np.gradient(d["Btot"], d["Spol"]) / d["Btot"],
        **basestyle,
    )
    ax.scatter(
        d["Spol"][d["Xpoint"]] + Spol_shift_base,
        (np.gradient(d["Btot"], d["Spol"]) / d["Btot"])[d["Xpoint"]],
        **xstyle,
    )
    for i, p in enumerate(profiles):
        Spol_shift = S_pol_xpoint_max - p["Spol"][p["Xpoint"]]
        ax.plot(
            p["Spol"] + Spol_shift,
            np.gradient(p["Btot"], p["Spol"]) / p["Btot"],
            **profstyle,
            marker=markers[i],
        )
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{tot}$ $[T]$")

    ax = axes[1, 0]
    ax.set_title(r"$B_{tot}$")

    ax.plot(d["Spol"] + Spol_shift_base, d["Btot"], **basestyle)
    ax.scatter(
        d["Spol"][d["Xpoint"]] + Spol_shift_base, d["Btot"][d["Xpoint"]], **xstyle
    )
    for i, p in enumerate(profiles):
        Spol_shift = S_pol_xpoint_max - p["Spol"][p["Xpoint"]]
        ax.plot(p["Spol"] + Spol_shift, p["Btot"], **profstyle, marker=markers[i])
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{tot}$ $[T]$")

    ax = axes[0, 1]
    ax.set_title(r"Field line pitch $B_{pol}/B_{tot}$")

    ax.plot(d["Spol"] + Spol_shift_base, d["Bpol"] / d["Btot"], **basestyle)
    ax.scatter(
        d["Spol"][d["Xpoint"]] + Spol_shift_base,
        (d["Bpol"] / d["Btot"])[d["Xpoint"]],
        **xstyle,
    )
    for i, p in enumerate(profiles):
        Spol_shift = S_pol_xpoint_max - p["Spol"][p["Xpoint"]]
        ax.plot(
            p["Spol"] + Spol_shift,
            p["Bpol"] / p["Btot"],
            **profstyle,
            marker=markers[i],
        )
    ax.set_xlabel(r"$S_{\theta} \   [m]$")
    ax.set_ylabel(r"$B_{pol} \ / B_{tot}$ ")

    ax = axes[1, 1]
    ax.set_title(r"$B_{pol}$")

    ax.plot(d["Spol"] + Spol_shift_base, d["Bpol"], **basestyle)
    ax.scatter(
        d["Spol"][d["Xpoint"]] + Spol_shift_base, (d["Bpol"])[d["Xpoint"]], **xstyle
    )
    for i, p in enumerate(profiles):
        Spol_shift = S_pol_xpoint_max - p["Spol"][p["Xpoint"]]
        ax.plot(p["Spol"] + Spol_shift, p["Bpol"], **profstyle, marker=markers[i])
        ax.scatter(
            p["Spol"][p["Xpoint"]] + Spol_shift, (p["Bpol"])[p["Xpoint"]], **xstyle
        )
    ax.set_xlabel(r"$S_{\theta} \   [m]$")
    ax.set_ylabel(r"$B_{\theta}$ $[T]$")

    fig.tight_layout()


def cord_spline(x, y, return_spline=False):
    """
    Do cord interpolation of x and y. This parametrises them
    by the cord length and allows them to go back on themselves,
    i.e. to have non-unique X values and non-monotonicity.
    I think you need to have at least 4 points.

    https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#parametric-spline-curves
    """
    # FIXME: Function should have a consistent return type, this breaks type checking
    p = np.stack((x, y))
    u_cord = get_cord_distance(x, y)

    spl = sp.interpolate.make_interp_spline(u_cord, p, axis=1)

    # FIXME: 200 is arbitrary
    uu = np.linspace(u_cord[0], u_cord[-1], 200)
    R, Z = spl(uu)

    if return_spline:
        return spl

    return R, Z


def get_cord_distance(x, y):
    """
    Return array of distances along a curve defined by x and y.
    """
    p = np.stack((x, y))
    dp = p[:, 1:] - p[:, :-1]  # 2-vector distances between points
    l_norm = (dp**2).sum(axis=0)  # squares of lengths of 2-vectors between points
    u_cord = np.sqrt(l_norm).cumsum()  # Cumulative sum of 2-norms
    u_cord /= u_cord[-1]  # normalize to interval [0,1]
    return np.r_[0, u_cord]  # the first point is parameterized at zero


def returnll(R, Z):
    # return the poloidal distances from the target for a given configuration
    # TODO: There's probably a way to do this without a loop in NumPy
    PrevR = R[0]
    ll = []
    currentl = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        currentl += dl
        ll.append(currentl)
        PrevR = R[i]
        PrevZ = Z[i]
    return np.asarray(ll)


def returnS(R, Z, B, Bpol):
    # return the real total distances from the target for a given configuration
    # TODO: There's probably a way to do this without a loop in NumPy
    PrevR = R[0]
    s = []
    currents = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        ds = dl * np.abs(B[i]) / np.abs(Bpol[i])
        currents += ds
        s.append(currents)
        PrevR = R[i]
        PrevZ = Z[i]
    return np.asarray(s)
