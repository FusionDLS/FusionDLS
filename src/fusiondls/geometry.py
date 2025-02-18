import pickle
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from typing_extensions import Self

from .Profile import Profile
from .typing import FloatArray, PathLike, Scalar


@dataclass
class MagneticGeometry:
    r"""Magnetic geometry for a diverator leg

    .. todo:: Make X-point variables properties?
              Means ensuring they're dropped when reading in

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
        :math:`S_\parallel`, distance from the target
    Spol:
        :math:`S_{poloidal}`
    zl:

    Xpoint:
        Index of the X-point in the leg arrays
    Bx:
        Value of the magnetic field at the X-point
    Sx:
        Value of :math:`S` at the X-point
    Spolx:
        Value of :math:`S_{pol}` at the X-point
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
    Xpoint: int
    zl: FloatArray | None = None
    R_full: FloatArray | None = None
    Z_full: FloatArray | None = None
    R_ring: FloatArray | None = None
    Z_ring: FloatArray | None = None

    @property
    def Sx(self):
        return self.S[self.Xpoint]

    @property
    def Spolx(self):
        return self.Spol[self.Xpoint]

    @property
    def Bx(self):
        return self.Btot[self.Xpoint]

    @property
    def zx(self):
        return self.zl[self.Xpoint]

    @classmethod
    def from_profile(cls, profile: Profile) -> Self:
        """Create a ``MagneticGeometry`` from a ``Profile`` object.

        These classes are expected to be merged in a future build.
        """
        return cls(
            Bpol=profile.Bpol,
            Btot=profile.Btot,
            R=profile.R,
            Z=profile.Z,
            S=profile.S,
            Spol=profile.Spol,
            Xpoint=profile.Xpoint,
        )

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

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def B(self, s: float) -> float:
        try:
            return self._B(s)
        except AttributeError:
            self._B = interpolate.interp1d(self.S, self.Btot, kind="cubic")
            return self._B(s)

    def scale_flux_expansion(
        self,
        *,
        scale_factor: Scalar | None = None,
        expansion: Scalar | None = None,
    ) -> Self:
        r"""Scale a :math:`B_\mathrm{total}` profile to have an arbitrary
        flux expansion (ratio of :math:`B_\mathrm{X-point}` to
        :math:`B_\mathrm{target}`), return a new `MagneticGeometry`.

        Specify either a scale factor (``scale_factor``) or required flux
        expansion (``expansion``) as a keyword argument

        Parameters
        ----------
        scale_factor:
            Multiplicative factor applied to initial ``Btot``
        expansion:
            Desired flux expansion

        Example
        -------

        >>> new_geometry = geometry.scale_flux_expansion(scale_factor=2)

        """

        if scale_factor is None and expansion is None:
            raise ValueError(
                "Missing required argument: one of `scale_factor` or `BxBt`"
            )
        if scale_factor is not None and expansion is not None:
            raise ValueError(
                "Exactly one of `scale_factor` or `BxBt` must be supplied (both given)"
            )

        Bt_base = self.Btot[0]
        Bx_base = self.Bx
        BxBt_base = Bx_base / Bt_base

        if scale_factor is not None:
            expansion = BxBt_base * scale_factor

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

        new_data = asdict(self)
        new_data["Btot"] = Btot_new

        return self.__class__(**new_data)

    def scale_connection_length(
        self,
        *,
        scale_factor: Scalar | None = None,
        connection_length: Scalar | None = None,
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

        if scale_factor is None and connection_length is None:
            raise ValueError("Missing required argument: one of `scale_factor` or `Lc`")
        if scale_factor is not None and connection_length is not None:
            raise ValueError(
                "Exactly one of `scale_factor` or `Lc` must be supplied (both given)"
            )

        Lc_base = self.S[self.Xpoint]

        # Having Lc non-zero
        if connection_length is not None:
            scale_factor = connection_length / Lc_base

        # This is only required to keep mypy happy
        assert scale_factor is not None

        # Scale up to get correct length
        S_new = self.S * scale_factor
        Spol_new = self.Spol * scale_factor

        # Align Xpoints
        S_new += self.S[self.Xpoint] - S_new[self.Xpoint]
        Spol_new += self.Spol[self.Xpoint] - Spol_new[self.Xpoint]

        # Make both the same upstream of self.Xpoint
        S_new[self.Xpoint :] = self.S[self.Xpoint :]
        Spol_new[self.Xpoint :] = self.Spol[self.Xpoint :]

        # Offset to make both targets at S = 0
        S_new -= S_new[0]
        Spol_new -= Spol_new[0]

        new_data = asdict(self)
        new_data["S"] = S_new
        new_data["Spol"] = Spol_new

        return self.__class__(**new_data)

    def scale_midplane_length(
        self,
        *,
        scale_factor: Scalar | None = None,
        midplane_length: Scalar | None = None,
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

        if scale_factor is None and midplane_length is None:
            raise ValueError("Missing required argument: one of `scale_factor` or `Lm`")
        if scale_factor is not None and midplane_length is not None:
            raise ValueError(
                "Exactly one of `scale_factor` or `Lm` must be supplied (both given)"
            )

        Lm_base = self.S[-1]

        if midplane_length is not None:
            scale_factor = midplane_length / Lm_base

        # This is only required to keep mypy happy
        assert scale_factor is not None

        # Scale up to get correct length
        S_new = self.S * scale_factor
        Spol_new = self.Spol * scale_factor

        # Align Xpoints
        S_new += self.S[self.Xpoint] - S_new[self.Xpoint]
        Spol_new += self.Spol[self.Xpoint] - Spol_new[self.Xpoint]

        # Make both the same upstream of self.Xpoint
        S_new[: self.Xpoint] = self.S[: self.Xpoint]
        Spol_new[: self.Xpoint] = self.Spol[: self.Xpoint]

        # Offset to make both targets at S = 0
        S_new -= S_new[0]
        Spol_new -= Spol_new[0]

        new_data = asdict(self)
        new_data["S"] = S_new
        new_data["Spol"] = Spol_new

        return self.__class__(**new_data)

    def refine(
        self,
        Sfront,
        fine_ratio: float = 1.5,
        width: float = 4,
        resolution: int | None = None,
        diagnostic_plot: bool = False,
        tolerance: float = 1e-3,
        maxiter: int = 50,
    ) -> Self:
        """Refines the grid around the front location.

        Refinement is in the form of a Gaussian distribution with a peak
        determined by fine_ratio and a given width.

        Parameters
        ----------
        fine_ratio
            Ratio of coarse cell size to fine cell size
        width
            Width of the fine region in meters parallel
        resolution
            resolution of resulting grid. If None, use same resolution as
            original grid.
        """

        if resolution is None:
            resolution = len(self.S)

        ## Grid generation is an iterative process because dSnew must know where to put the gaussian
        # refinement in S space, so it needs an initial S guess. Then we calculate new S from the dS
        # and loop again until it stops changing.
        # Initialise S with uniform spacing
        Snew = np.linspace(self.S[0], self.S[-1], resolution - 1)

        if diagnostic_plot:
            fig, axes = plt.subplots(2, 1, figsize=(5, 5), height_ratios=(8, 4))

        dSnew2 = np.ones_like(Snew) * 1e-18
        for i in range(maxiter):
            dSnew = 1 / (
                (width * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((Snew - Sfront) / (width)) ** 2)
                * (fine_ratio - 1)
                + 1
            )
            dSnew *= self.S[-1] / dSnew.sum()  # Normalise to the original S
            Snew = np.cumsum(dSnew)
            residual = abs((dSnew2[-1] - dSnew[-1]) / dSnew2[-1])
            dSnew2 = dSnew

            if diagnostic_plot:
                axes[0].plot(Snew, dSnew, label=i)
                axes[1].scatter(
                    Snew,
                    np.ones_like(Snew) * -i,
                    marker="|",
                    s=5,
                    linewidths=0.5,
                    alpha=0.1,
                )

            if residual < tolerance:
                # print(f"Residual is {residual}, breaking")
                break

        else:
            raise RuntimeError(
                f"Iterative grid adaption iteration limit ({maxiter}) reached, "
                f"try reducing refinement ratio ({fine_ratio}) and running with 'diagnostic_plot=True'"
            )

        # len(dS) = len(S) - 1
        Snew = np.insert(Snew, 0, 0)

        # Grid width diagnostics plot settings
        if diagnostic_plot:
            axes[1].set_yticklabels([])
            fig.tight_layout()
            fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncols=5)
            axes[0].set_title("Adaptive grid iterations")

            axes[0].set_ylabel("dS [m]")
            axes[0].set_xlabel("S [m]")
            axes[1].set_title("S spacing")
            axes[1].set_xlabel("S [m]")
            fig.tight_layout()

        ## Interpolate geometry and field onto the new S coordinate
        pnew = {}
        pnew["S"] = Snew
        for par in ["S", "Spol", "R", "Z", "Btot", "Bpol"]:
            if par not in {"Xpoint", "S"}:
                pnew[par] = interpolate.make_interp_spline(
                    self.S, getattr(self, par), k=2
                )(Snew)

                if diagnostic_plot:
                    fig, ax = plt.subplots(dpi=100)
                    ax.plot(
                        self.S,
                        getattr(self, par),
                        label="Original",
                        marker="o",
                        color="darkorange",
                        alpha=0.3,
                        ms=10,
                        markerfacecolor="None",
                    )
                    ax.plot(pnew["S"], pnew[par], label="New", marker="o", ms=3)
                    ax.set_title(par)

        pnew["Xpoint"] = int(np.argmin(np.abs(Snew - self.S[self.Xpoint])))

        return self.__class__(**pnew)
