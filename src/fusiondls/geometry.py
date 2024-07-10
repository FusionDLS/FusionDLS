import pickle
from dataclasses import asdict, dataclass
from typing import Optional

from typing_extensions import Self

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
    def from_pickle(cls, filename: PathLike, design: str, side: str) -> Self:
        """Read a particular design and side from a pickle balance file."""

        with open(filename, "rb") as f:
            eqb = pickle.load(f)

        return cls(**eqb[design][side])

    @classmethod
    def read_design(cls, filename: PathLike, design: str) -> dict[str, Self]:
        """Read all divertor legs for a single design from a pickle balance file."""

        with open(filename, "rb") as f:
            eqb = pickle.load(f)

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
