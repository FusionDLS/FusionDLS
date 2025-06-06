from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from contourpy import contour_generator
from freegs import Equilibrium, critical, fieldtracer, jtor, machine
from freeqdsk import geqdsk
from matplotlib import path as mpath
from numpy.typing import NDArray
from pyloidal.cocos import Transform as TransformCocos
from pyloidal.cocos import identify_cocos
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

from . import MagneticGeometry
from .typing import FloatArray

COCOS = 11


def _transform_geqdsk(
    data: geqdsk.GEQDSKFile,
    cocos_in: int | None,
    cocos_out: int = COCOS,
    clockwise_phi: bool = False,
) -> geqdsk.GEQDSKFile:
    """Transform a G-EQDSK file from its original COCOS convention to 11."""
    if cocos_in is None:
        cocos_in = identify_cocos(
            b_toroidal=data.bcentr,
            plasma_current=data.cpasma,
            safety_factor=data.qpsi,
            poloidal_flux=np.linspace(data.simagx, data.sibdry, data.nx),
            clockwise_phi=clockwise_phi,
            minor_radii=_get_minor_radii(data),
        )[0]
    transform = TransformCocos(cocos_in, cocos_out)
    return geqdsk.GEQDSKFile(
        comment=data.comment,
        shot=data.shot,
        nx=data.nx,
        ny=data.ny,
        rdim=data.rdim,
        zdim=data.zdim,
        rcentr=data.rcentr,
        rleft=data.rleft,
        zmid=data.zmid,
        rmagx=data.rmagx,
        zmagx=data.zmagx,
        simagx=data.simagx * transform.psi,
        sibdry=data.sibdry * transform.psi,
        bcentr=data.bcentr * transform.b_toroidal,
        cpasma=data.cpasma * transform.plasma_current,
        fpol=data.fpol * transform.f,
        pres=data.pres,
        ffprime=data.ffprime * transform.ffprime,
        pprime=data.pprime * transform.pprime,
        psi=data.psi * transform.psi,
        qpsi=data.qpsi * transform.q,
        nbdry=data.nbdry,
        nlim=data.nlim,
        rbdry=data.rbdry,
        zbdry=data.zbdry,
        rlim=data.rlim,
        zlim=data.zlim,
    )


def _get_minor_radii(data: geqdsk.GEQDSKFile) -> NDArray[np.floating]:
    """Get the minor radii of each contour in the G-EQDSK file.

    This is needed to determine the COCOS convention used in the file.
    As G-EQDSK files do not contain the minor radius directly, we must
    fit contours to the data.
    """
    r = np.linspace(data.rleft, data.rleft + data.rdim, data.nx)
    z = np.linspace(data.zmid - 0.5 * data.zdim, data.zmid + 0.5 * data.zdim, data.ny)
    cont_gen = contour_generator(x=z, y=r, z=data.psi)
    psi_grid = np.linspace(data.simagx, data.sibdry, data.nx)
    rz_axis = np.array([data.rmagx, data.zmagx])
    minor_radii = []
    for psi in psi_grid:
        contours = cont_gen.lines(psi)
        if not contours:
            if len(minor_radii) == 0:
                # We've failed because we're at the magnetic axis
                minor_radii.append(0.0)
                continue
            raise ValueError(f"Could not find contour for {psi=}")
        # Get contour closest to the magnetic axis
        if len(contours) > 1:
            contour = min(
                contours, key=lambda c: np.mean(np.linalg.norm(c - rz_axis, axis=1))
            )
        else:
            contour = contours[0]
        # Get minor radius of the contour
        # The contour will be arranged as [[Z0, R0], [Z1, R1], ..., [ZN, RN], [Z0, R0]]
        rmin, rmax = np.min(contour[:, 1]), np.max(contour[:, 1])
        minor_radii.append(0.5 * (rmax - rmin))
    return np.array(minor_radii)


class WallCoords(NamedTuple):
    R: FloatArray
    Z: FloatArray


class GeqdskReader:
    path: Path
    wall: WallCoords
    eq: Equilibrium
    opoint: NDArray[np.floating]
    xpoints: list[NDArray[np.floating]]

    def __init__(
        self,
        path: Path,
        wall: tuple[FloatArray, FloatArray] | None = None,
        cocos: int | None = None,
        clockwise_phi: bool = False,
    ):
        """Class for extracting magnetic geometries from G-EQDSK files.

        Parameters
        ----------
        path
            Path to a G-EQDSK file
        wall
            Coordinates along the wall, in ``(R, Z)`` format. If ``None``, uses
            boundary data from the G-EQDSK file.
        cocos
            The COCOS convention used in the G-EQDSK file.  If ``None``, the
            COCOS convention will be identified from the contents of the G-EQDSK
            file and the value provided to ``clockwise_phi``.
        clockwise_phi
            Wheter the  direction of increasing toroidal angle is positive when
            the tokamak is viewed from above. Used to infer the COCOS convention
            when ``cocos`` is ``None``, and is otherwise ignored.
        """
        self.path = Path(path)

        with Path(path).open() as fh:
            data = geqdsk.read(fh)

        if cocos != COCOS:
            data = _transform_geqdsk(
                data, cocos_in=cocos, cocos_out=COCOS, clockwise_phi=clockwise_phi
            )

        if wall is None:
            if data.rlim is None or data.zlim is None:
                raise ValueError("G-EQDSK file does not contain wall data.")
            rlim = np.asarray(data.rlim)
            zlim = np.asarray(data.zlim)
            # Ensure that the wall is closed
            if rlim[0] != rlim[-1] or zlim[0] != zlim[-1]:
                rlim = np.append(rlim, rlim[0])
                zlim = np.append(zlim, zlim[0])
            # Add intermediate points to smooth the wall.
            # Also helps in cases where only the corners of a box are given.
            rinter = rlim[:-1] + 0.5 * np.diff(rlim)
            zinter = zlim[:-1] + 0.5 * np.diff(zlim)
            rnew = np.empty(rlim.size + rinter.size, dtype=rlim.dtype)
            znew = np.empty(zlim.size + zinter.size, dtype=zlim.dtype)
            rnew[::2] = rlim
            rnew[1::2] = rinter
            znew[::2] = zlim
            znew[1::2] = zinter
            self.wall = WallCoords(R=rnew, Z=znew)
        else:
            self.wall = WallCoords(R=np.asarray(wall[0]), Z=np.asarray(wall[1]))

        self.eq = Equilibrium(
            tokamak=machine.EmptyTokamak(),
            Rmin=data.rleft,
            Rmax=data.rleft + data.rdim,
            Zmin=data.zmid - 0.5 * data.zdim,
            Zmax=data.zmid + 0.5 * data.zdim,
            nx=data.nx,
            ny=data.ny,
            psi=data.psi,
        )

        # Get profiles, particularly f needed for toroidal field
        psinorm = np.linspace(0.0, 1.0, data.nx)
        f_spl = InterpolatedUnivariateSpline(psinorm, data.fpol)
        self.eq._profiles = jtor.ProfilesPprimeFfprime(
            pprime_func=None,
            ffprime_func=None,
            fvac=data.rcentr * data.bcentr,
            f_func=f_spl,
        )

        # Set tokamak wall
        self.eq.tokamak.wall = machine.Wall(*self.wall)

        # Find x-points and o-points
        opoints, xpoints = critical.find_critical(self.eq.R, self.eq.Z, self.eq.psi())
        # Reject any xpoints outside the walls
        polygon = mpath.Path(np.asarray(self.wall).T)
        self.xpoints = [x for x in xpoints if polygon.contains_point(x)]
        # Retain only the opoint closest to that reported in the G-EQDSK file
        self.opoint = min(
            opoints,
            key=lambda x: (x[0] - data.rmagx) ** 2 + (x[1] - data.zmagx) ** 2,
        )

    def trace_field_line(
        self, leg: str = "ol", solwidth: float = 1.0e-3, npoints=1000
    ) -> MagneticGeometry:
        """Get midpoint-to-target field lines.

        Parameters
        ----------
        leg
            The divertor leg to trace. Either "ol" for outboard-lower, "ou" for
            outboard-upper, "il" for inboard-lower, or "iu" for inboard-upper.
        solwidth
            The radius from the plasma edge to begin, in meters.
        npoints
            Number of points to trace along the field line.
        """
        if leg not in {"ol", "ou", "il", "iu"}:
            err = f"Invalid leg '{leg}', should be 'ol', 'ou', 'il', or 'iu'."
            raise ValueError(err)
        outer = leg[0] == "o"
        upper = leg[1] == "u"
        # Select appropriate x-point.
        if upper:
            xpoint = max(self.xpoints, key=lambda x: x[1])
        else:
            xpoint = min(self.xpoints, key=lambda x: x[1])

        r0, z0, psi0 = self.opoint
        psi_bdry = self.eq.psi_bndry
        psifunc = RectBivariateSpline(
            self.eq.R[:, 0],
            self.eq.Z[0, :],
            (self.eq.psi() - psi0) / (psi_bdry - psi0),
        )
        psix = psifunc(xpoint[0], xpoint[1])[0][0]

        # Find the flux surface for the appropriate leg
        # Need to make sure that r1 is inside the walls
        wall_angle = np.arctan2(self.wall[1] - z0, self.wall[0] - r0)
        wall_idx = np.argmin(wall_angle**2) if outer else np.argmax(wall_angle**2)
        r1 = self.wall[0][wall_idx]
        rmid, zmid = critical.find_psisurface(
            self.eq, psifunc, r0, z0, r1, z0, psival=psix
        )

        # Starting location, just outside the separatrix
        rstart = rmid + solwidth if outer else rmid - solwidth

        # Sweep through 20 turns
        # This is probably overkill, but it's a simple way to ensure we get to
        # the target. N.B. I tried 4 turns and it failed for the inner legs!
        angles = np.linspace(0.0, 20 * np.pi, npoints)

        # Follow field line
        ft = fieldtracer.FieldTracer(self.eq)
        line = ft.follow([rstart], [zmid], angles)
        # If we got it in the wrong direction, try again in the other direction
        if not self._valid_field_line(upper, line[:, 0, 1]):
            line = ft.follow([rstart], [zmid], -angles)
        # If it's still wrong, then this leg couldn't be found
        if not self._valid_field_line(upper, line[:, 0, 1]):
            err = f"Could not calculate field line for leg '{leg}'."
            raise ValueError(err)
        R = line[:, :, 0]
        Z = line[:, :, 1]
        length = line[:, :, 2]

        # Remove repeated points at the end
        max_length = length[-1]
        end_idx = 1 + np.argmin(np.abs(length - max_length))
        R = R[:end_idx]
        Z = Z[:end_idx]
        length = length[:end_idx]

        # Should go from the target to the midplane
        if upper == (Z[0] < Z[-1]):
            R = R[::-1]
            Z = Z[::-1]
            length = length[-1] - length

        # Get other parameters for MagneticGeometry
        dR = np.diff(R, prepend=0.0)
        dZ = np.diff(Z, prepend=0.0)
        Spol = np.cumsum(np.sqrt(dR**2 + dZ**2))
        xpoint_idx = np.argmin((R - xpoint[0]) ** 2 + (Z - xpoint[1]) ** 2)
        Bpol = np.hypot(self.eq.Br(R, Z), self.eq.Bz(R, Z))
        Btor = self.eq.Btor(R, Z)
        Btot = np.hypot(Bpol, Btor)

        return MagneticGeometry(
            R=R,
            Z=Z,
            Spar=length,
            Spol=Spol,
            Bpol=Bpol,
            Btot=Btot,
            Xpoint=int(xpoint_idx),
        )

    @staticmethod
    def _valid_field_line(upper: bool, Z: NDArray[np.floating]) -> bool:
        # Converting to bool to keep type checker happy
        if upper:
            return bool(np.all(np.diff(Z) >= 0.0))
        return bool(np.all(np.diff(Z) <= 0.0))


def read_geqdsk(
    path: Path,
    wall: tuple[FloatArray, FloatArray] | None = None,
    solwidth: float = 1.0e-3,
    npoints: int = 1000,
    cocos: int | None = None,
    clockwise_phi: bool = False,
) -> dict[str, MagneticGeometry | None]:
    """Read a G-EQDSK file and return all field lines.

    Returns a dictionary with keys "ol", "ou", "il", and "iu" for the
    outboard-lower, outboard-upper, inboard-lower, and inboard-upper field
    lines. If a field line could not be calculated, the value will be ``None``.

    Parameters
    ----------
    path
        Path to a G-EQDSK file
    wall
        Coordinates along the wall, in ``(R, Z)`` format. If ``None``, uses
        boundary data from the G-EQDSK file.
    solwidth
        The radius from the plasma edge to begin, in meters.
    npoints
        Number of points to trace along the field line.
    cocos
        The COCOS convention used in the G-EQDSK file.  If ``None``, the
        COCOS convention will be identified from the contents of the G-EQDSK
        file and the value provided to ``clockwise_phi``.
    clockwise_phi
        Wheter the  direction of increasing toroidal angle is positive when
        the tokamak is viewed from above. Used to infer the COCOS convention
        when ``cocos`` is ``None``, and is otherwise ignored.
    """
    results = {}
    reader = GeqdskReader(path, wall=wall, cocos=cocos, clockwise_phi=clockwise_phi)
    for leg in ["ol", "ou", "il", "iu"]:
        try:
            results[leg] = reader.trace_field_line(leg, solwidth, npoints)
        except ValueError as exc:
            err = f"Could not calculate field line for leg '{leg}'."
            if exc.args[0] != err:
                raise exc
            results[leg] = None
    return results
