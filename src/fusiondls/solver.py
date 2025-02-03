from collections import defaultdict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Any

import numpy as np
from scipy import interpolate
from scipy.constants import elementary_charge
from scipy.integrate import cumulative_trapezoid, solve_ivp, trapezoid

from .DLScommonTools import pad_profile
from .geometry import MagneticGeometry
from .settings import SimulationInputs
from .typing import FloatArray


@dataclass
class SimulationState(MutableMapping):
    """A collection of all variables and data needed to a simulation.

    The state is passed around different functions, which allows more of the
    algorithm to be abstracted away from the main function.
    """

    nu: float = field(init=False)
    """Current upstream density"""

    cz: float = field(init=False)
    """Current impurity fraction"""

    T: FloatArray = field(init=False)
    """Current temperature profile"""

    q: FloatArray = field(init=False)
    """Current parallel electron heat flux profile"""

    Pu0: float = field(init=False)
    """Initial upstream electron pressure"""

    verbosity: int
    """Level of verbosity. Higher is more verbose"""

    s: FloatArray = field(init=False)
    """Parallel connection length profile"""

    SparFront: float = field(init=False)
    """Current detachment front position  in [m]"""

    cvar: float = field(init=False)
    """Control variable (density, impurity fraction or 1/power)"""

    Tu: float = field(init=False)
    """Upstream temperature"""

    Tucalc: float = field(init=False)
    """New calculation of upstream temperature"""

    lower_bound: float = 0.0
    """Lower estimate of the cvar solution"""

    upper_bound: float = 0.0
    """Upper estimate of the cvar solution"""

    error1: float = 1.0
    """Control variable (inner) loop error based on upstream heat flux
    approaching qpllu0 or 0 depending on settings"""

    error0: float = 1.0
    """Temperature (outer) loop error based on upstream temperature converging
    to steady state"""

    qpllu1: float = 0.0
    """Calculated upstream heat flux"""

    qradial: float = 0.0
    """qpllu1 converted into a source term representing radial heat flux
    between upstream and X-point"""

    qpllt: float = 0.0
    """Virtual target heat flux (typically 0)"""

    point: int = 0
    """Location of front position in index space"""

    log: dict = field(default_factory=dict)
    """Log of guesses of Tu and cvar, errors and bounds. Dictionary keys are
    front positions in index space"""

    def __post_init__(self):
        self.singleLog = {
            "error0": [],
            "error1": [],
            "cvar": [],
            "qpllu1": [],
            "Tu": [],
            "lower_bound": [],
            "upper_bound": [],
        }

    def update_log(self) -> None:
        """Update primary log"""
        for param in self.singleLog:
            self.singleLog[param].append(self.get(param))

        self.log[self.SparFront] = self.singleLog  # Put in global log

        if self.verbosity >= 2:
            log = self.singleLog

            if len(log["error0"]) == 1:  # Print header on first iteration
                print(f"\n\n Solving at parallel location {self.SparFront}")
                print("--------------------------------")

            print(
                f"error0: {log['error0'][-1]:.3E}"
                f"Tu: {log['Tu'][-1]:.3E}"
                f"error1: {log['error1'][-1]:.3E}"
                f"cvar: {log['cvar'][-1]:.3E}"
                f"lower_bound: {log['lower_bound'][-1]:.3E}"
                f"upper_bound: {log['upper_bound'][-1]:.3E}"
            )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("Deletion of items is not allowed")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


@dataclass
class SimulationOutput(MutableMapping):
    r"""Output from the fusiondls model.

    Most of the output are lists where each element corresponds
    to a front location.

    Attributes
    ----------
    Spar_front: FloatArray
        :math:`S_\parallel` of each front location
    Spol_front: FloatArray
        :math:`S_{poloidal}` of each front locations
    cvar: FloatArray
        Control variable
    Spar_profiles: FloatArray
        :math:`S_\parallel` profiles for each front location
    Te_profiles: FloatArray
        Temperature profiles
    Qrad_profiles: FloatArray
        Profiles of radiation in W/m^3
    qpar_profiles: FloatArray
        Heat flux in W/m^2
    Spol_profiles: FloatArray
        Profiles of poloidal position
    Btot_profiles: FloatArray
        Profiles of Btot
    Bpol_profiles: FloatArray
        Profiles of Bpol
    Xpoints: FloatArray
        Xpoint locations
    Wradials: FloatArray
        Upstream power sources
    logs: dict
        Performance logs. This is a dict where the key is the Spar location
    inputs: SimulationInputs
        Simulation inputs
    geometry: MagneticGeometry
        Profile geometry
    state: SimulationState
        State object of the final simulation for debug purposes
    runtime: float
        Number of seconds taken to run all front positions
    """

    Spar_front: FloatArray
    Spol_front: FloatArray
    cvar: list[FloatArray]
    Spar_profiles: list[FloatArray]
    Te_profiles: list[FloatArray]
    Qrad_profiles: list[FloatArray]
    qpar_profiles: list[FloatArray]
    Spol_profiles: list[FloatArray]
    Btot_profiles: list[FloatArray]
    Bpol_profiles: list[FloatArray]
    Xpoints: list[FloatArray]
    Wradials: list[FloatArray]
    logs: dict
    inputs: SimulationInputs
    geometry: MagneticGeometry
    state: SimulationState
    runtime: float

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, val: Any) -> None:
        setattr(self, name, val)

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("Deletion of items is not allowed")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    @property
    def cvar_norm(self) -> FloatArray:
        return self.cvar / self.cvar[0]


def run_dls(
    inputs: SimulationInputs,
    geometry: MagneticGeometry,
    verbosity: int = 0,
    diagnostic_plot: bool = False,
) -> SimulationOutput:
    """Run the DLS-extended model.

    Returns the impurity fraction required for a given temperature at
    the target. Can request a low temperature at a given position to
    mimic a detachment front at that position.

    Note: radiation output is very sensitive to grid resolution. Ensure
    you achieve grid convergence.

    Parameters
    ----------
    inputs
        General settings for the simulation.
    geometry
        The magnetic geometry profile.
    verbosity
        Level of verbosity. Higher is more verbose.
    diagnostic_plot
        Plot grid refinement.
    """
    # Start timer
    t0 = timer()

    # Initialise simulation state object
    st = SimulationState(verbosity=verbosity)

    # Get reference to starting geometry
    start_geometry = geometry

    # Initialise output dictionary
    output = defaultdict(list)

    if verbosity > 0:
        print("Solving...", end="")

    """------SOLVE------"""
    # For each detachment front location:
    for idx, SparFront in enumerate(inputs.SparRange):
        # Current prescribed parallel front location
        st.SparFront = SparFront

        if inputs.static_grid:
            geometry = start_geometry
            point = int(np.argmin(abs(geometry.Spar - SparFront)))
        else:
            geometry = start_geometry.refine(
                SparFront,
                fine_ratio=inputs.grid_refinement_ratio,
                width=inputs.grid_refinement_width,
                resolution=inputs.grid_resolution,
                diagnostic_plot=diagnostic_plot,
            )

            # Find index of front location on new grid
            SparFrontOld = inputs.SparRange[idx]
            point = int(np.argmin(abs(geometry.Spar - SparFrontOld)))
        st.point = point

        if verbosity > 0:
            print(f"{SparFront:.2f}...", end="")

        """------INITIAL GUESSES------"""

        # Current set of parallel position coordinates
        st.s = geometry.Spar[point:]
        output["Spar_front"].append(geometry.Spar[point])
        output["Spol_front"].append(geometry.Spol[point])

        # Inital guess for the value of qpll integrated across connection length
        qavLguess = 0.0
        if inputs.upstreamGrid:
            if st.s[0] < geometry.Spar[geometry.Xpoint]:
                qavLguess = (
                    (inputs.qpllu0) * (geometry.Spar[geometry.Xpoint] - st.s[0])
                    + (inputs.qpllu0 / 2) * (st.s[-1] - geometry.Spar[geometry.Xpoint])
                ) / (st.s[-1] - geometry.Spar[0])
            else:
                qavLguess = inputs.qpllu0 / 2
        else:
            qavLguess = inputs.qpllu0

        # Inital guess for upstream temperature based on guess of qpll ds integral
        st.Tu = ((7 / 2) * qavLguess * (st.s[-1] - st.s[0]) / inputs.kappa0) ** (2 / 7)
        # Initial upstream pressure in Pa, calculated so it can be kept constant if required
        st.Pu0 = st.Tu * inputs.nu0 * elementary_charge

        # Cooling curve integral
        Lint = cumulative_trapezoid(
            inputs.Lz[1] * np.sqrt(inputs.Lz[0]), inputs.Lz[0], initial=0
        )
        integralinterp = interpolate.interp1d(inputs.Lz[0], Lint)

        # Guesses/initialisations for control variables assuming qpll0 everywhere and qpll=0 at target

        if inputs.control_variable == "impurity_frac":
            # Initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
            cz0_guess = (inputs.qpllu0**2) / (
                2 * inputs.kappa0 * inputs.nu0**2 * st.Tu**2 * integralinterp(st.Tu)
            )
            st.cvar = cz0_guess

        elif inputs.control_variable == "density":
            # Initial guess of nu0 assuming qpll0 everywhere and qpll=0 at target
            nu0_guess = np.sqrt(
                (inputs.qpllu0**2)
                / (2 * inputs.kappa0 * inputs.cz0 * st.Tu**2 * integralinterp(st.Tu))
            )
            st.cvar = nu0_guess

        elif inputs.control_variable == "power":
            # nu0 and cz0 guesses are from Lengyel which depends on an estimate of Tu using qpllu0
            # This means we cannot make a more clever guess for qpllu0 based on cz0 or nu0
            qpllu0_guess = inputs.qpllu0
            # qradial_guess = qpllu0_guess / trapezoid(si.Btot[si.Xpoint:] / si.Btot[si.Xpoint], x = si.S[si.Xpoint:])
            qradial_guess = (qpllu0_guess / geometry.Btot[geometry.Xpoint]) / trapezoid(
                1 / geometry.Btot[geometry.Xpoint :], x=geometry.Spar[geometry.Xpoint :]
            )
            st.cvar = 1 / qradial_guess

        # Assumption of target heat flux
        if inputs.front_sheath:
            st.qpllt = (
                inputs.gamma_sheath
                / 2
                * inputs.nu0
                * st.Tu
                * elementary_charge
                * np.sqrt(2 * inputs.Tt * elementary_charge / inputs.mi)
            )
        else:
            st.qpllt = inputs.qpllu0 * inputs.qpllt_fraction

        """------INITIALISATION------"""
        st.error1 = 1  # Inner loop error (error in qpllu based on provided cz/ne)
        st.error0 = 1  # Outer loop residual in upstream temperature
        # Upstream conditions
        st.nu = inputs.nu0
        st.cz = inputs.cz0
        st.qradial = (inputs.qpllu0 / geometry.Btot[geometry.Xpoint]) / trapezoid(
            1 / geometry.Btot[geometry.Xpoint :], x=geometry.Spar[geometry.Xpoint :]
        )

        st.update_log()

        # Tu convergence loop
        for k0 in range(inputs.timeout):
            # Initialise
            st = iterate(inputs, geometry, st, verbosity=verbosity)

            """------INITIAL SOLUTION BOUNDING------"""

            # Double or halve cvar until the error flips sign
            for k1 in range(inputs.timeout * 2):
                if st.error1 > 0:
                    st.cvar /= 2
                elif st.error1 < 0:
                    st.cvar *= 2

                st = iterate(inputs, geometry, st, verbosity=verbosity)

                if np.sign(st.log[st.SparFront]["error1"][k1 + 1]) != np.sign(
                    st.log[st.SparFront]["error1"][k1 + 2]
                ):  # It's initialised with a 1 already, hence k1+1 and k1+2
                    break

                if k1 == inputs.timeout - 1:
                    raise Exception("Initial bounding failed")

            if st.cvar < 1e-6 and inputs.control_variable == "impurity_fraction":
                raise Exception("Required impurity fraction is tending to zero")

            # We have bounded the problem -  the last two iterations
            # are on either side of the solution
            st.lower_bound = min(
                st.log[st.SparFront]["cvar"][-1], st.log[st.SparFront]["cvar"][-2]
            )
            st.upper_bound = max(
                st.log[st.SparFront]["cvar"][-1], st.log[st.SparFront]["cvar"][-2]
            )

            """------INNER LOOP------"""

            for k2 in range(inputs.timeout):
                # New cvar guess is halfway between the upper and lower bound.
                st.cvar = st.lower_bound + (st.upper_bound - st.lower_bound) / 2

                st = iterate(inputs, geometry, st, verbosity=verbosity)

                # Narrow bounds based on the results.
                if st.error1 < 0:
                    st.lower_bound = st.cvar
                elif st.error1 > 0:
                    st.upper_bound = st.cvar

                # Looser tolerance for the first two T iterations
                tolerance = 1e-2 if k0 < 2 else inputs.Ctol

                # Break on success
                if abs(st.error1) < tolerance:
                    break

                if k2 == inputs.timeout - 1 and verbosity > 0:
                    print("\nWARNING: Failed to converge control variable loop")

            """------OUTER LOOP------"""
            # Upstream temperature error
            st.error0 = (st.Tu - st.Tucalc) / st.Tu

            # Calculate new Tu, under-relax by URF
            st.Tu = (1 - inputs.URF) * st.Tu + inputs.URF * st.Tucalc

            st.update_log()

            # Break on outer (temperature) loop success
            if abs(st.error0) < inputs.Ttol:
                if verbosity > 2:
                    print(f"\n Converged temperature loop in {k0} iterations")
                break
            if k0 == inputs.timeout:
                output["logs"] = st.log
                print("Failed to converge temperature loop, exiting and returning logs")
                return output

        """------COLLECT PROFILE DATA------"""

        if inputs.control_variable == "power":
            output["cvar"].append(1 / st.cvar)  # so that output is in Wm-2
        else:
            output["cvar"].append(st.cvar)

        Qrad = []
        Lfunc = inputs.cooling_curve
        for Tf in st.T:
            if inputs.control_variable == "impurity_frac":
                Qrad.append(((inputs.nu0**2 * st.Tu**2) / Tf**2) * st.cvar * Lfunc(Tf))
            elif inputs.control_variable == "density":
                Qrad.append(((st.cvar**2 * st.Tu**2) / Tf**2) * inputs.cz0 * Lfunc(Tf))
            elif inputs.control_variable == "power":
                Qrad.append(
                    ((inputs.nu0**2 * st.Tu**2) / Tf**2) * inputs.cz0 * Lfunc(Tf)
                )

        # Pad some profiles with zeros to ensure same length as S
        output["Spar_profiles"].append(geometry.Spar)
        output["Te_profiles"].append(pad_profile(geometry.Spar, st.T))
        output["Qrad_profiles"].append(  # Radiation in W/m3
            pad_profile(geometry.Spar, Qrad)
        )
        output["qpar_profiles"].append(  # Heat flux in W/m2
            pad_profile(geometry.Spar, st.q)
        )
        output["Spol_profiles"].append(geometry.Spol)
        output["Btot_profiles"].append(np.array(geometry.Btot))
        output["Bpol_profiles"].append(np.array(geometry.Bpol))
        output["Xpoints"].append(geometry.Xpoint)
        output["Wradials"].append(st.qradial)

    output["logs"] = st.log  # Append log with all front positions

    t1 = timer()
    runtime = t1 - t0
    if verbosity > 0:
        print(f"Complete in {runtime:.1f} seconds")
    output["runtime"] = runtime

    # return output
    return SimulationOutput(inputs=inputs, geometry=geometry, state=st, **output)


def LengFunc(
    s: float,
    y: tuple[float, float],
    inputs: SimulationInputs,
    geometry: MagneticGeometry,
    st: SimulationState,
) -> tuple[float, float]:
    """Lengyel function.

    This is passed to ODEINT in ``integrate()`` and used to solve for ``q`` and
    ``T`` along the field line.

    Returns tuple of heat flux gradient :math:`dq/ds` and temperature gradient
    :math:`dT/ds`.

    Parameters
    ----------
    s
        Parallel coordinate of front position
    y
        List containing ratio of target q to target total B and target temperature
    inputs
        Simulation input object containing all constant parameters
    geometry
        Magnetic profile
    st
        Simulation state object containing all evolved parameters
    """

    qoverB, T = y
    fieldValue = geometry.B(np.clip(s, geometry.Spar[0], geometry.Spar[-1]))
    Lfunc = inputs.cooling_curve

    # add a constant radial source of heat above the X point, which is
    # qradial = qpll at Xpoint/np.abs(S[-1]-S[Xpoint]) i.e. radial heat
    # entering SOL evenly spread between midplane and xpoint needs to be
    # sufficient to get the correct qpll at the xpoint.

    # working on neutral/ionisation model
    # dqoverBds = dqoverBds/fieldValue
    dqoverBds = ((st.nu**2 * st.Tu**2) / T**2) * st.cz * Lfunc(T) / fieldValue

    if inputs.upstreamGrid and s > geometry.Spar[geometry.Xpoint]:
        # The second term here converts the x point qpar to a radial heat
        # source acting between midplane and the xpoint account for flux
        # expansion to Xpoint
        dqoverBds -= st.qradial / fieldValue

    dtds = qoverB * fieldValue / (inputs.kappa0 * T ** (5 / 2))

    return dqoverBds, dtds


def iterate(
    inputs: SimulationInputs,
    geometry: MagneticGeometry,
    st: SimulationState,
    verbosity: int = 0,
):
    """
    Solves the Lengyel function for q and T profiles along field line.
    Calculates error1 by looking at upstream q and comparing it to 0
    (when upstreamGrid=True) or to qpllu0 (when upstreamGrid=False).

    State modifications:
    - st.q : np.array, profile of heat flux along field line
    - st.T : np.array, profile of temperature along field line
    - st.Tucalc : float, upstream temperature for later use in outer loop
      to calculate error0
    - st.qpllu1 : float, upstream heat flux
    - st.error1 : float, error in upstream heat flux


    Parameters
    ----------
    inputs
        Simulation input object containing all constant parameters
    geometry
        Magnetic profile.
    st
        Simulation state object containing all evolved parameters
    verbosity
        Level of verbosity. Higher is more verbose.
    """
    if inputs.control_variable == "impurity_frac":
        st.cz = st.cvar
        st.nu = inputs.nu0
    elif inputs.control_variable == "density":
        st.cz = inputs.cz0
        st.nu = st.cvar

    st.qradial = (inputs.qpllu0 / geometry.Btot[geometry.Xpoint]) / trapezoid(
        1 / geometry.Btot[geometry.Xpoint :], x=geometry.Spar[geometry.Xpoint :]
    )

    if inputs.control_variable == "power":
        st.cz = inputs.cz0
        st.nu = inputs.nu0
        # This is needed so that too high a cvar gives positive error
        st.qradial = (1 / st.cvar / geometry.Btot[geometry.Xpoint]) / trapezoid(
            1 / geometry.Btot[geometry.Xpoint :], x=geometry.Spar[geometry.Xpoint :]
        )

    if verbosity > 2:
        print(
            f"qpllu0: {inputs.qpllu0:.3E} | nu: {st.nu:.3E} | Tu: {st.Tu:.1f} | cz: {st.cz:.3E} | cvar: {st.cvar:.2E}",
            end="",
        )

    qoverBresult, Tresult = solve_ivp(
        LengFunc,
        t_span=(st.s[0], st.s[-1]),
        t_eval=st.s,
        y0=[st.qpllt / geometry.B(st.s[0]), inputs.Tt],
        rtol=inputs.rtol,
        atol=inputs.atol,
        method=inputs.solver,
        args=(inputs, geometry, st),
    ).y

    # Sometimes when solve_ivp returns negative q upstream, it will trim
    # the output instead of giving nans. This pads it back to correct length
    if len(qoverBresult) < len(st.s):
        if verbosity > 3:
            print("Warning: solver output contains NaNs")

        qoverBresult = np.insert(
            qoverBresult, -1, np.zeros(len(st.s) - len(qoverBresult))
        )
        Tresult = np.insert(Tresult, -1, np.zeros(len(st.s) - len(qoverBresult)))

    st.q = qoverBresult * geometry.B(st.s)  # q profile
    st.T = Tresult  # Temp profile

    st.Tucalc = st.T[-1]  # Upstream temperature. becomes st.Tu in outer loop

    # Set qpllu1 to lowest q value in array.
    # Prevents unphysical results when ODEINT bugs causing negative q in middle
    # but still positive q at end, fooling solver to go in wrong direction
    # Sometimes this also creates a single NaN which breaks np.min(), hence
    # nanmin()
    if len(st.q[st.q < 0]) > 0:
        st.qpllu1 = np.nanmin(st.q)  # minimum q
    else:
        st.qpllu1 = st.q[-1]  # upstream q

    # If upstream grid, qpllu1 is at the midplane and is solved until it's 0.
    # It then gets radial transport so that the xpoint Q is qpllu0. If
    # uypstramGrid=False, qpllu1 is solved to match qpllu0 at the Xpoint.
    if inputs.upstreamGrid:
        st.error1 = (st.qpllu1 - 0) / inputs.qpllu0
    else:
        st.error1 = (st.qpllu1 - inputs.qpllu0) / inputs.qpllu0

    if verbosity > 2:
        print(
            f" -> qpllu1: {st.qpllu1:.3E} | Tucalc: {st.Tucalc:.1f} | error1: {st.error1:.3E}"
        )

    st.update_log()

    if st.Tucalc == 0:
        raise Exception("Tucalc is 0")

    return st
