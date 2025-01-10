from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from timeit import default_timer as timer
from typing import Any

import numpy as np
from scipy import interpolate
from scipy.constants import elementary_charge, physical_constants
from scipy.integrate import cumulative_trapezoid, solve_ivp, trapezoid

from .AnalyticCoolingCurves import cooling_curves
from .DLScommonTools import pad_profile
from .geometry import MagneticGeometry
from .typing import FloatArray

deuterium_mass = physical_constants["deuteron mass"][0]


@dataclass
class SimulationState:
    """A collection of all variables and data needed to a simulation.

    The state is passed around different functions, which allows more of the
    algorithm to be abstracted away from the main function.
    """

    nu: float = field(init=False)

    cz: float = field(init=False)

    T: FloatArray = field(init=False)

    q: FloatArray = field(init=False)

    Pu0: float = field(init=False)

    verbosity: int
    """Level of verbosity. Higher is more verbose"""

    s: FloatArray = field(init=False)
    """Working set of parallel coordinates.

    Includes points between front location and X-point."""

    SparFront: FloatArray = field(init=False)

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

    # Return parameter from state
    def get(self, param):
        return self.__dict__[param]


@dataclass
class SimulationInputs:
    """The inputs used to set up a simulation.

    This class functions the same as SimulationState, but is used to store the
    inputs instead. The separation is to make it easier to see which variables
    should be unchangeable.
    """

    nu: float

    gamma_sheath: float
    """Heat transfer coefficient of the virtual target [-]"""

    qpllu0: float
    """Upstream heat flux setting.

    Overriden if control_variable is power [:math:`Wm^{-2}`]"""

    nu0: float
    """Upstream density setting.

    Overriden if control_variable is density [:math:`m^{-3}`]"""

    cz0: float
    """Impurity fraction setting.

    Overriden if control_variable is impurity_frac [-]"""

    Tt: float
    """Desired virtual target temperature [:math:`eV`]"""

    cooling_curve: str
    """Cooling curve function.

    Can be ``"Kallenbachx"`` where ``"x"`` is ``"Ne"``, ``"Ar"`` or ``"N"``.
    """

    SparRange: FloatArray
    """List of :math:`S_parallel` locations to solve for"""

    Xpoint: int
    """Index of X-point in parallel space"""

    S: FloatArray
    """Parallel distance [:math:`m`]"""

    Spol: FloatArray
    """Poloidal distance [:math:`m`]"""

    B: Callable[[FloatArray], float]
    """Interpolator function returning :math:`B_{tot}` for a given :math:`S`"""

    Btot: FloatArray
    """Total B field [:math:`T`]"""

    Bpol: FloatArray
    """Poloidal magnetic field [:math:`T`]"""

    kappa0: float = 2500
    """Electron conductivity"""

    mi: float = deuterium_mass
    """Ion mass [:math:`kg`]"""

    control_variable: str = "impurity_frac"
    """One of 'density', 'impurity_frac' or 'power'"""

    verbosity: int = 0
    """Level of verbosity. Higher is more verbose"""

    Ctol: float = 1e-3
    """Control variable (inner) loop convergence tolerance"""

    Ttol: float = 1e-2
    """Temperature (outer) loop convergence tolerance"""

    URF: float = 1.0
    """Under-relaxation factor to smooth out temperature convergence.

    This usually doesn't help with anything, so it's best to keep it at 1."""

    timeout: int = 20
    """Maximum number of iterations for each loop before warning or error"""

    Lz: list[FloatArray] = field(init=False)
    """Cooling curve data.

    [0] contains temperatures in [:math:`eV`] and [1] the corresponding cooling
    values in [:math:`Wm^{-3}`]"""

    upstreamGrid: bool = True
    """Determine whether to include domain above the X-point.

    If true, includes domain above X-point and source of divertor heat flux
    comes from radial transport upstream, with :math:`T_u` at the midplane.

    If false, heat flux simply enters at the X-point as :math:`q_i`, and
    :math:`T_u` is at the X-point"""

    def __post_init__(self):
        ALLOWED_VARIABLES = ["density", "impurity_frac", "power"]
        if self.control_variable not in ALLOWED_VARIABLES:
            err = (
                "Unexpected value for 'control_variable' "
                f"(got {self.control_variable}, expected one of {ALLOWED_VARIABLES})"
            )
            raise ValueError(err)

        # Initialise cooling curve
        Tcool = np.linspace(0.3, 500, 1000)
        Tcool = np.append(0, Tcool)
        Lalpha = np.array([cooling_curves[self.cooling_curve](dT) for dT in Tcool])
        self.Lz = [Tcool, Lalpha]


@dataclass
class SimulationOutput(Mapping):
    r"""Output from the fusiondls model

    Attributes
    ----------
    Splot: FloatArray
        :math:`S_\parallel` of each front location
    SpolPlot: FloatArray
        :math:`S_{poloidal}` of each front locations
    cvar: FloatArray
        Control variable
    Sprofiles: FloatArray
        :math:`S_\parallel` profiles for each front location
    Tprofiles: FloatArray
        Temperature profiles
    Rprofiles: FloatArray
        Radiation in W/m^3
    Qprofiles: FloatArray
        Heat flux in W/m^2
    Spolprofiles: FloatArray
    Btotprofiles: FloatArray
    Bpolprofiles: FloatArray
    Xpoints: FloatArray
    Wradials: FloatArray
    logs: dict
    spar_onset: int
    spol_onset: int
    splot: FloatArray
    crel: FloatArray
    cvar_trim: FloatArray
    crel_trim: FloatArray
    threshold: float
    window: float
    window_frac: float
    window_ratio: float
    inputs: SimulationInputs
    state: SimulationState

    """

    Splot: FloatArray
    SpolPlot: FloatArray
    cvar: list[FloatArray]
    Sprofiles: list[FloatArray]
    Tprofiles: list[FloatArray]
    Rprofiles: list[FloatArray]
    Qprofiles: list[FloatArray]
    Spolprofiles: list[FloatArray]
    Btotprofiles: list[FloatArray]
    Bpolprofiles: list[FloatArray]
    Xpoints: list[FloatArray]
    Wradials: list[FloatArray]
    logs: dict
    spar_onset: int
    spol_onset: int
    splot: list[FloatArray]
    crel: list[FloatArray]
    cvar_trim: list[FloatArray]
    crel_trim: list[FloatArray]
    threshold: float
    window: float
    window_frac: float
    window_ratio: float
    inputs: SimulationInputs
    state: SimulationState

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, val: Any) -> None:
        setattr(self, name, val)

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self))

    def __len__(self) -> int:
        return len(asdict(self))

    @property
    def cvar_norm(self) -> FloatArray:
        return self.cvar / self.cvar[0]


def run_dls(
    constants: dict,
    geometry: MagneticGeometry,
    SparRange: FloatArray,
    control_variable: str = "impurity_frac",
    Ctol: float = 1e-3,
    Ttol: float = 1e-2,
    URF: float = 1,
    timeout: int = 20,
    grid_refinement_ratio: float = 5,
    grid_refinement_width: float = 1,
    grid_resolution: int | None = 500,
    zero_qpllt: bool = False,
    static_grid: bool = False,
    verbosity: int = 0,
    diagnostic_plot: bool = False,
) -> dict[str, FloatArray]:
    """Run the DLS-extended model.

    Returns the impurity fraction required for a given temperature at
    the target. Can request a low temperature at a given position to
    mimic a detachment front at that position.

    Note: radiation output is very sensitive to grid resolution. Ensure
    you achieve grid convergence.

    Parameters
    ----------
    constants
        dict of options.
    geometry
        Dataclass describing the magnetic geometry profile.
    SparRange
        :math:`S_parallel` locations to solve for.
    control_variable
        Either ``"impurity_frac"``, ``"density"`` or ``"power"``.
    Ctol
        Error tolerance target for the inner loop, i.e. density/impurity/heat
        flux.
    Ttol
        Error tolerance target for the outer loop, i.e. rerrunning until Tu
        convergence.
    URF
      Under-relaxation factor for temperature. If URF is 0.2, :math:`Tu_{new} =
      0.8Tu_{old} + 0.2Tu_{calculated}. Always set to 1.
    timeout
        Controls timeout for all three loops within the code. Each has
        different message on timeout. Default 20.
    grid_refinement_ratio
        Ratio of finest to coarsest cell width.
    grid_refinement_width
        Size of grid refinement region in metres parallel.
    grid_resolution
        Resolution of the refined grid. If set to ``None``, uses the same resolution
        as the original grid.
    zero_qpllt
        Set the initial guess of ``qpllt``, the virtual target temperature , to
        zero.
    static_grid
        Do not perform dynamic grid refinement. ``grid_refinement_ratio``,
        ``grid_refinement_width`` and ``grid_resolution`` will be ignored, as
        will ``diagnostic_plot``.
    verbosity
        Level of verbosity. Higher is more verbose.
    diagnostic_plot
        Plot grid refinement.
    """
    # Start timer
    t0 = timer()

    # Initialise simulation inputs object
    si = SimulationInputs(
        verbosity=verbosity,
        Ctol=Ctol,
        Ttol=Ttol,
        URF=URF,
        timeout=timeout,
        control_variable=control_variable,
        Xpoint=geometry.Xpoint,
        S=geometry.S,
        Spol=geometry.Spol,
        Btot=geometry.Btot,
        Bpol=geometry.Bpol,
        B=interpolate.interp1d(geometry.S, geometry.Btot, kind="cubic"),
        SparRange=SparRange,
        **constants,
    )

    # Initialise simulation state object
    st = SimulationState(verbosity=verbosity)

    # Initialise output dictionary
    output = defaultdict(list)

    print("Solving...", end="")

    """------SOLVE------"""
    # For each detachment front location:
    for idx, SparFront in enumerate(si.SparRange):
        # Current prescribed parallel front location
        st.SparFront = SparFront

        if static_grid:
            point = st.point = int(np.argmin(abs(geometry.S - SparFront)))
        else:
            newProfile = geometry.refine(
                SparFront,
                fine_ratio=grid_refinement_ratio,
                width=grid_refinement_width,
                resolution=grid_resolution,
                diagnostic_plot=diagnostic_plot,
            )
            si.Xpoint = newProfile.Xpoint
            si.S = newProfile.S
            si.Spol = newProfile.Spol
            si.Btot = newProfile.Btot
            si.Bpol = newProfile.Bpol
            # TODO: is this necessary?  We have Btot already
            si.B = interpolate.interp1d(si.S, si.Btot, kind="cubic")

            # Find index of front location on new grid
            SparFrontOld = si.SparRange[idx]
            point = st.point = int(np.argmin(abs(si.S - SparFrontOld)))

        print(f"{SparFront:.2f}...", end="")

        """------INITIAL GUESSES------"""

        # Current set of parallel position coordinates
        st.s = si.S[point:]
        output["Splot"].append(si.S[point])
        output["SpolPlot"].append(si.Spol[point])

        # Inital guess for the value of qpll integrated across connection length
        qavLguess = 0.0
        if si.upstreamGrid:
            if st.s[0] < si.S[si.Xpoint]:
                qavLguess = (
                    (si.qpllu0) * (si.S[si.Xpoint] - st.s[0])
                    + (si.qpllu0 / 2) * (st.s[-1] - si.S[si.Xpoint])
                ) / (st.s[-1] - si.S[0])
            else:
                qavLguess = si.qpllu0 / 2
        else:
            qavLguess = si.qpllu0

        # Inital guess for upstream temperature based on guess of qpll ds integral
        st.Tu = ((7 / 2) * qavLguess * (st.s[-1] - st.s[0]) / si.kappa0) ** (2 / 7)
        # Initial upstream pressure in Pa, calculated so it can be kept constant if required
        st.Pu0 = st.Tu * si.nu0 * elementary_charge

        # Cooling curve integral
        Lint = cumulative_trapezoid(si.Lz[1] * np.sqrt(si.Lz[0]), si.Lz[0], initial=0)
        integralinterp = interpolate.interp1d(si.Lz[0], Lint)

        # Guesses/initialisations for control variables assuming qpll0 everywhere and qpll=0 at target

        if si.control_variable == "impurity_frac":
            # Initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
            cz0_guess = (si.qpllu0**2) / (
                2 * si.kappa0 * si.nu0**2 * st.Tu**2 * integralinterp(st.Tu)
            )
            st.cvar = cz0_guess

        elif si.control_variable == "density":
            # Initial guess of nu0 assuming qpll0 everywhere and qpll=0 at target
            nu0_guess = np.sqrt(
                (si.qpllu0**2)
                / (2 * si.kappa0 * si.cz0 * st.Tu**2 * integralinterp(st.Tu))
            )
            st.cvar = nu0_guess

        elif si.control_variable == "power":
            # nu0 and cz0 guesses are from Lengyel which depends on an estimate of Tu using qpllu0
            # This means we cannot make a more clever guess for qpllu0 based on cz0 or nu0
            qpllu0_guess = si.qpllu0
            # qradial_guess = qpllu0_guess / trapezoid(si.Btot[si.Xpoint:] / si.Btot[si.Xpoint], x = si.S[si.Xpoint:])
            qradial_guess = (qpllu0_guess / si.Btot[si.Xpoint]) / trapezoid(
                1 / si.Btot[si.Xpoint :], x=si.S[si.Xpoint :]
            )
            st.cvar = 1 / qradial_guess

        # Initial guess of qpllt, the virtual target temperature (typically 0).
        if zero_qpllt:
            st.qpllt = si.qpllu0 * 1e-2
        else:
            st.qpllt = (
                si.gamma_sheath
                / 2
                * si.nu0
                * st.Tu
                * elementary_charge
                * np.sqrt(2 * si.Tt * elementary_charge / si.mi)
            )

        """------INITIALISATION------"""
        st.error1 = 1  # Inner loop error (error in qpllu based on provided cz/ne)
        st.error0 = 1  # Outer loop residual in upstream temperature
        # Upstream conditions
        st.nu = si.nu0
        st.cz = si.cz0
        st.qradial = (si.qpllu0 / si.Btot[si.Xpoint]) / trapezoid(
            1 / si.Btot[si.Xpoint :], x=si.S[si.Xpoint :]
        )

        st.update_log()

        # Tu convergence loop
        for k0 in range(si.timeout):
            # Initialise
            st = iterate(si, st)

            """------INITIAL SOLUTION BOUNDING------"""

            # Double or halve cvar until the error flips sign
            for k1 in range(si.timeout * 2):
                if st.error1 > 0:
                    st.cvar /= 2
                elif st.error1 < 0:
                    st.cvar *= 2

                st = iterate(si, st)

                if np.sign(st.log[st.SparFront]["error1"][k1 + 1]) != np.sign(
                    st.log[st.SparFront]["error1"][k1 + 2]
                ):  # It's initialised with a 1 already, hence k1+1 and k1+2
                    break

                if k1 == si.timeout - 1:
                    raise Exception("Initial bounding failed")

            if st.cvar < 1e-6 and si.control_variable == "impurity_fraction":
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

            for k2 in range(si.timeout):
                # New cvar guess is halfway between the upper and lower bound.
                st.cvar = st.lower_bound + (st.upper_bound - st.lower_bound) / 2

                st = iterate(si, st)

                # Narrow bounds based on the results.
                if st.error1 < 0:
                    st.lower_bound = st.cvar
                elif st.error1 > 0:
                    st.upper_bound = st.cvar

                # Looser tolerance for the first two T iterations
                tolerance = 1e-2 if k0 < 2 else si.Ctol

                # Break on success
                if abs(st.error1) < tolerance:
                    break

                if k2 == si.timeout - 1 and verbosity > 0:
                    print("\nWARNING: Failed to converge control variable loop")

            """------OUTER LOOP------"""
            # Upstream temperature error
            st.error0 = (st.Tu - st.Tucalc) / st.Tu

            # Calculate new Tu, under-relax by URF
            st.Tu = (1 - si.URF) * st.Tu + si.URF * st.Tucalc

            st.update_log()

            # Break on outer (temperature) loop success
            if abs(st.error0) < si.Ttol:
                if verbosity > 2:
                    print(f"\n Converged temperature loop in {k0} iterations")
                break
            if k0 == si.timeout:
                output["logs"] = st.log
                print("Failed to converge temperature loop, exiting and returning logs")
                return output

        """------COLLECT PROFILE DATA------"""

        if si.control_variable == "power":
            output["cvar"].append(1 / st.cvar)  # so that output is in Wm-2
        else:
            output["cvar"].append(st.cvar)

        Qrad = []
        Lfunc = cooling_curves[si.cooling_curve]
        for Tf in st.T:
            if si.control_variable == "impurity_frac":
                Qrad.append(((si.nu0**2 * st.Tu**2) / Tf**2) * st.cvar * Lfunc(Tf))
            elif si.control_variable == "density":
                Qrad.append(((st.cvar**2 * st.Tu**2) / Tf**2) * si.cz0 * Lfunc(Tf))
            elif si.control_variable == "power":
                Qrad.append(((si.nu0**2 * st.Tu**2) / Tf**2) * si.cz0 * Lfunc(Tf))

        # Pad some profiles with zeros to ensure same length as S
        output["Sprofiles"].append(si.S)
        output["Tprofiles"].append(pad_profile(si.S, st.T))
        output["Rprofiles"].append(pad_profile(si.S, Qrad))  # Radiation in W/m3
        output["Qprofiles"].append(pad_profile(si.S, st.q))  # Heat flux in W/m2
        output["Spolprofiles"].append(si.Spol)
        output["Btotprofiles"].append(np.array(si.Btot))
        output["Bpolprofiles"].append(np.array(si.Bpol))
        output["Xpoints"].append(si.Xpoint)
        output["Wradials"].append(st.qradial)

    output["logs"] = st.log  # Append log with all front positions

    """------COLLECT RESULTS------"""
    if len(SparRange) > 1:
        # Here we calculate things like window, threshold etc from a whole scan.

        # Relative control variable:
        cvar_list = np.array(output["cvar"])
        crel_list = cvar_list / cvar_list[0]

        # S parallel and poloidal locations of each front location (for plotting against cvar/crel):
        splot = output["Splot"]
        spolplot = output["SpolPlot"]

        # Trim any unstable detachment (negative gradient) region for post-processing reasons
        crel_list_trim = crel_list.copy()
        cvar_list_trim = cvar_list.copy()

        # Find values on either side of C = 1 and interpolate onto 1
        if len(crel_list) > 1:
            for i in range(len(crel_list) - 1):
                if np.sign(crel_list[i] - 1) != np.sign(crel_list[i + 1] - 1) and i > 0:
                    interp_par = interpolate.interp1d(
                        [crel_list[i], crel_list[i + 1]], [splot[i], splot[i + 1]]
                    )
                    interp_pol = interpolate.interp1d(
                        [crel_list[i], crel_list[i + 1]], [spolplot[i], spolplot[i + 1]]
                    )

                    spar_onset = float(interp_par(1))
                    spol_onset = float(interp_pol(1))
                    break
                if i == len(crel_list) - 2:
                    spar_onset = 0
                    spol_onset = 0

            output["spar_onset"] = spar_onset
            output["spol_onset"] = spol_onset

            grad = np.gradient(crel_list)
            for i, _val in enumerate(grad):
                if i > 0 and np.sign(_val) != np.sign(grad[i - 1]):
                    crel_list_trim[:i] = np.nan
                    cvar_list_trim[:i] = np.nan

        # Pack things into the output dictionary.

        output["splot"] = splot
        output["cvar"] = cvar_list
        output["crel"] = crel_list
        output["cvar_trim"] = cvar_list_trim
        output["crel_trim"] = crel_list_trim
        output["threshold"] = cvar_list[0]
        # Ct
        output["window"] = cvar_list[-1] - cvar_list[0]  # Cx - Ct
        output["window_frac"] = output["window"] / output["threshold"]  # (Cx - Ct) / Ct
        output["window_ratio"] = cvar_list[-1] / cvar_list[0]  # Cx / Ct

    elif len(SparRange) == 1:
        output["crel"] = 1
        output["threshold"] = st.cvar

    t1 = timer()

    print(f"Complete in {t1 - t0:.1f} seconds")

    return SimulationOutput(inputs=si, state=st, **output)


def LengFunc(
    s: float, y: tuple[float, float], si: SimulationInputs, st: SimulationState
) -> tuple[float, float]:
    """
    Lengyel function.
    This is passed to ODEINT in integrate() and used to solve for q and T along the field line.

    Inputs
    -------
    y:
        List containing ratio of target q to target total B and target temperature
    s:
        Parallel coordinate of front position
    st:
        Simulation state object containing all evolved parameters
    si:
        Simulation input object containing all constant parameters

    Outputs
    -------
    [dqoverBds,dtds] :
        Heat flux gradient dq/ds and temperature gradient dT/ds
    """

    qoverB, T = y
    fieldValue = si.B(np.clip(s, si.S[0], si.S[-1]))
    Lfunc = cooling_curves[si.cooling_curve]

    # add a constant radial source of heat above the X point, which is qradial = qpll at Xpoint/np.abs(S[-1]-S[Xpoint]
    # i.e. radial heat entering SOL evenly spread between midplane and xpoint needs to be sufficient to get the
    # correct qpll at the xpoint.

    # working on neutral/ionisation model
    # dqoverBds = dqoverBds/fieldValue
    dqoverBds = ((st.nu**2 * st.Tu**2) / T**2) * st.cz * Lfunc(T) / fieldValue

    if si.upstreamGrid and s > si.S[si.Xpoint]:
        # The second term here converts the x point qpar to a radial heat source acting between midplane and the xpoint
        # account for flux expansion to Xpoint
        dqoverBds -= st.qradial / fieldValue

    dtds = qoverB * fieldValue / (si.kappa0 * T ** (5 / 2))

    return [dqoverBds, dtds]


def iterate(si, st):
    """
    Solves the Lengyel function for q and T profiles along field line.
    Calculates error1 by looking at upstream q and comparing it to 0
    (when upstreamGrid=True) or to qpllu0 (when upstreamGrid=False).

    Inputs
    ------
    st : SimulationState
        Simulation state object containing all evolved parameters
    si : SimulationInput
        Simulation input object containing all constant parameters

    State modifications
    -------------------
    st.q : np.array
        Profile of heat flux along field line
    st.T : np.array
        Profile of temperature along field line
    st.Tucalc : float
        Upstream temperature for later use in outer loop to calculate error0
    st.qpllu1 : float
        Upstream heat flux
    st.error1 : float
        Error in upstream heat flux

    """
    if si.control_variable == "impurity_frac":
        st.cz = st.cvar
        st.nu = si.nu0

    elif si.control_variable == "density":
        st.cz = si.cz0
        st.nu = st.cvar

    st.qradial = (si.qpllu0 / si.Btot[si.Xpoint]) / trapezoid(
        1 / si.Btot[si.Xpoint :], x=si.S[si.Xpoint :]
    )

    if si.control_variable == "power":
        st.cz = si.cz0
        st.nu = si.nu0
        # This is needed so that too high a cvar gives positive error
        st.qradial = (1 / st.cvar / si.Btot[si.Xpoint]) / trapezoid(
            1 / si.Btot[si.Xpoint :], x=si.S[si.Xpoint :]
        )

    if si.verbosity > 2:
        print(
            f"qpllu0: {si.qpllu0:.3E} | nu: {st.nu:.3E} | Tu: {st.Tu:.1f} | cz: {st.cz:.3E} | cvar: {st.cvar:.2E}",
            end="",
        )

    result = solve_ivp(
        LengFunc,
        t_span=(st.s[0], st.s[-1]),
        t_eval=st.s,
        y0=[st.qpllt / si.B(st.s[0]), si.Tt],
        rtol=1e-5,
        atol=1e-10,
        method="LSODA",
        args=(si, st),
    )

    # Update state with results
    qoverBresult = result.y[0]
    Tresult = result.y[1]

    # Sometimes when solve_ivp returns negative q upstream, it will trim
    # the output instead of giving nans. This pads it back to correct length
    if len(qoverBresult) < len(st.s):
        if si.verbosity > 3:
            print("Warning: solver output contains NaNs")

        qoverBresult = np.insert(
            qoverBresult, -1, np.zeros(len(st.s) - len(qoverBresult))
        )
        Tresult = np.insert(Tresult, -1, np.zeros(len(st.s) - len(qoverBresult)))

    st.q = qoverBresult * si.B(st.s)  # q profile
    st.T = Tresult  # Temp profile

    st.Tucalc = st.T[-1]  # Upstream temperature. becomes st.Tu in outer loop

    # Set qpllu1 to lowest q value in array.
    # Prevents unphysical results when ODEINT bugs causing negative q in middle but still positive q at end, fooling solver to go in wrong direction
    # Sometimes this also creates a single NaN which breaks np.min(), hence nanmin()
    if len(st.q[st.q < 0]) > 0:
        st.qpllu1 = np.nanmin(st.q)  # minimum q
    else:
        st.qpllu1 = st.q[-1]  # upstream q

    # If upstream grid, qpllu1 is at the midplane and is solved until it's 0. It then gets radial transport
    # so that the xpoint Q is qpllu0. If uypstramGrid=False, qpllu1 is solved to match qpllu0 at the Xpoint.
    if si.upstreamGrid:
        st.error1 = (st.qpllu1 - 0) / si.qpllu0
    else:
        st.error1 = (st.qpllu1 - si.qpllu0) / si.qpllu0

    if si.verbosity > 2:
        print(
            f" -> qpllu1: {st.qpllu1:.3E} | Tucalc: {st.Tucalc:.1f} | error1: {st.error1:.3E}"
        )

    st.update_log()

    if st.Tucalc == 0:
        raise Exception("Tucalc is 0")

    return st
