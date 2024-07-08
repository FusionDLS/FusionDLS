from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from timeit import default_timer as timer

import numpy as np
from scipy import interpolate
from scipy.constants import elementary_charge, physical_constants
from scipy.integrate import cumulative_trapezoid, trapezoid

from .DLScommonTools import pad_profile
from .Iterate import iterate
from .refineGrid import refineGrid
from .typing import FloatArray


deuterium_mass = physical_constants["deuteron mass"][0]


class SimulationState:
    """
    This class represents the simulation state and contains all the variables and data
    needed to run the simulation. The state is passed around different functions, which
    allows more of the algorithm to be abstracted away from the main function.

    Parameters
    ----------
    si : SimulationInputs
        Simulation inputs object containing all constant parameters
    log : dict
        Log of guesses of Tu and cvar, errors and bounds. Dictionary keys are front positions in index space
    point : int
        Location of front position in index space
    s : array
        Working set of parallel coordinates (between front location and X-point)
    cvar : float
        Control variable (density, impurity fraction or 1/power)
    lower_bound : float
        Lower estimate of the cvar solution
    upper_bound : float
        Upper estimate of the cvar solution
    Tu : float
        Upstream temperature
    Tucalc : float
        New calculation of upstream temperature
    error1 : float
        Control variable (inner) loop error based on upstream heat flux approaching qpllu0 or 0 depending on settings
    error0 : float
        Temperature (outer) loop error based on upstream temperature converging to steady state
    qpllu1 : float
        Calculated upstream heat flux
    qradial : float
        qpllu1 converted into a source term representing radial heat flux between upstream and X-point
    qpllt : float
        Virtual target heat flux (typically 0)
    """

    def __init__(self, si):
        self.si = si  # Add input object to state

        # Initialise variables
        self.error1 = 1
        self.error0 = 1
        self.qpllu1 = 0
        self.lower_bound = 0
        self.upper_bound = 0

        # Initialise log: one per front position index
        self.singleLog = {}

        logParams = [
            "error0",
            "error1",
            "cvar",
            "qpllu1",
            "Tu",
            "lower_bound",
            "upper_bound",
        ]
        for x in logParams:
            self.singleLog[x] = []

        self.log = {}  # Log for all front positions

    ## Update primary log
    def update_log(self):
        for param in [
            "error0",
            "error1",
            "cvar",
            "qpllu1",
            "Tu",
            "lower_bound",
            "upper_bound",
        ]:
            self.singleLog[param].append(self.get(param))

        self.log[self.SparFront] = self.singleLog  # Put in global log

        if self.si.verbosity >= 2:
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

    # Update many variables
    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    # Will return this if called as string
    def __repr__(self):
        return str(self.__dict__)

    # Return parameter from state
    def get(self, param):
        return self.__dict__[param]


@dataclass
class SimulationInputs:
    """
    This class functions the same as SimulationState, but is used to store the inputs instead.
    The separation is to make it easier to see which variables should be unchangeable.
    """

    nu: float
    nu0: float
    cz0: float
    gamma_sheath: float
    """Heat transfer coefficient of the virtual target [-]"""
    qpllu0: float
    """Upstream heat flux setting, overriden if control_variable is power [Wm^-2]"""
    nu0: float
    """Upstream density setting, overriden if control_variable is density [m^-3]"""
    cz0: float
    """Impurity fraction setting, overriden if control_variable is impurity_frac [-]"""
    Tt: float
    """Desired virtual target temperature [eV]"""
    Lfunc: Callable[[float], float]
    """Cooling curve function, can be LfuncKallenbachx where x is Ne, Ar or N."""
    Lz: list
    """Cooling curve data: [0] contains temperatures in [eV] and [1] the corresponding cooling values in [W/m^3]"""
    radios: dict
    """Contains flags for ionisation (WIP do not use), upstreamGrid (allows full flux tube)"""
    SparRange: list
    """List of S parallel locations to solve for"""
    Xpoint: int
    """Index of X-point in parallel space"""
    S: FloatArray
    """Parallel distance [m]"""
    Spol: FloatArray
    """Poloidal distance [m]"""
    B: Callable[[float], float]
    """Interpolator function returning a Btot for a given S"""
    Btot: FloatArray
    """Total B field [T]"""
    Bpol: FloatArray
    kappa0: float = 2500
    """Electron conductivity"""
    mi: float = deuterium_mass
    """Ion mass [kg]"""
    control_variable: str = "impurity_frac"
    """density, impurity_frac or power"""
    verbosity: int = 0
    """Level of verbosity. Higher is more verbose"""
    Ctol: float = 1e-3
    """Control variable (inner) loop convergence tolerance"""
    Ttol: float = 1e-2
    """Temperature (outer) loop convergence tolerance"""
    URF: float = 1.0
    """Under-relaxation factor to smooth out temperature convergence (usually doesn't help with anything, keep at 1)"""
    timeout: int = 20
    """Maximum number of iterations for each loop before warning or error"""
    Lz: FloatArray = field(init=False)
    """Array of temperatures and corresponding cooling"""

    def __post_init__(self):
        # Initialise cooling curve
        Tcool = np.linspace(0.3, 500, 1000)
        Tcool = np.append(0, Tcool)
        Lalpha = np.array([self.Lfunc(dT) for dT in Tcool])
        self.Lz = [Tcool, Lalpha]


def run_dls(
    constants: dict,
    radios: dict,
    d: dict,
    SparRange: FloatArray,
    control_variable: str = "impurity_frac",
    verbosity: int = 0,
    Ctol: float = 1e-3,
    Ttol: float = 1e-2,
    URF: float = 1,
    timeout: int = 20,
    dynamicGrid: bool = False,
    dynamicGridRefinementRatio: float = 5,
    dynamicGridRefinementWidth: float = 1,
    dynamicGridDiagnosticPlot: bool = False,
    zero_qpllt: bool = False,
) -> dict[str, FloatArray]:
    """Run the DLS-extended model

    Returns the impurity fraction required for a given temperature at
    the target. Can request a low temperature at a given position to
    mimic a detachment front at that position.

    Parameters
    ----------
    constants:
        dict of options
    radios:
        dict of options
    control_variable:
        either impurity_frac, density or power
    Ctol:
        error tolerance target for the inner loop (i.e. density/impurity/heat flux)
    Ttol:
        error tolerance target for the outer loop (i.e. rerrunning until Tu convergence)
    URF:
        under-relaxation factor for temperature. If URF is 0.2, Tu_new = Tu_old*0.8 + Tu_calculated*0.2. Always set to 1.
    Timeout:
        controls timeout for all three loops within the code. Each has different message on timeout. Default 20
    dynamicGrid:
        enables iterative grid refinement around the front (recommended)
    dynamicGridRefinementRatio:
        ratio of finest to coarsest cell width in dynamic grid
    dynamicGridRefinementWidth:
        size of dynamic grid refinement region in metres parallel

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
        radios=radios,
        control_variable=control_variable,
        Xpoint=d["Xpoint"],
        S=d["S"],
        Spol=d["Spol"],
        Btot=d["Btot"],
        Bpol=d["Bpol"],
        B=interpolate.interp1d(d["S"], d["Btot"], kind="cubic"),
        SparRange=SparRange,
        **constants,
    )

    # Initialise simulation state object
    st = SimulationState(si)

    # Initialise output dictionary
    output = defaultdict(list)

    print("Solving...", end="")

    """------SOLVE------"""
    # For each detachment front location:
    for idx, SparFront in enumerate(si.SparRange):
        # Current prescribed parallel front location
        st.SparFront = SparFront

        if dynamicGrid:
            newProfile = refineGrid(
                d,
                SparFront,
                fine_ratio=dynamicGridRefinementRatio,
                width=dynamicGridRefinementWidth,
                diagnostic_plot=dynamicGridDiagnosticPlot,
            )
            si.Xpoint = newProfile["Xpoint"]
            si.S = newProfile["S"]
            si.Spol = newProfile["Spol"]
            si.Btot = newProfile["Btot"]
            si.Bpol = newProfile["Bpol"]
            si.B = interpolate.interp1d(
                si.S, si.Btot, kind="cubic"
            )  # TODO: is this necessary?  We have Btot already

            # Find index of front location on new grid
            SparFrontOld = si.SparRange[idx]
            point = st.point = np.argmin(abs(si.S - SparFrontOld))

        else:
            point = st.point = np.argmin(abs(d["S"] - SparFront))

        print(f"{SparFront:.2f}...", end="")

        """------INITIAL GUESSES------"""

        # Current set of parallel position coordinates
        st.s = si.S[point:]
        output["Splot"].append(si.S[point])
        output["SpolPlot"].append(si.Spol[point])

        # Inital guess for the value of qpll integrated across connection length
        qavLguess = 0
        if si.radios["upstreamGrid"]:
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
        Tu0 = ((7 / 2) * qavLguess * (st.s[-1] - st.s[0]) / si.kappa0) ** (2 / 7)
        st.Tu = Tu0
        # Initial upstream pressure in Pa, calculated so it can be kept constant if required
        st.Pu0 = Tu0 * si.nu0 * elementary_charge

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
        for Tf in st.T:
            if si.control_variable == "impurity_frac":
                Qrad.append(
                    ((si.nu0**2 * st.Tu**2) / Tf**2) * st.cvar * si.Lfunc(Tf)
                )
            elif si.control_variable == "density":
                Qrad.append(
                    ((st.cvar**2 * st.Tu**2) / Tf**2) * si.cz0 * si.Lfunc(Tf)
                )
            elif si.control_variable == "power":
                Qrad.append(
                    ((si.nu0**2 * st.Tu**2) / Tf**2) * si.cz0 * si.Lfunc(Tf)
                )

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
    output["constants"] = constants
    output["radios"] = si.radios
    output["state"] = st

    # Convert back to regular dict
    output = dict(output)
    t1 = timer()

    print(f"Complete in {t1 - t0:.1f} seconds")

    return output
