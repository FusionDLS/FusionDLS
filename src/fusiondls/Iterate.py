import numpy as np
from scipy.integrate import solve_ivp


def LengFunc(s, y, si, st):
    """
    Lengyel function.
    This is passed to ODEINT in integrate() and used to solve for q and T along the field line.

    Inputs
    -------
    y : list
        List containing ratio of target q to target total B and target temperature
    s : float
        Parallel coordinate of front position
    st : SimulationState
        Simulation state object containing all evolved parameters
    si : SimulationInput
        Simulation input object containing all constant parameters

    Outputs
    -------
    [dqoverBds,dtds] : list
        Heat flux gradient dq/ds and temperature gradient dT/ds
    """

    nu, Tu, cz, qradial = st.nu, st.Tu, st.cz, st.qradial
    kappa0, _qpllu0, alpha, radios, S, B, Xpoint, Lfunc = (
        si.kappa0,
        si.qpllu0,
        si.alpha,
        si.radios,
        si.S,
        si.B,
        si.Xpoint,
        si.Lfunc,
    )

    qoverB, T = y

    fieldValue = 0
    if s > S[-1]:
        fieldValue = B(S[-1])
    elif s < S[0]:
        fieldValue = B(S[0])
    else:
        fieldValue = B(s)

    # add a constant radial source of heat above the X point, which is qradial = qpll at Xpoint/np.abs(S[-1]-S[Xpoint]
    # i.e. radial heat entering SOL evenly spread between midplane and xpoint needs to be sufficient to get the
    # correct qpll at the xpoint.

    # working on neutral/ionisation model
    # dqoverBds = dqoverBds/fieldValue
    dqoverBds = ((nu**2 * Tu**2) / T**2) * cz * Lfunc(T) / fieldValue

    if radios["upstreamGrid"] and s > S[Xpoint]:
        # The second term here converts the x point qpar to a radial heat source acting between midplane and the xpoint
        # account for flux expansion to Xpoint
        dqoverBds -= qradial / fieldValue

    # Flux limiter
    dtds = 0
    if radios["fluxlim"]:
        dtds = (
            qoverB
            * fieldValue
            / (
                kappa0 * T ** (5 / 2)
                - qoverB
                * fieldValue
                * kappa0
                * T ** (1 / 2)
                / (alpha * ne * np.sqrt(9e-31))
            )
        )
    else:
        dtds = qoverB * fieldValue / (kappa0 * T ** (5 / 2))
    # return gradient of q and T
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

    st.qradial = (si.qpllu0 / si.Btot[si.Xpoint]) / np.trapezoid(
        1 / si.Btot[si.Xpoint :], x=si.S[si.Xpoint :]
    )

    if si.control_variable == "power":
        st.cz = si.cz0
        st.nu = si.nu0
        # This is needed so that too high a cvar gives positive error
        st.qradial = (1 / st.cvar / si.Btot[si.Xpoint]) / np.trapezoid(
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
    if si.radios["upstreamGrid"]:
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
