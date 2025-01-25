import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp


class FrontLocationScan:
    """A collection of DLS solutions at different front locations.

    Contains the whole detachment front movement profile and calculates
    useful statistics: detachment window and unstable region size.
    """

    def __init__(self, store):
        num_locations = len(store["Sprofiles"])
        self.cases = []
        for i in range(num_locations):
            self.cases.append(FrontLocation(store, index=i))

        self.data = pd.DataFrame()
        self.data["Spar"] = store["Splot"]
        self.data["Spol"] = store["SpolPlot"]
        self.data["cvar"] = store["cvar"]  # cvar at detachment threshold
        self.data["crel"] = store["cvar"] / store["cvar"][0]

        self.single_case = "window" not in store

        if self.single_case:
            print(
                "Warning, deck contains only one case! Detachment window and unstable region not available."
            )
            self.window = 0
            self.window_frac = 0
            self.window_ratio = 0
        else:
            self.window = store["window"]  # Cx - Ct
            self.window_frac = store["window_frac"]  # (Cx - Ct) / Ct
            self.window_ratio = store["window_ratio"]  # Cx / Ct

        if len(self.data) != len(self.data.drop_duplicates(subset="Spar")):
            print("Warning: Duplicate Spar values found, removing!")
            self.data = self.data.drop_duplicates(subset="Spar")

        if not self.single_case:
            self.get_stable_region()

    def get_stable_region(self, diagnostic_plot=False):
        """Calculate the size of the unstable region (when under flux compression)

        The unstable region is where there is no detachment front solution and can
        happen on the inner. The size is different depending on whether the front
        is moving forward or backward (Cowley 2022). Here it's calculated in both
        directions in the poloidal and parallel.

        Parameters
        ----------
        diagnostic_plot : bool
            If True, plot the crel vs Spar with stable region highlighted.
        """

        self.data.loc[:, "crel_grad"] = np.gradient(self.data["crel"])
        self.data.loc[:, "stable"] = False
        self.data.loc[self.data["crel_grad"] > 0, "stable"] = True
        data_stable = self.data[self.data["stable"]]
        data_unstable = self.data[~self.data["stable"]]

        ## Size of unstable region when going backward
        if len(data_stable) == 0:
            self.unstable_Lpol_backward = self.data.iloc[-1]["Spol"]
            self.unstable_Lpar_backward = self.data.iloc[-1]["Spar"]
        elif len(data_unstable) > 0:
            self.unstable_Lpol_backward = data_stable.iloc[0]["Spol"]
            self.unstable_Lpar_backward = data_stable.iloc[0]["Spar"]
        else:
            self.unstable_Lpol_backward = 0
            self.unstable_Lpar_backward = 0

        if self.data["crel"].iloc[-1] < 1:
            self.unstable_Lpol_forward = self.data["Spol"].iloc[-1]
            self.unstable_Lpar_forward = self.data["Spar"].iloc[-1]
        else:
            self.unstable_Lpol_forward = sp.interpolate.interp1d(
                data_stable["crel"], data_stable["Spol"], kind="linear"
            )(1)
            self.unstable_Lpar_forward = sp.interpolate.interp1d(
                data_stable["crel"], data_stable["Spar"], kind="linear"
            )(1)

        if diagnostic_plot:
            fig, ax = plt.subplots(dpi=120)
            ax.plot(self.data["crel"], self.data["Spol"], label="Crel")
            ax.plot(data_stable["crel"], data_stable["Spol"])
            ax.hlines(
                self.unstable_Lpol_backward,
                self.data["crel"].min(),
                self.data["crel"].max(),
                ls="--",
                color="red",
                label="forward stability breakpoint",
            )
            ax.hlines(
                self.unstable_Lpol_forward,
                self.data["crel"].min(),
                self.data["crel"].max(),
                ls="--",
                color="orange",
                label="backward stability breakpoint",
            )
            ax.legend()

    def plot_front_movement(
        self, ax=None, label="", parallel=False, relative=True, **kwargs
    ):
        """Plot the front movement profile.

        Parameters
        ----------
        ax : matplotlib axis
            If provided, plot on this axis.
        label : str
            Label for the plot (optional).
        parallel : bool
            If True, plot the parallel front movement, otherwise poloidal
        relative : bool
            If True, plot the relative control parameter (crel). Otherwise cvar.
        kwargs : dict
            Additional plot settings passed to ax.plot().
        """

        if ax is None:
            fig, ax = plt.subplots()
        data = self.data

        if parallel:
            y = data["Spar"]
            ylabel = r"$S_{\parallel} [m]$"
        else:
            y = data["Spol"]
            ylabel = "$S_{pol} [m]$"

        if relative:
            x = data["cvar"] / data["cvar"].iloc[0]
            xlabel = "$C_{rel}$"
        else:
            x = data["cvar"]
            xlabel = "C"

        default_plot_settings = {"marker": "o", "lw": 2, "ms": 4}
        plot_settings = {**default_plot_settings, **kwargs}

        ax.plot(x, y, label=label, **plot_settings)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


class FrontLocation:
    """A single DLS front position solution.

    Contains a dataframe with all of the underlying 1D profiles in DLScase.data.
    Also calculates a number of scalar statistics in DLScase.stats.

    Parameters
    ----------
    SimulationOutputs : dict
        The output object from a DLS simulation. Can contain an arbitrary number of
        front locations.
    index : int
        Which front location to extract.
    """

    def __init__(self, SimulationOutputs, index=0):
        out = SimulationOutputs
        inputs = out["inputs"]

        dls = pd.DataFrame()
        dls["Qrad"] = out["Rprofiles"][index]
        dls["Spar"] = out["Sprofiles"][index]
        dls["Spol"] = out["Spolprofiles"][index]
        dls["Te"] = out["Tprofiles"][index]
        dls["qpar"] = out["Qprofiles"][index]
        dls["Btot"] = out["Btotprofiles"][index]
        dls["Ne"] = (
            out["cvar"][index] * dls["Te"].iloc[-1] / dls["Te"]
        )  ## Assuming cvar is ne
        dls["cz"] = inputs.cz0
        Xpoint = out["Xpoints"][index]
        dls.loc[Xpoint, "Xpoint"] = 1

        # qradial is the uniform upstream heat source
        dls["qradial"] = 1.0
        # dls["qradial"].iloc[Xpoint:] = out["state"].qradial
        dls.loc[Xpoint:, "qradial"] = out["state"].qradial

        # Radiative power loss without flux expansion effect.
        # Units are W, bit integrated assuming unity cross-sectional area, so really W/m2
        # Done by reconstructing the RHS of the qpar equation
        dls["Prad_per_area"] = (
            np.gradient(dls["qpar"] / dls["Btot"], dls["Spar"])
            + dls["qradial"] / dls["Btot"]
        )
        dls["Prad_per_area_cum"] = sp.integrate.cumulative_trapezoid(
            y=dls["Prad_per_area"], x=dls["Spar"], initial=0
        )  # W/m2
        dls["Prad_per_area_cum_norm"] = (
            dls["Prad_per_area_cum"] / dls["Prad_per_area_cum"].max()
        )
        # Proper radiative power integral [W]
        dls["Prad_cum"] = sp.integrate.cumulative_trapezoid(
            y=dls["Qrad"] / dls["Btot"], x=dls["Spar"], initial=0
        )  # Radiation integral over volume
        dls["Prad_cum_norm"] = dls["Prad_cum"] / dls["Prad_cum"].max()

        dls["Pe"] = dls["Te"] * dls["Ne"] * 1.60217662e-19
        dls["qpar_over_B"] = dls["qpar"] / dls["Btot"]

        ### Calculate scalar properties
        s = {}
        s["cvar"] = out["state"].cvar
        s["kappa0"] = inputs.kappa0  # Electron conductivity
        s["Bf"] = dls["Btot"].iloc[0]
        s["Bx"] = dls[dls["Xpoint"] == 1]["Btot"].iloc[0]
        s["Beff"] = (
            np.sqrt(  # Radiation weighted average B field (term alpha in Kryjak 2025)
                sp.integrate.trapezoid(y=dls["qpar"] * dls["Qrad"], x=dls["Spar"])
                / sp.integrate.trapezoid(
                    y=dls["qpar"] * dls["Qrad"] / dls["Btot"] ** 2, x=dls["Spar"]
                )
            )
        )
        s["BxBt"] = s["Bx"] / s["Bf"]  # Total flux expansion
        s["BxBteff"] = s["Bx"] / s["Beff"]
        s["Lc"] = dls["Spol"].iloc[-1]  # Total connection length
        s["Wradial"] = out["state"].qradial  # Radial heat source [W/m3]
        s["Tu"] = dls["Te"].iloc[-1]  # Upstream temperature [eV]

        dlsx = dls[dls["Xpoint"] == 1]  # Profile quantities at the X-point
        dls_div = dls[
            dls["Spar"] <= dlsx["Spar"].iloc[0]
        ]  # Profile quantities below X-point
        avgB_div = (
            sp.integrate.trapz(dls_div["Btot"], x=dls_div["Spar"])
            / dls_div["Spar"].iloc[-1]
        )

        s["avgB_ratio"] = dlsx["Btot"].iloc[0] / avgB_div  # avgB term from Cowley 2022

        # print(s["avgB_ratio"])

        ## DLS-Extended effects (see Kryjak 2025)
        # Impact of qpar profile changing upstream due to B field and radiation,
        # leading to a different qpar at the X-point
        s["upstream_rad"] = np.sqrt(  # Term delta from Kryjak 2025
            2
            * sp.integrate.trapz(
                y=dls["qpar"].iloc[Xpoint:]
                / (dls["Btot"].iloc[Xpoint:] ** 2 * s["Wradial"]),
                x=dls["Spar"].iloc[Xpoint:],
            )
        )

        # Tu proportional term calculated from heat flux integral. Includes effects of Lpar and B/averageB.
        # Simple version is just the Tu proportionality.
        s["W_Tu"] = (s["Wradial"] ** (2 / 7)) / (  # Term beta from Kryjak 2025
            sp.integrate.trapezoid(y=dls["qpar"], x=dls["Spar"]) ** (2 / 7)
        )
        s["W_Tu_simple"] = (
            sp.integrate.trapezoid(
                y=dls["Btot"][Xpoint:]
                / s["Bx"]
                * (s["Lc"] - dls["Spar"].iloc[Xpoint:])
                / (s["Lc"] - dls["Spar"].iloc[Xpoint]),
                x=dls["Spar"].iloc[Xpoint:],
            )
            + sp.integrate.trapezoid(
                y=dls["Btot"].iloc[:Xpoint] / s["Bx"], x=dls["Spar"].iloc[:Xpoint]
            )
        ) ** (-2 / 7)

        # Cooling curve integral which includes effect of Tu clipping integral limit
        self.Lfunc = lambda x: inputs.cooling_curve(x)
        Lz = [self.Lfunc(x) for x in dls["Te"]]
        s["curveclip"] = (  # Term gamma from Kryjak 2025
            np.sqrt(
                2
                * sp.integrate.trapz(y=s["kappa0"] * dls["Te"] ** 0.5 * Lz, x=dls["Te"])
            )
            ** -1
        )

        self.data = dls
        self.stats = s
