import pathlib

import numpy as np

from fusiondls import MagneticGeometry, SimulationInputs, run_dls
from fusiondls.Analytic_DLS import CfInt


def test_analytic():
    filename = (
        pathlib.Path(__file__).parent.parent / "docs/examples/eqb_store_lores.pkl"
    )
    geometry = MagneticGeometry.from_pickle(filename, "V10", "ou")

    s_parallel = np.linspace(0, geometry.S[geometry.Xpoint - 1], 30)

    inputs = SimulationInputs(
        SparRange = s_parallel,
        nu = 1e20,
        gamma_sheath = 7,
        qpllu0 = 4e8,
        nu0 = 1e20,
        cz0 = 0.02,
        Tt = 1,
        cooling_curve = "N",
        control_variable="density",
    )

    result = run_dls(inputs, geometry)

    analytic = [
        CfInt(geometry.S, geometry.Btot, geometry.Sx, np.max(geometry.S), s)
        for s in s_parallel
    ]

    analytical_norm = analytic / analytic[0]

    error = result.cvar_norm - analytical_norm
    l2_error = np.sqrt(np.mean(error**2))

    # RMS error should only be a few percent
    assert l2_error < 0.05
