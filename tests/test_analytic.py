import pathlib

import numpy as np
from fusiondls import LfuncN, run_dls, file_read
from fusiondls.Analytic_DLS import CfInt


def test_analytic():
    filename = (
        pathlib.Path(__file__).parent.parent / "docs/examples/eqb_store_lores.pkl"
    )
    eqb = file_read(filename)
    d = eqb["V10"]["ou"]

    radios = {"ionisation": False, "upstreamGrid": True}
    constants = {
        "gamma_sheath": 7,
        "Tt": 1,
        "qpllu0": 4e8,
        "nu": 1e20,
        "nu0": 1e20,
        "cz0": 0.02,
        "Lfunc": LfuncN
    }

    s_parallel = np.linspace(0, d["S"][d["Xpoint"] - 1], 30)

    result = run_dls(constants, radios, d, s_parallel, control_variable="density")
    density_norm = result["cvar"] / result["cvar"][0]

    analytic = [
        CfInt(d["S"], d["Btot"], d["Sx"], np.max(d["S"]), s) for s in s_parallel
    ]

    analytical_norm = analytic / analytic[0]

    error = density_norm - analytical_norm
    l2_error = np.sqrt(np.mean(error**2))

    # RMS error should only be a few percent
    assert l2_error < 0.05
