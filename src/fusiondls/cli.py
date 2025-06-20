r"""
Defines Command Line Interface (CLI) for fusiondls.

Input files should follow the following format:

```toml
# input.toml
[fusiondls]

# One of 'density', 'impurity_frac' or 'power'
control_variable = [str]

# Either:
# - List of S_parallel locations to solve for
# - Table specifying the range in terms of MagneticGeometry (see below)
# If the latter, the table should contain `mode = [str]`, where mode is
# one of "equally_spaced_poloidal", "equally_spaced_parallel",
# "target_and_xpoint", or "target". If one of the equally spaced modes, the
# table should also contain `npoints = [int]`. For example:
# SparRange = {mode = "equally_spaced_poloidal", npoints = 10}
SparRange = [List[float] | Table]

# Upstream heat flux setting.
qpllu0 = [float]

# Upstream density setting.
nu0 = [float]

# Impurity fraction setting.
cz0 = [float]

# Should be set to a built-in cooling curve via a string e.g. "KallenbachX",
# where "X" is "Ne", "Ar" or "N". See `cooling_curves.py` for all examples.
cooling_curve = [str]

# Heat transfer coefficient of the virtual target.
# Optional, default is 7.
gamma_sheath = [float]

# Desired virtual target temperature in eV. Aim for <1eV
# Optional, default is 0.5.
Tt = [float]

# Electron conductivity
# Optional, default is 2500.
kappa0 = [float]

# Ion mass in kg
# Optional, defaults to the mass of deuterium.
mi = [float]

# Control variable (inner) loop convergence tolerance.
# Optional, default is 1e-3.
Ctol = [float]

# Temperature (outer) loop convergence tolerance.
# Optional, default is 1e-3.
Ttol = [float]

# Solver absolute tolerance
# Optional, default is 1e-10.
atol = [float]

# Solver relative tolerance
# Optional, default is 1e-5.
rtol = [float]

# Solver to use
# Optional, default is "RK23"
solver = [str]

# Under-relaxation factor to smooth out temperature convergence.
# Optional, default is 1.0
URF = [float]

# Maximum number of iterations for each loop before warning or error.
# Optional, default is 20.
timeout = [int]

# Determines whether to include domain above the X-point.
# Optional, default is True.
upstreamGrid = [bool]

# Ratio of finest to coarsest cell width.
# Optional, default is 5.
grid_refinement_ratio = [float]

# Size of grid refinement region in metres parallel.
# Optional, default is 1.
grid_refinement_width = [float]

# Resolution of the refined grid. Should be set to an integer or "None".
# Optional, default is 500.
grid_resolution = [int | str]

# Do not perform dynamic grid refinement.
# Optional, default is False.
static_grid = [bool]

# Enable a sheath gamma style model for heat flux through the front.
# Optional, default is False.
front_sheath = [bool]

# Fraction of the upstream heat flux at the target.
# Optional, default is 0.05.
qpllt_fraction = [float]

# Options for setting the magnetic geometry.
# Set one of [fusiondls.MagneticGeometry.geqdsk] or
# [fusiondls.MagneticGeometry.pickle]

[fusiondls.MagneticGeometry.geqdsk]
# Path to the G-EQDSK file. Preferably, this should be relative to the directory
# containing the input file. It otherwise may be an absolute path. Currently,
# the provided G-EQDSK file must contain the walls within the file itself.
# Consider using the Python interface directly if you need to read in the walls
# from an external source.
path = [str]

# One of "ol", "ou", "il", "iu"
# Optional, default is "ol"
leg = [str]

# The radius of from the plasma edge on the midplane to trace, in meters.
# Optional, default is 0.001
solwidth = [float]

# The number of points to trace along the magnetic field line.
# Optional, default is 1000
npoints = [int]

[fusiondls.MagneticGeometry.pickle]
path = [str]
design = [str]
side = [str]
"""

import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import tomli as toml

from .geometry import MagneticGeometry
from .settings import SimulationInputs
from .solver import run_dls


def read_toml(path: Path) -> tuple[SimulationInputs, MagneticGeometry]:
    """Reads a TOML file and returns a SimulationInputs object."""
    path = Path(path).absolute()
    parent_dir = path.parent
    with path.open("rb") as fh:
        data = toml.load(fh)

    if "fusiondls" not in data:
        raise ValueError("No fusiondls section found in input file")

    # Extract the magnetic geometry
    if "MagneticGeometry" not in data["fusiondls"]:
        raise ValueError("No MagneticGeometry section found in input file")
    geometry_data = data["fusiondls"]["MagneticGeometry"]
    if "geqdsk" not in geometry_data and "pickle" not in geometry_data:
        raise ValueError("No geqdsk or pickle section found in MagneticGeometry")
    if "geqdsk" in geometry_data and "pickle" in geometry_data:
        raise ValueError("Both geqdsk and pickle sections found in MagneticGeometry")
    if "geqdsk" in geometry_data:
        geqdsk_data = geometry_data["geqdsk"]
        geqdsk_path = geqdsk_data.pop("path")
        if not Path(geqdsk_path).is_absolute():
            geqdsk_path = parent_dir / geqdsk_path
        geometry = MagneticGeometry.from_geqdsk(geqdsk_path, **geqdsk_data)
    else:
        pickle_data = geometry_data["pickle"]
        pickle_path = pickle_data.pop("path")
        if not Path(pickle_path).is_absolute():
            pickle_path = parent_dir / pickle_path
        geometry = MagneticGeometry.from_pickle(pickle_path, **pickle_data)

    # Extract the simulation inputs
    # Pick out any keys that require special handling, pass through the rest
    spar_range_data = data["fusiondls"]["SparRange"]
    try:
        # Assume spar_range_data is a table containing mode and npoints
        spar_range = geometry.spar_range(**spar_range_data)
    except TypeError:
        # If the above fails, assume it's a list of floats
        spar_range = np.asarray(spar_range_data, dtype=float)

    grid_resolution = data["fusiondls"].get("grid_resolution", 500)
    if grid_resolution == "None":
        grid_resolution = None

    simple_data = {
        key: value
        for key, value in data["fusiondls"].items()
        if key not in ["SparRange", "grid_resolution", "MagneticGeometry"]
    }

    inputs = SimulationInputs(
        SparRange=spar_range,
        grid_resolution=grid_resolution,
        **simple_data,
    )

    return inputs, geometry


def parse_args() -> Namespace:
    """Defines the command line interface for fusiondls."""

    # TODO This is very simplistic for now! It could be expanded to permit
    # command line overrides of the values in the input file.

    parser = ArgumentParser(description="Run a DLS simulation")
    parser.add_argument("input", type=Path, help="Path to the input file")
    parser.add_argument("output", type=Path, help="Path to the output file")
    return parser.parse_args()


def run_dls_cli(toml_path: Path, output_path: Path):
    inputs, geometry = read_toml(toml_path)
    results = run_dls(inputs, geometry)
    # TODO Better output format than pickle!
    with output_path.open("wb") as fh:
        pickle.dump(results, fh)


def main():
    args = parse_args()
    run_dls_cli(args.input, args.output)
