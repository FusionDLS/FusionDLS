import pickle
import shutil
from pathlib import Path

import numpy.testing as npt
import pytest

from fusiondls.cli import run_dls_cli
from fusiondls.postprocessing import FrontLocationScan
from fusiondls.solver import SimulationOutput

TOML_FILE = Path(__file__).parent / "test_files" / "input.toml"
PKL_FILE = Path(__file__).parents[1] / "docs" / "examples" / "eqb_store.pkl"


@pytest.fixture
def cli_test_path(tmp_path: Path) -> Path:
    d = tmp_path / "cli_test"
    d.mkdir()
    # Copy toml file and pickle file to new dir
    shutil.copy(TOML_FILE, d)
    shutil.copy(PKL_FILE, d)
    return d


def test_cli(cli_test_path: Path):
    """Test that the CLI can be run and produces the expected output.

    A variant on a test in `basic_use.ipynb`. Expect slightly different `cvar`
    values as it uses `LfuncKallenbachAr` instead of `LfuncKallenbach("Ar")`.
    """
    input_path = cli_test_path / "input.toml"
    output_path = cli_test_path / "output.pkl"
    run_dls_cli(input_path, output_path)
    assert output_path.exists()

    # Check that some of the output data is as expected
    with output_path.open("rb") as fh:
        data: SimulationOutput = pickle.load(fh)
    scan = FrontLocationScan(data)
    cvar_expected = [
        6.39460377e19,
        8.50407130e19,
        1.04820590e20,
        1.24066606e20,
        1.43924591e20,
    ]
    spol_expected = [0.0, 1.58654208, 2.57608158, 2.97983262, 3.10262434]
    npt.assert_allclose(scan.data["cvar"], cvar_expected, rtol=5e-4)
    npt.assert_allclose(scan.data["Spol"], spol_expected)
