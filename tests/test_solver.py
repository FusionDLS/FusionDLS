import pytest

from fusiondls import SimulationInputs


def test_invalid_control_variable():
    with pytest.raises(ValueError, match="bad_var"):
        SimulationInputs(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, control_variable="bad_var"
        )
