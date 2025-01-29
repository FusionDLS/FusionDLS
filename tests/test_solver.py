import pytest

from fusiondls import SimulationInputs
from fusiondls.analytic_cooling_curves import LfuncKallenbachNe


def test_invalid_control_variable():
    with pytest.raises(ValueError, match="bad_var"):
        SimulationInputs("bad_var", [0], 0, 0, 0)


def test_cooling_curve():
    """Set cooling curve via either function or string"""
    by_string = SimulationInputs(
        "impurity_frac", [0], 0, 0, 0, cooling_curve="KallenbachNe"
    )
    by_func = SimulationInputs(
        "impurity_frac", [0], 0, 0, 0, cooling_curve=LfuncKallenbachNe
    )
    assert by_string.cooling_curve is LfuncKallenbachNe
    assert by_func.cooling_curve is LfuncKallenbachNe


def test_invalid_cooling_curve():
    """Test that invalid cooling curves are caught with useful errors"""
    with pytest.raises(ValueError, match="Unknown cooling curve 'hello'"):
        SimulationInputs("impurity_frac", [0], 0, 0, 0, cooling_curve="hello")
