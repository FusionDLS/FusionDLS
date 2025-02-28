from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.constants import physical_constants

from .analytic_cooling_curves import cooling_curves
from .typing import FloatArray

deuterium_mass = physical_constants["deuteron mass"][0]

__all__ = ["SimulationInputs"]


class CoolingCurve:
    """Descriptor object, allows setting via function or string.

    Adapted from the `Python docs <https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields>`_
    """

    def __set_name__(self, _, name):
        self._name = "_" + name

    def __get__(self, obj, _) -> Callable[[float], float]:
        # Bug: This should raise AttributeError when self._name is not defined,
        # as cooling_curve should not have a default value. However, this causes
        # a bug with IPython autoreload. As a result, creating a
        # SimulationInputs with too few arguments will not warn of missing
        # positional arguments.
        if not hasattr(obj, self._name):
            return None
        return getattr(obj, self._name)

    def __set__(self, obj, value: str | Callable[[float], float]):
        if isinstance(value, str):
            try:
                value = cooling_curves[value]
            except KeyError as e:
                msg = f"Unknown cooling curve '{value}'"
                raise ValueError(msg) from e
        if not callable(value):
            raise ValueError("Cooling curve must be a callable or a string")
        setattr(obj, self._name, value)


@dataclass
class SimulationInputs(Mapping):
    """The inputs used to set up a simulation.

    This class functions the same as SimulationState, but is used to store the
    inputs instead. The separation is to make it easier to see which variables
    should be unchangeable.
    """

    control_variable: str
    """One of 'density', 'impurity_frac' or 'power'"""

    SparRange: FloatArray
    """List of :math:`S_parallel` locations to solve for"""

    qpllu0: float
    """Upstream heat flux setting.

    Overriden if control_variable is power [:math:`Wm^{-2}`]"""

    nu0: float
    """Upstream density setting.

    Overriden if control_variable is density [:math:`m^{-3}`]"""

    cz0: float
    """Impurity fraction setting.

    Overriden if control_variable is impurity_frac [-]"""

    cooling_curve: CoolingCurve = CoolingCurve()
    """Cooling curve function.

    May be to a built-in cooling curve via a string such as ``"KallenbachX"``,
    where ``"X"`` is ``"Ne"``, ``"Ar"`` or ``"N"``. See ``cooling_curves`` for
    all examples.

    Alternatively, a custom cooling curve may be set by supplying a
    ``Callable`` that takes a single ``float`` argument and returns a
    ``float``. A cooling curve should be a function of temperature in
    [:math:`eV`].

    The results are very sensitive to cooling curve choice, so care should be
    taken to set this correctly.
    """

    gamma_sheath: float = 7
    """Heat transfer coefficient of the virtual target [-]"""

    Tt: float = 0.5
    """Desired virtual target temperature. Aim for <1eV [:math:`eV`]"""

    kappa0: float = 2500
    """Electron conductivity"""

    mi: float = deuterium_mass
    """Ion mass [:math:`kg`]"""

    Ctol: float = 1e-3
    """Control variable (inner) loop convergence tolerance"""

    Ttol: float = 1e-3
    """Temperature (outer) loop convergence tolerance"""

    atol: float = 1e-10
    """Solver absolute tolerance"""

    rtol: float = 1e-5
    """Solver relative tolerance"""

    solver: str = "RK23"

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

    grid_refinement_ratio: float = 5
    """Ratio of finest to coarsest cell width."""

    grid_refinement_width: float = 1
    """Size of grid refinement region in metres parallel."""

    grid_resolution: int | None = 500
    """Resolution of the refined grid.

    If set to ``None``, uses the same resolution as the original grid.
    """

    static_grid: bool = False
    """Do not perform dynamic grid refinement.

    ``grid_refinement_ratio``, ``grid_refinement_width`` and
    ``grid_resolution`` will be ignored, as will ``diagnostic_plot``.
    """

    front_sheath: bool = False
    """Enables a sheath gamma style model for heat flux through the front."""

    qpllt_fraction: float = 0.05
    """Fraction of the upstream heat flux at the target.

    Applies if ``front_sheath`` is ``False``.
    """

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"Unknown key: {key}")

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"Unknown key: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

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
        Lalpha = np.fromiter((self.cooling_curve(dT) for dT in Tcool), float)
        self.Lz = [Tcool, Lalpha]
