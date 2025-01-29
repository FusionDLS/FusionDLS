from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fusiondls")
except PackageNotFoundError:
    from setuptools_scm import get_version  # type: ignore[import]

    __version__ = get_version(root="..", relative_to=__file__)

from .analytic_cooling_curves import cooling_curves
from .DLScommonTools import file_read, file_write, make_arrays
from .geometry import MagneticGeometry
from .Profile import Profile
from .settings import SimulationInputs
from .solver import SimulationOutput, run_dls

__all__ = [
    "MagneticGeometry",
    "Profile",
    "SimulationInputs",
    "SimulationOutput",
    "cooling_curves",
    "file_read",
    "file_write",
    "make_arrays",
    "run_dls",
]
