from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fusiondls")
except PackageNotFoundError:
    from setuptools_scm import get_version  # type: ignore[import]

    __version__ = get_version(root="..", relative_to=__file__)

from .AnalyticCoolingCurves import LfuncN
from .DLScommonTools import file_read, file_write, make_arrays
from .geometry import MagneticGeometry
from .solver import run_dls

__all__ = [
    "LfuncN",
    "file_read",
    "file_write",
    "make_arrays",
    "run_dls",
    "MagneticGeometry",
]
