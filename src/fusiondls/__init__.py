from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fusiondls")
except PackageNotFoundError:
    from setuptools_scm import get_version  # type: ignore[import]

    __version__ = get_version(root="..", relative_to=__file__)

from .DLScommonTools import LfuncN, file_read, file_write, make_arrays
from .LRBv21 import LRBv21

__all__ = ["file_read", "LfuncN", "LRBv21", "file_write", "make_arrays"]
