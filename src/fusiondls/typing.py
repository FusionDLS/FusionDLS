from os import PathLike as os_PathLike
from typing import Union

from numpy import floating, integer, ndarray
from numpy.typing import ArrayLike as ArrayLike  # Re-export

try:
    from numpy.typing import NDArray

    FloatArray = NDArray[floating]
except ImportError:
    FloatArray = ndarray  # type: ignore[misc]

Scalar = Union[float, integer, floating]
PathLike = Union[os_PathLike, str]
