from os import PathLike as os_PathLike
from typing import Union

from numpy import floating, integer
from numpy.typing import ArrayLike  # noqa

try:
    from numpy.typing import NDArray

    FloatArray = NDArray[floating]
except ImportError:
    FloatArray = np.ndarray  # type: ignore

Scalar = Union[float, integer, floating]
PathLike = Union[os_PathLike, str]
