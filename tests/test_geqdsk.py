from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from freeqdsk import geqdsk

from fusiondls import read_geqdsk

TEST_DIR = Path(__file__).parent / "geqdsk_test_files"


def test_single_null():
    path = TEST_DIR / "test_single-null.eqdsk"
    with path.open() as fh:
        data = geqdsk.read(fh)
    geometries = read_geqdsk(path)
    # Should have only lower divertors
    assert geometries["iu"] is None
    assert geometries["ou"] is None
    assert geometries["il"] is not None
    assert geometries["ol"] is not None
    # Field lines should begin at the midpoint (accurate to within a mm)
    npt.assert_allclose(geometries["il"].Z[0], data.zmagx, atol=1e-3)
    npt.assert_allclose(geometries["ol"].Z[0], data.zmagx, atol=1e-3)
    # Field lines should end at the wall (accurate to within a cm)
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["il"].Z[-1])), 0.0, atol=1e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["ol"].Z[-1])), 0.0, atol=1e-2
    )
    # Field lines should only travel downwards
    npt.assert_array_less(np.diff(geometries["il"].Z), 0.0)
    npt.assert_array_less(np.diff(geometries["ol"].Z), 0.0)
    # Expect x point somewhere between the magnetic axis and the wall
    assert 0 < geometries["il"].Xpoint < len(geometries["il"].Spar)
    assert 0 < geometries["ol"].Xpoint < len(geometries["ol"].Spar)
    # Expect end point to be to the right/left of the xpoint for ol/il
    assert geometries["il"].R[-1] < geometries["il"].R[geometries["il"].Xpoint]
    assert geometries["ol"].R[-1] > geometries["ol"].R[geometries["ol"].Xpoint]


@pytest.mark.parametrize("null_type", ["connected", "disconnected"])
def test_double_null(null_type: str):
    path = TEST_DIR / f"test_{null_type}-double-null.eqdsk"
    with path.open() as fh:
        data = geqdsk.read(fh)
    geometries = read_geqdsk(path)
    # All divertors should be present
    assert geometries["iu"] is not None
    assert geometries["ou"] is not None
    assert geometries["il"] is not None
    assert geometries["ol"] is not None
    # Field lines should begin at the midpoint (accurate to within a cm)
    npt.assert_allclose(geometries["il"].Z[0], data.zmagx, atol=1e-2)
    npt.assert_allclose(geometries["ol"].Z[0], data.zmagx, atol=1e-2)
    npt.assert_allclose(geometries["iu"].Z[0], data.zmagx, atol=1e-2)
    npt.assert_allclose(geometries["ou"].Z[0], data.zmagx, atol=1e-2)
    # Field lines should end at the wall (accurate to within 1.5 cm)
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["il"].Z[-1])), 0.0, atol=1.5e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["ol"].Z[-1])), 0.0, atol=1.5e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["iu"].Z[-1])), 0.0, atol=1.5e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["ou"].Z[-1])), 0.0, atol=1.5e-2
    )
    # Field lines should only travel downwards for il/ol, upwards for iu/ou
    npt.assert_array_less(np.diff(geometries["il"].Z), 0.0)
    npt.assert_array_less(np.diff(geometries["ol"].Z), 0.0)
    npt.assert_array_less(0.0, np.diff(geometries["iu"].Z))
    npt.assert_array_less(0.0, np.diff(geometries["ou"].Z))
    # Expect x point somewhere between the magnetic axis and the wall
    assert 0 < geometries["il"].Xpoint < len(geometries["il"].Spar)
    assert 0 < geometries["ol"].Xpoint < len(geometries["ol"].Spar)
    assert 0 < geometries["iu"].Xpoint < len(geometries["iu"].Spar)
    assert 0 < geometries["ou"].Xpoint < len(geometries["ou"].Spar)
    # Expect end point to be to the right/left of the xpoint for outer/inner
    assert geometries["il"].R[-1] < geometries["il"].R[geometries["il"].Xpoint]
    assert geometries["iu"].R[-1] < geometries["iu"].R[geometries["iu"].Xpoint]
    assert geometries["ol"].R[-1] > geometries["ol"].R[geometries["ol"].Xpoint]
    assert geometries["ou"].R[-1] > geometries["ol"].R[geometries["ou"].Xpoint]
