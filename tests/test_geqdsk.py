from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from freeqdsk import geqdsk

from fusiondls import MagneticGeometry, read_geqdsk
from fusiondls.geqdsk import _transform_geqdsk

TEST_DIR = Path(__file__).parent / "test_files"


def test_single_null():
    path = TEST_DIR / "test_single-null.eqdsk"
    with path.open() as fh:
        data = geqdsk.read(fh)
    # COCOS auto-identification doesn't seem to work, so we manually set it to 11.
    geometries = read_geqdsk(path, cocos=11)
    # Should have only lower divertors
    assert geometries["iu"] is None
    assert geometries["ou"] is None
    assert geometries["il"] is not None
    assert geometries["ol"] is not None
    # Field lines should start at the wall (accurate to within a cm)
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["il"].Z[0])), 0.0, atol=1e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["ol"].Z[0])), 0.0, atol=1e-2
    )
    # Field lines should end at the midpoint (accurate to within a mm)
    npt.assert_allclose(geometries["il"].Z[-1], data.zmagx, atol=1e-3)
    npt.assert_allclose(geometries["ol"].Z[-1], data.zmagx, atol=1e-3)
    # Field lines should only travel upwards
    npt.assert_array_less(0.0, np.diff(geometries["il"].Z))
    npt.assert_array_less(0.0, np.diff(geometries["ol"].Z))
    # Expect x point somewhere between the midplane and the wall
    assert 0 < geometries["il"].Xpoint < len(geometries["il"].Spar)
    assert 0 < geometries["ol"].Xpoint < len(geometries["ol"].Spar)
    # Expect target to be to the right/left of the xpoint for ol/il
    assert geometries["il"].R[0] < geometries["il"].R[geometries["il"].Xpoint]
    assert geometries["ol"].R[0] > geometries["ol"].R[geometries["ol"].Xpoint]


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
    # Field lines should start at the wall (accurate to within 1.5 cm)
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["il"].Z[0])), 0.0, atol=1.5e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["ol"].Z[0])), 0.0, atol=1.5e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["iu"].Z[0])), 0.0, atol=1.5e-2
    )
    npt.assert_allclose(
        np.min(np.abs(data.zlim - geometries["ou"].Z[0])), 0.0, atol=1.5e-2
    )
    # Field lines should end at the midpoint (accurate to within a cm)
    npt.assert_allclose(geometries["il"].Z[-1], data.zmagx, atol=1e-2)
    npt.assert_allclose(geometries["ol"].Z[-1], data.zmagx, atol=1e-2)
    npt.assert_allclose(geometries["iu"].Z[-1], data.zmagx, atol=1e-2)
    npt.assert_allclose(geometries["ou"].Z[-1], data.zmagx, atol=1e-2)
    # Field lines should only travel upwards for il/ol, downwards for iu/ou
    npt.assert_array_less(0.0, np.diff(geometries["il"].Z))
    npt.assert_array_less(0.0, np.diff(geometries["ol"].Z))
    npt.assert_array_less(np.diff(geometries["iu"].Z), 0.0)
    npt.assert_array_less(np.diff(geometries["ou"].Z), 0.0)
    # Expect x point somewhere between the midplane and the wall
    assert 0 < geometries["il"].Xpoint < len(geometries["il"].Spar)
    assert 0 < geometries["ol"].Xpoint < len(geometries["ol"].Spar)
    assert 0 < geometries["iu"].Xpoint < len(geometries["iu"].Spar)
    assert 0 < geometries["ou"].Xpoint < len(geometries["ou"].Spar)
    # Expect target to be to the right/left of the xpoint for outer/inner
    assert geometries["il"].R[0] < geometries["il"].R[geometries["il"].Xpoint]
    assert geometries["iu"].R[0] < geometries["iu"].R[geometries["iu"].Xpoint]
    assert geometries["ol"].R[0] > geometries["ol"].R[geometries["ol"].Xpoint]
    assert geometries["ou"].R[0] > geometries["ol"].R[geometries["ou"].Xpoint]


@pytest.mark.parametrize(
    "filename",
    [
        "test_single-null.eqdsk",
        "test_connected-double-null.eqdsk",
        "test_disconnected-double-null.eqdsk",
    ],
)
def test_geqdsk_cocos(filename: str, tmp_path: Path):
    """Similar to test_single_null, tests that the same results are obtained
    when the input file follows a different COCOS convention."""

    path = TEST_DIR / filename
    # Single-null COCOS auto-identification doesn't seem to work, so we manually
    # set it to 11.
    cocos = 11 if "single" in filename else None
    geometries = read_geqdsk(path, cocos=cocos)

    # Make a version of the same file with a different COCOS convention.
    with path.open() as fh:
        data = geqdsk.read(fh)
    new_data = _transform_geqdsk(data, cocos_in=cocos, cocos_out=8)
    new_path = tmp_path / filename
    with new_path.open("w") as fh:
        geqdsk.write(new_data, fh)

    # Generate geometries, check that the end points are the same.
    # Transformed G-EQDSK files also don't seem to be identified correctly, so
    # we must
    new_geometries = read_geqdsk(new_path, cocos=8)
    for key in ["il", "ol", "iu", "ou"]:
        if geometries[key] is None:
            assert new_geometries[key] is None
            continue

        # Keep type checker happy...
        geometry = geometries[key]
        new_geometry = new_geometries[key]
        assert isinstance(geometry, MagneticGeometry)
        assert isinstance(new_geometry, MagneticGeometry)

        # Check that the end points are the same
        # N.B. Very large tolerances here, as G-EQDSK files are not very precise
        # and the results are surprisingly sensitive to the transformations.
        npt.assert_allclose(geometry.Z[0], new_geometry.Z[0], atol=1.2e-2)
        npt.assert_allclose(geometry.Z[-1], new_geometry.Z[-1], atol=1.2e-2)
        npt.assert_allclose(geometry.R[0], new_geometry.R[0], atol=1.2e-2)
        npt.assert_allclose(geometry.R[-1], new_geometry.R[-1], atol=1.2e-2)
        npt.assert_allclose(geometry.Btot[0], new_geometry.Btot[0], rtol=1e-2)
        npt.assert_allclose(geometry.Btot[-1], new_geometry.Btot[-1], rtol=1e-2)
