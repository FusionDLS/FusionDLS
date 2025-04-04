import pathlib

import numpy as np
import pytest

from fusiondls import MagneticGeometry


@pytest.fixture(scope="module")
def geometry():
    filename = (
        pathlib.Path(__file__).parent.parent / "docs/examples/eqb_store_lores.pkl"
    )
    return MagneticGeometry.from_pickle(filename, "V10", "ou")


def test_from_pickle():
    filename = (
        pathlib.Path(__file__).parent.parent / "docs/examples/eqb_store_lores.pkl"
    )
    geometry = MagneticGeometry.from_pickle(filename, "V10", "ou")

    assert isinstance(geometry, MagneticGeometry)


def test_read_design():
    filename = (
        pathlib.Path(__file__).parent.parent / "docs/examples/eqb_store_lores.pkl"
    )
    design = MagneticGeometry.read_design(filename, "V10")

    assert sorted(design.keys()) == sorted(("iu", "ou", "il", "ol"))
    assert isinstance(design["iu"], MagneticGeometry)


def test_xpoint_properties(geometry):
    assert geometry.Sparx == geometry.Spar[geometry.Xpoint]
    assert geometry.Spolx == geometry.Spol[geometry.Xpoint]
    assert geometry.Bx == geometry.Btot[geometry.Xpoint]
    assert geometry.zx == geometry.zl[geometry.Xpoint]


def test_scale_flux_expansion(geometry):
    current_expansion = geometry.Bx / geometry.Btot[0]

    scaled_geometry = geometry.scale_flux_expansion(scale_factor=2)

    new_expansion = scaled_geometry.Bx / scaled_geometry.Btot[0]
    expected_expansion = 2 * current_expansion

    assert np.isclose(new_expansion, expected_expansion)


def test_scale_flux_expansion_set_value(geometry):
    current_expansion = geometry.Bx / geometry.Btot[0]
    expected_expansion = 2 * current_expansion

    scaled_geometry = geometry.scale_flux_expansion(expansion=expected_expansion)
    new_expansion = scaled_geometry.Bx / scaled_geometry.Btot[0]

    assert np.isclose(new_expansion, expected_expansion)


def test_refine(geometry):
    # Some point halfway through the domain
    half_S = geometry.Spar[-1] / 2
    refined_geometry = geometry.refine(half_S, width=2)

    new_spacing = np.gradient(refined_geometry.Spar)

    smallest_spacing = np.min(new_spacing)
    smallest_spacing_index = np.argmin(np.abs(new_spacing - smallest_spacing))

    # Expect that the smallest spacing is close to the location we asked for
    S_at_smallest_spacing = refined_geometry.Spar[smallest_spacing_index]
    assert np.isclose(S_at_smallest_spacing, half_S, atol=1)

    # Expect that the smallest spacing is roughly halfway through the array
    mid_index = len(geometry.Spar) // 2
    assert mid_index - 2 <= smallest_spacing_index <= mid_index + 2
