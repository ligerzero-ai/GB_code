"""Shared fixtures for GB_code tests."""

import pytest
import numpy as np
from gb_code.gb_generator import GB_character


@pytest.fixture
def sigma5_100_fcc():
    """Return a GB_character configured for Sigma-5 [1,0,0] fcc, (0,3,1) tilt.

    Not yet built — call .build() in the test.
    """
    gb = GB_character()
    gb.ParseGB(axis=[1, 0, 0], basis='fcc', LatP=4.05, m=2, n=1, gb=[0, 3, 1])
    gb.CSL_Bicrystal_Atom_generator()
    return gb


@pytest.fixture
def sigma5_built(sigma5_100_fcc):
    """Return a built Sigma-5 GB with default parameters."""
    sigma5_100_fcc.build(overlap=0.0, dim=[2, 1, 1])
    return sigma5_100_fcc


@pytest.fixture
def sigma5_100_fcc_012():
    """Sigma-5 [1,0,0] fcc, (0,1,2) tilt — larger unit cell."""
    gb = GB_character()
    gb.ParseGB(axis=[1, 0, 0], basis='fcc', LatP=4.05, m=2, n=1, gb=[0, 1, 2])
    gb.CSL_Bicrystal_Atom_generator()
    return gb
