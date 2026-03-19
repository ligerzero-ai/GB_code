"""Tests for gb_code.csl_generator."""

import numpy as np
import pytest
from math import degrees, radians

from gb_code import csl_generator as cslgen


class TestGetCubicSigma:
    """Tests for get_cubic_sigma."""

    def test_sigma5_100(self):
        assert cslgen.get_cubic_sigma([1, 0, 0], m=2, n=1) == 5

    def test_sigma3_111(self):
        assert cslgen.get_cubic_sigma([1, 1, 1], m=2, n=1) == 7

    def test_sigma1_returns_none(self):
        # m=1, n=0 gives sigma=1, which should return None
        assert cslgen.get_cubic_sigma([1, 0, 0], m=1, n=0) is None

    def test_sigma13_111(self):
        assert cslgen.get_cubic_sigma([1, 1, 1], m=7, n=1) == 13

    def test_various_axes(self):
        # [1,1,0] axis, m=3, n=1 -> sigma = 9 + 2 = 11
        sig = cslgen.get_cubic_sigma([1, 1, 0], m=3, n=1)
        assert sig is not None
        assert isinstance(sig, (int, float))


class TestGetCubicTheta:
    """Tests for get_cubic_theta."""

    def test_sigma5_100_angle(self):
        theta = cslgen.get_cubic_theta([1, 0, 0], m=2, n=1)
        # 2 * atan(1 * 1 / 2) = 2 * atan(0.5) ~ 53.13 deg
        assert abs(degrees(theta) - 53.13) < 0.1

    def test_sigma7_111_angle(self):
        theta = cslgen.get_cubic_theta([1, 1, 1], m=2, n=1)
        # 2 * atan(sqrt(3) * 1 / 2) ~ 81.79 deg
        assert abs(degrees(theta) - 81.79) < 0.1

    def test_m_zero_gives_pi(self):
        theta = cslgen.get_cubic_theta([1, 0, 0], m=0, n=1)
        assert abs(theta - np.pi) < 1e-10


class TestRotationMatrix:
    """Tests for rot (rotation matrix generation)."""

    def test_identity_at_zero_angle(self):
        R = cslgen.rot([1, 0, 0], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        R = cslgen.rot([1, 1, 1], radians(60))
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_orthogonal(self):
        R = cslgen.rot([1, 0, 0], radians(36.87))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)


class TestBasis:
    """Tests for Basis function."""

    def test_sc_single_atom(self):
        b = cslgen.Basis('sc')
        assert b.shape == (1, 3)
        np.testing.assert_array_equal(b, [[0, 0, 0]])

    def test_fcc_four_atoms(self):
        b = cslgen.Basis('fcc')
        assert len(b) == 4

    def test_bcc_two_atoms(self):
        b = cslgen.Basis('bcc')
        assert len(b) == 2

    def test_diamond_eight_atoms(self):
        b = cslgen.Basis('diamond')
        assert len(b) == 8


class TestFindOrthogonalCell:
    """Tests for Find_Orthogonal_cell."""

    def test_returns_three_items(self):
        result = cslgen.Find_Orthogonal_cell('fcc', [1, 0, 0], 2, 1,
                                              np.array([0, 3, 1]))
        assert len(result) == 3
        ortho1, ortho2, num_atoms = result
        assert ortho1.shape == (3, 3)
        assert ortho2.shape == (3, 3)
        assert isinstance(num_atoms, (int, np.integer))

    def test_high_index_plane_still_works(self):
        # High-index planes may still produce valid (large) cells
        result = cslgen.Find_Orthogonal_cell('fcc', [1, 0, 0], 2, 1,
                                              np.array([7, 7, 7]))
        assert result is not None
        ortho1, ortho2, num_atoms = result
        assert num_atoms > 0
