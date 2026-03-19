"""Tests for gb_code.gb_generator.GB_character."""

import numpy as np
import pytest
from numpy.linalg import norm

from gb_code.gb_generator import GB_character


# ------------------------------------------------------------------ #
#  Basic construction / parsing                                        #
# ------------------------------------------------------------------ #

class TestParseGB:
    """Tests for ParseGB and CSL_Bicrystal_Atom_generator."""

    def test_sigma_assigned(self, sigma5_100_fcc):
        assert sigma5_100_fcc.sigma == 5

    def test_lattice_parameter(self, sigma5_100_fcc):
        assert sigma5_100_fcc.LatP == 4.05

    def test_ortho_cells_populated(self, sigma5_100_fcc):
        assert sigma5_100_fcc.ortho1.shape == (3, 3)
        assert sigma5_100_fcc.ortho2.shape == (3, 3)

    def test_atoms_populated_after_bicrystal(self, sigma5_100_fcc):
        assert sigma5_100_fcc.atoms1.shape[1] == 3
        assert sigma5_100_fcc.atoms2.shape[1] == 3
        assert len(sigma5_100_fcc.atoms1) > 0
        assert len(sigma5_100_fcc.atoms2) > 0

    def test_invalid_basis_exits(self):
        gb = GB_character()
        with pytest.raises(SystemExit):
            gb.ParseGB([1, 0, 0], 'hcp', 4.05, 2, 1, [0, 3, 1])


# ------------------------------------------------------------------ #
#  build()                                                             #
# ------------------------------------------------------------------ #

class TestBuild:
    """Tests for the build() method."""

    def test_returns_self(self, sigma5_100_fcc):
        result = sigma5_100_fcc.build(overlap=0.0, dim=[1, 1, 1])
        assert result is sigma5_100_fcc

    def test_built_flag(self, sigma5_100_fcc):
        assert not sigma5_100_fcc._built
        sigma5_100_fcc.build()
        assert sigma5_100_fcc._built

    def test_default_dim(self, sigma5_100_fcc):
        sigma5_100_fcc.build()
        np.testing.assert_array_equal(sigma5_100_fcc.dim, [1, 1, 1])

    def test_custom_dim(self, sigma5_100_fcc):
        sigma5_100_fcc.build(dim=[3, 2, 2])
        np.testing.assert_array_equal(sigma5_100_fcc.dim, [3, 2, 2])

    def test_atom_count_scales_with_dim(self, sigma5_100_fcc):
        n1_base = len(sigma5_100_fcc.atoms1)
        sigma5_100_fcc.build(dim=[1, 1, 1])
        n1_1x = len(sigma5_100_fcc.atoms1)
        # atoms1 was already populated by CSL_Bicrystal_Atom_generator,
        # build() expands it. With dim=[1,1,1], count should equal base.
        assert n1_1x == n1_base

    def test_overlap_removes_atoms(self, sigma5_100_fcc_012):
        gb = sigma5_100_fcc_012
        gb.build(overlap=0.0, dim=[1, 1, 1])
        n_no_overlap = len(gb.atoms1) + len(gb.atoms2)

        gb2 = GB_character()
        gb2.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 1, 2])
        gb2.CSL_Bicrystal_Atom_generator()
        gb2.build(overlap=0.3, whichG='g1', dim=[1, 1, 1])
        n_with_overlap = len(gb2.atoms1) + len(gb2.atoms2)

        assert n_with_overlap <= n_no_overlap

    def test_overlap_whichg_g2(self, sigma5_100_fcc_012):
        gb = sigma5_100_fcc_012
        gb.build(overlap=0.3, whichG='g2', dim=[1, 1, 1])
        # Should not raise and should have fewer atoms than no-overlap
        assert gb._built

    def test_invalid_whichg_raises(self, sigma5_100_fcc):
        with pytest.raises(ValueError, match="whichG must be"):
            sigma5_100_fcc.build(overlap=0.3, whichG='g3')

    def test_export_before_build_raises(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        with pytest.raises(RuntimeError, match="Call build"):
            gb._get_cartesian_positions_and_grains()


# ------------------------------------------------------------------ #
#  min_inplane_dist                                                    #
# ------------------------------------------------------------------ #

class TestMinInplaneDist:
    """Tests for the min_inplane_dist parameter."""

    def test_dim_increased_to_meet_minimum(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        gb.build(dim=[2, 1, 1], min_inplane_dist=15.0)
        cell = gb._get_cell_vectors()
        # In-plane dims (y and z) must be >= 15 A
        assert cell[1, 1] >= 15.0
        assert cell[2, 2] >= 15.0

    def test_dimx_not_affected(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        gb.build(dim=[2, 1, 1], min_inplane_dist=15.0)
        # dimX should still be 2 (not increased)
        assert gb.dim[0] == 2

    def test_dim_floor_respected(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        # If user provides dim=[2, 10, 10] and min_inplane_dist is small,
        # the user's dim values should be kept
        gb.build(dim=[2, 10, 10], min_inplane_dist=1.0)
        assert gb.dim[1] == 10
        assert gb.dim[2] == 10

    def test_none_means_no_override(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        gb.build(dim=[2, 1, 1], min_inplane_dist=None)
        assert gb.dim[1] == 1
        assert gb.dim[2] == 1

    def test_exact_multiple(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        base_y = norm(gb.ortho1[:, 1]) * gb.LatP
        # Ask for exactly 2x the base cell
        gb.build(dim=[1, 1, 1], min_inplane_dist=base_y * 2)
        assert gb.dim[1] == 2


# ------------------------------------------------------------------ #
#  gb_normal                                                           #
# ------------------------------------------------------------------ #

class TestGBNormal:
    """Tests for the gb_normal parameter (axis permutation)."""

    def test_default_is_x(self, sigma5_built):
        assert sigma5_built._gb_normal == 'x'
        assert sigma5_built._normal_axis_index == 0

    def test_invalid_gb_normal_raises(self, sigma5_100_fcc):
        with pytest.raises(ValueError, match="gb_normal must be"):
            sigma5_100_fcc.build(gb_normal='w')

    def test_permutation_matrix_x_is_identity(self, sigma5_built):
        P = sigma5_built._permutation_matrix()
        np.testing.assert_array_equal(P, np.eye(3))

    def test_permutation_matrix_y(self, sigma5_100_fcc):
        sigma5_100_fcc.build(gb_normal='y')
        P = sigma5_100_fcc._permutation_matrix()
        # Internal x -> output y
        out = P @ np.array([1, 0, 0])
        np.testing.assert_array_equal(out, [0, 1, 0])

    def test_permutation_matrix_z(self, sigma5_100_fcc):
        sigma5_100_fcc.build(gb_normal='z')
        P = sigma5_100_fcc._permutation_matrix()
        # Internal x -> output z
        out = P @ np.array([1, 0, 0])
        np.testing.assert_array_equal(out, [0, 0, 1])

    def test_cell_permuted_y(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        gb.build(dim=[2, 1, 1], gb_normal='x')
        cell_x = gb._get_cell_vectors().diagonal().copy()

        gb2 = GB_character()
        gb2.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb2.CSL_Bicrystal_Atom_generator()
        gb2.build(dim=[2, 1, 1], gb_normal='y')
        cell_y = gb2._get_cell_vectors().diagonal()

        # x-normal's Lx should become y-normal's Ly
        np.testing.assert_allclose(cell_y[1], cell_x[0])
        np.testing.assert_allclose(cell_y[2], cell_x[1])
        np.testing.assert_allclose(cell_y[0], cell_x[2])

    def test_cell_permuted_z(self, sigma5_100_fcc):
        gb = sigma5_100_fcc
        gb.build(dim=[2, 1, 1], gb_normal='x')
        cell_x = gb._get_cell_vectors().diagonal().copy()

        gb2 = GB_character()
        gb2.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb2.CSL_Bicrystal_Atom_generator()
        gb2.build(dim=[2, 1, 1], gb_normal='z')
        cell_z = gb2._get_cell_vectors().diagonal()

        # x-normal's Lx should become z-normal's Lz
        np.testing.assert_allclose(cell_z[2], cell_x[0])
        np.testing.assert_allclose(cell_z[0], cell_x[1])
        np.testing.assert_allclose(cell_z[1], cell_x[2])

    def test_positions_permuted(self, sigma5_100_fcc):
        gb_x = GB_character()
        gb_x.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb_x.CSL_Bicrystal_Atom_generator()
        gb_x.build(dim=[2, 1, 1], gb_normal='x')
        pos_x, ids_x = gb_x._get_cartesian_positions_and_grains()

        gb_z = GB_character()
        gb_z.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb_z.CSL_Bicrystal_Atom_generator()
        gb_z.build(dim=[2, 1, 1], gb_normal='z')
        pos_z, ids_z = gb_z._get_cartesian_positions_and_grains()

        # Same number of atoms, same grain IDs
        assert len(pos_x) == len(pos_z)
        np.testing.assert_array_equal(ids_x, ids_z)

        # The x column in pos_x should match z column in pos_z
        np.testing.assert_allclose(
            np.sort(pos_x[:, 0]), np.sort(pos_z[:, 2]), atol=1e-8)

    def test_atom_count_same_across_normals(self, sigma5_100_fcc):
        counts = []
        for normal in ('x', 'y', 'z'):
            gb = GB_character()
            gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
            gb.CSL_Bicrystal_Atom_generator()
            gb.build(dim=[2, 1, 1], gb_normal=normal)
            counts.append(len(gb.atoms1) + len(gb.atoms2))
        assert counts[0] == counts[1] == counts[2]


# ------------------------------------------------------------------ #
#  get_grain_info()                                                    #
# ------------------------------------------------------------------ #

class TestGetGrainInfo:
    """Tests for get_grain_info."""

    def test_before_build_raises(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        with pytest.raises(RuntimeError):
            gb.get_grain_info()

    def test_keys_present(self, sigma5_built):
        info = sigma5_built.get_grain_info()
        assert 'gb_normal' in info
        assert 'n_grain1' in info
        assert 'n_grain2' in info
        assert 'cell' in info
        assert 'sigma' in info
        assert 'axis' in info
        assert 'gbplane' in info
        assert 'theta_rad' in info

    def test_gb_normal_x_key(self, sigma5_built):
        info = sigma5_built.get_grain_info()
        assert info['gb_normal'] == 'x'
        assert 'gb_plane_x_angstrom' in info

    def test_gb_normal_z_key(self, sigma5_100_fcc):
        sigma5_100_fcc.build(dim=[2, 1, 1], gb_normal='z')
        info = sigma5_100_fcc.get_grain_info()
        assert info['gb_normal'] == 'z'
        assert 'gb_plane_z_angstrom' in info

    def test_sigma_value(self, sigma5_built):
        info = sigma5_built.get_grain_info()
        assert info['sigma'] == 5

    def test_cell_shape(self, sigma5_built):
        info = sigma5_built.get_grain_info()
        assert info['cell'].shape == (3, 3)


# ------------------------------------------------------------------ #
#  pymatgen export                                                     #
# ------------------------------------------------------------------ #

class TestToPymatgen:
    """Tests for to_pymatgen."""

    def test_num_sites(self, sigma5_built):
        struct = sigma5_built.to_pymatgen(element='Al')
        expected = len(sigma5_built.atoms1) + len(sigma5_built.atoms2)
        assert struct.num_sites == expected

    def test_element(self, sigma5_built):
        struct = sigma5_built.to_pymatgen(element='Cu')
        assert all(site.specie.symbol == 'Cu' for site in struct)

    def test_elements_dict(self, sigma5_built):
        struct = sigma5_built.to_pymatgen(elements={1: 'Al', 2: 'Cu'})
        symbols = [site.specie.symbol for site in struct]
        assert 'Al' in symbols
        assert 'Cu' in symbols

    def test_grain_id_property(self, sigma5_built):
        struct = sigma5_built.to_pymatgen(element='Al')
        grain_ids = struct.site_properties['grain_id']
        assert set(grain_ids) == {1, 2}

    def test_positions_inside_cell(self, sigma5_built):
        struct = sigma5_built.to_pymatgen(element='Al')
        frac = struct.frac_coords
        assert np.all(frac >= -0.01)
        assert np.all(frac <= 1.01)

    def test_gb_normal_z_lattice(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        gb.build(dim=[2, 1, 1], gb_normal='z')
        struct = gb.to_pymatgen(element='Al')
        # c should be the long (normal) direction
        assert struct.lattice.c > struct.lattice.a
        assert struct.lattice.c > struct.lattice.b


# ------------------------------------------------------------------ #
#  ASE export                                                          #
# ------------------------------------------------------------------ #

class TestToAse:
    """Tests for to_ase."""

    def test_num_atoms(self, sigma5_built):
        atoms = sigma5_built.to_ase(element='Al')
        expected = len(sigma5_built.atoms1) + len(sigma5_built.atoms2)
        assert len(atoms) == expected

    def test_tags(self, sigma5_built):
        atoms = sigma5_built.to_ase(element='Al')
        tags = set(atoms.get_tags())
        assert tags == {1, 2}

    def test_grain_id_array(self, sigma5_built):
        atoms = sigma5_built.to_ase(element='Al')
        assert 'grain_id' in atoms.arrays
        assert set(atoms.arrays['grain_id']) == {1, 2}

    def test_info_keys(self, sigma5_built):
        atoms = sigma5_built.to_ase(element='Al')
        assert 'gb_plane_x' in atoms.info
        assert 'gb_normal' in atoms.info
        assert 'sigma' in atoms.info
        assert atoms.info['gb_normal'] == 'x'

    def test_gb_normal_z_info(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        gb.build(dim=[2, 1, 1], gb_normal='z')
        atoms = gb.to_ase(element='Al')
        assert 'gb_plane_z' in atoms.info
        assert atoms.info['gb_normal'] == 'z'

    def test_pbc(self, sigma5_built):
        atoms = sigma5_built.to_ase(element='Al')
        assert all(atoms.pbc)

    def test_elements_dict(self, sigma5_built):
        atoms = sigma5_built.to_ase(elements={1: 'Al', 2: 'Cu'})
        symbols = atoms.get_chemical_symbols()
        assert 'Al' in symbols
        assert 'Cu' in symbols


# ------------------------------------------------------------------ #
#  File writers                                                        #
# ------------------------------------------------------------------ #

class TestFileWriters:
    """Tests for write_lammps and write_vasp."""

    def test_write_lammps(self, sigma5_built, tmp_path):
        f = tmp_path / 'test.lammps'
        sigma5_built.write_lammps(str(f))
        content = f.read_text()
        assert 'atoms' in content
        assert 'atom types' in content
        assert 'xlo xhi' in content

    def test_write_vasp(self, sigma5_built, tmp_path):
        f = tmp_path / 'POSCAR'
        sigma5_built.write_vasp(str(f))
        content = f.read_text()
        assert 'POSCAR written by GB_code' in content
        assert 'Cartesian' in content

    def test_write_before_build_raises(self, tmp_path):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        with pytest.raises(RuntimeError, match="Call build"):
            gb.write_lammps(str(tmp_path / 'fail.lammps'))

    def test_lammps_atom_count(self, sigma5_built, tmp_path):
        f = tmp_path / 'test.lammps'
        sigma5_built.write_lammps(str(f))
        content = f.read_text()
        expected = len(sigma5_built.atoms1) + len(sigma5_built.atoms2)
        # First data line after header should contain atom count
        assert '{} atoms'.format(expected) in content


# ------------------------------------------------------------------ #
#  __str__                                                             #
# ------------------------------------------------------------------ #

class TestStr:
    """Tests for __str__."""

    def test_not_built(self):
        gb = GB_character()
        assert 'not yet built' in str(gb)

    def test_built(self, sigma5_built):
        s = str(sigma5_built)
        assert 'sigma=5' in s
        assert 'n_grain1=' in s


# ------------------------------------------------------------------ #
#  Combined features                                                   #
# ------------------------------------------------------------------ #

class TestCombinedFeatures:
    """Test min_inplane_dist + gb_normal together."""

    def test_min_dist_with_gb_normal_z(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        gb.build(dim=[2, 1, 1], min_inplane_dist=15.0, gb_normal='z')
        cell = gb._get_cell_vectors()
        diag = cell.diagonal()
        # GB normal is z, in-plane are x and y
        assert diag[0] >= 15.0  # in-plane x
        assert diag[1] >= 15.0  # in-plane y
        # z is the normal direction (should be > in-plane since dim[0]=2)

    def test_min_dist_with_gb_normal_y(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        gb.build(dim=[2, 1, 1], min_inplane_dist=10.0, gb_normal='y')
        cell = gb._get_cell_vectors()
        diag = cell.diagonal()
        # GB normal is y, in-plane are x and z
        assert diag[0] >= 10.0  # in-plane x
        assert diag[2] >= 10.0  # in-plane z
