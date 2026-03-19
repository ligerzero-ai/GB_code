"""Tests for gb_code.inplane_shift."""

import numpy as np
import pytest

from gb_code.gb_generator import GB_character
from gb_code.inplane_shift import (
    compute_shift_vectors,
    generate_shifts,
    apply_shift,
    get_all_shifted_structures,
    write_all_shifts,
)


@pytest.fixture
def built_gb():
    """A built Sigma-5 GB for shift tests."""
    gb = GB_character()
    gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
    gb.CSL_Bicrystal_Atom_generator()
    gb.build(overlap=0.0, dim=[2, 1, 1])
    return gb


@pytest.fixture
def built_gb_overlap():
    """A built Sigma-5 GB with overlap removal."""
    gb = GB_character()
    gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 1, 2])
    gb.CSL_Bicrystal_Atom_generator()
    gb.build(overlap=0.3, whichG='g1', dim=[2, 1, 1])
    return gb


# ------------------------------------------------------------------ #
#  compute_shift_vectors                                               #
# ------------------------------------------------------------------ #

class TestComputeShiftVectors:
    """Tests for compute_shift_vectors."""

    def test_returns_four_items(self, built_gb):
        result = compute_shift_vectors(built_gb, a=10, b=5)
        assert len(result) == 4
        shift1, shift2, a_eff, b_eff = result
        assert shift1.shape == (3,)
        assert shift2.shape == (3,)

    def test_shift_vectors_are_inplane(self, built_gb):
        shift1, shift2, _, _ = compute_shift_vectors(built_gb, a=10, b=5)
        # In-plane means x component (GB normal direction) should be ~0
        assert abs(shift1[0]) < 1e-10
        assert abs(shift2[0]) < 1e-10

    def test_effective_grid_sizes(self, built_gb):
        _, _, a_eff, b_eff = compute_shift_vectors(built_gb, a=10, b=5)
        assert a_eff == 10
        assert b_eff == 5


# ------------------------------------------------------------------ #
#  generate_shifts                                                     #
# ------------------------------------------------------------------ #

class TestGenerateShifts:
    """Tests for generate_shifts."""

    def test_count(self, built_gb):
        shifts = generate_shifts(built_gb, a=10, b=5)
        assert len(shifts) == 50

    def test_first_is_zero(self, built_gb):
        shifts = generate_shifts(built_gb, a=10, b=5)
        np.testing.assert_allclose(shifts[0], [0, 0, 0], atol=1e-15)

    def test_shift_shapes(self, built_gb):
        shifts = generate_shifts(built_gb, a=3, b=2)
        for s in shifts:
            assert s.shape == (3,)

    def test_all_inplane(self, built_gb):
        shifts = generate_shifts(built_gb, a=5, b=3)
        for s in shifts:
            assert abs(s[0]) < 1e-10


# ------------------------------------------------------------------ #
#  apply_shift                                                         #
# ------------------------------------------------------------------ #

class TestApplyShift:
    """Tests for apply_shift."""

    def test_pymatgen_output(self, built_gb):
        shifts = generate_shifts(built_gb, a=3, b=2)
        struct = apply_shift(built_gb, shifts[1], output='pymatgen',
                             element='Al')
        expected = len(built_gb.atoms1) + len(built_gb.atoms2)
        assert struct.num_sites == expected

    def test_ase_output(self, built_gb):
        shifts = generate_shifts(built_gb, a=3, b=2)
        atoms = apply_shift(built_gb, shifts[1], output='ase', element='Al')
        expected = len(built_gb.atoms1) + len(built_gb.atoms2)
        assert len(atoms) == expected

    def test_arrays_output(self, built_gb):
        shifts = generate_shifts(built_gb, a=3, b=2)
        result = apply_shift(built_gb, shifts[1], output='arrays')
        positions, grain_ids, cell = result
        assert positions.shape[1] == 3
        assert len(grain_ids) == len(positions)
        assert cell.shape == (3, 3)

    def test_invalid_output_raises(self, built_gb):
        shifts = generate_shifts(built_gb, a=2, b=2)
        with pytest.raises(ValueError, match="output must be"):
            apply_shift(built_gb, shifts[0], output='invalid')

    def test_does_not_mutate_original(self, built_gb):
        original_atoms1 = built_gb.atoms1.copy()
        shifts = generate_shifts(built_gb, a=3, b=2)
        apply_shift(built_gb, shifts[1], output='arrays')
        np.testing.assert_array_equal(built_gb.atoms1, original_atoms1)

    def test_shift_changes_positions(self, built_gb):
        shifts = generate_shifts(built_gb, a=5, b=3)
        pos0, _, _ = apply_shift(built_gb, shifts[0], output='arrays')
        pos1, _, _ = apply_shift(built_gb, shifts[1], output='arrays')
        # Positions should differ (shift is non-zero for index > 0)
        assert not np.allclose(pos0, pos1)

    def test_elements_dict(self, built_gb):
        shifts = generate_shifts(built_gb, a=2, b=2)
        struct = apply_shift(built_gb, shifts[0], output='pymatgen',
                             elements={1: 'Al', 2: 'Cu'})
        symbols = [site.specie.symbol for site in struct]
        assert 'Al' in symbols
        assert 'Cu' in symbols


# ------------------------------------------------------------------ #
#  apply_shift with gb_normal                                          #
# ------------------------------------------------------------------ #

class TestApplyShiftWithGBNormal:
    """Test that apply_shift works with non-default gb_normal."""

    def test_shift_with_gb_normal_z(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        gb.build(dim=[2, 1, 1], gb_normal='z')

        shifts = generate_shifts(gb, a=3, b=2)
        struct = apply_shift(gb, shifts[1], output='pymatgen', element='Al')
        expected = len(gb.atoms1) + len(gb.atoms2)
        assert struct.num_sites == expected
        # c should be the long (normal) direction
        assert struct.lattice.c > struct.lattice.a


# ------------------------------------------------------------------ #
#  get_all_shifted_structures                                          #
# ------------------------------------------------------------------ #

class TestGetAllShiftedStructures:
    """Tests for get_all_shifted_structures."""

    def test_pymatgen_list(self, built_gb):
        structs = get_all_shifted_structures(
            built_gb, a=3, b=2, output='pymatgen', element='Al')
        assert len(structs) == 6
        for s in structs:
            assert hasattr(s, 'num_sites')

    def test_ase_list(self, built_gb):
        atoms_list = get_all_shifted_structures(
            built_gb, a=3, b=2, output='ase', element='Al')
        assert len(atoms_list) == 6
        for a in atoms_list:
            assert hasattr(a, 'get_tags')


# ------------------------------------------------------------------ #
#  write_all_shifts                                                    #
# ------------------------------------------------------------------ #

class TestWriteAllShifts:
    """Tests for write_all_shifts (file output)."""

    def test_lammps_files_created(self, built_gb, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_all_shifts(built_gb, a=3, b=2, fmt='LAMMPS')
        lammps_files = list(tmp_path.glob('input_G*'))
        assert len(lammps_files) == 6

    def test_vasp_files_created(self, built_gb, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_all_shifts(built_gb, a=2, b=2, fmt='VASP')
        vasp_files = list(tmp_path.glob('POS_G*'))
        assert len(vasp_files) == 4

    def test_invalid_fmt_raises(self, built_gb):
        with pytest.raises(ValueError, match="fmt must be"):
            write_all_shifts(built_gb, a=2, b=2, fmt='XYZ')

    def test_original_not_mutated(self, built_gb, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original = built_gb.atoms1.copy()
        write_all_shifts(built_gb, a=3, b=2, fmt='LAMMPS')
        np.testing.assert_array_equal(built_gb.atoms1, original)
