#!/usr/bin/env python
"""
Standalone utility for applying rigid-body in-plane translations to a
grain boundary structure built by GB_character.

This was previously built into the GB generation workflow (Translate method).
Now it is a separate tool that operates on an already-built GB_character
and returns shifted structures as pymatgen, ASE, or file outputs.

Usage
-----
    from gb_code.gb_generator import GB_character
    from gb_code.inplane_shift import generate_shifts, apply_shift

    # Build GB
    gb = GB_character()
    gb.ParseGB([1,0,0], 'fcc', 4.05, 5, 1, [3,1,0])
    gb.CSL_Bicrystal_Atom_generator()
    gb.build(overlap=0.0, dim=[1,1,1])

    # Get all shift vectors
    shifts = generate_shifts(gb, a=10, b=5)

    # Apply a single shift and get pymatgen Structure
    struct = apply_shift(gb, shifts[0], output='pymatgen', element='Al')

    # Or iterate over all shifts
    for i, s in enumerate(shifts):
        atoms = apply_shift(gb, s, output='ase', element='Al')
        # ... do something with atoms

    # Write all shifted structures to LAMMPS files
    write_all_shifts(gb, a=10, b=5, fmt='LAMMPS')
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm
import copy

try:
    from . import csl_generator as cslgen
except ImportError:
    import csl_generator as cslgen


def compute_shift_vectors(gb, a=10, b=5):
    """
    Compute the two basis shift vectors for in-plane rigid body translation.

    For twist boundaries (GB plane parallel to rotation axis), uses DSC
    vectors projected onto the GB plane.
    For tilt/mixed boundaries, uses the in-plane orthogonal cell vectors.

    Parameters
    ----------
    gb : GB_character
        A built grain boundary object.
    a, b : int
        Number of divisions along each in-plane direction.

    Returns
    -------
    shift1 : (3,) array
        First shift basis vector (in lattice units, pre-LatP).
    shift2 : (3,) array
        Second shift basis vector (in lattice units, pre-LatP).
    a_eff : int
        Effective a (may be overridden to 3 for twist).
    b_eff : int
        Effective b (may be overridden to 3 for twist).
    """
    tol = 0.001
    is_twist = (1 - cslgen.ang(gb.gbplane, gb.axis) < tol)

    if is_twist:
        M1, _ = cslgen.Create_minimal_cell_Method_1(
            gb.sigma, gb.axis, gb.R)
        D = (1 / gb.sigma * cslgen.DSC_vec(gb.basis, gb.sigma, M1))
        Dvecs = cslgen.DSC_on_plane(D, gb.gbplane)
        TransDvecs = np.round(dot(gb.rot1, Dvecs), 7)
        shift1 = TransDvecs[:, 0] / 2
        shift2 = TransDvecs[:, 1] / 2
        a_eff = b_eff = 3
    else:
        a_eff, b_eff = a, b
        if norm(gb.ortho1[:, 1]) > norm(gb.ortho1[:, 2]):
            shift1 = (1 / a_eff) * (norm(gb.ortho1[:, 1]) *
                                     np.array([0, 1, 0]))
            shift2 = (1 / b_eff) * (norm(gb.ortho1[:, 2]) *
                                     np.array([0, 0, 1]))
        else:
            shift1 = (1 / a_eff) * (norm(gb.ortho1[:, 2]) *
                                     np.array([0, 0, 1]))
            shift2 = (1 / b_eff) * (norm(gb.ortho1[:, 1]) *
                                     np.array([0, 1, 0]))

    return shift1, shift2, a_eff, b_eff


def generate_shifts(gb, a=10, b=5):
    """
    Generate all in-plane shift vectors for a GB.

    Parameters
    ----------
    gb : GB_character
        A built grain boundary object.
    a, b : int
        Grid divisions along each in-plane direction.

    Returns
    -------
    list of (3,) arrays
        Each element is a shift vector (in lattice units) to be added
        to grain 1 atom positions.
    """
    shift1, shift2, a_eff, b_eff = compute_shift_vectors(gb, a, b)

    shifts = []
    for i in range(a_eff):
        for j in range(b_eff):
            shifts.append(i * shift1 + j * shift2)
    return shifts


def apply_shift(gb, shift, output='pymatgen', element='X', elements=None):
    """
    Apply a rigid-body in-plane shift to grain 1 and return the structure.

    Parameters
    ----------
    gb : GB_character
        A built grain boundary object (build() must have been called).
    shift : array-like of shape (3,)
        The shift vector to apply to grain 1 (in lattice units).
    output : str
        One of 'pymatgen', 'ase', 'arrays'.
        - 'pymatgen': returns pymatgen.core.Structure
        - 'ase': returns ase.Atoms
        - 'arrays': returns (positions_ang, grain_ids, cell) tuple
    element : str
        Chemical symbol for atoms. Default 'X'.
    elements : dict, optional
        Map grain id to element, e.g. {1: 'Al', 2: 'Cu'}.

    Returns
    -------
    Depends on *output* parameter.
    """
    shift = np.asarray(shift, dtype=float)

    # Create a shallow copy so we don't mutate the original
    gb_shifted = copy.copy(gb)
    gb_shifted.atoms1 = gb.atoms1.copy() + shift
    gb_shifted.atoms2 = gb.atoms2.copy()

    if output == 'pymatgen':
        return gb_shifted.to_pymatgen(element=element, elements=elements)
    elif output == 'ase':
        return gb_shifted.to_ase(element=element, elements=elements)
    elif output == 'arrays':
        return gb_shifted._get_cartesian_positions_and_grains() + \
            (gb_shifted._get_cell_vectors(),)
    else:
        raise ValueError("output must be 'pymatgen', 'ase', or 'arrays'")


def write_all_shifts(gb, a=10, b=5, fmt='LAMMPS'):
    """
    Write all shifted GB structures to files (legacy file output).

    Parameters
    ----------
    gb : GB_character
        A built grain boundary object.
    a, b : int
        Grid divisions for in-plane translations.
    fmt : str
        'LAMMPS' or 'VASP'.
    """
    shifts = generate_shifts(gb, a, b)
    print("<<------ {} GB structures are being created! ------>>"
          .format(len(shifts)))

    original_atoms1 = gb.atoms1.copy()
    for count, shift in enumerate(shifts, start=1):
        gb.atoms1 = original_atoms1 + shift
        if fmt == 'LAMMPS':
            gb.write_lammps(count=count)
        elif fmt == 'VASP':
            gb.write_vasp(count=count)
        else:
            raise ValueError("fmt must be 'LAMMPS' or 'VASP'")

    # Restore original
    gb.atoms1 = original_atoms1


def get_all_shifted_structures(gb, a=10, b=5, output='pymatgen',
                                element='X', elements=None):
    """
    Convenience: generate all shifted structures as a list.

    Parameters
    ----------
    gb : GB_character
        A built grain boundary object.
    a, b : int
        Grid divisions.
    output : str
        'pymatgen' or 'ase'.
    element : str
        Chemical symbol.
    elements : dict, optional
        Map grain id to element.

    Returns
    -------
    list
        List of pymatgen Structures or ASE Atoms objects.
    """
    shifts = generate_shifts(gb, a, b)
    structures = []
    for shift in shifts:
        structures.append(
            apply_shift(gb, shift, output=output,
                        element=element, elements=elements)
        )
    return structures


def main():
    """CLI entry point for in-plane shifting."""
    import sys
    import yaml

    usage = """
    Usage: inplane_shift.py io_file

    Reads the same io_file as gb_generator.py and produces shifted structures.
    The io_file must have rigid_trans: yes and valid a, b values.
    """
    if len(sys.argv) != 2:
        print(usage)
        sys.exit()

    io_file = sys.argv[1]
    with open(io_file, 'r') as f:
        in_params = yaml.safe_load(f)

    from .gb_generator import GB_character

    axis = np.array(in_params['axis'])
    m = int(in_params['m'])
    n = int(in_params['n'])
    basis = str(in_params['basis'])
    gbplane = np.array(in_params['GB_plane'])
    LatP = in_params['lattice_parameter']
    overlap = in_params['overlap_distance']
    whichG = in_params.get('which_g', 'g1')
    a_grid = int(in_params.get('a', 10))
    b_grid = int(in_params.get('b', 5))
    dim = in_params.get('dimensions', [1, 1, 1])
    file_type = in_params.get('File_type', 'LAMMPS')

    gb = GB_character()
    gb.ParseGB(axis, basis, LatP, m, n, gbplane)
    gb.CSL_Bicrystal_Atom_generator()
    gb.build(overlap=overlap, whichG=whichG, dim=dim)

    write_all_shifts(gb, a=a_grid, b=b_grid, fmt=file_type)


if __name__ == '__main__':
    main()
