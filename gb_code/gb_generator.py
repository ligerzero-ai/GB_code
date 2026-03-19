
# !/usr/bin/env python
"""
This module produces GB structures. You need to run csl_generator first
to get the info necessary for your grain boundary of interest.

Supports output to pymatgen Structure, ASE Atoms, LAMMPS, or VASP formats.
Each atom is tagged with its grain membership (grain 1 or grain 2) and
the GB plane position is tracked.

Usage:
    gb = GB_character()
    gb.ParseGB(axis, basis, LatP, m, n, gbplane)
    gb.CSL_Bicrystal_Atom_generator()
    gb.build(overlap=0.0, dim=[1,1,1])

    # Get pymatgen Structure
    structure = gb.to_pymatgen(element='Al')

    # Get ASE Atoms
    atoms = gb.to_ase(element='Al')

    # Write LAMMPS/VASP as before
    gb.write_lammps()
    gb.write_vasp()
"""

import sys
import numpy as np
from numpy import dot, cross
from numpy.linalg import det, norm

try:
    from . import csl_generator as cslgen
except ImportError:
    import csl_generator as cslgen

import warnings


class GB_character:
    """
    A grain boundary class encompassing all the characteristics of a GB.

    After building, atoms are stored with grain labels:
      - self.atoms1: positions of grain 1 atoms (Nx3 array)
      - self.atoms2: positions of grain 2 atoms (Nx3 array)
      - self.gb_plane_x: x-coordinate of the GB plane (in lattice units)

    The GB plane sits at x=0 in the coordinate system. Grain 1 occupies
    x >= 0, grain 2 occupies x < 0.
    """
    def __init__(self):
        self.axis = np.array([1, 0, 0])
        self.sigma = 1
        self.theta = 0
        self.m = 1
        self.n = 1
        self.R = np.eye(1)
        self.basis = 'fcc'
        self.LatP = 4.05
        self.gbplane = np.array([1, 1, 1])
        self.ortho1 = np.eye(3)
        self.ortho2 = np.eye(3)
        self.ortho = np.eye(3)
        self.atoms = np.eye(3)
        self.atoms1 = np.eye(3)
        self.atoms2 = np.eye(3)
        self.rot1 = np.eye(3)
        self.rot2 = np.eye(3)
        self.Num = 0
        self.dim = np.array([1, 1, 1])
        self.overD = 0
        self.whichG = 'g1'
        self.trans = False
        self.File = 'LAMMPS'
        self.gb_plane_x = 0.0  # GB plane location in x (lattice units)
        self._built = False

    def ParseGB(self, axis, basis, LatP, m, n, gb):
        """
        Parses the GB input: axis, basis, lattice parameter,
        m and n integers, and GB plane.
        """
        self.axis = np.array(axis)
        self.m = int(m)
        self.n = int(n)
        self.sigma = cslgen.get_cubic_sigma(self.axis, self.m, self.n)
        self.theta = cslgen.get_cubic_theta(self.axis, self.m, self.n)
        self.R = cslgen.rot(self.axis, self.theta)

        if str(basis) in ('fcc', 'bcc', 'sc', 'diamond'):
            self.basis = str(basis)
            self.LatP = float(LatP)
            self.gbplane = np.array(gb)

            try:
                self.ortho1, self.ortho2, self.Num = \
                    cslgen.Find_Orthogonal_cell(self.basis,
                                                self.axis,
                                                self.m,
                                                self.n,
                                                self.gbplane)
            except Exception:
                print("""
                    Could not find the orthogonal cells.... Most likely the
                    input GB_plane is "NOT" a CSL plane. Go back to the first
                    script and double check!
                    """)
                sys.exit()
        else:
            print("Sorry! For now only works for cubic lattices ... ")
            sys.exit()

    def CSL_Ortho_unitcell_atom_generator(self):
        """
        Populates a unit cell from the orthogonal vectors.
        """
        Or = self.ortho.T
        Orint = cslgen.integerMatrix(Or)
        LoopBound = np.zeros((3, 2), dtype=float)
        transformed = []
        CubeCoords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                              [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]],
                              dtype=float)
        for i in range(len(CubeCoords)):
            transformed.append(np.dot(Orint.T, CubeCoords[i]))

        LoopBound[0, :] = [min(np.array(transformed)[:, 0]),
                           max(np.array(transformed)[:, 0])]
        LoopBound[1, :] = [min(np.array(transformed)[:, 1]),
                           max(np.array(transformed)[:, 1])]
        LoopBound[2, :] = [min(np.array(transformed)[:, 2]),
                           max(np.array(transformed)[:, 2])]

        Tol = 1
        x = np.arange(LoopBound[0, 0] - Tol, LoopBound[0, 1] + Tol + 1, 1)
        y = np.arange(LoopBound[1, 0] - Tol, LoopBound[1, 1] + Tol + 1, 1)
        z = np.arange(LoopBound[2, 0] - Tol, LoopBound[2, 1] + Tol + 1, 1)
        V = len(x) * len(y) * len(z)
        indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(V, 3)
        Base = cslgen.Basis(str(self.basis))
        Atoms = []
        tol = 0.001
        if V > 5e6:
            print("Warning! It may take a very long time "
                  "to produce this cell!")

        for i in range(V):
            for j in range(len(Base)):
                Atoms.append(indice[i, 0:3] + Base[j, 0:3])
        Atoms = np.array(Atoms)

        Con1 = dot(Atoms, Or[0]) / norm(Or[0]) + tol
        Con2 = dot(Atoms, Or[1]) / norm(Or[1]) + tol
        Con3 = dot(Atoms, Or[2]) / norm(Or[2]) + tol
        Atoms = (Atoms[(Con1 >= 0) & (Con1 <= norm(Or[0])) & (Con2 >= 0) &
                 (Con2 <= norm(Or[1])) &
                 (Con3 >= 0) & (Con3 <= norm(Or[2]))])

        if len(Atoms) == (round(det(Or) * len(Base), 7)).astype(int):
            self.Atoms = Atoms
        else:
            self.Atoms = None
        return

    def CSL_Bicrystal_Atom_generator(self):
        """
        Builds the unit cells for both grains g1 and g2.
        """
        Or_1 = self.ortho1.T
        Or_2 = self.ortho2.T
        self.rot1 = np.array([Or_1[0, :] / norm(Or_1[0, :]),
                             Or_1[1, :] / norm(Or_1[1, :]),
                             Or_1[2, :] / norm(Or_1[2, :])])
        self.rot2 = np.array([Or_2[0, :] / norm(Or_2[0, :]),
                             Or_2[1, :] / norm(Or_2[1, :]),
                             Or_2[2, :] / norm(Or_2[2, :])])

        self.ortho = self.ortho1.copy()
        self.CSL_Ortho_unitcell_atom_generator()
        self.atoms1 = self.Atoms

        self.ortho = self.ortho2.copy()
        self.CSL_Ortho_unitcell_atom_generator()
        self.atoms2 = self.Atoms

        self.atoms1 = dot(self.rot1, self.atoms1.T).T
        self.atoms2 = dot(self.rot2, self.atoms2.T).T
        self.atoms2[:, 0] = self.atoms2[:, 0] - norm(Or_2[0, :])
        return

    def Expand_Super_cell(self):
        """
        Expands the smallest CSL unit cell to the given dimensions.
        """
        a = norm(self.ortho1[:, 0])
        b = norm(self.ortho1[:, 1])
        c = norm(self.ortho1[:, 2])
        dimX, dimY, dimZ = self.dim

        X = self.atoms1.copy()
        Y = self.atoms2.copy()

        X_new = []
        Y_new = []
        for i in range(dimX):
            for j in range(dimY):
                for k in range(dimZ):
                    Position1 = [i * a, j * b, k * c]
                    Position2 = [-i * a, j * b, k * c]
                    for l in range(len(X)):
                        X_new.append(Position1[0:3] + X[l, 0:3])
                    for m in range(len(Y)):
                        Y_new.append(Position2[0:3] + Y[m, 0:3])

        self.atoms1 = np.array(X_new)
        self.atoms2 = np.array(Y_new)
        return

    def Find_overlapping_Atoms(self):
        """
        Finds the overlapping atoms near the GB plane.
        """
        periodic_length = norm(self.ortho1[:, 0]) * self.dim[0]
        periodic_image = self.atoms2 + [periodic_length * 2, 0, 0]
        IndX = np.where([(self.atoms1[:, 0] < 1) |
                         (self.atoms1[:, 0] > (periodic_length - 1))])[1]
        IndY = np.where([self.atoms2[:, 0] > -1])[1]
        IndY_image = np.where([periodic_image[:, 0] <
                              (periodic_length + 1)])[1]
        X_new = self.atoms1[IndX]
        Y_new = np.concatenate((self.atoms2[IndY], periodic_image[IndY_image]))
        IndY_new = np.concatenate((IndY, IndY_image))
        x = np.arange(0, len(X_new), 1)
        y = np.arange(0, len(Y_new), 1)
        indice = (np.stack(np.meshgrid(x, y)).T).reshape(len(x) * len(y), 2)
        norms = norm(X_new[indice[:, 0]] - Y_new[indice[:, 1]], axis=1)
        indice_x = indice[norms < self.overD][:, 0]
        indice_y = indice[norms < self.overD][:, 1]
        X_del = X_new[indice_x]
        Y_del = Y_new[indice_y]

        if (len(X_del) != len(Y_del)):
            print("Warning! the number of deleted atoms "
                  "in the two grains are not equal!")
        return (X_del, Y_del, IndX[indice_x], IndY_new[indice_y])

    # ------------------------------------------------------------------ #
    #  New unified build / export API                                      #
    # ------------------------------------------------------------------ #

    def build(self, overlap=0.0, whichG='g1', dim=None):
        """
        Build the bicrystal supercell (without translations or file I/O).

        Parameters
        ----------
        overlap : float
            Overlap distance (fraction of lattice parameter). Atoms closer
            than this across the GB are removed.
        whichG : str
            Which grain to remove overlapping atoms from ('g1' or 'g2').
        dim : list of int, optional
            Supercell dimensions [dimX, dimY, dimZ]. Default [1,1,1].

        After calling this, use to_pymatgen(), to_ase(), write_lammps(),
        or write_vasp() to get the structure.
        """
        if dim is None:
            dim = [1, 1, 1]
        self.dim = np.array(dim, dtype=int)
        self.overD = float(overlap)
        self.whichG = whichG

        self.Expand_Super_cell()

        if self.overD > 0:
            xdel, _, x_indice, y_indice = self.Find_overlapping_Atoms()
            print("<<------ {} atoms are being removed! ------>>"
                  .format(len(xdel)))
            wg = self.whichG.lower()
            if wg == 'g1':
                self.atoms1 = np.delete(self.atoms1, x_indice, axis=0)
            elif wg == 'g2':
                self.atoms2 = np.delete(self.atoms2, y_indice, axis=0)
            else:
                raise ValueError("whichG must be 'g1' or 'g2'")

        # Record GB plane location (always at x=0 in this coordinate system)
        self.gb_plane_x = 0.0
        self._built = True
        return self

    def _get_cell_vectors(self):
        """Return the three cell vectors in Cartesian coords (Angstroms)."""
        dimx, dimy, dimz = self.dim
        Lx = 2 * norm(self.ortho1[:, 0]) * dimx * self.LatP
        Ly = norm(self.ortho1[:, 1]) * dimy * self.LatP
        Lz = norm(self.ortho1[:, 2]) * dimz * self.LatP
        return np.diag([Lx, Ly, Lz])

    def _get_cartesian_positions_and_grains(self):
        """
        Return (positions, grain_ids) in Angstroms.

        positions : (N, 3) array of Cartesian coordinates
        grain_ids : (N,) array of integers (1 for grain 1, 2 for grain 2)
        """
        if not self._built:
            raise RuntimeError(
                "Call build() before exporting. Example:\n"
                "  gb.CSL_Bicrystal_Atom_generator()\n"
                "  gb.build(overlap=0.0, dim=[1,1,1])"
            )
        X = self.atoms1.copy() * self.LatP
        Y = self.atoms2.copy() * self.LatP
        positions = np.concatenate((X, Y), axis=0)
        grain_ids = np.concatenate([
            np.ones(len(X), dtype=int),
            2 * np.ones(len(Y), dtype=int)
        ])
        return positions, grain_ids

    def get_grain_info(self):
        """
        Return a dict summarising the GB geometry.

        Keys
        ----
        gb_plane_x_angstrom : float
            x-coordinate of the GB plane in Angstroms (always 0.0 in
            the generated coordinate frame).
        periodic_image_x_angstrom : float
            x-coordinate of the periodic image of the GB.
        n_grain1, n_grain2 : int
            Number of atoms in each grain.
        cell : (3,3) array
            Cell vectors in Angstroms.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")
        cell = self._get_cell_vectors()
        return {
            'gb_plane_x_angstrom': 0.0,
            'periodic_image_x_angstrom': cell[0, 0] / 2.0,
            'n_grain1': len(self.atoms1),
            'n_grain2': len(self.atoms2),
            'cell': cell,
            'sigma': self.sigma,
            'axis': self.axis.tolist(),
            'gbplane': self.gbplane.tolist(),
            'theta_rad': self.theta,
        }

    # ------------------------------------------------------------------ #
    #  pymatgen export                                                     #
    # ------------------------------------------------------------------ #

    def to_pymatgen(self, element='X', elements=None):
        """
        Return a pymatgen Structure with grain labels as site properties.

        Parameters
        ----------
        element : str
            Chemical symbol for atoms (used when both grains are the same
            element). Default 'X' (dummy).
        elements : dict, optional
            Map grain id to element, e.g. {1: 'Al', 2: 'Cu'}.
            Overrides *element* if given.

        Returns
        -------
        pymatgen.core.Structure
            Structure with site property 'grain_id' (1 or 2) and
            'gb_plane_x' (the x position of the GB plane in Angstroms).
        """
        from pymatgen.core import Structure, Lattice

        positions, grain_ids = self._get_cartesian_positions_and_grains()
        cell = self._get_cell_vectors()
        lattice = Lattice(cell)

        if elements is not None:
            species = [elements[g] for g in grain_ids]
        else:
            species = [element] * len(grain_ids)

        # Shift positions so they sit inside the cell [0, L)
        # Grain 2 has negative x; shift by +Lx/2 so GB is at Lx/2
        Lx = cell[0, 0]
        positions[:, 0] += Lx / 2.0

        struct = Structure(
            lattice,
            species,
            positions,
            coords_are_cartesian=True,
            site_properties={
                'grain_id': grain_ids.tolist(),
            }
        )
        return struct

    # ------------------------------------------------------------------ #
    #  ASE export                                                          #
    # ------------------------------------------------------------------ #

    def to_ase(self, element='X', elements=None):
        """
        Return an ASE Atoms object with grain tags.

        Parameters
        ----------
        element : str
            Chemical symbol for atoms. Default 'X' (dummy).
        elements : dict, optional
            Map grain id to element, e.g. {1: 'Al', 2: 'Cu'}.

        Returns
        -------
        ase.Atoms
            Atoms object where atoms.arrays['grain_id'] holds grain
            membership (1 or 2). atoms.info['gb_plane_x'] stores the
            GB plane position in Angstroms.
            ASE tags are also set: tag=1 for grain 1, tag=2 for grain 2.
        """
        from ase import Atoms as ASEAtoms

        positions, grain_ids = self._get_cartesian_positions_and_grains()
        cell = self._get_cell_vectors()

        if elements is not None:
            symbols = [elements[g] for g in grain_ids]
        else:
            symbols = [element] * len(grain_ids)

        # Shift so everything is inside the cell
        Lx = cell[0, 0]
        positions[:, 0] += Lx / 2.0

        atoms = ASEAtoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True,
        )
        atoms.set_tags(grain_ids)
        atoms.arrays['grain_id'] = grain_ids
        atoms.info['gb_plane_x'] = Lx / 2.0
        atoms.info['sigma'] = self.sigma
        atoms.info['axis'] = self.axis.tolist()
        atoms.info['gbplane'] = self.gbplane.tolist()
        return atoms

    # ------------------------------------------------------------------ #
    #  Legacy file writers (kept for backward compatibility)               #
    # ------------------------------------------------------------------ #

    def write_vasp(self, filename=None, count=0):
        """Write structure to VASP POSCAR format."""
        if not self._built:
            raise RuntimeError("Call build() first.")
        if filename is None:
            plane = ''.join(str(x) for x in self.gbplane)
            overD = str(self.overD) if self.overD > 0 else str(None)
            filename = 'POS_G{}_{}'.format(plane, overD)
            if count > 0:
                filename += '_{}'.format(count)

        X = self.atoms1.copy() * self.LatP
        Y = self.atoms2.copy() * self.LatP
        cell = self._get_cell_vectors()

        Wf = np.concatenate((X, Y))

        with open(filename, 'w') as f:
            f.write('#POSCAR written by GB_code \n')
            f.write('1 \n')
            f.write('{0:.8f} 0.0 0.0 \n'.format(cell[0, 0]))
            f.write('0.0 {0:.8f} 0.0 \n'.format(cell[1, 1]))
            f.write('0.0 0.0 {0:.8f} \n'.format(cell[2, 2]))
            f.write('{} {} \n'.format(len(X), len(Y)))
            f.write('Cartesian\n')
            np.savetxt(f, Wf, fmt='%.8f %.8f %.8f')

    def write_lammps(self, filename=None, count=0):
        """Write structure to LAMMPS data format."""
        if not self._built:
            raise RuntimeError("Call build() first.")
        if filename is None:
            plane = ''.join(str(x) for x in self.gbplane)
            overD = str(self.overD) if self.overD > 0 else str(None)
            filename = 'input_G{}_{}'.format(plane, overD)
            if count > 0:
                filename += '_{}'.format(count)

        X = self.atoms1.copy() * self.LatP
        Y = self.atoms2.copy() * self.LatP
        NumberAt = len(X) + len(Y)
        dimx, dimy, dimz = self.dim

        xlo = -1 * np.round(norm(self.ortho1[:, 0]) * dimx * self.LatP, 8)
        xhi = np.round(norm(self.ortho1[:, 0]) * dimx * self.LatP, 8)
        ylo = 0.0
        yhi = np.round(norm(self.ortho1[:, 1]) * dimy * self.LatP, 8)
        zlo = 0.0
        zhi = np.round(norm(self.ortho1[:, 2]) * dimz * self.LatP, 8)

        Type1 = np.ones(len(X), int).reshape(1, -1)
        Type2 = 2 * np.ones(len(Y), int).reshape(1, -1)
        Counter = np.arange(1, NumberAt + 1).reshape(1, -1)

        W1 = np.concatenate((Type1.T, X), axis=1)
        W2 = np.concatenate((Type2.T, Y), axis=1)
        Wf = np.concatenate((W1, W2))
        FinalMat = np.concatenate((Counter.T, Wf), axis=1)

        with open(filename, 'w') as f:
            f.write('#Header \n \n')
            f.write('{} atoms \n \n'.format(NumberAt))
            f.write('2 atom types \n \n')
            f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
            f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
            f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
            f.write('Atoms \n \n')
            np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')

    # ------------------------------------------------------------------ #
    #  Legacy WriteGB (kept for backward compat with io_file workflow)     #
    # ------------------------------------------------------------------ #

    def WriteGB(self, overlap=0.0, rigid=False,
                dim1=1, dim2=1, dim3=1, file='LAMMPS',
                **kwargs):
        """
        Legacy method: parses arguments and writes the final structure.
        Possible keys: (whichG, a, b)
        """
        self.overD = float(overlap)
        self.trans = rigid
        self.dim = np.array([int(dim1), int(dim2), int(dim3)])
        self.File = file
        if self.overD > 0:
            try:
                self.whichG = kwargs['whichG']
            except KeyError:
                print('decide on whichG!')
                sys.exit()
            if self.trans:
                try:
                    a = int(kwargs['a'])
                    b = int(kwargs['b'])
                except KeyError:
                    print('Make sure the a and b integers are there!')
                    sys.exit()
            self.Expand_Super_cell()
            xdel, _, x_indice, y_indice = self.Find_overlapping_Atoms()
            print("<<------ {} atoms are being removed! ------>>"
                  .format(len(xdel)))

            if self.whichG == "G1" or self.whichG == "g1":
                self.atoms1 = np.delete(self.atoms1, x_indice, axis=0)
            elif self.whichG == "G2" or self.whichG == "g2":
                self.atoms2 = np.delete(self.atoms2, y_indice, axis=0)
            else:
                print("You must choose either 'g1', 'g2' ")
                sys.exit()

            self._built = True
            if not self.trans:
                count = 0
                print("<<------ 1 GB structure is being created! ------>>")
                if self.File == "LAMMPS":
                    self.Write_to_Lammps(count)
                elif self.File == "VASP":
                    self.Write_to_Vasp(count)
                else:
                    print("The output file must be either LAMMPS or VASP!")
            elif self.trans:
                self.Translate(a, b)

        elif self.overD == 0:
            if self.trans:
                try:
                    a = int(kwargs['a'])
                    b = int(kwargs['b'])
                except KeyError:
                    print('Make sure the a and b integers are there!')
                    sys.exit()
                print("<<------ 0 atoms are being removed! ------>>")
                self.Expand_Super_cell()
                self._built = True
                self.Translate(a, b)
            else:
                self.Expand_Super_cell()
                self._built = True
                count = 0
                print("<<------ 1 GB structure is being created! ------>>")
                if self.File == "LAMMPS":
                    self.Write_to_Lammps(count)
                elif self.File == "VASP":
                    self.Write_to_Vasp(count)
                else:
                    print("The output file must be either LAMMPS or VASP!")
        else:
            print('Overlap distance is not inputted incorrectly!')
            sys.exit()

    def Translate(self, a, b):
        """
        Legacy in-place translate + write.
        For the standalone utility, see inplane_shift.py.
        """
        tol = 0.001
        if (1 - cslgen.ang(self.gbplane, self.axis) < tol):
            M1, _ = cslgen.Create_minimal_cell_Method_1(
                     self.sigma, self.axis, self.R)
            D = (1 / self.sigma * cslgen.DSC_vec(self.basis, self.sigma, M1))
            Dvecs = cslgen.DSC_on_plane(D, self.gbplane)
            TransDvecs = np.round(dot(self.rot1, Dvecs), 7)
            shift1 = TransDvecs[:, 0] / 2
            shift2 = TransDvecs[:, 1] / 2
            a = b = 3
        else:
            if norm(self.ortho1[:, 1]) > norm(self.ortho1[:, 2]):
                shift1 = (1 / a) * (norm(self.ortho1[:, 1]) *
                                    np.array([0, 1, 0]))
                shift2 = (1 / b) * (norm(self.ortho1[:, 2]) *
                                    np.array([0, 0, 1]))
            else:
                shift1 = (1 / a) * (norm(self.ortho1[:, 2]) *
                                    np.array([0, 0, 1]))
                shift2 = (1 / b) * (norm(self.ortho1[:, 1]) *
                                    np.array([0, 1, 0]))
        print("<<------ {} GB structures are being created! ------>>"
              .format(int(a*b)))

        XX = self.atoms1
        count = 0
        if self.File == 'LAMMPS':
            for i in range(a):
                for j in range(b):
                    count += 1
                    shift = i * shift1 + j * shift2
                    atoms1_new = XX.copy() + shift
                    self.atoms1 = atoms1_new
                    self.Write_to_Lammps(count)
        elif self.File == 'VASP':
            for i in range(a):
                for j in range(b):
                    count += 1
                    shift = i * shift1 + j * shift2
                    atoms1_new = XX.copy() + shift
                    self.atoms1 = atoms1_new
                    self.Write_to_Vasp(count)
        else:
            print("The output file must be either LAMMPS or VASP!")

    # Legacy aliases for the old Write_to_* names
    def Write_to_Vasp(self, trans):
        """Legacy wrapper — delegates to write_vasp."""
        plane = str(self.gbplane[0])+str(self.gbplane[1])+str(self.gbplane[2])
        if self.overD > 0:
            overD = str(self.overD)
        else:
            overD = str(None)
        Trans = str(trans)
        X = self.atoms1.copy()
        Y = self.atoms2.copy()
        X_new = X * self.LatP
        Y_new = Y * self.LatP
        dimx, dimy, dimz = self.dim

        xlo = -1 * np.round(norm(self.ortho1[:, 0]) * dimx * self.LatP, 8)
        xhi = np.round(norm(self.ortho1[:, 0]) * dimx * self.LatP, 8)
        LenX = xhi - xlo
        ylo = 0.0
        yhi = np.round(norm(self.ortho1[:, 1]) * dimy * self.LatP, 8)
        LenY = yhi - ylo
        zlo = 0.0
        zhi = np.round(norm(self.ortho1[:, 2]) * dimz * self.LatP, 8)
        LenZ = zhi - zlo

        Wf = np.concatenate((X_new, Y_new))
        name = 'POS_G'
        with open(name + plane + '_' + overD + '_' + Trans, 'w') as f:
            f.write('#POSCAR written by GB_code \n')
            f.write('1 \n')
            f.write('{0:.8f} 0.0 0.0 \n'.format(LenX))
            f.write('0.0 {0:.8f} 0.0 \n'.format(LenY))
            f.write('0.0 0.0 {0:.8f} \n'.format(LenZ))
            f.write('{} {} \n'.format(len(X), len(Y)))
            f.write('Cartesian\n')
            np.savetxt(f, Wf, fmt='%.8f %.8f %.8f')

    def Write_to_Lammps(self, trans):
        """Legacy wrapper — delegates to write_lammps."""
        name = 'input_G'
        plane = str(self.gbplane[0])+str(self.gbplane[1])+str(self.gbplane[2])
        if self.overD > 0:
            overD = str(self.overD)
        else:
            overD = str(None)
        Trans = str(trans)
        X = self.atoms1.copy()
        Y = self.atoms2.copy()

        NumberAt = len(X) + len(Y)
        X_new = X * self.LatP
        Y_new = Y * self.LatP
        dimx, dimy, dimz = self.dim

        xlo = -1 * np.round(norm(self.ortho1[:, 0]) * dimx * self.LatP, 8)
        xhi = np.round(norm(self.ortho1[:, 0]) * dimx * self.LatP, 8)
        ylo = 0.0
        yhi = np.round(norm(self.ortho1[:, 1]) * dimy * self.LatP, 8)
        zlo = 0.0
        zhi = np.round(norm(self.ortho1[:, 2]) * dimz * self.LatP, 8)

        Type1 = np.ones(len(X_new), int).reshape(1, -1)
        Type2 = 2 * np.ones(len(Y_new), int).reshape(1, -1)
        Counter = np.arange(1, NumberAt + 1).reshape(1, -1)

        W1 = np.concatenate((Type1.T, X_new), axis=1)
        W2 = np.concatenate((Type2.T, Y_new), axis=1)
        Wf = np.concatenate((W1, W2))
        FinalMat = np.concatenate((Counter.T, Wf), axis=1)

        with open(name + plane + '_' + overD + '_' + Trans, 'w') as f:
            f.write('#Header \n \n')
            f.write('{} atoms \n \n'.format(NumberAt))
            f.write('2 atom types \n \n')
            f.write('{0:.8f} {1:.8f} xlo xhi \n'.format(xlo, xhi))
            f.write('{0:.8f} {1:.8f} ylo yhi \n'.format(ylo, yhi))
            f.write('{0:.8f} {1:.8f} zlo zhi \n\n'.format(zlo, zhi))
            f.write('Atoms \n \n')
            np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')

    def __str__(self):
        if self._built:
            return ("GB_character(sigma={}, axis={}, plane={}, "
                    "n_grain1={}, n_grain2={})".format(
                        self.sigma, self.axis, self.gbplane,
                        len(self.atoms1), len(self.atoms2)))
        return "GB_character(not yet built)"


def main():
    import yaml
    if len(sys.argv) == 2:
        io_file = sys.argv[1]
        file = open(io_file, 'r')
        in_params = yaml.safe_load(file)

        try:
            axis = np.array(in_params['axis'])
            m = int(in_params['m'])
            n = int(in_params['n'])
            basis = str(in_params['basis'])
            gbplane = np.array(in_params['GB_plane'])
            LatP = in_params['lattice_parameter']
            overlap = in_params['overlap_distance']
            whichG = in_params['which_g']
            rigid = in_params['rigid_trans']
            a = in_params['a']
            b = in_params['b']
            dim1, dim2, dim3 = in_params['dimensions']
            file = in_params['File_type']
        except Exception:
            print('Make sure the input arguments in io_file are '
                  'put in correctly!')
            sys.exit()

        gbI = GB_character()
        gbI.ParseGB(axis, basis, LatP, m, n, gbplane)
        gbI.CSL_Bicrystal_Atom_generator()

        if overlap > 0 and rigid:
            gbI.WriteGB(
                overlap=overlap, whichG=whichG, rigid=rigid, a=a,
                b=b, dim1=dim1, dim2=dim2, dim3=dim3, file=file
                )
        elif overlap > 0 and not rigid:
            gbI.WriteGB(
                overlap=overlap, whichG=whichG, rigid=rigid,
                dim1=dim1, dim2=dim2, dim3=dim3, file=file
                )
        elif overlap == 0 and rigid:
            gbI.WriteGB(
                overlap=overlap, rigid=rigid, a=a,
                b=b, dim1=dim1, dim2=dim2, dim3=dim3,
                file=file
                )
        elif overlap == 0 and not rigid:
            gbI.WriteGB(
                overlap=overlap, rigid=rigid,
                dim1=dim1, dim2=dim2, dim3=dim3, file=file
                )
    else:
        print(__doc__)
    return


if __name__ == '__main__':
    main()
