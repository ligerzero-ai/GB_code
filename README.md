# GB_code [![DOI](http://joss.theoj.org/papers/10.21105/joss.00900/status.svg)](https://doi.org/10.21105/joss.00900) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1433531.svg)](https://doi.org/10.5281/zenodo.1433531)

A Python package for creating orthogonal grain boundary supercells for atomistic calculations. Based on the coincident site lattice (CSL) formulation for cubic materials (sc, bcc, fcc, diamond).

Structures can be exported to [pymatgen](https://pymatgen.org/) `Structure`, [ASE](https://wiki.fysik.dtu.dk/ase/) `Atoms`, [LAMMPS](https://lammps.sandia.gov/) or [VASP](https://www.vasp.at/) file formats.

For more details please read the [paper](https://doi.org/10.21105/joss.00900).

## Installation

```bash
pip install .
```

With optional pymatgen/ASE support:

```bash
pip install ".[all]"      # both pymatgen and ASE
pip install ".[pymatgen]"  # pymatgen only
pip install ".[ase]"       # ASE only
```

Requirements: Python >= 3.5.1, numpy >= 1.14.

## Quick start (Python API)

The new v2 API separates structure **building** from **export**, so you can work with GB structures programmatically without writing intermediate files.

```python
from gb_code.gb_generator import GB_character

# 1. Set up the GB: Sigma-5 [1,0,0] fcc Al, (0,3,1) symmetric tilt
gb = GB_character()
gb.ParseGB(axis=[1, 0, 0], basis='fcc', LatP=4.05, m=2, n=1, gb=[0, 3, 1])

# 2. Build the bicrystal unit cells
gb.CSL_Bicrystal_Atom_generator()

# 3. Build the supercell (remove overlapping atoms, expand)
gb.build(overlap=0.0, whichG='g1', dim=[2, 1, 1])

# 4. Export to pymatgen or ASE
structure = gb.to_pymatgen(element='Al')   # pymatgen Structure
atoms = gb.to_ase(element='Al')            # ASE Atoms

# Or write files directly
gb.write_lammps('my_gb.lammps')
gb.write_vasp('POSCAR_gb')
```

Each exported structure carries grain membership labels (`grain_id` = 1 or 2), making it easy to distinguish the two grains for analysis or visualisation.

A complete worked example is in [`examples/minimal_gb_example.ipynb`](./examples/minimal_gb_example.ipynb).

## In-plane rigid body translations

To scan microscopic degrees of freedom (rigid body translations on the GB plane), use the standalone `inplane_shift` module:

```python
from gb_code.gb_generator import GB_character
from gb_code.inplane_shift import generate_shifts, apply_shift, get_all_shifted_structures

# Build GB as above ...
gb = GB_character()
gb.ParseGB([1, 0, 0], 'fcc', 4.05, 2, 1, [0, 3, 1])
gb.CSL_Bicrystal_Atom_generator()
gb.build(overlap=0.3, whichG='g1', dim=[2, 1, 1])

# Get all shift vectors on a 10x5 grid
shifts = generate_shifts(gb, a=10, b=5)

# Apply one shift and get a pymatgen Structure
struct = apply_shift(gb, shifts[0], output='pymatgen', element='Al')

# Or get all 50 shifted structures at once
all_structs = get_all_shifted_structures(gb, a=10, b=5, output='pymatgen', element='Al')
```

## Command-line usage (legacy workflow)

The original CLI workflow is still fully supported.

### Step 1: Find CSL sigma values

```
csl_generator.py 1 1 1 50

   List of possible CSLs for [1 1 1] axis sorted by Sigma
Sigma:     1  Theta:   0.00
Sigma:     3  Theta:  60.00
Sigma:     7  Theta:  38.21
Sigma:    13  Theta:  27.80
...
```

### Step 2: List GB planes for a given sigma

```
csl_generator.py 1 1 1 diamond 13

----------List of possible CSL planes for Sigma 13---------
 GB1-------------------GB2-------------------Type----------Number of Atoms
[ 2  1 -2]             [ 1  2 -2]             Mixed                  3744
[-1 -1 -1]             [-1 -1 -1]             Twist                  1248
[ 1  3 -4]             [-1  4 -3]             Symmetric Tilt         1248
...
```

This writes an `io_file` (YAML) with your chosen axis, basis and sigma.

### Step 3: Customise the io_file

```yaml
## input parameters for gb_generator.py ###
GB_plane: [2, 1, -2]
lattice_parameter: 4
overlap_distance: 0.3
which_g: g1
rigid_trans: no
a: 10
b: 5
dimensions: [1,1,1]
File_type: LAMMPS

# Written by csl_generator — do not change:
axis: [1, 1, 1]
m: 7
n: 1
basis: diamond
```

### Step 4: Generate the GB structure

```
gb_generator.py io_file
<<------ 32 atoms are being removed! ------>>
<<------ 1 GB structure is being created! ------>>
```

<img src="./exGB.png" width="50%">

## Key parameters

| Parameter | Description |
|---|---|
| `axis` | Rotation axis, e.g. `[1,0,0]`, `[1,1,0]`, `[1,1,1]` |
| `basis` | Crystal structure: `fcc`, `bcc`, `sc`, or `diamond` |
| `LatP` | Lattice parameter in Angstrom |
| `m`, `n` | Integers that define the CSL sigma and rotation angle |
| `gbplane` | GB plane (Miller indices), chosen from the CSL plane list |
| `overlap` | Fraction of lattice parameter; atoms closer than this across the GB are removed |
| `whichG` | Which grain to remove overlapping atoms from (`g1` or `g2`) |
| `dim` | Supercell dimensions `[dimX, dimY, dimZ]`. Make `dimX` large enough so the GB and its periodic image don't interact |

## Notes on energy minimisation

To find the minimum-energy GB structure, microscopic degrees of freedom must be explored:

1. **Atom removal**: set `overlap > 0` to delete atoms that are too close across the GB plane.
2. **Rigid body translations**: scan in-plane shifts using `inplane_shift`. A 10x5 grid (50 structures) is typically sufficient for fcc metals.

A typical LAMMPS minimisation protocol: conjugate gradient on atoms, then on the box, then on atoms again — followed by NVT annealing and damped dynamics.

## Jupyter notebooks

- [`examples/minimal_gb_example.ipynb`](./examples/minimal_gb_example.ipynb) — Minimal working example of the Python API
- [`Test/Usage_of_GB_code.ipynb`](./Test/Usage_of_GB_code.ipynb) — General usage with tips for locating GBs of interest
- [`Test/Dichromatic_pattern_CSL.ipynb`](./Test/Dichromatic_pattern_CSL.ipynb) — CSL construction for various purposes

## Citation

If you find this code useful, please cite:

> Hadian et al., (2018). GB code: A grain boundary generation code. *Journal of Open Source Software*, 3(29), 900. [https://doi.org/10.21105/joss.00900](https://doi.org/10.21105/joss.00900)

## License

[MIT](./LICENSE).
