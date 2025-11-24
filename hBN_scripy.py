import re
import subprocess
import sys
import os
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from scipy.linalg import block_diag
import sympy as sp

# ==============================================================================
# Main preprocessing pipeline for tight-binding model setup
# ==============================================================================
# This script orchestrates the complete preprocessing workflow:
# 1. Parse configuration file
# 2. Validate input data (sanity checks)
# 3. Generate space group representations
# 4. Complete orbital basis under symmetry
# 5. Find neighboring atoms
#
# The script chains multiple Python subscripts together, passing data via
# JSON through stdin/stdout.


# ==============================================================================
# STEP 1: Validate command line arguments
# ==============================================================================
argErrCode = 20
if (len(sys.argv) != 2):
    print("wrong number of arguments")
    print("example: python preprocessing.py /path/to/mc.conf")
    exit(argErrCode)

confFileName = str(sys.argv[1])

# ==============================================================================
# STEP 2: Parse configuration file
# ==============================================================================
# Run parse_conf.py to read and parse the configuration file
confResult = subprocess.run(
    ["python3", "./parse_files/parse_conf.py", confFileName],
    capture_output=True,
    text=True
)

# Check if the subprocess ran successfully
if confResult.returncode != 0:
    print("Error running parse_conf.py:")
    print(confResult.stderr)
    exit(confResult.returncode)

# Parse the JSON output from parse_conf.py
try:
    parsed_config = json.loads(confResult.stdout)

    # Display parsed configuration in a formatted way
    print("=" * 60)
    print("COMPLETE PARSED CONFIGURATION")
    print("=" * 60)

    # Print basic configuration parameters
    print(f"Name: {parsed_config['name']}")
    print(f"Dimensions: {parsed_config['dim']}")
    print(f"Spin: {parsed_config['spin']}")
    print(f"Neighbors: {parsed_config['neighbors']}")
    print(f"Atom Type Number: {parsed_config['atom_type_num']}")
    print(f"Lattice Type: {parsed_config['lattice_type']}")
    print(f"Space Group: {parsed_config['space_group']}")

    # Print space group origin (fractional coordinates)
    print(f"Space Group Origin: [{', '.join(map(str, parsed_config['space_group_origin']))}]")

    # Print lattice basis vectors (primitive cell)
    print("Lattice Basis:")
    for i, vector in enumerate(parsed_config['lattice_basis']):
        print(f"  Vector {i + 1}: [{', '.join(map(str, vector))}]")

    # Print space group basis vectors
    print("Space Group Basis:")
    for i, vector in enumerate(parsed_config['space_group_basis']):
        print(f"  Vector {i + 1}: [{', '.join(map(str, vector))}]")

    # Print atom types and their orbital information
    print("\nAtom Types:")
    for atom_type, info in parsed_config['atom_types'].items():
        print(f"  {atom_type}:")
        print(f"    Count: {info['count']}")
        print(f"    Orbitals: {info['orbitals']}")

    # Print atom positions in the unit cell
    print(f"\nAtom Positions (Total: {len(parsed_config['atom_positions'])}):")
    for i, pos in enumerate(parsed_config['atom_positions']):
        print(f"  Position {i + 1}:")
        print(f"    Name: {pos['position_name']}")
        print(f"    Atom Type: {pos['atom_type']}")
        print(f"    fractional_coordinates: [{', '.join(map(str, pos['fractional_coordinates']))}]")

except json.JSONDecodeError as e:
    print("Error parsing JSON output from parse_conf.py:")
    print(f"JSON Error: {e}")
    print("Raw output was:")
    print(confResult.stdout)
    exit(1)

# Convert parsed_config to JSON string for passing to other subprocesses
config_json = json.dumps(parsed_config)

# ==============================================================================
# STEP 3: Run sanity checks on parsed configuration
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING SANITY CHECK")
print("=" * 60)

# Run sanity_check.py and pass the JSON data via stdin
sanity_result = subprocess.run(
    ["python3", "./parse_files/sanity_check.py"],
    input=config_json,
    capture_output=True,
    text=True
)

print(f"Exit code: {sanity_result.returncode}")

# Check sanity check results
if sanity_result.returncode != 0:
    print("Sanity check failed!")
    print(f"return code={sanity_result.returncode}")
    print("Error output:")
    print(sanity_result.stderr)
    exit(sanity_result.returncode)
else:
    print("Sanity check passed!")
    print("Output:")
    print(sanity_result.stdout)

# ==============================================================================
# STEP 4: Generate space group representations
# ==============================================================================
print("\n" + "=" * 60)
print("COMPUTING SPACE GROUP REPRESENTATIONS")
print("=" * 60)

# Run generate_space_group_representations.py
sgr_result = subprocess.run(
    ["python3", "./symmetry/generate_space_group_representations.py"],
    input=config_json,
    capture_output=True,
    text=True
)

print(f"Exit code: {sgr_result.returncode}")

# Check if space group representations were generated successfully
if sgr_result.returncode != 0:
    print("Space group representations generation failed!")
    print(f"return code={sgr_result.returncode}")
    print("Error output:")
    print(sgr_result.stderr)
    print("Standard output:")
    print(sgr_result.stdout)
    exit(sgr_result.returncode)
else:
    print("Space group representations generated successfully!")

    # Parse the JSON output
    try:
        space_group_representations = json.loads(sgr_result.stdout)

        print("\n" + "=" * 60)
        print("SPACE GROUP REPRESENTATIONS SUMMARY")
        print("=" * 60)

        # Get number of space group operations
        num_operations = len(space_group_representations["space_group_matrices"])
        print(f"Number of space group operations: {num_operations}")

        # Print space group origin in different coordinate systems
        print("\nSpace Group Origin:")
        origin_cart = space_group_representations["space_group_origin_cartesian"]
        origin_frac_prim = space_group_representations["space_group_origin_fractional_primitive"]
        print(
            f"  Bilbao (fractional in space group basis): [{', '.join(map(str, parsed_config['space_group_origin']))}]")
        print(f"  Cartesian: [{', '.join(f'{x:.6f}' for x in origin_cart)}]")
        print(f"  Fractional (primitive cell basis): [{', '.join(f'{x:.6f}' for x in origin_frac_prim)}]")

        # Extract orbital representations (s, p, d, f)
        repr_s, repr_p, repr_d, repr_f = space_group_representations["repr_s_p_d_f"]

        # Print dimensions of representation matrices
        print(f"\nOrbital Representations:")
        print(f"  s orbitals: {len(repr_s)} operations × {len(repr_s[0])}×{len(repr_s[0][0])} matrices")
        print(f"  p orbitals: {len(repr_p)} operations × {len(repr_p[0])}×{len(repr_p[0][0])} matrices")
        print(f"  d orbitals: {len(repr_d)} operations × {len(repr_d[0])}×{len(repr_d[0][0])} matrices")
        print(f"  f orbitals: {len(repr_f)} operations × {len(repr_f[0])}×{len(repr_f[0][0])} matrices")

        # Convert to NumPy arrays for further processing
        space_group_matrices = np.array(space_group_representations["space_group_matrices"])
        space_group_matrices_cartesian = np.array(space_group_representations["space_group_matrices_cartesian"])
        space_group_matrices_primitive = np.array(space_group_representations["space_group_matrices_primitive"])
        #  space group matrices in Cartesian coordinates , a list
        space_group_bilbao_cart = [np.array(item) for item in space_group_matrices_cartesian]
        repr_s_np = np.array(repr_s)
        repr_p_np = np.array(repr_p)
        repr_d_np = np.array(repr_d)
        repr_f_np = np.array(repr_f)

        print("\nSpace group representations loaded and converted to NumPy arrays.")
        print(f"Available matrices:")
        print(f"  - space_group_matrices: {space_group_matrices.shape}")
        print(f"  - space_group_matrices_cartesian: {space_group_matrices_cartesian.shape}")
        print(f"  - space_group_matrices_primitive: {space_group_matrices_primitive.shape}")
        print(f"  - s orbital representations: {repr_s_np.shape}")
        print(f"  - p orbital representations: {repr_p_np.shape}")
        print(f"  - d orbital representations: {repr_d_np.shape}")
        print(f"  - f orbital representations: {repr_f_np.shape}")

    except json.JSONDecodeError as e:
        print("Error parsing JSON output from space group representations:")
        print(f"JSON Error: {e}")
        print("Raw output was:")
        print(sgr_result.stdout)
        exit(1)

    except KeyError as e:
        print(f"Missing key in space group representations output: {e}")
        print("Available keys:", list(
            space_group_representations.keys()) if 'space_group_representations' in locals() else "Could not parse JSON")
        exit(1)

# ==============================================================================
# STEP 5: Define orbital mapping for 78-dimensional orbital space
# ==============================================================================
# Maps orbital names (like '3dxy') to their index in the orbital vector
# Total: 78 orbitals from 1s to 7f
orbital_map = {
    # n=1: 1s (index 0)
    '1s': 0,

    # n=2: 2s, 2p (indices 1-4)
    '2s': 1,
    '2px': 2, '2py': 3, '2pz': 4,

    # n=3: 3s, 3p, 3d (indices 5-13)
    '3s': 5,
    '3px': 6, '3py': 7, '3pz': 8,
    '3dxy': 9, '3dyz': 10, '3dxz': 11, '3dx2-y2': 12, '3dz2': 13,

    # n=4: 4s, 4p, 4d, 4f (indices 14-29)
    '4s': 14,
    '4px': 15, '4py': 16, '4pz': 17,
    '4dxy': 18, '4dyz': 19, '4dxz': 20, '4dx2-y2': 21, '4dz2': 22,
    '4fxyz': 23, '4fz3': 24, '4fxz2': 25, '4fyz2': 26,
    '4fz(x2-y2)': 27, '4fx(x2-3y2)': 28, '4fy(3x2-y2)': 29,

    # n=5: 5s, 5p, 5d, 5f (indices 30-45)
    '5s': 30,
    '5px': 31, '5py': 32, '5pz': 33,
    '5dxy': 34, '5dyz': 35, '5dxz': 36, '5dx2-y2': 37, '5dz2': 38,
    '5fxyz': 39, '5fz3': 40, '5fxz2': 41, '5fyz2': 42,
    '5fz(x2-y2)': 43, '5fx(x2-3y2)': 44, '5fy(3x2-y2)': 45,

    # n=6: 6s, 6p, 6d, 6f (indices 46-61)
    '6s': 46,
    '6px': 47, '6py': 48, '6pz': 49,
    '6dxy': 50, '6dyz': 51, '6dxz': 52, '6dx2-y2': 53, '6dz2': 54,
    '6fxyz': 55, '6fz3': 56, '6fxz2': 57, '6fyz2': 58,
    '6fz(x2-y2)': 59, '6fx(x2-3y2)': 60, '6fy(3x2-y2)': 61,

    # n=7: 7s, 7p, 7d, 7f (indices 62-77)
    '7s': 62,
    '7px': 63, '7py': 64, '7pz': 65,
    '7dxy': 66, '7dyz': 67, '7dxz': 68, '7dx2-y2': 69, '7dz2': 70,
    '7fxyz': 71, '7fz3': 72, '7fxz2': 73, '7fyz2': 74,
    '7fz(x2-y2)': 75, '7fx(x2-3y2)': 76, '7fy(3x2-y2)': 77,
}

# ==============================================================================
# STEP 6: Complete orbital basis under symmetry operations
# ==============================================================================
print("\n" + "=" * 60)
print("COMPLETING ORBITALS UNDER SYMMETRY")
print("=" * 60)

# Combine parsed_config and space_group_representations
combined_input = {
    "parsed_config": parsed_config,
    "space_group_representations": space_group_representations
}

# Convert to JSON for subprocess
combined_input_json = json.dumps(combined_input)

# Run complete_orbitals.py
completing_result = subprocess.run(
    ["python3", "./symmetry/complete_orbitals.py"],
    input=combined_input_json,
    capture_output=True,
    text=True
)

# Check if orbital completion succeeded
if completing_result.returncode != 0:
    print("Orbital completion failed!")
    print(f"Return code: {completing_result.returncode}")
    print("Error output:")
    print(completing_result.stderr)
    exit(completing_result.returncode)

# Parse the output
try:
    orbital_completion_data = json.loads(completing_result.stdout)

    print("Orbital completion successful!")

    # Display which orbitals were added by symmetry
    print("\n" + "-" * 40)
    print("ORBITALS ADDED BY SYMMETRY:")
    print("-" * 40)

    added_orbitals = orbital_completion_data["added_orbitals"]
    if any(added_orbitals.values()):
        for atom_name, orbitals in added_orbitals.items():
            if orbitals:
                print(f"  {atom_name}: {', '.join(orbitals)}")
    else:
        print("  No additional orbitals needed - input was already complete")

    # Display final active orbitals for each atom
    print("\n" + "-" * 40)
    print("FINAL ACTIVE ORBITALS PER ATOM:")
    print("-" * 40)

    updated_vectors = orbital_completion_data["updated_orbital_vectors"]
    orbital_map_reverse = {v: k for k, v in orbital_map.items()}  # Reverse lookup

    for atom_name, vector in updated_vectors.items():
        # Find indices where orbital is active (value = 1)
        active_indices = [i for i, val in enumerate(vector) if val == 1]
        # Convert indices back to orbital names
        active_orbital_names = [orbital_map_reverse.get(idx, f"unknown_{idx}") for idx in active_indices]
        print(f"  {atom_name} ({len(active_orbital_names)} orbitals): {', '.join(active_orbital_names)}")

    # Display symmetry representation information
    print("\n" + "-" * 40)
    print("SYMMETRY REPRESENTATIONS ON ACTIVE ORBITALS:")
    print("-" * 40)

    representations = orbital_completion_data["representations_on_active_orbitals"]
    for atom_name, repr_matrices in representations.items():
        if repr_matrices:
            repr_array = np.array(repr_matrices)
            print(
                f"  {atom_name}: {repr_array.shape[0]} operations, {repr_array.shape[1]}×{repr_array.shape[2]} matrices")

    # Update parsed_config with completed orbitals
    for atom_pos in parsed_config['atom_positions']:
        atom_name = atom_pos['position_name']
        atom_type = atom_pos['atom_type']

        # Get the updated orbital vector for this atom
        if atom_name in updated_vectors:
            vector = updated_vectors[atom_name]
            active_indices = [i for i, val in enumerate(vector) if val == 1]
            active_orbital_names = [orbital_map_reverse.get(idx, f"unknown_{idx}") for idx in active_indices]

            # Update atom_types with completed orbital list
            parsed_config['atom_types'][atom_type]['orbitals'] = active_orbital_names
            parsed_config['atom_types'][atom_type]['orbitals_completed'] = True

    # Store completion results for later use
    orbital_completion_results = {
        "status": "completed",
        "added_orbitals": added_orbitals,
        "orbital_vectors": updated_vectors,
        "representations_on_active_orbitals": representations,
    }


except json.JSONDecodeError as e:
    print("Error parsing JSON output from complete_orbitals.py:")
    print(f"JSON Error: {e}")
    print("Raw output:")
    print(completing_result.stdout)
    print("Error output:")
    print(completing_result.stderr)
    exit(1)

except KeyError as e:
    print(f"Missing key in orbital completion output: {e}")
    print("Available keys:",
          list(orbital_completion_data.keys()) if 'orbital_completion_data' in locals() else "Could not parse JSON")
    exit(1)

except Exception as e:
    print(f"Unexpected error processing orbital completion: {e}")
    print("Type:", type(e).__name__)
    exit(1)

print("\n" + "=" * 60)
print("ORBITAL COMPLETION FINISHED")
print("=" * 60)

# ==============================================================================
# Helper function: orbital_to_submatrix
# ==============================================================================
def orbital_to_submatrix(orbitals, Vs, Vp, Vd, Vf):
    """
    Extract submatrix from full orbital representation for specific orbitals

    Args:
        orbitals: List of orbital names (e.g., ['2s', '2px', '2py', '2pz'])
        Vs, Vp, Vd, Vf: Representation matrices for s, p, d, f orbitals

    Returns:
        numpy array: Submatrix for the specified orbitals
    """
    full_orbitals = [
        's',
        'px', 'py', 'pz',
        'dxy', 'dyz', 'dzx', 'd(x2-y2)', 'd(3z2-r2)',
        'fz3', 'fxz3', 'fyz3', 'fxyz', 'fz(x2-y2)', 'fx(x2-3y2)', 'fy(3x2-y2)'
    ]
    # Remove leading numbers from orbitals (e.g., '2s' -> 's', '2pz' -> 'pz')
    orbital_types = []
    for orb in orbitals:
        # Remove all leading digits
        orbital_type = orb.lstrip('0123456789')
        orbital_types.append(orbital_type)
    # Sort orbitals by their position in full_orbitals
    sorted_orbital_types = sorted(orbital_types, key=lambda orb: full_orbitals.index(orb))
    # Get the indices in full_orbitals
    orbital_indices = [full_orbitals.index(orb) for orb in sorted_orbital_types]
    # Build full representation matrix
    hopping_matrix_full = block_diag(Vs, Vp, Vd, Vf)
    # Extract submatrix for the specific orbitals
    V_submatrix = hopping_matrix_full[np.ix_(orbital_indices, orbital_indices)]
    return V_submatrix



# ==============================================================================
# atomIndex class with orbital representations
# ==============================================================================

def frac_to_cartesian(cell, frac_coord, basis):
    """Convert fractional coordinates to Cartesian"""
    n0, n1, n2 = cell
    f0, f1, f2 = frac_coord
    a0, a1, a2 = basis
    return (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2


class atomIndex:
    def __init__(self, cell, frac_coord, atom_name, basis, parsed_config=None,
                 repr_s_np=None, repr_p_np=None, repr_d_np=None, repr_f_np=None):
        """
        Initialize an atom with position, orbital, and representation information

        Args:
            cell: [n0, n1, n2] unit cell indices
            frac_coord: [f0, f1, f2] fractional coordinates
            atom_name: atom type name (e.g., 'B', 'N')
            basis: lattice basis vectors [a0, a1, a2]
            parsed_config: configuration dict containing orbital information (optional)
            repr_s_np, repr_p_np, repr_d_np, repr_f_np: representation matrices for s,p,d,f orbitals
        """
        # Deep copy mutable inputs
        self.n0 = deepcopy(cell[0])
        self.n1 = deepcopy(cell[1])
        self.n2 = deepcopy(cell[2])
        self.atom_name = atom_name  # string is immutable
        self.frac_coord = deepcopy(frac_coord)
        self.basis = deepcopy(basis)
        self.parsed_config=deepcopy(parsed_config)

        # Calculate Cartesian coordinates using frac_to_cartesian helper
        # The basis vectors a0, a1, a2 are primitive lattice vectors expressed in
        # Cartesian coordinates using Bilbao's origin, so the result is
        # Cartesian coordinates using Bilbao's origin
        self.cart_coord = frac_to_cartesian(cell, frac_coord, basis)

        # Store orbital information if config is provided
        if parsed_config is not None and atom_name in parsed_config['atom_types']:
            self.orbitals = deepcopy(parsed_config['atom_types'][atom_name]['orbitals'])
            self.num_orbitals = len(self.orbitals)
        else:
            self.orbitals = None
            self.num_orbitals = 0

        # Deep copy representation matrices
        self.repr_s_np = deepcopy(repr_s_np) if repr_s_np is not None else None
        self.repr_p_np = deepcopy(repr_p_np) if repr_p_np is not None else None
        self.repr_d_np = deepcopy(repr_d_np) if repr_d_np is not None else None
        self.repr_f_np = deepcopy(repr_f_np) if repr_f_np is not None else None

        # Pre-compute representation matrices for this atom's orbitals if available
        self.orbital_representations = None
        if (self.orbitals is not None and repr_s_np is not None):
            self._compute_orbital_representations()

    def _compute_orbital_representations(self):
        """
        Pre-compute orbital representation matrices for all space group operations
        Returns a list where each element is the representation matrix for one operation
        """
        num_operations = len(self.repr_s_np)
        self.orbital_representations = []
        for op_idx in range(num_operations):
            Vs = self.repr_s_np[op_idx]
            Vp = self.repr_p_np[op_idx]
            Vd = self.repr_d_np[op_idx]
            Vf = self.repr_f_np[op_idx]
            # Get submatrix for this atom's specific orbitals
            V_submatrix = orbital_to_submatrix(self.orbitals, Vs, Vp, Vd, Vf)
            self.orbital_representations.append(V_submatrix)

    def get_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix for a specific space group operation

        Args:
            operation_idx: index of the space group operation

        Returns:
            numpy array: representation matrix for this atom's orbitals
        """
        if self.orbital_representations is None:
            raise ValueError(f"Orbital representations not computed for atom {self.atom_name}")

        if operation_idx >= len(self.orbital_representations):
            raise IndexError(f"Operation index {operation_idx} out of range")

        return self.orbital_representations[operation_idx]

    def get_sympy_representation_matrix(self, operation_idx):
        """
        Get the orbital representation matrix as a sympy Matrix

        Args:
            operation_idx: index of the space group operation

        Returns:
            sympy.Matrix: representation matrix for this atom's orbitals
        """
        return sp.Matrix(self.get_representation_matrix(operation_idx))

    def __str__(self):
        """String representation for print()"""
        orbital_info = f", Orbitals: {self.num_orbitals}" if self.orbitals else ""
        repr_info = f", Repr: ✓" if self.orbital_representations is not None else ""
        return (f"Atom: {self.atom_name}, "
                f"Cell: [{self.n0}, {self.n1}, {self.n2}], "
                f"Frac: {self.frac_coord}, "
                f"Cart: {self.cart_coord}"
                f"{orbital_info}{repr_info}")

    def __repr__(self):
        """Detailed representation for debugging"""
        return (f"atomIndex(cell=[{self.n0}, {self.n1}, {self.n2}], "
                f"frac_coord={self.frac_coord}, "
                f"atom_name='{self.atom_name}', "
                f"orbitals={self.num_orbitals})")

    def get_orbital_names(self):
        """Get list of orbital names for this atom"""
        return self.orbitals if self.orbitals is not None else []

    def has_orbital(self, orbital_name):
        """Check if this atom has a specific orbital"""
        if self.orbitals is None:
            return False
        # Handle both '2s' and 's' format
        orbital_type = orbital_name.lstrip('0123456789')
        return any(orb.lstrip('0123456789') == orbital_type for orb in self.orbitals)


# ==============================================================================
# Helper functions for atom operations
# ==============================================================================
def compute_dist(center_atom, unit_cell_atoms, search_range=8, radius=None, search_dim=2):
    """
    Find all atoms within a specified radius of a center atom by searching neighboring cells.
    Returns constructed atomIndex objects for all neighbors found.

    Args:
        center_atom: atomIndex object for the center atom
        unit_cell_atoms: list of atomIndex objects in the reference unit cell [0,0,0]
        search_range: how many cells to search in each direction (default: 1)
        radius: cutoff distance in Cartesian coordinates (default: None means all atoms)
        search_dim: dimension to search (1, 2, or 3) (default: 3)
            - 1: search along n0 only
            - 2: search along n0 and n1
            - 3: search along n0, n1, and n2

    Returns:
        list: atomIndex objects within the specified radius, sorted by distance
    """
    neighbor_atoms = []
    center_cart = center_atom.cart_coord
    lattice_basis = center_atom.basis

    # Determine search ranges based on search_dim
    if search_dim == 1:
        n0_range = range(-search_range, search_range + 1)
        n1_range = [0]
        n2_range = [0]
    elif search_dim == 2:
        n0_range = range(-search_range, search_range + 1)
        n1_range = range(-search_range, search_range + 1)
        n2_range = [0]
    else:  # search_dim == 3
        n0_range = range(-search_range, search_range + 1)
        n1_range = range(-search_range, search_range + 1)
        n2_range = range(-search_range, search_range + 1)

    # Search through neighboring cells
    for n0 in n0_range:
        for n1 in n1_range:
            for n2 in n2_range:
                cell = [n0, n1, n2]

                # Check each atom in the unit cell
                for unit_atom in unit_cell_atoms:
                    # Compute Cartesian coordinates for this atom in the proposed cell
                    candidate_cart = frac_to_cartesian(cell, unit_atom.frac_coord, lattice_basis)

                    # Calculate distance
                    dist = np.linalg.norm(candidate_cart - center_cart)

                    # Only construct atom if it passes the distance check
                    if radius is None or dist <= radius:
                        # Create atomIndex for this atom in the current cell with deep copies
                        neighbor_atom = atomIndex(
                            cell=deepcopy(cell),
                            frac_coord=deepcopy(unit_atom.frac_coord),
                            atom_name=unit_atom.atom_name,  # string is immutable, safe
                            basis=deepcopy(lattice_basis),
                            parsed_config=deepcopy(unit_atom.parsed_config),
                            repr_s_np=deepcopy(unit_atom.repr_s_np) if unit_atom.repr_s_np is not None else None,
                            repr_p_np=deepcopy(unit_atom.repr_p_np) if unit_atom.repr_p_np is not None else None,
                            repr_d_np=deepcopy(unit_atom.repr_d_np) if unit_atom.repr_d_np is not None else None,
                            repr_f_np=deepcopy(unit_atom.repr_f_np) if unit_atom.repr_f_np is not None else None
                        )

                        # Deep copy orbital information from unit cell atom
                        neighbor_atom.orbitals = deepcopy(unit_atom.orbitals)
                        neighbor_atom.num_orbitals = unit_atom.num_orbitals
                        neighbor_atom.orbital_representations = deepcopy(unit_atom.orbital_representations)

                        neighbor_atoms.append((dist, neighbor_atom))

    # Sort by distance and return only the atomIndex objects
    neighbor_atoms.sort(key=lambda x: x[0])
    return [atom for dist, atom in neighbor_atoms]






def find_identity_operation(space_group_bilbao_cart, tolerance=1e-9, verbose=True):
    """
    Find the index of the identity operation in space group matrices.

    The identity operation has:
    - Rotation part: 3×3 identity matrix
    - Translation part: zero vector

    Args:
        space_group_bilbao_cart: List or array of 4×4 or 3×4 space group matrices [R|t]
                                 in Cartesian coordinates
        tolerance: Numerical tolerance for comparison (default: 1e-9)
        verbose: Whether to print status messages (default: True)

    Returns:
        int: Index of the identity operation

    Raises:
        ValueError: If identity operation is not found
    """
    identity_idx = None

    for idx, group_mat in enumerate(space_group_bilbao_cart):
        # Check if rotation part is identity and translation part is zero
        if np.allclose(group_mat[:3, :3], np.eye(3), atol=tolerance) and \
                np.allclose(group_mat[:3, 3], 0, atol=tolerance):
            identity_idx = idx
            if verbose:
                print(f"Identity operation found at index {identity_idx}")
            break

    if identity_idx is None:
        error_msg = "Identity operation not found in space_group_bilbao_cart!"
        if verbose:
            print(f"WARNING: {error_msg}")
        raise ValueError(error_msg)

    return identity_idx


# ==============================================================================
# hopping class
# ==============================================================================
class hopping:
    """
    Represents a single hopping term between two atoms.
    """

    def __init__(self, to_atom, from_atom, class_id, operation_idx, rotation_matrix, translation_vector):
        self.to_atom = deepcopy(to_atom)  # Deep copy of Atom object (destination)
        self.from_atom = deepcopy(from_atom)  # Deep copy of Atom object (source)
        self.class_id = class_id  # Equivalence class identifier (immutable)
        self.operation_idx = operation_idx  # Which space group operation transforms parent to this hopping (immutable)
        self.rotation_matrix = deepcopy(rotation_matrix)  # Deep copy of 3×3 rotation matrix R
        self.translation_vector = deepcopy(translation_vector)  # Deep copy of 3D translation vector b
        self.distance = None  # Will be computed
        self.T = None  # Hopping matrix (sympy Matrix)
        # self.T_full=None

    def conjugate(self):
        return [deepcopy(self.from_atom), deepcopy(self.to_atom)]

    def compute_distance(self):
        """
        Compute the Euclidean distance for this hopping.
        """
        pos_to = self.to_atom.cart_coord
        pos_from = self.from_atom.cart_coord

        # Real space position difference
        delta_pos = pos_to - pos_from

        self.distance = np.linalg.norm(delta_pos, ord=2)

    def __repr__(self):
        return (f"hopping(to={self.to_atom.atom_name}, from={self.from_atom.atom_name}, "
                f"class={self.class_id}, op={self.operation_idx}, "
                f"distance={self.distance:.4f if self.distance is not None else 'None'})")

# ==============================================================================
# vertex class
# ==============================================================================
class vertex():
    def __init__(self, hopping, type, identity_idx, parent=None):
        self.hopping = deepcopy(hopping)
        self.type = type  # "linear" or "hermitian" for child, None for root (string is immutable)
        self.is_root = (hopping.operation_idx == identity_idx)  # boolean is immutable
        self.children = []  # List of child vertex objects
        self.parent = parent  # Reference to parent vertex (None for root) - we DON'T deep copy to avoid circular reference issues

    def add_child(self, child_vertex):
        """Add a child vertex to this vertex"""
        self.children.append(child_vertex)
        child_vertex.parent = self  # Set this vertex as the child's parent

    def __repr__(self):
        root_str = "ROOT" if self.is_root else "CHILD"
        parent_str = "None" if self.parent is None else f"op={self.parent.hopping.operation_idx}"
        return (f"vertex(type={self.type}, {root_str}, "
                f"op={self.hopping.operation_idx}, "
                f"parent={parent_str}, "
                f"children={len(self.children)})")
# ==============================================================================
# Helper function for symmetry operations
# ==============================================================================

def get_rotation_translation(space_group_bilbao_cart, operation_idx):
    """
    Extract rotation/reflection matrix R and translation vector t from a space group operation.

    The space group operation is in the form [R|t], represented as a 3×4 matrix:
        [R | t] = [R00 R01 R02 | t0]
                  [R10 R11 R12 | t1]
                  [R20 R21 R22 | t2]

    The operation transforms a position vector r as: r' = R @ r + t

    Args:
        space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
                                 using Bilbao origin (shape: num_ops × 3 × 4)
        operation_idx: Index of the space group operation

    Returns:
        tuple: (R, t)
            - R (ndarray): 3×3 rotation/reflection matrix
            - t (ndarray): 3D translation vector
    """
    operation = space_group_bilbao_cart[operation_idx]
    R = operation[:3, :3]  # Rotation/reflection part
    t = operation[:3, 3]  # Translation part

    return R, t

def is_lattice_vector(vector, lattice_basis, tolerance=1e-6):
    """
    Check if a vector can be expressed as an integer linear combination of lattice basis vectors.

    A vector v is a lattice vector if:
        v = n0*a0 + n1*a1 + n2*a2
    where n0, n1, n2 are integers and a0, a1, a2 are primitive lattice basis vectors.

    Args:
        vector: 3D vector to check (Cartesian coordinates)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using Bilbao origin
        tolerance: Numerical tolerance for checking if coefficients are integers (default: 1e-6)

    Returns:
        tuple: (is_lattice, n_vector)
            - is_lattice (bool): True if vector is a lattice vector
            - n_vector (ndarray): The integer coefficients [n0, n1, n2]
    """
    # Extract basis vectors (each row is a basis vector)
    a0, a1, a2 = lattice_basis

    # Create matrix with basis vectors as columns
    lattice_matrix = np.column_stack([a0, a1, a2])

    # Solve: vector = lattice_matrix @ [n0, n1, n2]
    # So: [n0, n1, n2] = lattice_matrix^(-1) @ vector
    n_vector_float = np.linalg.solve(lattice_matrix, vector)

    # Round to nearest integers
    n_vector = np.round(n_vector_float)

    # Check if coefficients are integers (within tolerance)
    is_lattice = np.allclose(n_vector_float, n_vector, atol=tolerance)

    return is_lattice, n_vector


def check_center_invariant(center_atom, operation_idx, space_group_bilbao_cart,
                           lattice_basis, tolerance=1e-6, verbose=False):
    """
    Check if a center atom is invariant under a specific space group operation.

    An atom is invariant if the symmetry operation maps it to itself, possibly
    translated by a lattice vector. The actual operation is:
        r' = R @ r + t + n0*a0 + n1*a1 + n2*a2
    where n0, n1, n2 are integers and a0, a1, a2 are primitive lattice basis vectors.

    For invariance, we need: r' = r, which means:
        R @ r + t + n0*a0 + n1*a1 + n2*a2 = r
        => (R - I) @ r + t = -(n0*a0 + n1*a1 + n2*a2)

    Args:
        center_atom: atomIndex object representing the center atom
        operation_idx: Index of the space group operation to check
        space_group_bilbao_cart: List of space group matrices in Cartesian coordinates
                                 using Bilbao origin (shape: num_ops × 3 × 4)
        lattice_basis: Primitive lattice basis vectors (3×3 array, each row is a basis vector)
                      expressed in Cartesian coordinates using Bilbao origin
        tolerance: Numerical tolerance for comparison (default: 1e-6)
        verbose: Whether to print debug information (default: False)

    Returns:
        bool: True if the atom is invariant under the operation, False otherwise
    """
    # Get the symmetry operation [R|t]
    R, t = get_rotation_translation(space_group_bilbao_cart, operation_idx)

    # Get center atom's Cartesian position (using Bilbao origin)
    r_center = center_atom.cart_coord

    # Compute (R - I) @ r + t
    # This should equal -(n0*a0 + n1*a1 + n2*a2) for some integers n0, n1, n2
    lhs = (R - np.eye(3)) @ r_center + t

    # Check if -lhs is a lattice vector
    is_invariant, n_vector = is_lattice_vector(-lhs, lattice_basis, tolerance)

    if verbose:
        a0, a1, a2 = lattice_basis
        print(f"\nChecking invariance for operation {operation_idx}:")
        print(f"  Basis vectors:")
        print(f"    a0 = {a0}")
        print(f"    a1 = {a1}")
        print(f"    a2 = {a2}")
        print(f"  Center position r: {r_center}")
        print(f"  Rotation R:")
        print(f"    {R}")
        print(f"  Translation t: {t}")
        print(f"  (R - I) @ r + t: {lhs}")
        print(f"  Required lattice shift: n0*a0 + n1*a1 + n2*a2")
        print(f"  n_vector [n0, n1, n2]: {n_vector}")
        print(f"  Is invariant: {is_invariant}")

    return is_invariant


# ==============================================================================
# STEP 7: Find neighboring atoms and partition into equivalence classes
# ==============================================================================


# ==============================================================================
# Extract atom type information
# ==============================================================================
def initialize_unit_cell_atoms(parsed_config,
                               repr_s_np=None, repr_p_np=None,
                               repr_d_np=None, repr_f_np=None):
    """
    Initialize all atoms in the unit cell [0, 0, 0] from parsed configuration.

    Args:
        parsed_config: Dictionary containing atom positions, types, and lattice basis
        repr_s_np, repr_p_np, repr_d_np, repr_f_np: Representation matrices for orbitals (optional)

    Returns:
        tuple: (atom_types, fractional_positions, unit_cell_atoms) where:
            - atom_types: list of atom type names (e.g., ['B', 'N', 'B', ...])
            - fractional_positions: list of numpy arrays with fractional coordinates
            - unit_cell_atoms: list of atomIndex objects for all atoms in cell [0,0,0]
    """
    atom_types = []
    fractional_positions = []
    unit_cell_atoms = []

    # Extract lattice basis from parsed config
    lattice_basis = np.array(parsed_config['lattice_basis'])

    # Reference cell is [0, 0, 0]
    reference_cell = [0, 0, 0]

    # Extract and construct atoms from parsed configuration
    for i, pos in enumerate(parsed_config['atom_positions']):
        # Extract atom information
        type_name = pos["atom_type"]
        frac_pos = np.array(pos["fractional_coordinates"])

        # Store basic information
        atom_types.append(type_name)
        fractional_positions.append(frac_pos)

        # Create atomIndex object
        atom = atomIndex(
            cell=reference_cell,
            frac_coord=frac_pos,
            atom_name=type_name,
            basis=lattice_basis,
            parsed_config=parsed_config,
            repr_s_np=repr_s_np,
            repr_p_np=repr_p_np,
            repr_d_np=repr_d_np,
            repr_f_np=repr_f_np
        )

        unit_cell_atoms.append(atom)

    return atom_types, fractional_positions, unit_cell_atoms


# First initialize the unit cell atoms
atom_types, fractional_positions, unit_cell_atoms = initialize_unit_cell_atoms(parsed_config)
search_range=8
radius=1.05 * np.sqrt(3)
search_dim=2
# Then for each atom in unit_cell_atoms, compute its neighbors
all_neighbors = {}  # Dictionary with index as key
for i, unit_atom in enumerate(unit_cell_atoms):
    # Find all neighbors within the specified radius for this atom
    neighbors = compute_dist(
        center_atom=unit_atom,
        unit_cell_atoms=unit_cell_atoms,
        search_range=search_range,  # e.g., 1, 2, 3, etc.
        radius=radius,  # e.g., 5.0 Angstroms
        search_dim=search_dim  # 1, 2, or 3
    )
    # Store neighbors using the unit cell atom index as key
    all_neighbors[i] = neighbors
    print(f"Unit cell atom {i} ({unit_atom.atom_name}): found {len(neighbors)} neighbors within radius {radius}")

