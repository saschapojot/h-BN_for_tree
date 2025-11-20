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
        self.n0 = cell[0]
        self.n1 = cell[1]
        self.n2 = cell[2]
        self.atom_name = atom_name
        self.frac_coord = frac_coord
        self.basis = basis

        # Calculate Cartesian coordinates
        a0, a1, a2 = basis
        f0, f1, f2 = frac_coord
        cart_coord = (self.n0 + f0) * a0 + (self.n1 + f1) * a1 + (self.n2 + f2) * a2
        self.cart_coord = cart_coord

        # Store orbital information if config is provided
        if parsed_config is not None and atom_name in parsed_config['atom_types']:
            self.orbitals = parsed_config['atom_types'][atom_name]['orbitals']
            self.num_orbitals = len(self.orbitals)
        else:
            self.orbitals = None
            self.num_orbitals = 0

        # Store representation matrices
        self.repr_s_np = repr_s_np
        self.repr_p_np = repr_p_np
        self.repr_d_np = repr_d_np
        self.repr_f_np = repr_f_np

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
def compute_dist(center_frac, center_cell, dest_frac, basis, search_range, radius):
    """
    Find all atoms within a radius from a center atom

    :param center_frac: Fractional coordinates of center
    :param center_cell: Cell indices of center
    :param dest_frac: Fractional coordinates of destination atom type
    :param basis: Lattice basis vectors
    :param search_range: Range to search in each direction
    :param radius: Maximum distance
    :return: List of [cell, frac_coord] pairs within radius
    """
    f0, f1 = center_frac
    n0, n1 = center_cell

    g0, g1 = dest_frac

    a0, a1, a2 = basis

    center_coord = np.array(
        (f0 + n0) * a0 + (f1 + n1) * a1
    )

    rst = []
    for j0 in range(-search_range, search_range + 1):
        for j1 in range(-search_range, search_range + 1):
            dest_coord = np.array(
                (g0 + j0) * a0 + (g1 + j1) * a1
            )
            dist = np.linalg.norm(center_coord - dest_coord, ord=2)
            if dist <= radius:
                rst.append([[j0, j1, 0], [g0, g1, 0]])

    return rst


def frac_to_cartesian(cell, frac_coord, basis):
    """Convert fractional coordinates to Cartesian"""
    n0, n1, n2 = cell
    f0, f1, f2 = frac_coord
    a0, a1, a2 = basis
    return (n0 + f0) * a0 + (n1 + f1) * a1 + (n2 + f2) * a2


# ==============================================================================
# Extract atom type information
# ==============================================================================
atom_types = []
fractional_positions = []
for i, pos in enumerate(parsed_config['atom_positions']):
    type_name = pos["atom_type"]
    frac_pos = pos["fractional_coordinates"]
    atom_types.append(type_name)
    fractional_positions.append(np.array(frac_pos))

# ==============================================================================
# STEP 7: Find neighboring atoms and partition into equivalence classes
# ==============================================================================
print("\n" + "=" * 60)
print("FINDING NEIGHBORING ATOMS")
print("=" * 60)

ind0 = 0
atm0 = atom_types[ind0]
center_frac0 = (fractional_positions[ind0])[:2]
center_cell = [0, 0]
lattice_basis = np.array(parsed_config['lattice_basis'])
l = 1.05 * np.sqrt(3)

# Find BB neighbors
neigboring_BB = compute_dist(center_frac0, center_cell, center_frac0, lattice_basis, 7, l)
print(f"Found {len(neigboring_BB)} BB neighbors")

# Convert to Cartesian for verification
neigboring_BB_cartesian = []
for item in neigboring_BB:
    cell, frac_coord = item
    cart_coord = frac_to_cartesian(cell, frac_coord, lattice_basis)
    neigboring_BB_cartesian.append([cell, cart_coord])

# Get space group matrices in Cartesian coordinates
eps = 1e-8
space_group_bilbao_cart = []
for item in space_group_representations["space_group_matrices_cartesian"]:
    space_group_bilbao_cart.append(np.array(item))

# Create BB_atoms with orbital representations
BB_atoms = []
for item in neigboring_BB:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "B", lattice_basis, parsed_config,
                    repr_s_np, repr_p_np, repr_d_np, repr_f_np)
    BB_atoms.append(atm)

B_center_frac = list(center_frac0) + [0]
B_center_atom = atomIndex([0, 0, 0], B_center_frac, atm0, lattice_basis, parsed_config,
                          repr_s_np, repr_p_np, repr_d_np, repr_f_np)

# ==============================================================================
# Find identity operation
# ==============================================================================
identity_idx = None
for idx, group_mat in enumerate(space_group_bilbao_cart):
    if np.allclose(group_mat[:3, :3], np.eye(3)) and np.allclose(group_mat[:3, 3], 0):
        identity_idx = idx
        print(f"Identity operation found at index {identity_idx}")
        break

if identity_idx is None:
    print("WARNING: Identity operation not found in space_group_bilbao_cart!")
    exit(1)

# ==============================================================================
# Verify atom orbital representations
# ==============================================================================
print("\n" + "=" * 80)
print("VERIFYING ATOM ORBITAL REPRESENTATIONS")
print("=" * 80)

print(f"\nB center atom:")
print(f"  {B_center_atom}")
print(f"  Orbitals: {B_center_atom.get_orbital_names()}")
if B_center_atom.orbital_representations:
    print(f"  Number of operations: {len(B_center_atom.orbital_representations)}")
    V_identity = B_center_atom.get_representation_matrix(identity_idx)
    print(f"  Identity matrix shape: {V_identity.shape}")
    print(f"  Is identity: {np.allclose(V_identity, np.eye(V_identity.shape[0]))}")


# ==============================================================================
# hopping class
# ==============================================================================
class hopping:
    """
    Represents a single hopping term between two atoms.
    """

    def __init__(self, to_atom, from_atom, class_id, operation_idx, rotation_matrix, translation_vector):
        self.to_atom = to_atom  # Atom object (destination)
        self.from_atom = from_atom  # Atom object (source)
        self.class_id = class_id  # Equivalence class identifier
        self.operation_idx = operation_idx  # Which space group operation transforms parent to this hopping
        self.rotation_matrix = rotation_matrix  # 3×3 rotation matrix R
        self.translation_vector = translation_vector  # 3D translation vector b
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
        self.type = type  # "linear" or "hermitian" for child, None for root
        self.is_root = (hopping.operation_idx == identity_idx)
        self.children = []  # List of child vertex objects
        self.parent = parent  # Reference to parent vertex (None for root)

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
def get_next(center_atom, nghb_atom, group_mat):
    """
    Relocate origin to center_atom, and get symmetry transformed atoms

    :param center_atom: Center atom object
    :param nghb_atom: Neighbor atom object
    :param group_mat: Space group matrix [R|b]
    :return: Transformed Cartesian coordinate
    """
    R = group_mat[:, :3]
    b = group_mat[:, 3]
    center_cart_coord = center_atom.cart_coord
    nghb_cart_coord = nghb_atom.cart_coord
    diff_vec = nghb_cart_coord - center_cart_coord
    next_cart_coord = center_cart_coord + R @ diff_vec + b
    return next_cart_coord


# ==============================================================================
# STEP 8: Partition BB_atoms into equivalent sets under symmetry
# ==============================================================================
print("\n" + "=" * 60)
print("PARTITIONING ALL BB_ATOMS INTO EQUIVALENT SETS")
print("=" * 60)

equivalent_atom_sets_BB = []
equivalent_hopping_sets_BB = []
set_counter = 0

while len(BB_atoms) > 0:
    set_counter += 1
    print(f"\n--- Equivalent Set {set_counter} ---")

    # Take the first atom from remaining BB_atoms as seed
    seed_atom = BB_atoms[0]
    print(f"Seed atom: {seed_atom}")

    # Calculate seed atom's distance to center
    seed_distance = np.linalg.norm(seed_atom.cart_coord - B_center_atom.cart_coord)
    print(f"Seed distance to center: {seed_distance:.6f}")

    # Dictionary to track which operation generated each equivalent atom
    equivalent_dict = {}
    current_hopping_set = []

    # First, assign seed atom to identity operation
    equivalent_dict[0] = (identity_idx, seed_atom)

    # Get identity matrix components
    identity_matrix = space_group_bilbao_cart[identity_idx]
    identity_rotation = identity_matrix[:3, :3]
    identity_translation = identity_matrix[:3, 3]

    # Create hopping for seed atom with identity operation
    seed_hop = hopping(
        to_atom=B_center_atom,
        from_atom=seed_atom,
        class_id=set_counter - 1,
        operation_idx=identity_idx,
        rotation_matrix=identity_rotation,
        translation_vector=identity_translation
    )
    current_hopping_set.append(seed_hop)

    # Apply all space group operations to the seed atom
    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        # Skip identity operation (already handled for seed)
        if op_idx == identity_idx:
            continue

        # Check if this operation leaves the center atom invariant
        center_transformed = get_next(B_center_atom, B_center_atom, group_mat)
        diff_center = center_transformed - B_center_atom.cart_coord

        if np.linalg.norm(diff_center) > 1e-6:
            continue

        # Get the transformed coordinate from seed atom
        next_coord = get_next(B_center_atom, seed_atom, group_mat)

        # Calculate distance of transformed coordinate to center
        next_distance = np.linalg.norm(next_coord - B_center_atom.cart_coord)

        # Check if distance is equal to seed distance (within tolerance)
        if abs(next_distance - seed_distance) > 1e-6:
            continue

        # Check if this coordinate matches any atom in BB_atoms
        for idx, atm in enumerate(BB_atoms):
            if idx not in equivalent_dict:
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    equivalent_dict[idx] = (op_idx, atm)

                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]

                    hop = hopping(
                        to_atom=B_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)

    equivalent_indices = sorted(equivalent_dict.keys())
    print(f"Found {len(equivalent_indices)} equivalent atoms")

    current_atom_set = [BB_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_BB.append(current_atom_set)
    equivalent_hopping_sets_BB.append(current_hopping_set)

    # Remove equivalent atoms from BB_atoms
    BB_atoms = [atm for i, atm in enumerate(BB_atoms) if i not in equivalent_indices]

print(f"\nTotal BB equivalent sets: {len(equivalent_atom_sets_BB)}")

# ==============================================================================
# Build trees for BB hoppings
# ==============================================================================
tree_roots = []
for set_idx, hopping_set in enumerate(equivalent_hopping_sets_BB):
    # Find the root hopping (identity operation)
    root_hopping = None
    child_hoppings = []

    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)

    if root_hopping is None:
        print(f"WARNING: No identity hopping found in set {set_idx}!")
        continue

    # Create root vertex
    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)

    # Create child vertices and link them to root
    for hop in child_hoppings:
        child_v = vertex(hopping=hop, type="linear", identity_idx=identity_idx)
        root_vertex.add_child(child_v)

    tree_roots.append(root_vertex)

# ==============================================================================
# STEP 9: Process BN atoms (Boron center, Nitrogen neighbors)
# ==============================================================================
ind1 = 1
atm1 = atom_types[ind1]
center_frac1 = (fractional_positions[ind1])[:2]

neigboring_BN = compute_dist(center_frac0, center_cell, center_frac1, lattice_basis, 7, l)

BN_atoms = []
for item in neigboring_BN:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "N", lattice_basis, parsed_config,
                    repr_s_np, repr_p_np, repr_d_np, repr_f_np)
    BN_atoms.append(atm)

print(f"\nFound {len(neigboring_BN)} BN neighbors")

# Partition BN atoms
equivalent_atom_sets_BN = []
equivalent_hopping_sets_BN = []
set_counter = 0

while len(BN_atoms) > 0:
    set_counter += 1
    seed_atom = BN_atoms[0]
    seed_distance = np.linalg.norm(seed_atom.cart_coord - B_center_atom.cart_coord)

    equivalent_dict = {}
    current_hopping_set = []

    equivalent_dict[0] = (identity_idx, seed_atom)

    identity_matrix = space_group_bilbao_cart[identity_idx]
    identity_rotation = identity_matrix[:3, :3]
    identity_translation = identity_matrix[:3, 3]

    seed_hop = hopping(
        to_atom=B_center_atom,
        from_atom=seed_atom,
        class_id=set_counter - 1,
        operation_idx=identity_idx,
        rotation_matrix=identity_rotation,
        translation_vector=identity_translation
    )
    current_hopping_set.append(seed_hop)

    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        if op_idx == identity_idx:
            continue

        center_transformed = get_next(B_center_atom, B_center_atom, group_mat)
        diff_center = center_transformed - B_center_atom.cart_coord

        if np.linalg.norm(diff_center) > 1e-6:
            continue

        next_coord = get_next(B_center_atom, seed_atom, group_mat)
        next_distance = np.linalg.norm(next_coord - B_center_atom.cart_coord)

        if abs(next_distance - seed_distance) > 1e-6:
            continue

        for idx, atm in enumerate(BN_atoms):
            if idx not in equivalent_dict:
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    equivalent_dict[idx] = (op_idx, atm)
                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]

                    hop = hopping(
                        to_atom=B_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)

    equivalent_indices = sorted(equivalent_dict.keys())
    current_atom_set = [BN_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_BN.append(current_atom_set)
    equivalent_hopping_sets_BN.append(current_hopping_set)

    BN_atoms = [atm for i, atm in enumerate(BN_atoms) if i not in equivalent_indices]

print(f"Total BN equivalent sets: {len(equivalent_atom_sets_BN)}")

# Build trees for BN hoppings
tree_roots_BN = []
for set_idx, hopping_set in enumerate(equivalent_hopping_sets_BN):
    root_hopping = None
    child_hoppings = []

    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)

    if root_hopping is None:
        continue

    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)

    for hop in child_hoppings:
        child_v = vertex(hopping=hop, type="linear", identity_idx=identity_idx)
        root_vertex.add_child(child_v)

    tree_roots_BN.append(root_vertex)

# ==============================================================================
# STEP 10: Process NB atoms (Nitrogen center, Boron neighbors)
# ==============================================================================
neigboring_NB = compute_dist(center_frac1, center_cell, center_frac0, lattice_basis, 7, l)

NB_atoms = []
for item in neigboring_NB:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "B", lattice_basis, parsed_config,
                    repr_s_np, repr_p_np, repr_d_np, repr_f_np)
    NB_atoms.append(atm)

N_center_frac = list(center_frac1) + [0]
N_center_atom = atomIndex([0, 0, 0], N_center_frac, atm1, lattice_basis, parsed_config,
                          repr_s_np, repr_p_np, repr_d_np, repr_f_np)

print(f"\nFound {len(neigboring_NB)} NB neighbors")

# Partition NB atoms
equivalent_atom_sets_NB = []
equivalent_hopping_sets_NB = []
set_counter = 0

while len(NB_atoms) > 0:
    set_counter += 1
    seed_atom = NB_atoms[0]
    seed_distance = np.linalg.norm(seed_atom.cart_coord - N_center_atom.cart_coord)

    equivalent_dict = {}
    current_hopping_set = []

    equivalent_dict[0] = (identity_idx, seed_atom)

    identity_matrix = space_group_bilbao_cart[identity_idx]
    identity_rotation = identity_matrix[:3, :3]
    identity_translation = identity_matrix[:3, 3]

    seed_hop = hopping(
        to_atom=N_center_atom,
        from_atom=seed_atom,
        class_id=set_counter - 1,
        operation_idx=identity_idx,
        rotation_matrix=identity_rotation,
        translation_vector=identity_translation
    )
    current_hopping_set.append(seed_hop)

    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        if op_idx == identity_idx:
            continue

        center_transformed = get_next(N_center_atom, N_center_atom, group_mat)
        diff_center = center_transformed - N_center_atom.cart_coord

        if np.linalg.norm(diff_center) > 1e-6:
            continue

        next_coord = get_next(N_center_atom, seed_atom, group_mat)
        next_distance = np.linalg.norm(next_coord - N_center_atom.cart_coord)

        if abs(next_distance - seed_distance) > 1e-6:
            continue

        for idx, atm in enumerate(NB_atoms):
            if idx not in equivalent_dict:
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    equivalent_dict[idx] = (op_idx, atm)
                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]

                    hop = hopping(
                        to_atom=N_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)

    equivalent_indices = sorted(equivalent_dict.keys())
    current_atom_set = [NB_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_NB.append(current_atom_set)
    equivalent_hopping_sets_NB.append(current_hopping_set)

    NB_atoms = [atm for i, atm in enumerate(NB_atoms) if i not in equivalent_indices]

print(f"Total NB equivalent sets: {len(equivalent_atom_sets_NB)}")

# Build trees for NB hoppings
tree_roots_NB = []
for set_idx, hopping_set in enumerate(equivalent_hopping_sets_NB):
    root_hopping = None
    child_hoppings = []

    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)

    if root_hopping is None:
        continue

    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)

    for hop in child_hoppings:
        child_v = vertex(hopping=hop, type="linear", identity_idx=identity_idx)
        root_vertex.add_child(child_v)

    tree_roots_NB.append(root_vertex)

# ==============================================================================
# STEP 11: Process NN atoms (Nitrogen center, Nitrogen neighbors)
# ==============================================================================
neigboring_NN = compute_dist(center_frac1, center_cell, center_frac1, lattice_basis, 7, l)

NN_atoms = []
for item in neigboring_NN:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "N", lattice_basis, parsed_config,
                    repr_s_np, repr_p_np, repr_d_np, repr_f_np)
    NN_atoms.append(atm)

print(f"\nFound {len(neigboring_NN)} NN neighbors")

# Partition NN atoms
equivalent_atom_sets_NN = []
equivalent_hopping_sets_NN = []
set_counter = 0

while len(NN_atoms) > 0:
    set_counter += 1
    seed_atom = NN_atoms[0]
    seed_distance = np.linalg.norm(seed_atom.cart_coord - N_center_atom.cart_coord)

    equivalent_dict = {}
    current_hopping_set = []

    equivalent_dict[0] = (identity_idx, seed_atom)

    identity_matrix = space_group_bilbao_cart[identity_idx]
    identity_rotation = identity_matrix[:3, :3]
    identity_translation = identity_matrix[:3, 3]

    seed_hop = hopping(
        to_atom=N_center_atom,
        from_atom=seed_atom,
        class_id=set_counter - 1,
        operation_idx=identity_idx,
        rotation_matrix=identity_rotation,
        translation_vector=identity_translation
    )
    current_hopping_set.append(seed_hop)

    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        if op_idx == identity_idx:
            continue

        center_transformed = get_next(N_center_atom, N_center_atom, group_mat)
        diff_center = center_transformed - N_center_atom.cart_coord

        if np.linalg.norm(diff_center) > 1e-6:
            continue

        next_coord = get_next(N_center_atom, seed_atom, group_mat)
        next_distance = np.linalg.norm(next_coord - N_center_atom.cart_coord)

        if abs(next_distance - seed_distance) > 1e-6:
            continue

        for idx, atm in enumerate(NN_atoms):
            if idx not in equivalent_dict:
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    equivalent_dict[idx] = (op_idx, atm)
                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]

                    hop = hopping(
                        to_atom=N_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)

    equivalent_indices = sorted(equivalent_dict.keys())
    current_atom_set = [NN_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_NN.append(current_atom_set)
    equivalent_hopping_sets_NN.append(current_hopping_set)

    NN_atoms = [atm for i, atm in enumerate(NN_atoms) if i not in equivalent_indices]

print(f"Total NN equivalent sets: {len(equivalent_atom_sets_NN)}")

# Build trees for NN hoppings
tree_roots_NN = []
for set_idx, hopping_set in enumerate(equivalent_hopping_sets_NN):
    root_hopping = None
    child_hoppings = []

    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)

    if root_hopping is None:
        continue

    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)

    for hop in child_hoppings:
        child_v = vertex(hopping=hop, type="linear", identity_idx=identity_idx)
        root_vertex.add_child(child_v)

    tree_roots_NN.append(root_vertex)


# ==============================================================================
# STEP 12: Check hermitian conjugate relations between BN and NB
# ==============================================================================
def check_hermitian(hopping1, hopping2):
    """Check if hopping2 is the Hermitian conjugate of hopping1"""
    to_atom1 = hopping1.to_atom
    from_atom1 = hopping1.from_atom

    to_atom2c, from_atom2c = hopping2.conjugate()

    # Get lattice basis vectors
    a0, a1, a2 = lattice_basis

    # Iterate through all space group operations
    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        R = group_mat[:3, :3]
        b = group_mat[:3, 3]

        # Transform atoms
        to_atom1_transformed_cart_coord = R @ to_atom1.cart_coord + b
        from_atom1_transformed_cart_coord = R @ from_atom1.cart_coord + b

        # Compute differences
        diff_to = to_atom2c.cart_coord - to_atom1_transformed_cart_coord
        diff_from = from_atom2c.cart_coord - from_atom1_transformed_cart_coord

        # Check if diff_to and diff_from are the same lattice vector
        if np.linalg.norm(diff_to - diff_from) < 1e-6:
            lattice_matrix = np.column_stack([a0, a1, a2])
            try:
                n_vector = np.linalg.solve(lattice_matrix, diff_to)
                n_rounded = np.round(n_vector)
                if np.allclose(n_vector, n_rounded, atol=1e-6):
                    return True, op_idx
            except np.linalg.LinAlgError:
                continue

    return False, None


print("\n" + "=" * 80)
print("GROUPING BN AND NB ROOTS BY HERMITIAN CONJUGATE RELATIONS")
print("=" * 80)

nb_matched = set()
bn_matched = set()
hermitian_groups = []
independent_groups = []

for nb_idx, nb_root in enumerate(tree_roots_NB):
    if nb_idx in nb_matched:
        continue

    found_match = False
    for bn_idx, bn_root in enumerate(tree_roots_BN):
        if bn_idx in bn_matched:
            continue

        exists, op_idx = check_hermitian(bn_root.hopping, nb_root.hopping)
        if exists:
            hermitian_groups.append([bn_root, nb_root, op_idx])
            nb_matched.add(nb_idx)
            bn_matched.add(bn_idx)
            found_match = True
            break

    if not found_match:
        independent_groups.append([nb_root])
        nb_matched.add(nb_idx)

# Process hermitian groups - add NB root as hermitian child of BN root
for bn_root, nb_root, op_idx in hermitian_groups:
    bn_root.children.append(nb_root)
    nb_root.type = "hermitian"
    nb_root.is_root = False
    nb_root.parent = bn_root
    nb_root.hopping.operation_idx = op_idx

print(f"Processed {len(hermitian_groups)} hermitian pairs")

# ==============================================================================
# STEP 13: Collect all roots and sort by distance
# ==============================================================================
all_roots = []
all_roots.extend(tree_roots)
all_roots.extend(tree_roots_BN)
all_roots.extend(tree_roots_NN)
all_roots.extend([group[0] for group in independent_groups])


def get_hopping_distance(root):
    """Calculate the distance for a root's hopping"""
    hopping = root.hopping
    return np.linalg.norm(hopping.from_atom.cart_coord - hopping.to_atom.cart_coord)


all_roots_sorted = sorted(all_roots, key=get_hopping_distance)

print(f"\nTotal roots: {len(all_roots_sorted)}")
print(f"  BB roots: {len(tree_roots)}")
print(f"  BN roots (with hermitian children): {len(tree_roots_BN)}")
print(f"  NN roots: {len(tree_roots_NN)}")
print(f"  Independent roots: {len(independent_groups)}")


# ==============================================================================
# Helper functions for constraint analysis
# ==============================================================================
def stabilizer(atom):
    """Find stabilizer operations for an atom"""
    stabilizer_op_id_list = []
    atom_cart_coord = atom.cart_coord

    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        R = group_mat[:3, :3]
        b = group_mat[:3, 3]
        atom_cart_transformed = R @ atom_cart_coord + b
        if np.linalg.norm(atom_cart_coord - atom_cart_transformed, ord=2) < 1e-6:
            stabilizer_op_id_list.append(op_idx)

    return set(stabilizer_op_id_list)


def find_root_stabilizer(root):
    """Find stabilizer operations for a hopping root"""
    to_atom = root.hopping.to_atom
    from_atom = root.hopping.from_atom

    to_atom_stabilizer = stabilizer(to_atom)
    from_atom_stabilizer = stabilizer(from_atom)

    root_stabilizer = to_atom_stabilizer.intersection(from_atom_stabilizer)

    return root_stabilizer


def create_hopping_matrix(root, parsed_config, tree_idx):
    """
    Create a symbolic hopping matrix for a root's hopping

    Args:
        root: vertex object containing the hopping
        parsed_config: configuration with orbital information
        tree_idx: tree number/index

    Returns:
        sympy.Matrix: Hopping matrix with symbolic elements T^{tree_idx}_{i,j}
    """
    hopping = root.hopping

    # Get orbitals directly from atom objects
    to_orbitals = hopping.to_atom.get_orbital_names()
    from_orbitals = hopping.from_atom.get_orbital_names()

    # Fallback to parsed_config if atoms don't have orbital info
    if not to_orbitals:
        to_atom_type = hopping.to_atom.atom_name
        to_orbitals = parsed_config['atom_types'][to_atom_type]['orbitals']

    if not from_orbitals:
        from_atom_type = hopping.from_atom.atom_name
        from_orbitals = parsed_config['atom_types'][from_atom_type]['orbitals']

    # Get dimensions
    n_to = len(to_orbitals)
    n_from = len(from_orbitals)

    # Create symbolic matrix
    T = sp.zeros(n_to, n_from)

    # Fill with symbolic elements
    for i, to_orb in enumerate(to_orbitals):
        for j, from_orb in enumerate(from_orbitals):
            symbol_name = f"T^{{{tree_idx}}}_{{{to_orb},{from_orb}}}"
            T[i, j] = sp.Symbol(symbol_name)

    return T


def get_stabilizer_constraints(root, parsed_config, tree_idx):
    """
    Get all constraint equations from stabilizer operations for a root's hopping matrix.

    Args:
        root: vertex object containing the hopping
        parsed_config: configuration with orbital information
        tree_idx: tree number/index

    Returns:
        dict: Contains 'T', 'constraints', 'equations'
    """
    # Create hopping matrix
    T = create_hopping_matrix(root, parsed_config, tree_idx)

    # Get stabilizer operations
    root_stabilizer = list(find_root_stabilizer(root))

    # Get atom information
    root_to_atom = root.hopping.to_atom
    root_from_atom = root.hopping.from_atom

    # Store all constraints
    all_constraints = []
    all_equations = []

    for stab_id, op_id in enumerate(root_stabilizer):
        # Get representation matrices directly from atoms
        V_to = root_to_atom.get_sympy_representation_matrix(op_id)
        V_from = root_from_atom.get_sympy_representation_matrix(op_id)

        # Compute transformed T: V_to * T * V_from^†
        transformed_T = V_to * T * V_from.H

        # Compute difference
        diff_T = T - transformed_T
        diff_T_simplified = sp.simplify(diff_T)

        # Extract non-zero equations
        equations = []
        for i in range(diff_T.shape[0]):
            for j in range(diff_T.shape[1]):
                if diff_T_simplified[i, j] != 0:
                    equations.append({
                        'element': (i, j),
                        'equation': diff_T_simplified[i, j]
                    })

        all_constraints.append({
            'op_id': op_id,
            'stab_id': stab_id,
            'V_to': root_to_atom.get_representation_matrix(op_id),
            'V_from': root_from_atom.get_representation_matrix(op_id),
            'diff_T': diff_T_simplified,
            'equations': equations
        })
        all_equations.extend(equations)

    return {
        'T': T,
        'root_stabilizer': root_stabilizer,
        'constraints': all_constraints,
        'all_equations': all_equations
    }


def are_equivalent_equations(eq1, eq2):
    """Check if two equations are mathematically equivalent"""
    return sp.simplify(eq1 - eq2) == 0


def get_unique_equations(all_equations):
    """Extract unique equations from a list of equation dictionaries"""
    unique_eqs = []
    for eq in all_equations:
        canonical = sp.simplify(eq['equation'])
        if not any(are_equivalent_equations(canonical, existing) for existing in unique_eqs):
            unique_eqs.append(canonical)
    return unique_eqs


def equations_to_matrix_form(equations, tolerance=1e-10):
    """
    Convert a list of linear equations to matrix form Ax = 0

    Args:
        equations: list of sympy expressions
        tolerance: numerical tolerance for treating values as zero

    Returns:
        A: coefficient matrix (sympy.Matrix)
        x: vector of variables (sympy.Matrix)
        symbols: list of all unique symbols
    """
    # Collect all unique symbols
    all_symbols = set()
    for eq in equations:
        all_symbols.update(eq.free_symbols)

    sorted_symbols = sorted(all_symbols, key=lambda s: str(s))

    # Create coefficient matrix
    n_equations = len(equations)
    n_variables = len(sorted_symbols)
    A_np = np.zeros((n_equations, n_variables))

    for i, eq in enumerate(equations):
        eq_expanded = sp.expand(eq)
        for j, symbol in enumerate(sorted_symbols):
            coeff = eq_expanded.coeff(symbol)
            try:
                coeff_val = float(coeff)
            except (TypeError, ValueError):
                coeff_val = complex(coeff).real if coeff != 0 else 0.0

            if abs(coeff_val) < tolerance:
                coeff_val = 0.0

            A_np[i, j] = coeff_val

    A = sp.Matrix(A_np)
    x = sp.Matrix(sorted_symbols)

    return A, x, sorted_symbols


def get_dependent_expressions(A_rref, pivot_cols, symbols, tolerance=1e-10):
    """Express dependent variables in terms of free variables"""
    free_var_indices = [i for i in range(len(symbols)) if i not in pivot_cols]
    dependent_expressions = {}

    for row_idx, col_idx in enumerate(pivot_cols):
        if row_idx < A_rref.shape[0]:
            dependent_var = symbols[col_idx]

            expr_terms = []
            for free_idx in free_var_indices:
                coeff = -A_rref[row_idx, free_idx]

                try:
                    coeff_val = float(coeff)
                    if abs(coeff_val) < tolerance:
                        coeff = 0
                    else:
                        coeff = sp.nsimplify(coeff, rational=True, tolerance=tolerance)
                except (TypeError, ValueError):
                    pass

                if coeff != 0:
                    expr_terms.append(coeff * symbols[free_idx])

            if expr_terms:
                expression = sum(expr_terms)
            else:
                expression = sp.Integer(0)

            dependent_expressions[dependent_var] = expression

    return dependent_expressions


def reconstruct_hopping_matrix(T_original, dependent_expressions):
    """Reconstruct hopping matrix with only free variables"""
    T_reconstructed = T_original.copy()

    for dep_var, expr in dependent_expressions.items():
        T_reconstructed = T_reconstructed.subs(dep_var, expr)

    return T_reconstructed


def analyze_tree_constraints(root, parsed_config, tree_idx, verbose=True, tolerance=1e-10):
    """
    Complete constraint analysis for a single tree

    Args:
        root: vertex object (root of tree)
        parsed_config: Configuration dictionary
        tree_idx: Tree index number
        verbose: Whether to print detailed output
        tolerance: numerical tolerance

    Returns:
        dict: Complete analysis results
    """
    if verbose:
        print("=" * 80)
        print(f"TREE {tree_idx} CONSTRAINT ANALYSIS")
        print("=" * 80)

    # Get stabilizer constraints
    root_stab_result = get_stabilizer_constraints(root, parsed_config, tree_idx)

    if verbose:
        print(f"\nRoot stabilizer: {root_stab_result['root_stabilizer']}")
        print(f"Number of stabilizer operations: {len(root_stab_result['root_stabilizer'])}")

    # Get unique equations
    unique_eqs = get_unique_equations(root_stab_result['all_equations'])

    if len(unique_eqs) > 0:
        A, x, symbols = equations_to_matrix_form(unique_eqs, tolerance=tolerance)

        A_cleaned = A.applyfunc(lambda x: 0 if abs(float(x)) < tolerance else x)
        A_rref, pivot_cols = A_cleaned.rref()

        A_rref = A_rref.applyfunc(
            lambda x: 0 if abs(float(x)) < tolerance else sp.nsimplify(x, rational=True, tolerance=tolerance))

        free_var_indices = [i for i in range(len(symbols)) if i not in pivot_cols]
        dependent_var_indices = list(pivot_cols)

        dependent_expressions = get_dependent_expressions(A_rref, pivot_cols, symbols, tolerance=tolerance)
        T_reconstructed = reconstruct_hopping_matrix(root_stab_result['T'], dependent_expressions)

        root_stab_result.update({
            'unique_equations': unique_eqs,
            'constraint_matrix': A,
            'constraint_matrix_rref': A_rref,
            'pivot_cols': pivot_cols,
            'symbols': symbols,
            'free_var_indices': free_var_indices,
            'dependent_var_indices': dependent_var_indices,
            'dependent_expressions': dependent_expressions,
            'T_reconstructed': T_reconstructed,
            'rank': len(pivot_cols),
            'nullity': len(symbols) - len(pivot_cols)
        })
    else:
        total_params = root_stab_result['T'].shape[0] * root_stab_result['T'].shape[1]
        root_stab_result.update({
            'unique_equations': [],
            'constraint_matrix': None,
            'constraint_matrix_rref': None,
            'pivot_cols': (),
            'symbols': [],
            'free_var_indices': list(range(total_params)),
            'dependent_var_indices': [],
            'dependent_expressions': {},
            'T_reconstructed': root_stab_result['T'].copy(),
            'rank': 0,
            'nullity': total_params
        })

    return root_stab_result




def print_tree(root, prefix="", is_last=True, show_details=True):
    """Print a tree structure in a visual format"""
    connector = "└── " if is_last else "├── "

    if root.is_root:
        node_label = "ROOT"
        style = "╔═══"
    else:
        node_label = f"CHILD ({root.type})"
        style = connector

    hop = root.hopping
    from_cell = [hop.from_atom.n0, hop.from_atom.n1, hop.from_atom.n2]
    to_atom_name = hop.to_atom.atom_name
    from_atom_name = hop.from_atom.atom_name
    distance = np.linalg.norm(hop.from_atom.cart_coord - hop.to_atom.cart_coord)

    if show_details:
        node_desc = (f"{node_label} | Op={hop.operation_idx:2d} | "
                     f"{to_atom_name}←{from_atom_name} | "
                     f"Cell=[{from_cell[0]:2d},{from_cell[1]:2d},{from_cell[2]:2d}] | "
                     f"Dist={distance:.4f}")
    else:
        node_desc = f"{node_label} | Op={hop.operation_idx:2d}"

    print(f"{prefix}{style}{node_desc}")

    if root.children:
        if root.is_root:
            new_prefix = prefix + "    "
        else:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

        for i, child in enumerate(root.children):
            is_last_child = (i == len(root.children) - 1)
            print_tree(child, new_prefix, is_last_child, show_details)

def propagate_T_to_child(parent_vertex, child_vertex):
    """
    Propagate hopping matrix T from parent to child using symmetry operations.

    For linear children: T_child = V_to * T_parent * V_from^†
    For hermitian children: T_child = (V_to * T_parent * V_from^†)^†

    :param parent_vertex: Parent vertex object
    :param child_vertex: Child vertex object (or None)
    :return: None (modifies child_vertex.hopping.T in place, or returns early if child is None)
    """
    # Early return if child is None
    if child_vertex is None:
        return

    # Get parent's hopping matrix
    T_parent = parent_vertex.hopping.T

    # Get the operation that transforms parent to child
    op_idx_parent_to_child = child_vertex.hopping.operation_idx

    # Get representation matrices for parent's atoms under this operation
    parent_to_atom_V = (parent_vertex.hopping.to_atom.orbital_representations)[op_idx_parent_to_child]
    parent_from_atom_V = (parent_vertex.hopping.from_atom.orbital_representations)[op_idx_parent_to_child]

    # Get child type
    child_type = child_vertex.type

    # Convert to SymPy matrices
    parent_to_atom_V_sp = sp.Matrix(parent_to_atom_V)
    parent_from_atom_V_sp = sp.Matrix(parent_from_atom_V)

    # Apply transformation based on child type
    if child_type == "linear":
        T_child = parent_to_atom_V_sp * T_parent * parent_from_atom_V_sp.H
    elif child_type == "hermitian":
        T_child = (parent_to_atom_V_sp * T_parent * parent_from_atom_V_sp.H).H
    else:
        raise ValueError(f"Unknown child type: {child_type}")

    # Simplify the result
    T_child = sp.simplify(T_child)

    # Assign to child
    child_vertex.hopping.T = T_child



# ==============================================================================
# Helper function: Recursively propagate to all children
# ==============================================================================
# def propagate_to_all_children(parent_vertex, verbose=False):
#     """
#         Recursively propagate T from parent to all descendants
#
#         Args:
#             parent_vertex: Parent vertex
#             verbose: Print progress
#         """
#
#     for child in parent_vertex.children:
#         propagate_T_to_child(parent_vertex, child)
#         if verbose:
#             print(f"  Propagated to child (op={child.hopping.operation_idx}, type={child.type})")
#         # Recursively propagate to grandchildren
#         if len(child.children) > 0:
#             propagate_to_all_children(child, verbose)

# ==============================================================================
# Helper function: Propagate to all children using BFS
# ==============================================================================
def propagate_to_all_children(parent_vertex, verbose=False):
    """
    Propagate T from parent to all descendants using BFS (breadth-first search)

    Processes vertices level-by-level:
    - Level 1: All immediate children
    - Level 2: All grandchildren
    - Level 3: All great-grandchildren
    - etc.

    Args:
        parent_vertex: Parent vertex (root of subtree)
        verbose: Print progress
    """
    from collections import deque

    if len(parent_vertex.children) == 0:
        if verbose:
            print("  No children to propagate to")
        return

    # Queue stores tuples: (parent_vertex, child_vertex, level)
    queue = deque()

    # Initialize queue with all immediate children (level 1)
    for child in parent_vertex.children:
        queue.append((parent_vertex, child, 1))

    # Track statistics
    total_propagated = 0
    max_level = 0

    # Process queue level-by-level
    while queue:
        parent, child, level = queue.popleft()

        # Propagate T from parent to child
        propagate_T_to_child(parent, child)
        total_propagated += 1
        max_level = max(max_level, level)

        if verbose:
            indent = "  " * level
            print(f"{indent}Level {level}: Propagated to child (op={child.hopping.operation_idx}, type={child.type})")

        # Add all grandchildren to queue (next level)
        for grandchild in child.children:
            queue.append((child, grandchild, level + 1))

    if verbose:
        print(f"\n  BFS completed: {total_propagated} vertices propagated, max depth = {max_level}")
# ==============================================================================
# Helper function: Print all T matrices in a tree
# ==============================================================================
def print_all_T_in_tree(root, tree_idx=None):
    """
    Print T matrices for all vertices in a tree

    Args:
        root: Root vertex of the tree
        tree_idx: Optional tree index for labeling
    """
    print("\n" + "=" * 80)
    if tree_idx is not None:
        print(f"ALL T MATRICES IN TREE {tree_idx}")
    else:
        print("ALL T MATRICES IN TREE")
    print("=" * 80)

    # Print root
    print("\n" + "-" * 80)
    print(f"ROOT | Op={root.hopping.operation_idx}")
    print(f"  {root.hopping.to_atom.atom_name} ← {root.hopping.from_atom.atom_name}")
    print(f"  Distance: {np.linalg.norm(root.hopping.from_atom.cart_coord - root.hopping.to_atom.cart_coord):.4f}")
    print("-" * 80)
    if root.hopping.T is not None:
        sp.pprint(root.hopping.T)
    else:
        print("T = None (not computed)")

    # Counter for children
    child_counter = [0]  # Use list to make it mutable in nested function

    def print_children_T(parent_vertex, level=1):
        """Recursively print children T matrices"""
        for child in parent_vertex.children:
            indent = "  " * level
            child_num = child_counter[0]
            child_counter[0] += 1

            print("\n" + "-" * 80)
            print(f"{indent}CHILD {child_num} | Op={child.hopping.operation_idx} | Type={child.type}")
            print(f"{indent}  {child.hopping.to_atom.atom_name} ← {child.hopping.from_atom.atom_name}")
            print(
                f"{indent}  Distance: {np.linalg.norm(child.hopping.from_atom.cart_coord - child.hopping.to_atom.cart_coord):.4f}")
            print("-" * 80)

            if child.hopping.T is not None:
                sp.pprint(child.hopping.T)
            else:
                print(f"{indent}T = None (not computed)")

            # Recursively print grandchildren
            if len(child.children) > 0:
                print_children_T(child, level + 1)

    # Print all children
    print_children_T(root)

    print("\n" + "=" * 80)
    print(f"Total vertices printed: {child_counter[0] + 1} (1 root + {child_counter[0]} children)")
    print("=" * 80)


def print_all_T_compact(root, tree_idx=None):
    """Print all T matrices in compact format"""

    print("\n" + "=" * 80)
    if tree_idx is not None:
        print(f"TREE {tree_idx}: ALL T MATRICES")
    else:
        print("ALL T MATRICES")
    print("=" * 80)

    vertices = [(root, "ROOT", 0)]  # (vertex, label, level)

    # Collect all vertices using BFS
    queue = [(child, f"CHILD", 1) for child in root.children]
    child_num = 0

    while queue:
        vertex, label_prefix, level = queue.pop(0)
        if label_prefix == "CHILD":
            label = f"{label_prefix}_{child_num}"
            child_num += 1
        else:
            label = label_prefix

        vertices.append((vertex, label, level))

        # Add children to queue
        for child in vertex.children:
            queue.append((child, "CHILD", level + 1))

    # Print all T matrices
    for vertex, label, level in vertices:
        indent = "  " * level
        print(f"\n{indent}{label} (op={vertex.hopping.operation_idx}, type={vertex.type}):")
        if vertex.hopping.T is not None:
            # Print with indentation
            T_str = sp.pretty(vertex.hopping.T)
            for line in T_str.split('\n'):
                print(f"{indent}{line}")
        else:
            print(f"{indent}T = None")

    print("\n" + "=" * 80)


# ==============================================================================
# Helper function: Print tree with T matrices
# ==============================================================================
def print_tree_with_T(root, prefix="", is_last=True, show_T=True):
    """
    Print tree structure with T matrices displayed inline

    Args:
        root: Root vertex of the tree
        prefix: Prefix for tree lines (used for recursion)
        is_last: Whether this is the last child (for formatting)
        show_T: Whether to show T matrices
    """
    # Determine connector style
    connector = "└── " if is_last else "├── "

    if root.is_root:
        node_label = "ROOT"
        style = "╔═══"
    else:
        node_label = f"CHILD ({root.type})"
        style = connector

    # Get hopping information
    hop = root.hopping
    from_cell = [hop.from_atom.n0, hop.from_atom.n1, hop.from_atom.n2]
    to_atom_name = hop.to_atom.atom_name
    from_atom_name = hop.from_atom.atom_name
    distance = np.linalg.norm(hop.from_atom.cart_coord - hop.to_atom.cart_coord)

    # Print node header
    node_desc = (f"{node_label} | Op={hop.operation_idx:2d} | "
                 f"{to_atom_name}←{from_atom_name} | "
                 f"Cell=[{from_cell[0]:2d},{from_cell[1]:2d},{from_cell[2]:2d}] | "
                 f"Dist={distance:.4f}")

    print(f"{prefix}{style}{node_desc}")

    # Print T matrix if requested and available
    if show_T and hop.T is not None:
        # Determine the continuation prefix for T matrix lines
        if root.is_root:
            T_prefix = prefix + "    "
        else:
            if is_last:
                T_prefix = prefix + "    "
            else:
                T_prefix = prefix + "│   "

        # Get pretty-printed T matrix
        T_str = sp.pretty(hop.T)
        T_lines = T_str.split('\n')

        # Print T matrix with proper indentation
        print(f"{T_prefix}T =")
        for line in T_lines:
            print(f"{T_prefix}{line}")

    elif show_T:
        # T is None
        if root.is_root:
            T_prefix = prefix + "    "
        else:
            if is_last:
                T_prefix = prefix + "    "
            else:
                T_prefix = prefix + "│   "
        print(f"{T_prefix}T = None")

    # Recursively print children
    if root.children:
        if root.is_root:
            new_prefix = prefix + "    "
        else:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

        for i, child in enumerate(root.children):
            is_last_child = (i == len(root.children) - 1)
            print_tree_with_T(child, new_prefix, is_last_child, show_T)


# ==============================================================================
# Helper function: Verify constraints for a vertex
# ==============================================================================
def verify_constraint(vertex, verbose=True):
    """
    Verify that a vertex's T matrix satisfies all stabilizer constraints.

    For each stabilizer operation g: T = V_to(g) * T * V_from(g)^†

    Args:
        vertex: vertex object with hopping.T assigned
        verbose: whether to print detailed results

    Returns:
        dict: {
            'satisfied': bool,
            'num_stabilizers': int,
            'violations': list of dicts with violation info
        }
    """
    # Get stabilizer operations
    vertex_stab_all = find_root_stabilizer(vertex)
    to_atom = vertex.hopping.to_atom
    from_atom = vertex.hopping.from_atom

    # Get T matrix
    T = vertex.hopping.T

    if T is None:
        if verbose:
            print("WARNING: T matrix is None, cannot verify constraints")
        return {
            'satisfied': False,
            'num_stabilizers': len(vertex_stab_all),
            'violations': [],
            'error': 'T is None'
        }

    # Convert T to sympy if needed
    T_sp = sp.Matrix(T) if not isinstance(T, sp.Matrix) else T

    violations = []

    if verbose:
        print(f"\nVerifying constraints for vertex (op={vertex.hopping.operation_idx}):")
        print(f"  Number of stabilizer operations: {len(vertex_stab_all)}")

    # Check each stabilizer operation
    for stab_idx, op_id in enumerate(vertex_stab_all):
        # Get representation matrices
        V_to = to_atom.get_sympy_representation_matrix(op_id)
        V_from = from_atom.get_sympy_representation_matrix(op_id)

        # Compute transformed T: V_to * T * V_from^†
        T_transformed = V_to * T_sp * V_from.H

        # Compute difference
        diff = T_sp - T_transformed
        diff_simplified = sp.simplify(diff)

        # Check if difference is zero matrix
        is_zero = diff_simplified.equals(sp.zeros(*diff_simplified.shape))

        if not is_zero:
            violations.append({
                'op_id': op_id,
                'stab_idx': stab_idx,
                'diff_matrix': diff_simplified
            })

            if verbose:
                print(f"  ❌ Stabilizer {stab_idx} (op={op_id}): VIOLATED")
                print(f"     Difference matrix:")
                sp.pprint(diff_simplified)
        else:
            if verbose:
                print(f"  ✓ Stabilizer {stab_idx} (op={op_id}): satisfied")

    # Summary
    satisfied = (len(violations) == 0)

    if verbose:
        print(f"\n{'=' * 60}")
        if satisfied:
            print("✓ ALL CONSTRAINTS SATISFIED")
        else:
            print(f"❌ CONSTRAINTS VIOLATED: {len(violations)} out of {len(vertex_stab_all)} stabilizers")
        print(f"{'=' * 60}")

    return {
        'satisfied': satisfied,
        'num_stabilizers': len(vertex_stab_all),
        'num_violations': len(violations),
        'violations': violations
    }


# ==============================================================================
# Helper function: Verify all vertices in a tree
# ==============================================================================
def verify_tree_constraints(root, verbose=True):
    """
    Verify constraints for all vertices in a tree

    Args:
        root: Root vertex of tree
        verbose: print detailed results

    Returns:
        dict: Summary of verification results
    """
    if verbose:
        print("\n" + "=" * 80)
        print("VERIFYING CONSTRAINTS FOR ENTIRE TREE")
        print("=" * 80)

    all_vertices = [root]
    vertex_labels = ["ROOT"]

    # Collect all vertices
    def collect_vertices(vertex, label_prefix="CHILD", child_num=[0]):
        for child in vertex.children:
            all_vertices.append(child)
            vertex_labels.append(f"{label_prefix}_{child_num[0]}")
            child_num[0] += 1
            collect_vertices(child, label_prefix, child_num)

    collect_vertices(root)

    # Verify each vertex
    results = []
    all_satisfied = True

    for vertex, label in zip(all_vertices, vertex_labels):
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"{label} (op={vertex.hopping.operation_idx}, type={vertex.type})")
            print(f"{'─' * 80}")

        result = verify_constraint(vertex, verbose=verbose)
        result['vertex_label'] = label
        result['vertex'] = vertex  # Store reference to vertex
        results.append(result)

        if not result['satisfied']:
            all_satisfied = False

    # Final summary
    if verbose:
        print("\n" + "=" * 80)
        print("TREE VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total vertices checked: {len(all_vertices)}")

        num_violations = sum(1 for r in results if not r['satisfied'])
        if all_satisfied:
            print("✓ ALL VERTICES SATISFY CONSTRAINTS")
        else:
            print(f"❌ {num_violations} vertices have violations")
            print("\n" + "─" * 80)
            print("DETAILED VIOLATION INFORMATION:")
            print("─" * 80)

            for r in results:
                if not r['satisfied']:
                    vertex = r['vertex']
                    hop = vertex.hopping

                    print(f"\n{r['vertex_label']}:")
                    print(f"  Operation index: {hop.operation_idx}")
                    print(f"  Vertex type: {vertex.type}")
                    print(
                        f"  To atom: {hop.to_atom.atom_name} at cell [{hop.to_atom.n0}, {hop.to_atom.n1}, {hop.to_atom.n2}]")
                    print(
                        f"  From atom: {hop.from_atom.atom_name} at cell [{hop.from_atom.n0}, {hop.from_atom.n1}, {hop.from_atom.n2}]")
                    print(f"  Distance: {np.linalg.norm(hop.from_atom.cart_coord - hop.to_atom.cart_coord):.6f}")
                    print(f"  Number of violated stabilizers: {r['num_violations']} out of {r['num_stabilizers']}")

                    print(f"\n  Violated stabilizer operations:")
                    for v in r['violations']:
                        print(f"    - Stabilizer {v['stab_idx']} (operation {v['op_id']})")
                        print(f"      Difference matrix (T - V_to * T * V_from^†):")
                        diff_str = sp.pretty(v['diff_matrix'])
                        for line in diff_str.split('\n'):
                            print(f"        {line}")

                    # Print the T matrix for this vertex
                    if hop.T is not None:
                        print(f"\n  Current T matrix:")
                        T_str = sp.pretty(hop.T)
                        for line in T_str.split('\n'):
                            print(f"    {line}")
                    else:
                        print(f"\n  T matrix: None")

                    print()  # Extra newline for readability

        print("=" * 80)

    return {
        'all_satisfied': all_satisfied,
        'total_vertices': len(all_vertices),
        'num_violations': sum(1 for r in results if not r['satisfied']),
        'results': results
    }
# ==============================================================================
# STEP 14: Example usage - analyze a single tree
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE: ANALYZING A SINGLE TREE")
print("=" * 80)

tree_idx = 2
root = all_roots_sorted[tree_idx]

# Verify atoms have representations
print(f"\nRoot hopping information:")
print(f"  To atom: {root.hopping.to_atom}")
print(f"  From atom: {root.hopping.from_atom}")
print(f"  To atom has representations: {root.hopping.to_atom.orbital_representations is not None}")
print(f"  From atom has representations: {root.hopping.from_atom.orbital_representations is not None}")

# Print the tree structure (without T matrices first)
print("\n" + "-" * 80)
print(f"TREE {tree_idx} STRUCTURE (without T)")
print("-" * 80)
print_tree(root, show_details=True)

# Run analysis
analysis_result = analyze_tree_constraints(root, parsed_config, tree_idx, verbose=True, tolerance=1e-8)

# Print summary
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print(f"Total parameters in T: {analysis_result['T'].shape[0]} × {analysis_result['T'].shape[1]} = {analysis_result['T'].shape[0] * analysis_result['T'].shape[1]}")
print(f"Number of constraints: {len(analysis_result['unique_equations'])}")
print(f"Rank of constraint matrix: {analysis_result['rank']}")
print(f"Number of free parameters: {analysis_result['nullity']}")

# CRITICAL: Assign T to root before propagation
root.hopping.T = analysis_result['T_reconstructed']

# Now propagate to all children
print("\nPropagating to children:")
propagate_to_all_children(root, verbose=True)

# Print the tree WITH T matrices
print("\n" + "=" * 80)
print(f"TREE {tree_idx} STRUCTURE (with T matrices)")
print("=" * 80)
print_tree_with_T(root, show_T=True)

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)

# Verify constraints for the root
print("\n" + "=" * 80)
print("VERIFYING ROOT CONSTRAINTS")
print("=" * 80)
verify_result = verify_constraint(root, verbose=True)

# Verify constraints for all vertices in the tree
tree_verification = verify_tree_constraints(root, verbose=True)