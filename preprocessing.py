import re
import subprocess
import sys
import os
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
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
        print(f"  Vector {i+1}: [{', '.join(map(str, vector))}]")

    # Print space group basis vectors
    print("Space Group Basis:")
    for i, vector in enumerate(parsed_config['space_group_basis']):
        print(f"  Vector {i+1}: [{', '.join(map(str, vector))}]")

    # Print atom types and their orbital information
    print("\nAtom Types:")
    for atom_type, info in parsed_config['atom_types'].items():
        print(f"  {atom_type}:")
        print(f"    Count: {info['count']}")
        print(f"    Orbitals: {info['orbitals']}")

    # Print atom positions in the unit cell
    print(f"\nAtom Positions (Total: {len(parsed_config['atom_positions'])}):")
    for i, pos in enumerate(parsed_config['atom_positions']):
        print(f"  Position {i+1}:")
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
        print(f"  Bilbao (fractional in space group basis): [{', '.join(map(str, parsed_config['space_group_origin']))}]")
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
        print("Available keys:", list(space_group_representations.keys()) if 'space_group_representations' in locals() else "Could not parse JSON")
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
            print(f"  {atom_name}: {repr_array.shape[0]} operations, {repr_array.shape[1]}×{repr_array.shape[2]} matrices")

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
    print("Available keys:", list(orbital_completion_data.keys()) if 'orbital_completion_data' in locals() else "Could not parse JSON")
    exit(1)

except Exception as e:
    print(f"Unexpected error processing orbital completion: {e}")
    print("Type:", type(e).__name__)
    exit(1)

print("\n" + "=" * 60)
print("ORBITAL COMPLETION FINISHED")
print("=" * 60)





def compute_dist(center_frac,center_cell, dest_frac, basis,search_range,radius):
    """

    :param center_frac:
    :param center_cell:
    :param dest_frac:
    :param basis:
    :param search_range:
    :param radius:
    :return:
    """
    f0,f1=center_frac
    n0,n1=center_cell
    
    g0,g1=dest_frac
    # m0,m1=dest_cell
    
    a0,a1,a2=basis
    
    center_coord=np.array(
        (f0+n0)*a0+(f1+n1)*a1
    )
    
    rst=[]
    for j0 in range(-search_range,search_range+1):
        for j1 in range(-search_range,search_range+1):
            dest_coord=np.array(
                (g0+j0)*a0+(g1+j1)*a1
            )
            dist=np.linalg.norm(center_coord-dest_coord,ord=2)
            if dist<=radius:
                rst.append([[j0,j1,0],[g0,g1,0]])

    return rst


atom_types=[]
fractional_positions=[]
for i, pos in enumerate(parsed_config['atom_positions']):
    type_name=pos["atom_type"]
    frac_pos=pos["fractional_coordinates"]
    # print(f"type_name={type_name}, frac_pos={frac_pos}")
    atom_types.append(type_name)
    fractional_positions.append(np.array(frac_pos))

# ==============================================================================
# STEP 7: Partition BB_atoms into equivalent sets under symmetry
# ==============================================================================
print("\n" + "=" * 60)
print("PARTITIONING BB ATOMS INTO EQUIVALENT SETS")
print("=" * 60)
ind0=0
atm0=atom_types[ind0]
center_frac0=(fractional_positions[ind0])[:2]
center_cell=[0,0]
lattice_basis=np.array(parsed_config['lattice_basis'])
l=1.05*np.sqrt(3)

neigboring_BB=compute_dist(center_frac0,center_cell,center_frac0,lattice_basis,7,l)
print(neigboring_BB)
def frac_to_cartesian(cell,frac_coord,basis):
    n0,n1,n2=cell
    f0,f1,f2=frac_coord
    a0,a1,a2=basis
    return (n0+f0)*a0+(n1+f1)*a1+(n2+f2)*a2

neigboring_BB_cartesian=[]
for item in neigboring_BB:
    cell,frac_coord=item
    cart_coord=frac_to_cartesian(cell,frac_coord,lattice_basis)
    neigboring_BB_cartesian.append([cell,cart_coord])

for item in neigboring_BB_cartesian:
    cell, cart_coord=item
    print(f"cell={cell}, cart_coord={cart_coord}")


eps=1e-8

space_group_bilbao_cart=[]
for item in space_group_representations["space_group_matrices_cartesian"]:
    space_group_bilbao_cart.append(np.array(item))

for i,mat in enumerate(space_group_bilbao_cart):
    print(f"===========matrix {i}: ")
    print(f"{mat}")

class atomIndex():
    def __init__(self,cell,frac_coord,atom_name,basis):
        self.n0=cell[0]
        self.n1=cell[1]
        self.n2=cell[2]
        self.atom_name=atom_name
        self.frac_coord=frac_coord
        self.basis=basis
        a0,a1,a2=basis
        f0,f1,f2=frac_coord
        cart_coord=(self.n0+f0)*a0+(self.n1+f1)*a1+(self.n2+f2)*a2
        self.cart_coord=cart_coord

    def __str__(self):
        """String representation for print()"""
        return (f"Atom: {self.atom_name}, "
                f"Cell: [{self.n0}, {self.n1}, {self.n2}], "
                f"Frac: {self.frac_coord}, "
                f"Cart: {self.cart_coord}")

    def __repr__(self):
        """Detailed representation for debugging"""
        return (f"atomIndex(cell=[{self.n0}, {self.n1}, {self.n2}], "
                f"frac_coord={self.frac_coord}, "
                f"atom_name='{self.atom_name}')")

#############center=B, neighbor=B
BB_atoms=[]
for item in neigboring_BB:
    cell,frac_coord=item
    atm=atomIndex(cell,frac_coord,"B",lattice_basis)
    BB_atoms.append(atm)
B_center_frac=list(center_frac0)+[0]

B_center_atom=atomIndex([0,0,0],B_center_frac,atm0,lattice_basis)

def get_next(center_atom,nghb_atom,group_mat):
    R=group_mat[:,:3]
    b=group_mat[:,3]
    # print(R)
    # print(b)
    center_cart_coord=center_atom.cart_coord
    nghb_cart_coord=nghb_atom.cart_coord
    diff_vec=nghb_cart_coord-center_cart_coord
    next_cart_coord=center_cart_coord+R@diff_vec+b
    return next_cart_coord


# ==============================================================================
# STEP 8: Partition all BB_atoms into equivalent sets
# ==============================================================================




class hopping():
    def __init__(self, to_atom, from_atom, class_id, operation_idx, rotation_matrix, translation_vector):
        self.to_atom = to_atom
        self.from_atom = from_atom
        self.class_id = class_id
        self.operation_idx = operation_idx  # Which space group operation
        self.rotation_matrix = rotation_matrix  # R (3x3 matrix)
        self.translation_vector = translation_vector  # b (3-vector)

    def conjugate(self):
        return [deepcopy(self.from_atom), deepcopy(self.to_atom)]

    def __repr__(self):
        return (f"hopping(from={self.from_atom}, to={self.to_atom}, "
                f"class_id={self.class_id}, op={self.operation_idx})")


print("\n" + "=" * 60)
print("PARTITIONING ALL BB_ATOMS INTO EQUIVALENT SETS")
print("=" * 60)

# Find identity operation index
identity_idx = None
for idx, group_mat in enumerate(space_group_bilbao_cart):
    if np.allclose(group_mat[:3, :3], np.eye(3)) and np.allclose(group_mat[:3, 3], 0):
        identity_idx = idx
        print(f"Identity operation found at index {identity_idx}")
        break

if identity_idx is None:
    print("WARNING: Identity operation not found in space_group_bilbao_cart!")

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
    # Key: atom index in BB_atoms, Value: (operation index, atom object)
    equivalent_dict = {}

    # List to store hoppings for this equivalent set
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
            # Skip this operation if it moves the center atom
            continue

        # Get the transformed coordinate from seed atom
        next_coord = get_next(B_center_atom, seed_atom, group_mat)

        # Calculate distance of transformed coordinate to center
        next_distance = np.linalg.norm(next_coord - B_center_atom.cart_coord)

        # Check if distance is equal to seed distance (within tolerance)
        if abs(next_distance - seed_distance) > 1e-6:
            # Skip this transformed atom if distance doesn't match
            continue

        # Check if this coordinate matches any atom in BB_atoms
        for idx, atm in enumerate(BB_atoms):
            if idx not in equivalent_dict:  # Skip already found atoms
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    # Record this atom and which operation generated it
                    equivalent_dict[idx] = (op_idx, atm)

                    # Extract rotation and translation from group matrix
                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]

                    # Create hopping: from equivalent atom to center
                    hop = hopping(
                        to_atom=B_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)

                    print(f"  Operation {op_idx} generates atom {idx}, distance={next_distance:.6f}")

    # Get list of equivalent atom indices
    equivalent_indices = sorted(equivalent_dict.keys())

    print(f"Found {len(equivalent_indices)} equivalent atoms: indices {equivalent_indices}")

    # Collect the actual atoms for this equivalent set
    current_atom_set = [BB_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_BB.append(current_atom_set)
    equivalent_hopping_sets_BB.append(current_hopping_set)

    # Print details of atoms in this set
    print("Atoms in this set:")
    for idx in equivalent_indices:
        atm = BB_atoms[idx]
        op_idx, _ = equivalent_dict[idx]
        dist = np.linalg.norm(atm.cart_coord - B_center_atom.cart_coord)
        op_str = f"op {op_idx}" + (" (identity)" if op_idx == identity_idx else "")
        print(f"  Atom {idx} ({op_str}): Cell=[{atm.n0:2d},{atm.n1:2d},{atm.n2:2d}], "
              f"Frac=[{atm.frac_coord[0]:.4f},{atm.frac_coord[1]:.4f},{atm.frac_coord[2]:.4f}], "
              f"Dist={dist:.6f}")

    # Remove equivalent atoms from BB_atoms
    BB_atoms = [atm for i, atm in enumerate(BB_atoms) if i not in equivalent_indices]

    print(f"Remaining BB_atoms: {len(BB_atoms)}")

print("\n" + "=" * 60)
print("PARTITIONING COMPLETE")
print("=" * 60)
print(f"Total number of equivalent sets: {len(equivalent_atom_sets_BB)}")
print(f"Total number of hopping sets: {len(equivalent_hopping_sets_BB)}")


for j,st in enumerate(equivalent_hopping_sets_BB):
    print(f"set {j} ********************************************************")
    for hp in st:
        print(hp)


# Updated vertex class with children and parent
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



# Store all trees (now just storing the root vertices)
tree_roots = []
for set_idx, hopping_set in enumerate(equivalent_hopping_sets_BB):
    print(f"\n--- Building tree for Set {set_idx} ---")
    print(f"Total hoppings in set: {len(hopping_set)}")
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
    print(f"Root hopping: {root_hopping}")
    print(f"Number of child hoppings: {len(child_hoppings)}")
    # Create root vertex
    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)
    # Create child vertices and link them to root
    for hop in child_hoppings:
        child_type = "linear"
        child_v = vertex(hopping=hop, type=child_type, identity_idx=identity_idx)
        # Add this child to the root's children list
        root_vertex.add_child(child_v)
    # Store the root (which now contains references to all its children)
    tree_roots.append(root_vertex)
    print(f"Tree built: 1 root with {len(root_vertex.children)} children")
    print(f"  Root: {root_vertex}")
    for i, child in enumerate(root_vertex.children[:3]):
        print(f"    Child {i}: {child}")
    if len(root_vertex.children) > 3:
        print(f"    ... and {len(root_vertex.children) - 3} more children")



# ==============================================================================
# STEP 9: Partition BN_atoms into equivalent sets under symmetry
# ==============================================================================
ind1=1
atm1=atom_types[ind1]
# print(f"ind1={ind1}, atm1={atm1}")
center_frac1=(fractional_positions[ind1])[:2]
center_cell=[0,0]
lattice_basis=np.array(parsed_config['lattice_basis'])
l=1.05*np.sqrt(3)

neigboring_BN=compute_dist(center_frac0,center_cell,center_frac1,lattice_basis,7,l)
neigboring_BN_cartesian=[]
for item in neigboring_BN:
    cell,frac_coord=item
    cart_coord = frac_to_cartesian(cell, frac_coord, lattice_basis)
    neigboring_BN_cartesian.append([cell,cart_coord])


# for item in neigboring_BN_cartesian:
#     cell, cart_coord=item
#     print(f"cell={cell}, cart_coord={cart_coord}")

# ==============================================================================
# STEP 10: Partition all BN_atoms into equivalent sets
# ==============================================================================
# Create BN_atoms list (analogous to BB_atoms)
BN_atoms = []
for item in neigboring_BN:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "N", lattice_basis)  # N atoms
    BN_atoms.append(atm)
print("\n" + "=" * 60)
print("PARTITIONING ALL BN_ATOMS INTO EQUIVALENT SETS")
print("=" * 60)

equivalent_atom_sets_BN = []
equivalent_hopping_sets_BN = []
set_counter = 0
while len(BN_atoms) > 0:
    set_counter += 1
    print(f"\n--- Equivalent Set {set_counter} ---")
    # Take the first atom from remaining BN_atoms as seed
    seed_atom = BN_atoms[0]
    print(f"Seed atom: {seed_atom}")
    # Calculate seed atom's distance to center
    seed_distance = np.linalg.norm(seed_atom.cart_coord - B_center_atom.cart_coord)
    print(f"Seed distance to center: {seed_distance:.6f}")
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
        # Check if this coordinate matches any atom in BN_atoms
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
                    print(f"  Operation {op_idx} generates atom {idx}, distance={next_distance:.6f}")

    equivalent_indices = sorted(equivalent_dict.keys())
    print(f"Found {len(equivalent_indices)} equivalent atoms: indices {equivalent_indices}")
    current_atom_set = [BN_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_BN.append(current_atom_set)
    equivalent_hopping_sets_BN.append(current_hopping_set)

    # Remove equivalent atoms from BN_atoms
    BN_atoms = [atm for i, atm in enumerate(BN_atoms) if i not in equivalent_indices]
    print(f"Remaining BN_atoms: {len(BN_atoms)}")

print("\n" + "=" * 60)
print("BN PARTITIONING COMPLETE")
print("=" * 60)
print(f"Total number of BN equivalent sets: {len(equivalent_atom_sets_BN)}")

# for j, st in enumerate(equivalent_atom_sets_BN):
#     print(f"equivalent class {j}: ----------------------------------------------")
#     for item in st:
#         print(item)



# ==============================================================================
# STEP 11: Build trees for BN hoppings
# ==============================================================================
print("\n" + "=" * 60)
print("BUILDING TREES FOR BN HOPPINGS")
print("=" * 60)

# Store all BN trees (root vertices)
tree_roots_BN = []

for set_idx, hopping_set in enumerate(equivalent_hopping_sets_BN):
    print(f"\n--- Building tree for BN Set {set_idx} ---")
    print(f"Total hoppings in set: {len(hopping_set)}")
    # Find the root hopping (identity operation)
    root_hopping = None
    child_hoppings = []
    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)
    if root_hopping is None:
        print(f"WARNING: No identity hopping found in BN set {set_idx}!")
        continue

    print(f"Root hopping: {root_hopping}")
    print(f"Number of child hoppings: {len(child_hoppings)}")
    # Create root vertex
    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)
    # Create child vertices and link them to root
    for hop in child_hoppings:
        child_type = "linear"
        child_v = vertex(hopping=hop, type=child_type, identity_idx=identity_idx)
        # Add this child to the root's children list
        root_vertex.add_child(child_v)
    # Store the root (which now contains references to all its children)
    tree_roots_BN.append(root_vertex)
    print(f"Tree built: 1 root with {len(root_vertex.children)} children")
    print(f"  Root: {root_vertex}")
    for i, child in enumerate(root_vertex.children[:3]):
        print(f"    Child {i}: {child}")
    if len(root_vertex.children) > 3:
        print(f"    ... and {len(root_vertex.children) - 3} more children")




def print_tree(root, prefix="", is_last=True, show_details=True):
    """
    Print a tree structure in a visual format

    Args:
        root: vertex object (root of tree or subtree)
        prefix: string prefix for indentation
        is_last: whether this is the last child at this level
        show_details: whether to show detailed hopping information
    """
    # Determine the connector symbol
    connector = "└── " if is_last else "├── "

    # Print current node
    if root.is_root:
        node_label = "ROOT"
        style = "╔═══"
    else:
        node_label = f"CHILD ({root.type})"
        style = connector

    # Build the node description
    hop = root.hopping
    from_cell = [hop.from_atom.n0, hop.from_atom.n1, hop.from_atom.n2]
    from_frac = hop.from_atom.frac_coord
    to_atom_name = hop.to_atom.atom_name
    from_atom_name = hop.from_atom.atom_name

    distance = np.linalg.norm(hop.from_atom.cart_coord - hop.to_atom.cart_coord)

    if show_details:
        node_desc = (f"{node_label} | Op={hop.operation_idx:2d} | "
                     f"{from_atom_name}→{to_atom_name} | "
                     f"Cell=[{from_cell[0]:2d},{from_cell[1]:2d},{from_cell[2]:2d}] | "
                     f"Dist={distance:.4f}")
    else:
        node_desc = f"{node_label} | Op={hop.operation_idx:2d}"

    print(f"{prefix}{style}{node_desc}")

    # Print children
    if root.children:
        # Update prefix for children
        if root.is_root:
            new_prefix = prefix + "    "
        else:
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension

        # Print each child
        for i, child in enumerate(root.children):
            is_last_child = (i == len(root.children) - 1)
            print_tree(child, new_prefix, is_last_child, show_details)


# ==============================================================================
# STEP 12: Partition NB_atoms into equivalent sets under symmetry
# ==============================================================================

ind1 = 1
atm1 = atom_types[ind1]
center_frac1 = (fractional_positions[ind1])[:2]
center_cell = [0, 0]
lattice_basis = np.array(parsed_config['lattice_basis'])
l = 1.05 * np.sqrt(3)

ind0 = 0
atm0 = atom_types[ind0]
center_frac0 = (fractional_positions[ind0])[:2]

neigboring_NB = compute_dist(center_frac1, center_cell, center_frac0, lattice_basis, 7, l)

neigboring_NB_cartesian = []
for item in neigboring_NB:
    cell, frac_coord = item
    cart_coord = frac_to_cartesian(cell, frac_coord, lattice_basis)
    neigboring_NB_cartesian.append([cell, cart_coord])

# Create NB_atoms list (Nitrogen center, Boron neighbors)
NB_atoms = []
for item in neigboring_NB:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "B", lattice_basis)  # B atoms
    NB_atoms.append(atm)

# Create N center atom
N_center_frac = list(center_frac1) + [0]
N_center_atom = atomIndex([0, 0, 0], N_center_frac, atm1, lattice_basis)

print("\n" + "=" * 60)
print("PARTITIONING ALL NB_ATOMS INTO EQUIVALENT SETS")
print("=" * 60)

equivalent_atom_sets_NB = []
equivalent_hopping_sets_NB = []
set_counter = 0

while len(NB_atoms) > 0:
    set_counter += 1
    print(f"\n--- Equivalent Set {set_counter} ---")

    # Take the first atom from remaining NB_atoms as seed
    seed_atom = NB_atoms[0]
    print(f"Seed atom: {seed_atom}")

    # Calculate seed atom's distance to center (N atom)
    seed_distance = np.linalg.norm(seed_atom.cart_coord - N_center_atom.cart_coord)
    print(f"Seed distance to center: {seed_distance:.6f}")

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
        to_atom=N_center_atom,
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

        # Check if this operation leaves the center atom (N) invariant
        center_transformed = get_next(N_center_atom, N_center_atom, group_mat)
        diff_center = center_transformed - N_center_atom.cart_coord

        if np.linalg.norm(diff_center) > 1e-6:
            # Skip this operation if it moves the center atom
            continue

        # Get the transformed coordinate from seed atom
        next_coord = get_next(N_center_atom, seed_atom, group_mat)

        # Calculate distance of transformed coordinate to center
        next_distance = np.linalg.norm(next_coord - N_center_atom.cart_coord)

        # Check if distance is equal to seed distance (within tolerance)
        if abs(next_distance - seed_distance) > 1e-6:
            # Skip this transformed atom if distance doesn't match
            continue

        # Check if this coordinate matches any atom in NB_atoms
        for idx, atm in enumerate(NB_atoms):
            if idx not in equivalent_dict:  # Skip already found atoms
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    # Record this atom and which operation generated it
                    equivalent_dict[idx] = (op_idx, atm)

                    # Extract rotation and translation from group matrix
                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]

                    # Create hopping: from equivalent atom to center
                    hop = hopping(
                        to_atom=N_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)

                    print(f"  Operation {op_idx} generates atom {idx}, distance={next_distance:.6f}")

    # Get list of equivalent atom indices
    equivalent_indices = sorted(equivalent_dict.keys())

    print(f"Found {len(equivalent_indices)} equivalent atoms: indices {equivalent_indices}")

    # Collect the actual atoms for this equivalent set
    current_atom_set = [NB_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_NB.append(current_atom_set)
    equivalent_hopping_sets_NB.append(current_hopping_set)

    # Print details of atoms in this set
    print("Atoms in this set:")
    for idx in equivalent_indices:
        atm = NB_atoms[idx]
        op_idx, _ = equivalent_dict[idx]
        dist = np.linalg.norm(atm.cart_coord - N_center_atom.cart_coord)
        op_str = f"op {op_idx}" + (" (identity)" if op_idx == identity_idx else "")
        print(f"  Atom {idx} ({op_str}): Cell=[{atm.n0:2d},{atm.n1:2d},{atm.n2:2d}], "
              f"Frac=[{atm.frac_coord[0]:.4f},{atm.frac_coord[1]:.4f},{atm.frac_coord[2]:.4f}], "
              f"Dist={dist:.6f}")

    # Remove equivalent atoms from NB_atoms
    NB_atoms = [atm for i, atm in enumerate(NB_atoms) if i not in equivalent_indices]

    print(f"Remaining NB_atoms: {len(NB_atoms)}")

print("\n" + "=" * 60)
print("NB PARTITIONING COMPLETE")
print("=" * 60)
print(f"Total number of NB equivalent sets: {len(equivalent_atom_sets_NB)}")

# ==============================================================================
# STEP 13: Build trees for NB hoppings
# ==============================================================================
print("\n" + "=" * 60)
print("BUILDING TREES FOR NB HOPPINGS")
print("=" * 60)

# Store all NB trees (root vertices)
tree_roots_NB = []
for set_idx, hopping_set in enumerate(equivalent_hopping_sets_NB):
    print(f"\n--- Building tree for NB Set {set_idx} ---")
    print(f"Total hoppings in set: {len(hopping_set)}")
    # Find the root hopping (identity operation)
    root_hopping = None
    child_hoppings = []
    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)
    if root_hopping is None:
        print(f"WARNING: No identity hopping found in NB set {set_idx}!")
        continue
    print(f"Root hopping: {root_hopping}")
    print(f"Number of child hoppings: {len(child_hoppings)}")
    # Create root vertex
    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)
    # Create child vertices and link them to root
    for hop in child_hoppings:
        child_type = "linear"
        child_v = vertex(hopping=hop, type=child_type, identity_idx=identity_idx)
        # Add this child to the root's children list
        root_vertex.add_child(child_v)
    # Store the root (which now contains references to all its children)
    tree_roots_NB.append(root_vertex)
    print(f"Tree built: 1 root with {len(root_vertex.children)} children")
    print(f"  Root: {root_vertex}")
    for i, child in enumerate(root_vertex.children[:3]):
        print(f"    Child {i}: {child}")
    if len(root_vertex.children) > 3:
        print(f"    ... and {len(root_vertex.children) - 3} more children")

# ==============================================================================
# STEP 14: Print tree structures for BB hoppings
# ==============================================================================
print("\n" + "=" * 80)
print("BB HOPPING TREE STRUCTURES")
print("=" * 80)

# Print all BB trees
for set_idx, root in enumerate(tree_roots):
    print(f"\n{'─' * 80}")
    print(f"EQUIVALENCE CLASS {set_idx} (BB: Boron center, Boron neighbors)")
    print(f"{'─' * 80}")
    print(f"Total nodes: {1 + len(root.children)} (1 root + {len(root.children)} children)")

    # Get distance for this equivalence class
    distance = np.linalg.norm(root.hopping.from_atom.cart_coord - root.hopping.to_atom.cart_coord)
    print(f"Distance to center: {distance:.6f}")
    print()

    print_tree(root, prefix="", is_last=True, show_details=True)

# ==============================================================================
# STEP 15: Print tree structures for BN hoppings
# ==============================================================================
print("\n\n" + "=" * 80)
print("BN HOPPING TREE STRUCTURES")
print("=" * 80)

# Print all BN trees
for set_idx, root in enumerate(tree_roots_BN):
    print(f"\n{'─' * 80}")
    print(f"EQUIVALENCE CLASS {set_idx} (BN: Boron center, Nitrogen neighbors)")
    print(f"{'─' * 80}")
    print(f"Total nodes: {1 + len(root.children)} (1 root + {len(root.children)} children)")

    # Get distance for this equivalence class
    distance = np.linalg.norm(root.hopping.from_atom.cart_coord - root.hopping.to_atom.cart_coord)
    print(f"Distance to center: {distance:.6f}")
    print()

    print_tree(root, prefix="", is_last=True, show_details=True)

# ==============================================================================
# STEP 16: Print tree structures for NB hoppings
# ==============================================================================
print("\n\n" + "=" * 80)
print("NB HOPPING TREE STRUCTURES")
print("=" * 80)

# Print all NB trees
for set_idx, root in enumerate(tree_roots_NB):
    print(f"\n{'─' * 80}")
    print(f"EQUIVALENCE CLASS {set_idx} (NB: Nitrogen center, Boron neighbors)")
    print(f"{'─' * 80}")
    print(f"Total nodes: {1 + len(root.children)} (1 root + {len(root.children)} children)")

    # Get distance for this equivalence class
    distance = np.linalg.norm(root.hopping.from_atom.cart_coord - root.hopping.to_atom.cart_coord)
    print(f"Distance to center: {distance:.6f}")
    print()

    print_tree(root, prefix="", is_last=True, show_details=True)

# ==============================================================================
# STEP 17: find hermitian relation between BN and NB
# ==============================================================================
def check_hermitian(hopping1, hopping2):
    to_atom1=hopping1.to_atom
    from_atom1=hopping1.from_atom

    to_atom2c,from_atom2c=hopping2.conjugate()
    # Get lattice basis vectors
    a0, a1, a2 = lattice_basis
    # Iterate through all space group operations
    for op_idx, group_mat in enumerate(space_group_bilbao_cart):
        R = group_mat[:3, :3]  # Rotation matrix
        b = group_mat[:3, 3]  # Translation vector
        # Transform to_atom1 and from_atom1
        to_atom1_transformed_cart_coord = R @ to_atom1.cart_coord + b
        from_atom1_transformed_cart_coord = R @ from_atom1.cart_coord + b
        # Check if there exist integers n0, n1, n2 such that:
        # to_atom1_transformed_cart_coord + n0*a0 + n1*a1 + n2*a2 = to_atom2c.cart_coord
        # from_atom1_transformed_cart_coord + n0*a0 + n1*a1 + n2*a2 = from_atom2c.cart_coord
        # Compute the difference vectors
        diff_to = to_atom2c.cart_coord - to_atom1_transformed_cart_coord
        diff_from = from_atom2c.cart_coord - from_atom1_transformed_cart_coord
        # Check if diff_to and diff_from are the same lattice vector
        if np.linalg.norm(diff_to - diff_from) < 1e-6:
            # They differ by the same lattice vector, now check if it's an integer combination
            # Solve: n0*a0 + n1*a1 + n2*a2 = diff_to
            # This is: [a0 a1 a2] @ [n0, n1, n2]^T = diff_to
            lattice_matrix = np.column_stack([a0, a1, a2])
            try:
                # Solve for [n0, n1, n2]
                n_vector = np.linalg.solve(lattice_matrix, diff_to)
                # Check if n_vector contains integers (within tolerance)
                n_rounded = np.round(n_vector)
                if np.allclose(n_vector, n_rounded, atol=1e-6):
                    # Found a valid Hermitian conjugate relationship
                    return True, op_idx
            except np.linalg.LinAlgError:
                # Singular matrix, skip this operation
                continue
    return False, None


print("\n" + "=" * 80)
print("GROUPING BN AND NB ROOTS BY HERMITIAN CONJUGATE RELATIONS")
print("=" * 80)
# Track which roots have been matched
nb_matched = set()
bn_matched = set()

# List to store grouped roots
hermitian_groups = []
independent_groups=[]

# For each NB root, try to find its Hermitian conjugate in BN roots
for nb_idx, nb_root in enumerate(tree_roots_NB):
    if nb_idx in nb_matched:
        continue
    found_match = False
    for bn_idx, bn_root in enumerate(tree_roots_BN):
        if bn_idx in bn_matched:
            continue
        exists, op_idx = check_hermitian(bn_root.hopping,nb_root.hopping)
        if exists:
            # Found a matching pair - bn_root first, nb_root second
            hermitian_groups.append([bn_root, nb_root,op_idx])
            nb_matched.add(nb_idx)
            bn_matched.add(bn_idx)
            print(f"Group {len(hermitian_groups) - 1}: BN root {bn_idx} to NB root {nb_idx} (op: {op_idx})")

            found_match = True
            break
    if not found_match:
        # No match found, put NB root alone
        independent_groups.append([nb_root])
        nb_matched.add(nb_idx)

# Process hermitian groups - add root2 as child of root1
for bn_root, nb_root, op_idx in hermitian_groups:
    # Add nb_root as child of bn_root
    bn_root.children.append(nb_root)

    # Update nb_root properties
    nb_root.type = "hermitian"
    nb_root.is_root = False
    nb_root.parent = bn_root
    nb_root.hopping.operation_idx = op_idx

    # Update the root's hopping operation_idx if not already set
    if bn_root.hopping.operation_idx is None:
        # Find the operation that maps bn_root to itself (identity-like operation)
        # This should already be set from the symmetry tree building, but just in case
        pass

print(f"\nProcessed {len(hermitian_groups)} hermitian pairs")
print(f"Added {len(hermitian_groups)} NB roots as hermitian children of BN roots")

# Print verification
print("\n" + "─" * 80)
print("VERIFICATION OF HERMITIAN RELATIONSHIPS")
print("─" * 80)
for idx, (bn_root, nb_root, op_idx) in enumerate(hermitian_groups):
    print(f"\nHermitian pair {idx}:")
    print(f"  BN root (parent): operation {bn_root.hopping.operation_idx}, {len(bn_root.children)} children")
    print(
        f"  NB root (child): operation {nb_root.hopping.operation_idx}, is_root={nb_root.is_root}, type={nb_root.type}")
    print(f"  Hermitian operation: {op_idx}")


# ==============================================================================
# STEP 18: Partition NN_atoms into equivalent sets under symmetry
# ==============================================================================
ind1=1
atm1=atom_types[ind1]
# print(f"ind1={ind1}, atm1={atm1}")
center_frac1=(fractional_positions[ind1])[:2]
center_cell=[0,0]
lattice_basis=np.array(parsed_config['lattice_basis'])
l=1.05*np.sqrt(3)

neigboring_NN=compute_dist(center_frac1,center_cell,center_frac1,lattice_basis,7,l)
neigboring_NN_cartesian=[]
for item in neigboring_NN:
    cell, frac_coord = item
    cart_coord = frac_to_cartesian(cell, frac_coord, lattice_basis)
    neigboring_NN_cartesian.append([cell, cart_coord])

# for item in neigboring_BN_cartesian:
#     cell, cart_coord=item
#     print(f"cell={cell}, cart_coord={cart_coord}")
# ==============================================================================
# STEP 19: Partition all NN_atoms into equivalent sets
# ==============================================================================
NN_atoms = []
for item in neigboring_NN:
    cell, frac_coord = item
    atm = atomIndex(cell, frac_coord, "N", lattice_basis)  # N atoms
    NN_atoms.append(atm)

equivalent_atom_sets_NN = []
equivalent_hopping_sets_NN = []
set_counter = 0
while len(NN_atoms) > 0:
    set_counter += 1
    print(f"\n--- Equivalent Set {set_counter} ---")
    # Take the first atom from remaining NN_atoms as seed
    seed_atom = NN_atoms[0]
    print(f"Seed atom: {seed_atom}")
    # Calculate seed atom's distance to center (N atom)
    seed_distance = np.linalg.norm(seed_atom.cart_coord - N_center_atom.cart_coord)
    print(f"Seed distance to center: {seed_distance:.6f}")
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
        to_atom=N_center_atom,
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
        # Check if this operation leaves the center atom (N) invariant
        center_transformed = get_next(N_center_atom, N_center_atom, group_mat)
        diff_center = center_transformed - N_center_atom.cart_coord
        if np.linalg.norm(diff_center) > 1e-6:
            # Skip this operation if it moves the center atom
            continue
        # Get the transformed coordinate from seed atom
        next_coord = get_next(N_center_atom, seed_atom, group_mat)
        # Calculate distance of transformed coordinate to center
        next_distance = np.linalg.norm(next_coord - N_center_atom.cart_coord)
        # Check if distance is equal to seed distance (within tolerance)
        if abs(next_distance - seed_distance) > 1e-6:
            # Skip this transformed atom if distance doesn't match
            continue
        # Check if this coordinate matches any atom in NN_atoms
        for idx, atm in enumerate(NN_atoms):
            if idx not in equivalent_dict:  # Skip already found atoms
                diff = next_coord - atm.cart_coord
                if np.linalg.norm(diff) < 1e-6:
                    # Record this atom and which operation generated it
                    equivalent_dict[idx] = (op_idx, atm)
                    # Extract rotation and translation from group matrix
                    rotation = group_mat[:3, :3]
                    translation = group_mat[:3, 3]
                    # Create hopping: from equivalent atom to center
                    hop = hopping(
                        to_atom=N_center_atom,
                        from_atom=atm,
                        class_id=set_counter - 1,
                        operation_idx=op_idx,
                        rotation_matrix=rotation,
                        translation_vector=translation
                    )
                    current_hopping_set.append(hop)
                    print(f"  Operation {op_idx} generates atom {idx}, distance={next_distance:.6f}")

    # Get list of equivalent atom indices
    equivalent_indices = sorted(equivalent_dict.keys())
    print(f"Found {len(equivalent_indices)} equivalent atoms: indices {equivalent_indices}")
    # Collect the actual atoms for this equivalent set
    current_atom_set = [NN_atoms[idx] for idx in equivalent_indices]
    equivalent_atom_sets_NN.append(current_atom_set)
    equivalent_hopping_sets_NN.append(current_hopping_set)
    # Print details of atoms in this set
    print("Atoms in this set:")
    for idx in equivalent_indices:
        atm = NN_atoms[idx]
        op_idx, _ = equivalent_dict[idx]
        dist = np.linalg.norm(atm.cart_coord - N_center_atom.cart_coord)
        op_str = f"op {op_idx}" + (" (identity)" if op_idx == identity_idx else "")
        print(f"  Atom {idx} ({op_str}): Cell=[{atm.n0:2d},{atm.n1:2d},{atm.n2:2d}], "
              f"Frac=[{atm.frac_coord[0]:.4f},{atm.frac_coord[1]:.4f},{atm.frac_coord[2]:.4f}], "
              f"Dist={dist:.6f}")
    # Remove equivalent atoms from NN_atoms
    NN_atoms = [atm for i, atm in enumerate(NN_atoms) if i not in equivalent_indices]
    print(f"Remaining NN_atoms: {len(NN_atoms)}")

print("\n" + "=" * 60)
print("NN PARTITIONING COMPLETE")
print("=" * 60)
print(f"Total number of NN equivalent sets: {len(equivalent_atom_sets_NN)}")



# ==============================================================================
# STEP 20: Build trees for NN hoppings
# ==============================================================================
print("\n" + "=" * 60)
print("BUILDING TREES FOR NN HOPPINGS")
print("=" * 60)
# Store all NN trees (root vertices)
tree_roots_NN = []

for set_idx, hopping_set in enumerate(equivalent_hopping_sets_NN):
    print(f"\n--- Building tree for NN Set {set_idx} ---")
    print(f"Total hoppings in set: {len(hopping_set)}")
    # Find the root hopping (identity operation)
    root_hopping = None
    child_hoppings = []

    for hop in hopping_set:
        if hop.operation_idx == identity_idx:
            root_hopping = hop
        else:
            child_hoppings.append(hop)

    if root_hopping is None:
        print(f"WARNING: No identity hopping found in NN set {set_idx}!")
        continue
    print(f"Root hopping: {root_hopping}")
    print(f"Number of child hoppings: {len(child_hoppings)}")
    # Create root vertex
    root_vertex = vertex(hopping=root_hopping, type=None, identity_idx=identity_idx)
    # Create child vertices and link them to root
    for hop in child_hoppings:
        child_type = "linear"
        child_v = vertex(hopping=hop, type=child_type, identity_idx=identity_idx)
        # Add this child to the root's children list
        root_vertex.add_child(child_v)
    # Store the root (which now contains references to all its children)

    tree_roots_NN.append(root_vertex)
    print(f"Tree built: 1 root with {len(root_vertex.children)} children")
    print(f"  Root: {root_vertex}")
    for i, child in enumerate(root_vertex.children[:3]):
        print(f"    Child {i}: {child}")
    if len(root_vertex.children) > 3:
        print(f"    ... and {len(root_vertex.children) - 3} more children")

# Collect all roots from different groups
all_roots = []

# Add BB roots
all_roots.extend(tree_roots)
print(f"Added {len(tree_roots)} BB roots")

# Add BN roots (which now include NB roots as hermitian children)
all_roots.extend(tree_roots_BN)
print(f"Added {len(tree_roots_BN)} BN roots")

# Add NN roots
all_roots.extend(tree_roots_NN)
print(f"Added {len(tree_roots_NN)} NN roots")

# Add independent NB and BN roots (leftovers without hermitian pairs)
all_roots.extend([group[0] for group in independent_groups])
print(f"Added {len(independent_groups)} independent roots")

print("\n" + "─" * 80)
print("SUMMARY OF ALL ROOTS")
print("─" * 80)
print(f"Total roots: {len(all_roots)}")
print(f"  BB roots: {len(tree_roots)}")
print(f"  BN roots (with hermitian children): {len(tree_roots_BN)}")
print(f"  NN roots: {len(tree_roots_NN)}")
print(f"  Independent roots: {len(independent_groups)}")
print("\n" + "=" * 80)

print("\n" + "=" * 80)
print("ALL SYMMETRY TREES")
print("=" * 80)

for tree_idx, root in enumerate(all_roots):
    print(f"\n{'=' * 80}")
    print(f"TREE {tree_idx}")
    print(f"{'=' * 80}")

    # Print root information
    hopping = root.hopping
    from_cell = [hopping.from_atom.n0, hopping.from_atom.n1, hopping.from_atom.n2]
    to_cell = [hopping.to_atom.n0, hopping.to_atom.n1, hopping.to_atom.n2]

    print(f"ROOT: {hopping.from_atom.atom_name}[{from_cell[0]},{from_cell[1]},{from_cell[2]}] -> "
          f"{hopping.to_atom.atom_name}[{to_cell[0]},{to_cell[1]},{to_cell[2]}]")
    print(f"  Distance: {np.linalg.norm(hopping.from_atom.cart_coord - hopping.to_atom.cart_coord):.6f} Å")
    print(f"  Operation: {hopping.operation_idx}")
    print(f"  Type: {root.type}")
    print(f"  Children: {len(root.children)}")
    print()

    # Print the tree using the previously defined function
    print_tree(root)

    # Print statistics for this tree
    def count_nodes(node):
        count = 1
        for child in node.children:
            count += count_nodes(child)
        return count

    total_nodes = count_nodes(root)
    print(f"\nTree statistics:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Root children: {len(root.children)}")

print("\n" + "=" * 80)
print(f"TOTAL TREES: {len(all_roots)}")
print("=" * 80)