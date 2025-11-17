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
