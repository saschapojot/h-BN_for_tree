import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# Parameters from the document
l = 1  # Å

# Define basis vectors (in-plane only for 2D view)
a0 = np.array([l, 0])
a1 = np.array([np.cos(2*np.pi/3) * l, np.sin(2*np.pi/3) * l])

# Atomic positions within one unit cell (only x, y coordinates)
r_B = (1/3) * a0 + (2/3) * a1
r_N = (2/3) * a0 + (1/3) * a1

# Create figure
fig, ax = plt.subplots(figsize=(12, 11))

# Define tiling range - increased to show more cells
n_tiles_x = 5  # Number of tiles in a0 direction
n_tiles_y = 5  # Number of tiles in a1 direction

# Store all atoms for plotting
all_B_atoms = []
all_N_atoms = []

# Draw tiled unit cells
for i in range(-1, n_tiles_x):
    for j in range(-1, n_tiles_y):
        # Calculate origin of this unit cell
        origin = i * a0 + j * a1

        # Unit cell corners
        cell_corners = np.array([
            origin,
            origin + a0,
            origin + a0 + a1,
            origin + a1,
            origin
        ])

        # Draw unit cell (no fill, black edges)
        if 0 <= i < n_tiles_x - 1 and 0 <= j < n_tiles_y - 1:
            unit_cell = Polygon(cell_corners[:-1], fill=False,
                                edgecolor='black', linewidth=1.5, alpha=0.3, zorder=1)
            ax.add_patch(unit_cell)

        # Draw cell edges - highlight the cell at (i=1, j=1)
        ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'k-',
                linewidth=1.5 if (i == 1 and j == 1) else 0.8,
                alpha=1.0 if (i == 1 and j == 1) else 0.4, zorder=2)

        # Calculate atomic positions in this cell
        B_pos = origin + r_B
        N_pos = origin + r_N

        all_B_atoms.append(B_pos)
        all_N_atoms.append(N_pos)

# Convert to arrays
all_B_atoms = np.array(all_B_atoms)
all_N_atoms = np.array(all_N_atoms)

# Draw B-N bonds for all cells
for i in range(len(all_B_atoms)):
    B_pos = all_B_atoms[i]
    # Find nearby N atoms and draw bonds
    for N_pos in all_N_atoms:
        distance = np.linalg.norm(B_pos - N_pos)
        if distance < l * 0.6:  # Only draw bonds shorter than this threshold
            ax.plot([B_pos[0], N_pos[0]], [B_pos[1], N_pos[1]],
                    'black', linewidth=1.5, linestyle='-', zorder=3, alpha=0.3)

# Plot all atoms (keeping original colors)
ax.scatter(all_B_atoms[:, 0], all_B_atoms[:, 1], s=400, c='pink',
           edgecolors='darkred', linewidths=2, marker='o', label='B atom', zorder=10)
ax.scatter(all_N_atoms[:, 0], all_N_atoms[:, 1], s=400, c='lightblue',
           edgecolors='darkblue', linewidths=2, marker='o', label='N atom', zorder=10)

# Highlight the primary unit cell (i=1, j=1) - shifted up and right
primary_origin = a0 + a1
primary_cell_corners = np.array([
    primary_origin,
    primary_origin + a0,
    primary_origin + a0 + a1,
    primary_origin + a1,
    primary_origin
])
ax.plot(primary_cell_corners[:, 0], primary_cell_corners[:, 1], 'k-',
        linewidth=3, label='Primary unit cell', zorder=5)

# Draw basis vectors for the primary cell - changed to black, starting from new origin
arrow_props = dict(head_width=0.2, head_length=0.15, linewidth=3, zorder=8)
ax.arrow(primary_origin[0], primary_origin[1], a0[0], a0[1], fc='black', ec='black',
         **arrow_props, alpha=0.8)
ax.arrow(primary_origin[0], primary_origin[1], a1[0], a1[1], fc='black', ec='black',
         **arrow_props, alpha=0.8)

# Add basis vector labels - changed to black
ax.text(primary_origin[0] + a0[0]/2, primary_origin[1] - 0.5, r'$\mathbf{a}_0$', fontsize=16, color='black',
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(primary_origin[0] + a1[0]/2 - 0.4, primary_origin[1] + a1[1]/2, r'$\mathbf{a}_1$', fontsize=16,
        color='black', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add atom labels in primary cell
primary_r_B = primary_origin + r_B
primary_r_N = primary_origin + r_N
ax.text(primary_r_B[0], primary_r_B[1], 'B', ha='center', va='center', fontsize=12,
        fontweight='bold', color='darkred', zorder=11)
ax.text(primary_r_N[0], primary_r_N[1], 'N', ha='center', va='center', fontsize=12,
        fontweight='bold', color='darkblue', zorder=11)

# Draw circle centered at B in primary cell with radius sqrt(3)*l
circle_radius = np.sqrt(3) * l
circle = Circle((primary_r_B[0], primary_r_B[1]), circle_radius, 
                fill=False, edgecolor='red', linewidth=2, 
                linestyle='--', alpha=0.7, zorder=4,
                label=f'Circle centered at B (r=√3l={circle_radius:.3f} Å)')
ax.add_patch(circle)

# Add grid for reference
ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, zorder=0)

# Set axis properties
ax.set_xlabel('x (Å)', fontsize=14, fontweight='bold')
ax.set_ylabel('y (Å)', fontsize=14, fontweight='bold')
ax.set_title('Tiling of 2D Hexagonal Unit Cells: hBN (Space Group P-6m2)',
             fontsize=16, fontweight='bold', pad=20)

# Set equal aspect ratio
ax.set_aspect('equal')

# Set axis limits
padding = 1.0
x_min = -1.5 * l
x_max = (n_tiles_x - 0.5) * l
y_min = -1.5 * l
y_max = (n_tiles_y - 0.5) * l * np.sin(2*np.pi/3)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Add legend
ax.legend(loc='best', fontsize=12, framealpha=0.95,
          edgecolor='black', fancybox=True)

# Add text box with information - changed to white background
textstr = f'Lattice parameter: a = {l} Å\n'
textstr += f'Angle: γ = 120°\n'
textstr += f'B-N distance: {np.linalg.norm(r_B - r_N):.4f} Å\n'
textstr += f'Unit cell area: {abs(np.cross(a0, a1)):.3f} Å²\n'
textstr += f'Tiles shown: {(n_tiles_x-1)} × {(n_tiles_y-1)}\n'
textstr += f'Circle radius: √3l = {circle_radius:.4f} Å'

props = dict(boxstyle='round', facecolor='white', alpha=0.95,
             edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('hBN_tiling_circle_at_B.png', dpi=300, bbox_inches='tight')
# plt.show()

# Print information
print("Hexagonal Tiling Information")
print("=" * 60)
print(f"\nLattice parameter: a = {l} Å")
print(f"Angle between basis vectors: γ = 120°")
print(f"B-N bond length: {np.linalg.norm(r_B - r_N):.4f} Å")
print(f"Unit cell area: {abs(np.cross(a0, a1)):.4f} Å²")
print(f"\nNumber of unit cells shown: {(n_tiles_x-1) * (n_tiles_y-1)}")
print(f"Total atoms shown: {len(all_B_atoms)} B atoms, {len(all_N_atoms)} N atoms")
print(f"\nThe primary unit cell is highlighted in black")
print(f"Each unit cell contains 1 B atom and 1 N atom")
print(f"\nCircle centered at B with radius √3·l = {circle_radius:.4f} Å")