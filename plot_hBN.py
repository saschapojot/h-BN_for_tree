import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Parameters from the document
l = 2.510  # Å

# Define basis vectors (in-plane only for 2D view)
a0 = np.array([l, 0])
a1 = np.array([np.cos(2*np.pi/3) * l, np.sin(2*np.pi/3) * l])

# Atomic positions within one unit cell (only x, y coordinates)
r_B = (1/3) * a0 + (2/3) * a1
r_N = (2/3) * a0 + (1/3) * a1

# Create figure
fig, ax = plt.subplots(figsize=(12, 11))

# Define tiling range
n_tiles_x = 4  # Number of tiles in a0 direction
n_tiles_y = 4  # Number of tiles in a1 direction

# Store all atoms for plotting
all_B_atoms = []
all_N_atoms = []

# Color maps for different cells (to distinguish them)
colors = plt.cm.Set3(np.linspace(0, 1, 12))
color_idx = 0

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

        # Draw unit cell with alternating colors
        if 0 <= i < n_tiles_x - 1 and 0 <= j < n_tiles_y - 1:
            unit_cell = Polygon(cell_corners[:-1], fill=True,
                                facecolor=colors[color_idx % len(colors)],
                                edgecolor='black', linewidth=1.5, alpha=0.15, zorder=1)
            ax.add_patch(unit_cell)
            color_idx += 1

        # Draw cell edges
        ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'k-',
                linewidth=1.5 if (i == 0 and j == 0) else 0.8,
                alpha=1.0 if (i == 0 and j == 0) else 0.4, zorder=2)

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
                    'gray', linewidth=1.5, linestyle='-', zorder=3, alpha=0.3)

# Plot all atoms
ax.scatter(all_B_atoms[:, 0], all_B_atoms[:, 1], s=400, c='pink',
           edgecolors='darkred', linewidths=2, marker='o', label='B atom', zorder=10)
ax.scatter(all_N_atoms[:, 0], all_N_atoms[:, 1], s=400, c='lightblue',
           edgecolors='darkblue', linewidths=2, marker='o', label='N atom', zorder=10)

# Highlight the primary unit cell (i=0, j=0)
primary_cell_corners = np.array([
    [0, 0],
    a0,
    a0 + a1,
    a1,
    [0, 0]
])
ax.plot(primary_cell_corners[:, 0], primary_cell_corners[:, 1], 'r-',
        linewidth=3, label='Primary unit cell', zorder=5)

# Draw basis vectors for the primary cell
arrow_props = dict(head_width=0.2, head_length=0.15, linewidth=3, zorder=8)
ax.arrow(0, 0, a0[0], a0[1], fc='darkgreen', ec='darkgreen',
         **arrow_props, alpha=0.8)
ax.arrow(0, 0, a1[0], a1[1], fc='darkorange', ec='darkorange',
         **arrow_props, alpha=0.8)

# Add basis vector labels
ax.text(a0[0]/2, -0.5, r'$\mathbf{a}_0$', fontsize=16, color='darkgreen',
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(a1[0]/2 - 0.4, a1[1]/2, r'$\mathbf{a}_1$', fontsize=16,
        color='darkorange', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add atom labels in primary cell
ax.text(r_B[0], r_B[1], 'B', ha='center', va='center', fontsize=12,
        fontweight='bold', color='darkred', zorder=11)
ax.text(r_N[0], r_N[1], 'N', ha='center', va='center', fontsize=12,
        fontweight='bold', color='darkblue', zorder=11)

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
ax.legend(loc='upper right', fontsize=12, framealpha=0.95,
          edgecolor='black', fancybox=True)

# Add text box with information
textstr = f'Lattice parameter: a = {l} Å\n'
textstr += f'Angle: γ = 120°\n'
textstr += f'B-N distance: {np.linalg.norm(r_B - r_N):.4f} Å\n'
textstr += f'Unit cell area: {abs(np.cross(a0, a1)):.3f} Å²\n'
textstr += f'Tiles shown: {(n_tiles_x-1)} × {(n_tiles_y-1)}'

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95,
             edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('hBN_tiling.png', dpi=300, bbox_inches='tight')
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
print(f"\nThe primary unit cell is highlighted in red")
print(f"Each unit cell contains 1 B atom and 1 N atom")