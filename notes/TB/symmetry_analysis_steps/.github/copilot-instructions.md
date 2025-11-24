# Copilot Instructions - Symmetry Analysis Steps

## Project Overview
This project documents a **crystallographic symmetry analysis system** using PlantUML diagrams and code examples. The core purpose is to model how atoms in a crystal lattice are identified and indexed using Wyckoff positions and unit cell coordinates.

## Key Concepts & Architecture

### AtomIndex - Immutable Value Object
The fundamental building block is `AtomIndex`, which uniquely identifies an atom in a crystal:
- **Components:** `θ = (k, r, n₀, n₁, n₂)`
  - `k` (wyckoff_position): Which Wyckoff site the atom occupies
  - `r` (atom_number): Which atom at that site (0-indexed)
  - `(n₀, n₁, n₂)`: Unit cell indices (translation in lattice vectors)
- **Key trait:** Immutable after creation - used as identity/key in collections
- **Location:** Documented in `test_atom_only.puml`

### Notation System
- Mathematical notation uses Greek letters and subscripts (e.g., θ, n₀)
- Zeta index (`to_zeta_index`) combines multiple Q-components into a single integer identifier
- Documentation uses clear examples (e.g., "First atom at Wyckoff site 0 in origin cell")

## Working with PlantUML Diagrams
- Store diagrams in `.puml` files with descriptive naming (e.g., `test_atom_only.puml`)
- Use comment headers (`' ============`) to section logical components
- Configure rendering with `skinparam` for consistency (white background, disabled shadows)
- Include notes alongside classes to explain domain concepts
- Use `<<ValueObject>>` stereotype for immutable domain objects

## Development Patterns
- **Value Objects:** Mark immutable types with `<<ValueObject>>` stereotype
- **Notation:** Always define mathematical notation in class diagrams or accompanying notes
- **Examples:** Provide concrete examples in notes (e.g., specific AtomIndex instances)
- **Comments:** Use structured comments with `'` separator for PlantUML organization

## File Organization
- PlantUML diagrams: `*.puml` files in root
- Diagrams should be granular and testable (e.g., individual class diagrams for validation)
- Future code: Likely organized by domain concepts (atoms, symmetry operations, unit cells)

## Recommended Next Steps
When expanding this project:
1. Define related domain objects (Wyckoff positions, SymmetryOperation)
2. Document transformation functions (Q-indices to Zeta conversion)
3. Add class relationship diagrams showing how AtomIndex interacts with other entities
4. Establish code generation or documentation generation from diagrams
