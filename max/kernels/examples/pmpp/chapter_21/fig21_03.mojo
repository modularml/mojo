# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Figure 21.3: Unoptimized DCS C code (CPU implementation).

This is the straightforward triple-nested loop implementation of Direct Coulomb
Summation. For each grid point (i, j), it iterates over all atoms and computes
the electrostatic potential contribution.

This serves as a baseline reference implementation.
"""

from std.math import sqrt

from dcs_utils import GridDim, init_atoms, compute_total_energy


def cenergy_cpu_unoptimized(
    energygrid: UnsafePointer[mut=True, Float32, _],
    grid: GridDim,
    gridspacing: Float32,
    z: Float32,
    atoms: UnsafePointer[mut=False, Float32, _],
    numatoms: Int,
):
    """Unoptimized CPU implementation of DCS energy calculation.

    This function computes electrostatic potential at each grid point by
    iterating over all atoms. The loops are ordered: j (rows), i (cols), atoms.

    Args:
        energygrid: Output energy grid.
        grid: Grid dimensions.
        gridspacing: Grid spacing.
        z: Z-coordinate of the grid slice.
        atoms: Atom array (4 floats per atom: x, y, z, charge).
        numatoms: Number of atoms.
    """
    var atomarrdim = numatoms * 4

    # Compute k index from z coordinate
    var k = Int(z / gridspacing + 0.5)

    for j in range(grid.y):
        var y = gridspacing * Float32(j)

        for i in range(grid.x):
            var x = gridspacing * Float32(i)
            var energy: Float32 = 0.0

            for n in range(0, atomarrdim, 4):
                var dx = x - atoms[n]
                var dy = y - atoms[n + 1]
                var dz = z - atoms[n + 2]
                energy += atoms[n + 3] / sqrt(dx * dx + dy * dy + dz * dz)

            # Store result
            energygrid[grid.x * grid.y * k + grid.x * j + i] = energy


def main() raises:
    print("Figure 21.3: CPU Unoptimized DCS")

    # Setup
    var grid = GridDim(64, 64, 1)  # Single slice for test
    var gridspacing: Float32 = 0.5
    var z_coord: Float32 = 0.0
    var numatoms = 256

    var atoms_size = numatoms * 4
    var grid_size = grid.x * grid.y

    # Allocate memory
    var h_atoms = alloc[Float32](atoms_size)
    var h_energygrid = alloc[Float32](grid_size)

    # Initialize
    init_atoms(h_atoms, numatoms, gridspacing, grid.x, grid.y, 1)
    for i in range(grid_size):
        h_energygrid[i] = 0.0

    # Run computation
    cenergy_cpu_unoptimized(
        h_energygrid, grid, gridspacing, z_coord, h_atoms, numatoms
    )

    print("Figure 21.3 CPU Unoptimized Execution Completed.")

    # Simple check: sum energy
    var total_energy = compute_total_energy(h_energygrid, grid_size)
    print("Total Energy:", total_energy)

    # Cleanup
    h_atoms.free()
    h_energygrid.free()
